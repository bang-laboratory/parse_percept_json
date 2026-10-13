#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import logging
import pathlib
from functools import cache
from typing import Collection, Optional, Set

import matplotlib.pyplot as plt
import mne
import numpy as np
import polars as pl
import polars.selectors as cs

logger = logging.getLogger(__name__)


def anonymize_file(file_path: pathlib.Path, prefix: str = "Sensitive_") -> pathlib.Path:
    """Creates a copy of the file with sensitive data removed, and moves the
    original file to {prefix}{file_path.name}

    """
    if not file_path.name.startswith("Sensitive_"):
        new_file_path = file_path.parent / f"{prefix}{file_path.name}"
        file_path.rename(new_file_path)
    else:
        new_file_path = file_path
    with open(new_file_path) as json_file:
        sensitive_content = json.load(json_file)
    with open(file_path, "w") as anonymized_file:
        json.dump(anonymize_data(sensitive_content), anonymized_file)
    del sensitive_content
    return file_path


def anonymize_data(data: dict) -> dict:
    """Returns a anonymized copy of the data dict with sensitive fields blanked
    out. Does not delete any sensitive data on disk.

    """
    new = copy.deepcopy(data)
    del data
    # 3 nested loops to rip through:
    # 3 layered dict of fields that should be blanked out
    for k1, v1 in {  # k,v for layer 1
        "PatientInformation": {
            "Initial": [
                "PatientLastName",
                "PatientFirstName",
                "PatientId",
                "PatientDateOfBirth",
                "PatientGender",
                "Diagnosis",
            ],
            "Final": [
                "PatientLastName",
                "PatientFirstName",
                "PatientId",
                "PatientDateOfBirth",
                "PatientGender",
                "Diagnosis",
            ],
        },
        "DeviceInformation": {"Final": ["NeurostimulatorSerialNumber"]},
    }.items():
        for k2, v2 in v1.items():  # k,v for layer 2
            for k3 in v2:  # k for layer 3 (it's just a list of keys)
                new[k1][k2][k3] = ""
    return new


@cache
def read_file(filename: pathlib.Path, anonymize: bool = True) -> dict:
    with open(filename) as json_file:
        data = json.load(json_file)
        if anonymize:
            return anonymize_data(data)
        else:
            return data


def convert_BrainSenseTimeDomain_to_mne(
    dataframe: pl.DataFrame,
    ch_names: Optional[list[str]] = None,
    sfreq: float = 250,
    ch_types: str = "eeg",
) -> mne.io.RawArray:
    """Convert polars dataframe representation, which supports missing data as
    polars "null" values into mne RawArray which crucially does NOT support
    missing data directly. Instead, we interpolate with 0s and add an mne
    Annotation to mark missing data.
    """

    if ch_names is None:
        # get channel names (and implicitly, count) from data
        ch_names = sorted(dataframe.get_column("Channel").unique().to_list())

    ms_per_sample = 1000 / sfreq
    start_time = dataframe.get_column("BlockTimeInterpolatedMs").explode().min()

    # Identify and add gap rows for missing time between blocks
    # Different channels can have the same block_id but different data,
    # so gap detection must be done independently for each channel
    # Get per-channel block boundaries sorted by time
    channel_block_boundaries = (
        dataframe.group_by(["Channel", "block_id"])
        .agg(
            pl.col("BlockTimeInterpolatedMs").explode().min().alias("block_start"),
            pl.col("BlockTimeInterpolatedMs").explode().max().alias("block_end"),
        )
        .sort(["Channel", "block_start"])
    )

    # Calculate gaps between consecutive blocks per channel - need separate steps for polars
    gap_info = channel_block_boundaries.with_columns(
        prev_end=pl.col("block_end").shift(1).over("Channel"),
    )
    gap_info = gap_info.filter(pl.col("block_start") > pl.col("prev_end") + 4)
    gap_info = gap_info.with_columns(
        gap_start=pl.col("prev_end") + 4,
    )
    gap_info = gap_info.with_columns(
        gap_end=pl.col("block_start") - 4,
    )
    gap_info = gap_info.with_columns(
        gap_samples=((pl.col("gap_end") - pl.col("gap_start")) / 4 + 1).cast(pl.Int64),
    )
    gap_info = gap_info.select(["Channel", "gap_start", "gap_end", "gap_samples"])

    # Create gap rows for each channel to ensure all channels get the same number of samples
    # TimeDomainData will be None, and the existing code will fill it with zeros
    all_channels = dataframe.get_column("Channel").unique().to_list()
    gap_rows = []
    for gap_idx, row in enumerate(gap_info.iter_rows(named=True)):
        gap_times = list(range(row["gap_start"], row["gap_end"] + 1, 4))
        channel = row["Channel"]
        gap_rows.append(
            {
                "GlobalSequences": -10000
                - gap_idx * 100,  # Unique negative sequence numbers
                "GlobalPacketSizes": None,
                "TicksInMses": None,
                "Channel": channel,
                "Gain": None,
                "FirstPacketDateTime": None,
                "PacketStartIndex": None,
                "TimeDomainData": None,
                "PacketTimeMs": None,
                "BlockTimeMs": None,
                "GlobalPacketSizesInterpolated": int(row["gap_samples"]),
                "BlockTimeInterpolatedMs": gap_times,
                "block_id": None,
            }
        )

    if gap_rows:
        gap_df = pl.DataFrame(gap_rows)
        # Ensure all columns match the original schema
        for col in dataframe.columns:
            if col in gap_df.columns:
                gap_df = gap_df.with_columns(pl.col(col).cast(dataframe[col].dtype))
        dataframe = pl.concat([dataframe, gap_df])

    missing_data_ms = (
        dataframe.filter(pl.col("GlobalPacketSizes").is_null())
        .unique("BlockTimeInterpolatedMs")
        .explode("BlockTimeInterpolatedMs")
        .with_columns((pl.col("BlockTimeInterpolatedMs") - start_time).alias("onset"))
        .with_columns(
            (
                pl.col("BlockTimeInterpolatedMs").max()
                - pl.col("BlockTimeInterpolatedMs").min()
                + ms_per_sample
            )
            .over("GlobalSequences", "Channel")
            .alias("duration")
        )
        .unique("GlobalSequences", keep="first")
        .select("onset", "duration")
        # .get_column("BlockTimeInterpolatedMs")
    )

    logger.debug(missing_data_ms)

    # because mne assumes an unbroken timeline, we set missing data to value=0
    # and then set an Annotation for the period
    # Use BlockTimeInterpolatedMs for alignment across channels to handle
    # cases where different channels have the same block_id but different time ranges
    dataframe = dataframe.with_columns(
        pl.when(pl.col("GlobalPacketSizes").is_null())
        .then(pl.col("GlobalPacketSizesInterpolated").map_elements(lambda n: [0] * n))
        .otherwise(pl.col("TimeDomainData"))
        .alias("TimeDomainData")
    )

    # Explode BlockTimeInterpolatedMs and TimeDomainData
    data = (
        dataframe.select(["Channel", "BlockTimeInterpolatedMs", "TimeDomainData"])
        .explode("BlockTimeInterpolatedMs", "TimeDomainData")
        .pivot("Channel", values="TimeDomainData")
    )

    # Fill nulls with 0 for missing data
    for col in data.columns:
        if col != "BlockTimeInterpolatedMs":
            data = data.with_columns(pl.col(col).fill_null(0))

    # Sort by BlockTimeInterpolatedMs and select only channel columns
    data = data.sort("BlockTimeInterpolatedMs").select(
        pl.exclude("BlockTimeInterpolatedMs")
    )

    logger.debug(data)

    # annotate missing data in mne
    annots = mne.Annotations(
        onset=missing_data_ms.get_column("onset") / 1000,
        duration=missing_data_ms.get_column("duration") / 1000,
        description=["Missing LFP packet"] * missing_data_ms.height,
    )
    logger.info(annots)

    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    info["description"] = (
        "Missing packets replaced with 0. "
        "Consult annotations labelled "
        "'Missing LFP packet'."
    )
    raw = mne.io.RawArray(data.transpose(), info)
    raw.set_annotations(annots)
    return raw


def convert_BrainSenseLfp_to_mne(
    dataframe: pl.DataFrame,
    ch_names: Optional[list[str]] = None,
    sfreq: Optional[float] = None,
    ch_types: str = "eeg",
) -> mne.io.RawArray:
    """Convert polars dataframe representation of BrainSenseLfp data to mne RawArray.

    Since BrainSenseLfp data is at 2Hz (one sample per packet), and each packet
    already has a timestamp (TicksInMs with 250ms resolution), we use the
    TicksInMs values directly to construct the timeline.

    Missing packets are represented as null rows in the dataframe and will be
    interpolated with 0s in the output, with annotations marking the missing periods.

    Parameters
    ----------
    dataframe : pl.DataFrame
        DataFrame from import_BrainSenseLfp
    ch_names : list[str] | None
        Channel names for MNE. If None, derived from Channel column.
    sfreq : float | None
        Sampling frequency. If None, derived from SampleRateInHz column.
    ch_types : str
        Channel type for MNE (default: "eeg")

    Returns
    -------
    mne.io.RawArray
        MNE RawArray with LFP data
    """
    if ch_names is None:
        # get channel names from data
        ch_names = sorted(dataframe.get_column("Channel").unique().to_list())

    # Get sample rate from data if not provided
    if sfreq is None:
        sample_rates = dataframe["SampleRateInHz"].unique().to_list()
        if len(sample_rates) > 0:
            sfreq = sample_rates[0]
        else:
            sfreq = 2  # Default to 2Hz for BrainSenseLfp

    # Get the start time (minimum TicksInMs)
    start_time_ms = dataframe["TicksInMs"].min()

    # Build annotations for missing data using polars operations
    # Missing packets have null LFP values
    # Identify contiguous blocks of missing packets based on Seq
    # First, add a flag for missing data
    df_with_missing = dataframe.with_columns(
        pl.col("LFP").is_null().alias("is_missing")
    )
    
    # Add a group identifier for contiguous missing sequences
    # Use cumulative sum of transitions from non-missing to missing
    df_with_groups = df_with_missing.with_columns(
        (pl.col("is_missing") & ~pl.col("is_missing").shift(1, fill_value=False)).cum_sum().alias("missing_group_start")
    )
    
    # Filter to only missing rows and group them
    missing_groups = (
        df_with_groups
        .filter(pl.col("is_missing"))
        .group_by("missing_group_start")
        .agg(
            pl.col("Seq").min().alias("min_seq"),
            pl.col("Seq").max().alias("max_seq"),
        )
    )
    
    annotations_list = []
    if missing_groups.height > 0:
        for row in missing_groups.iter_rows(named=True):
            min_seq = row["min_seq"]
            max_seq = row["max_seq"]
            
            # Find the packets immediately before and after this gap
            prev_packet = (
                dataframe
                .filter(
                    (pl.col("Seq") < min_seq) 
                    & (pl.col("LFP").is_not_null())
                )
                .sort("Seq", descending=True)
                .head(1)
            )
            
            next_packet = (
                dataframe
                .filter(
                    (pl.col("Seq") > max_seq) 
                    & (pl.col("LFP").is_not_null())
                )
                .sort("Seq")
                .head(1)
            )
            
            if prev_packet.height > 0 and next_packet.height > 0:
                prev_ticks = prev_packet["TicksInMs"].item()
                next_ticks = next_packet["TicksInMs"].item()
                
                # Estimate the time range
                # At 2Hz, each sample is 500ms apart
                # But TicksInMs has 250ms resolution
                gap_start_ms = float(prev_ticks) + 250  # Half step
                gap_end_ms = float(next_ticks) - 250  # Half step
                gap_duration_ms = gap_end_ms - gap_start_ms
                
                if gap_duration_ms > 0:
                    annotations_list.append({
                        "onset": (gap_start_ms - float(start_time_ms)) / 1000,
                        "duration": gap_duration_ms / 1000,
                        "description": "Missing LFP packet",
                    })
    
    annots = mne.Annotations(
        onset=[a["onset"] for a in annotations_list],
        duration=[a["duration"] for a in annotations_list],
        description=[a["description"] for a in annotations_list],
    )

    # Build the data matrix using polars operations
    # Get min and max TicksInMs to create a complete timeline
    min_ticks = dataframe["TicksInMs"].min()
    max_ticks = dataframe["TicksInMs"].max()
    
    # Create all ticks at 250ms intervals (every 250ms from min to max)
    # TicksInMs has 250ms resolution
    all_ticks = list(range(int(min_ticks), int(max_ticks) + 1, 250))
    
    # Get all channels present in the data
    channels_present = sorted(dataframe["Channel"].unique().to_list())
    
    # Ensure ch_names includes all present channels
    if ch_names is None:
        ch_names = channels_present
    else:
        # Add any present channels not in ch_names
        for ch in channels_present:
            if ch not in ch_names:
                ch_names.append(ch)
        ch_names = sorted(set(ch_names))
    
    # Create a complete grid of (Channel, TicksInMs) combinations
    # This ensures all channels have data at all time points including missing ones
    complete_grid = pl.DataFrame({
        "TicksInMs": all_ticks,
    }).join(
        pl.DataFrame({"Channel": ch_names}),
        how="cross",
    )
    
    # Join with the original data to get LFP values
    # This will have nulls where there's no data (both missing packets and missing ticks)
    data_with_all_ticks = complete_grid.join(
        dataframe.select(["Channel", "TicksInMs", "LFP"]),
        on=["Channel", "TicksInMs"],
        how="left",
    )
    
    # Fill null LFP values with 0
    data_with_all_ticks = data_with_all_ticks.with_columns(
        pl.col("LFP").fill_null(0.0).alias("LFP")
    )
    
    # Pivot to create the data matrix
    # Each column is a channel, rows are time points
    data_pivoted = (
        data_with_all_ticks
        .sort("TicksInMs")
        .pivot(
            values="LFP",
            index="TicksInMs",
            on="Channel",
        )
    )
    
    # Ensure all channels in ch_names are present and in the correct order
    for ch in ch_names:
        if ch not in data_pivoted.columns:
            # Add missing channel column filled with 0
            data_pivoted = data_pivoted.with_columns(
                pl.lit(0.0).alias(ch)
            )
    
    # Reorder columns to match ch_names
    data_pivoted = data_pivoted.select([
        pl.col(ch) for ch in ch_names
    ])
    
    # Extract data as numpy array (transposed because MNE expects channels x time)
    data_matrix = data_pivoted.to_numpy().T
    
    # Create MNE info and RawArray
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    info["description"] = (
        "Missing packets replaced with 0. "
        "Consult annotations labelled "
        "'Missing LFP packet'."
    )
    
    raw = mne.io.RawArray(data_matrix, info)
    raw.set_annotations(annots)
    return raw


def import_BrainSenseTimeDomain(filename: pathlib.Path) -> mne.io.RawArray | None:
    """Read a .json file from a percept system, extract the BrainSenseTimeDomain
    data as an mne raw array with missing data set to 0 and annotated as
    missing.
    """
    dataframe = import_BrainSenseTimeDomain_df(filename)
    if dataframe is not None:
        return convert_BrainSenseTimeDomain_to_mne(dataframe)
    else:
        return None


def reformat_BrainSenseTimeDomain_channelname(BrainSenseTimeDomain, target="LFP"):
    ch = (
        BrainSenseTimeDomain["Channel"]
        .replace("ZERO", "0")
        .replace("ONE", "1")
        .replace("TWO", "2")
        .replace("THREE", "3")
        .replace("FOUR", "4")
        .split("_")
    )
    return target + ch[2][0] + ch[0] + ch[1]


def _calc_BlockTimeMs(
    data_frame: pl.DataFrame,
    ms_per_sample: float,
    col_name_result: str,
    col_name_timestamp_start: str = "TicksInMses",
    col_name_timestamp_block: str = "PacketTimeMs",
    col_name_packet_size: str = "GlobalPacketSizes",
) -> pl.DataFrame:
    """Construct timeline that respects packet size. Use iter_rows() and native
    python instead of polars here because of the serial dependency on the
    previous row.

    """
    result = [None] * data_frame.height

    for i, row in enumerate(data_frame.iter_rows(named=True)):
        if i == 0:
            # first packet in block
            start_time = row[col_name_timestamp_start]
        else:
            # subsequent packets in block
            start_time = result[i - 1][-1] + int(ms_per_sample)
        result[i] = [
            int(start_time + x * ms_per_sample)
            for x in range(0, row[col_name_packet_size])
        ]
        # print(result[i], row[col_name_packet_size])
        assert len(result[i]) == row[col_name_packet_size]

    return data_frame.with_columns(
        pl.Series(col_name_result, values=result, dtype=pl.List(pl.Int64))
    )


def _process_BrainSenseTimeDomainBlock(
    raw_data: dict,
    known_accounted_for_packets: Collection[int] | None = None,
    block_id: int = 0,
) -> pl.DataFrame:
    """These data files can contain multiple blocks that are not guaranteed to
    be continuous

    They also contain packet-level metadata about sequence (used to check for
    missing packets), lengths, and timings. The packet timestamps claim a 50ms
    resolution in the manual, but in practice show only 250ms resolution.

    Input: a dict containing one block of BrainSenseTimeDomain data
    Output: A polars dataframe with one row per packet (not per sample; not per block)

    known_accounted_for_packets: the packets that are accounted for in other
    parts of the data and thus should not be considered missing

    """

    # one long sequence of samples that has already been reconstructed from the packets by the device
    # we index into it below to reconstruct timings using the packet-level data
    samples = pl.Series("sample", raw_data["TimeDomainData"])
    samples_per_ms = raw_data["SampleRateInHz"] / 1000
    ms_per_sample = 1 / samples_per_ms

    if known_accounted_for_packets is None:
        known_accounted_for_packets = set()
    else:
        known_accounted_for_packets = set(known_accounted_for_packets)

    data_frame = (
        pl.from_dict(raw_data)
        # unwrap lists of numbers disguised as strings
        .select(
            # split string col like "[1,2,3]" into polars list for all these 3 cols
            pl.col("GlobalSequences")
            .first()  # TODO: do I really need these first() calls
            .str.strip_suffix(",")
            .str.split(",")
            .list.eval(pl.element().str.to_integer()),
            pl.col("GlobalPacketSizes")
            .first()
            .str.strip_suffix(",")
            .str.split(",")
            .list.eval(pl.element().str.to_integer()),
            pl.col("TicksInMses")
            .first()
            .str.strip_suffix(",")
            .str.split(",")
            .list.eval(pl.element().str.to_integer()),
            pl.lit(reformat_BrainSenseTimeDomain_channelname(raw_data)).alias(
                "Channel"
            ),
            pl.col("Gain").first(),
            pl.col("FirstPacketDateTime").first(),
        )
        # compute the cumulative sum of packet sizes to get the slice indices into the packet data stream
        .with_columns(
            pl.col("GlobalPacketSizes")
            .list.eval(pl.element().cum_sum().shift(1, fill_value=0))
            .alias("PacketStartIndex")
        )
        # unwrap the "per-packet" data, but don't unwrap to per-sample level
        .explode(
            "PacketStartIndex", "GlobalPacketSizes", "GlobalSequences", "TicksInMses"
        )
        .with_columns(
            # map (≃ python zip) over the two columns to index into the samples
            pl.struct(["PacketStartIndex", "GlobalPacketSizes"])
            .map_elements(
                lambda row: samples.slice(
                    row["PacketStartIndex"], row["GlobalPacketSizes"]
                ),
                return_dtype=pl.List(pl.Float64),
            )
            .alias("TimeDomainData")
        )
        .with_columns(
            (
                # Construct timeline that is valid within packets, but not necessarily across packets
                pl.col("TicksInMses").min()
                + pl.int_ranges(pl.col("TimeDomainData").list.len()) / samples_per_ms
            )
            .cast(pl.List(pl.Int64))
            .over("GlobalSequences")
            .alias("PacketTimeMs"),
        )
    )

    # Construct timeline that respects packet size but not missing packets.
    # Use iter_rows() and native python instead of polars here because of the
    # serial dependency on the previous row.
    data_frame = _calc_BlockTimeMs(
        data_frame,
        ms_per_sample,
        "BlockTimeMs",
        "TicksInMses",
        "PacketTimeMs",
        "GlobalPacketSizes",
    )

    # Check for missing packets
    packets_found: set[int] = set(data_frame["GlobalSequences"])

    MODULO = 2**16  # GlobalSequences is a 16 bit integer
    HALF_RANGE = MODULO // 2
    packets_missing: set[int] = set()
    sequence_list = data_frame["GlobalSequences"].to_list()
    for prev, curr in zip(sequence_list[:-1], sequence_list[1:]):
        delta = (curr - prev) % MODULO  # handle integer overflows in GlobalSequence

        if delta == 1:
            continue

        if 1 < delta <= HALF_RANGE:
            # Missing packets
            for seq in range(1, delta):
                packets_missing.add((prev + seq) % MODULO)
        elif delta == 0:
            logger.error(f"Duplicate packet detected: {curr}")
            continue
        else:
            # delta > 32768
            logger.warning(
                f"Ignoring backwards sequence jump: prev={prev}, curr={curr}, delta={delta}"
            )

    packets_missing -= known_accounted_for_packets

    if len(packets_missing) == 0:
        logger.info(f"Missing Packets: 0")
    else:
        logger.warning(
            f"Missing Packets ({len(packets_missing)}): {sorted(packets_missing)}"
        )

    # explicitly represent missing packets as null rows (polars missingness)
    # found_or_missing = list(packets_found | packets_missing)
    channels_found_this_block = (
        data_frame.unique("Channel").get_column("Channel").to_list()
    )
    logger.debug(packets_missing)
    if len(packets_missing) > 0:
        full_sequence = pl.DataFrame(
            {
                "GlobalSequences": list(packets_missing),
                "Channel": channels_found_this_block * len(packets_missing),
            }
        )
        #  print(full_sequence)
        data_frame = data_frame.join(
            full_sequence, on=["GlobalSequences", "Channel"], how="full", coalesce=True
        ).sort("GlobalSequences")

    # Empirically, the number of samples per packet stabilizes to a 2-long
    # repeating sequence, ie 62 63 62 63 for 250Hz BrainSenseTimedomain data.
    #
    # As a heuristic for missing packets, we can copy a packet size from 2
    # packets back to capture this, and use it to estimate when the timeline
    # resumes. This could be helpful if the packets are dropped during
    # transport, but is probably not accurate if the packets are dropped due to
    # processing load on the implant.
    data_frame = forward_fill_nulls_shifted(
        data_frame, "GlobalPacketSizes", "GlobalPacketSizesInterpolated", 2
    )

    data_frame = data_frame.with_columns(
        pl.when(pl.col("TimeDomainData").is_null())
        .then(
            pl.col("GlobalPacketSizesInterpolated").map_elements(
                lambda n: [None] * n,
                return_dtype=pl.List(pl.Float64),
            )
        )
        .otherwise(pl.col("TimeDomainData"))
        .alias("TimeDomainData")
    )

    data_frame = _calc_BlockTimeMs(
        data_frame,
        ms_per_sample,
        "BlockTimeInterpolatedMs",
        "TicksInMses",
        "PacketTimeMs",
        "GlobalPacketSizesInterpolated",
    )

    # Add block_id to the dataframe before returning
    return data_frame.with_columns(pl.lit(block_id).alias("block_id"))


def forward_fill_nulls_shifted(
    data_frame: pl.DataFrame,
    from_col: str = "GlobalPacketSizes",
    to_col: str = "GlobalPacketSizesInterpolated",
    shift: int = 2,
) -> pl.DataFrame:
    """combine functionality of forward_fill and shift(2)"""
    result = [None] * data_frame.height

    for i, row in enumerate(data_frame.iter_rows(named=True)):
        if row[from_col] is None and i >= shift:
            result[i] = result[i - shift]
        else:
            result[i] = row[from_col]

    return data_frame.with_columns(pl.Series(to_col, values=result, dtype=pl.Int64))


def _get_LfpData_sequences(data: dict, filename) -> Set | None:
    """Extract the packet sequence numbers from the BrainSenseLfp -> LfpData
    part of the data. These packets contain the 2hz data stream showed on the
    tablet, and the sequence numbers are interleaved with the packet sequence
    for BrainSenseTimeDomain data.

    Input: a dict representing a whole percept json export file
    Output: a set of all the sequence numbers accounted for in the BrainSenseLfp
    data

    """
    if (
        "BrainSenseLfp" in data.keys()
        and len(data["BrainSenseLfp"]) > 0
        and "LfpData" in data["BrainSenseLfp"][0].keys()
    ):
        return set(
            packet["Seq"]
            for block in data["BrainSenseLfp"]
            for packet in block["LfpData"]
        )

    else:
        logger.warning(f"LfpData not found for {filename}")
        return None


def _get_TimeDomainData_sequences(data: dict, filename) -> Set | None:
    """Extract the packet sequence numbers from the BrainSenseTimeDomain data.
    These packets contain the 250Hz time domain data, and the sequence numbers
    are interleaved with the packet sequence for BrainSenseLfp data.

    Input: a dict representing a whole percept json export file
    Output: a set of all the sequence numbers accounted for in the BrainSenseTimeDomain
    data

    """
    if (
        "BrainSenseTimeDomain" in data.keys()
        and len(data["BrainSenseTimeDomain"]) > 0
    ):
        sequences = set()
        for block in data["BrainSenseTimeDomain"]:
            if "GlobalSequences" in block:
                # GlobalSequences can be a string like "[1,2,3]" or already a list
                seq_str = block["GlobalSequences"]
                if isinstance(seq_str, str):
                    # Parse string representation of list using vanilla python
                    # Remove brackets and split by commas
                    seq_str = seq_str.strip("[]")
                    sequences.update(int(s) for s in (x.strip() for x in seq_str.split(",")) if s)
                elif isinstance(seq_str, list):
                    sequences.update(seq_str)
        return sequences

    else:
        logger.warning(f"BrainSenseTimeDomain not found for {filename}")
        return None


def _process_BrainSenseLfpBlock(
    raw_data: dict,
    known_accounted_for_packets: Collection[int] | None = None,
    block_id: int = 0,
) -> pl.DataFrame:
    """Process a single BrainSenseLfp block to extract LfpData with missing packet detection.

    These data files can contain multiple blocks that are not guaranteed to
    be continuous. Each LfpData packet contains one sample per channel (Left/Right).

    Note: The LfpData entries appear to have duplicate Seq values with different
    TicksInMs. This suggests each Seq represents a "packet" that can contain multiple
    samples (one per hemisphere). We keep all entries and detect missing Seq values.

    Input: a dict containing one block of BrainSenseLfp data
    Output: A polars dataframe with one row per LfpData entry, or None if no data

    known_accounted_for_packets: the packets that are accounted for in other
    parts of the data and thus should not be considered missing

    """
    if known_accounted_for_packets is None:
        known_accounted_for_packets = set()
    else:
        known_accounted_for_packets = set(known_accounted_for_packets)

    sample_rate = raw_data.get("SampleRateInHz", 2)
    block_channel = raw_data.get("Channel", None)
    first_packet_datetime = raw_data.get("FirstPacketDateTime", None)

    # Extract LfpData entries
    lfp_data = raw_data.get("LfpData", [])
    if len(lfp_data) == 0:
        return None

    # Build dataframe from LfpData using pl.from_dicts with schema
    data_frame = (
        pl.from_dicts(
            lfp_data,
            schema={
                "Seq": pl.Int64,
                "TicksInMs": pl.Int64,
                "StatusBytes": pl.String,
                "Left": pl.Struct({"LFP": pl.Float64, "mA": pl.Float64}),
                "Right": pl.Struct({"LFP": pl.Float64, "mA": pl.Float64}),
            },
        )
        # Add block-level metadata
        .with_columns(
            pl.lit(block_channel).alias("Channel"),
            pl.lit(first_packet_datetime).alias("FirstPacketDateTime"),
            pl.lit(float(sample_rate)).alias("SampleRateInHz"),
            pl.lit(block_id).alias("block_id"),
        )
    )

    # Extract LFP and mA from Left/Right based on block_channel
    # In the data, only one of Left or Right is ever non-zero, matching the block's Channel
    if block_channel is not None and "LEFT" in block_channel:
        data_frame = data_frame.with_columns(
            pl.col("Left").struct["LFP"].alias("LFP"),
            pl.col("Left").struct["mA"].alias("mA"),
        )
    elif block_channel is not None and "RIGHT" in block_channel:
        data_frame = data_frame.with_columns(
            pl.col("Right").struct["LFP"].alias("LFP"),
            pl.col("Right").struct["mA"].alias("mA"),
        )
    else:
        # If channel is not specified, use Left (default)
        data_frame = data_frame.with_columns(
            pl.col("Left").struct["LFP"].alias("LFP"),
            pl.col("Left").struct["mA"].alias("mA"),
        )

    # Select final columns (including block_id)
    data_frame = data_frame.select([
        "Seq",
        "TicksInMs",
        "StatusBytes",
        "LFP",
        "mA",
        "Channel",
        "FirstPacketDateTime",
        "SampleRateInHz",
        "block_id",
    ])

    # Check for missing packets based on Seq values
    # Get all unique Seq values (each Seq can have multiple entries for different hemispheres)
    packets_found: set[int] = set(data_frame["Seq"].drop_nulls().to_list())

    if len(packets_found) == 0:
        # No data at all
        return None

    MODULO = 2**16  # Seq is a 16 bit integer
    HALF_RANGE = MODULO // 2
    packets_missing: set[int] = set()

    sorted_seqs = sorted(packets_found)

    for prev, curr in zip(sorted_seqs[:-1], sorted_seqs[1:]):
        delta = (curr - prev) % MODULO  # handle integer overflows in Seq

        if delta == 1:
            continue

        if 1 < delta <= HALF_RANGE:
            # Missing packets
            for seq in range(1, delta):
                packets_missing.add((prev + seq) % MODULO)
        elif delta == 0:
            # This can happen if we have duplicate Seq values (multiple entries per Seq)
            # This is expected for BrainSenseLfp, so don't log as error
            continue
        else:
            # delta > 32768
            logger.warning(
                f"Ignoring backwards sequence jump: prev={prev}, curr={curr}, delta={delta}"
            )

    packets_missing -= known_accounted_for_packets

    if len(packets_missing) == 0:
        logger.info(f"Missing Packets: 0")
    else:
        logger.warning(
            f"Missing Packets ({len(packets_missing)}): {sorted(packets_missing)}"
        )

    # Explicitly represent missing packets as null rows using polars operations
    if len(packets_missing) > 0:
        channels_found = data_frame["Channel"].unique().to_list()
        
        # Create missing packet rows
        missing_seq_df = pl.DataFrame({
            "Seq": sorted(packets_missing),
        })
        
        # Cross join with channels
        missing_df = missing_seq_df.join(
            pl.DataFrame({"Channel": channels_found}),
            how="cross",
        )
        
        # Add metadata columns with null values
        missing_df = missing_df.with_columns(
            pl.lit(None).cast(pl.Int64).alias("TicksInMs"),
            pl.lit(None).cast(pl.String).alias("StatusBytes"),
            pl.lit(None).cast(pl.Float64).alias("LFP"),
            pl.lit(None).cast(pl.Float64).alias("mA"),
            pl.lit(first_packet_datetime).alias("FirstPacketDateTime"),
            pl.lit(float(sample_rate)).alias("SampleRateInHz"),
            pl.lit(block_id).alias("block_id"),
        )
        
        # Reorder columns to match data_frame
        missing_df = missing_df.select(data_frame.columns)
        
        # Concatenate with original data
        data_frame = pl.concat([data_frame, missing_df])

    # Sort by TicksInMs (time order), then by Seq
    data_frame = data_frame.sort("TicksInMs")
    
    return data_frame


def import_BrainSenseTimeDomain_df(filename: pathlib.Path) -> pl.DataFrame | None:
    """Extract the BrainSenseTimeDomain data from a percept json file"""
    json_data = read_file(filename)
    lfddata_packet_sequences = _get_LfpData_sequences(json_data, filename.name)

    if "BrainSenseTimeDomain" in json_data.keys():
        return pl.concat(
            _process_BrainSenseTimeDomainBlock(
                block, lfddata_packet_sequences, block_id=i
            )
            for i, block in enumerate(json_data["BrainSenseTimeDomain"])
        )

    else:
        logger.warning(f"BrainSenseTimeDomain not found for {filename.name}")
        return None


def import_LfpTrendLogs(filename: pathlib.Path, tz: str = "UTC") -> pl.DataFrame | None:
    json_data = read_file(filename)
    if "DiagnosticData" in json_data and "LFPTrendLogs" in json_data["DiagnosticData"]:
        return (
            pl.from_dict(json_data["DiagnosticData"]["LFPTrendLogs"])
            .unpivot(
                on=cs.starts_with("Hemisphere"),
                variable_name="Hemisphere",
                value_name="block",
            )
            .unnest("block")
            .unpivot(index="Hemisphere", variable_name="TimeStart", value_name="Sample")
            .explode("Sample")
            .unnest("Sample")
            .select(
                pl.col("Hemisphere"),
                pl.col("DateTime").str.to_datetime(time_zone=tz),
                pl.col("LFP"),
                pl.col("AmplitudeInMilliAmps"),
            )
        )
    else:
        logger.warning("LFPTrendLogs not found")
        return None


def import_BrainSenseLfp(
    filename: pathlib.Path, tz: str = "UTC"
) -> pl.DataFrame | None:
    """Extract the BrainSenseLfp data from a percept json file.

    Returns a polars DataFrame with one row per LfpData packet, with columns:
    - Seq: Sequence number
    - TicksInMs: Timestamp in milliseconds
    - StatusBytes: Status bytes string
    - LFP: LFP value
    - mA: Current value in mA
    - Channel: Channel name
    - FirstPacketDateTime: First packet datetime for the block
    - SampleRateInHz: Sample rate in Hz
    - block_id: Block identifier
    """
    json_data = read_file(filename)

    if (
        "BrainSenseLfp" in json_data
        and len(json_data["BrainSenseLfp"]) > 0
        and "LfpData" in json_data["BrainSenseLfp"][0]
    ):
        # Get sequences from TimeDomain data to exclude from missing packet detection
        time_domain_sequences = _get_TimeDomainData_sequences(json_data, filename.name)

        return pl.concat(
            _process_BrainSenseLfpBlock(
                block,
                known_accounted_for_packets=time_domain_sequences,
                block_id=i,
            )
            for i, block in enumerate(json_data["BrainSenseLfp"])
        )
    else:
        logger.warning("BrainSenseLfp not found")
        return None


def import_BrainSenseLfp_mne(
    filename: pathlib.Path, tz: str = "UTC"
) -> mne.io.RawArray | None:
    """Extract the BrainSenseLfp data from a percept json file and convert to MNE RawArray.

    This is a convenience function that combines import_BrainSenseLfp and
    convert_BrainSenseLfp_to_mne.

    Parameters
    ----------
    filename : pathlib.Path
        Path to the JSON file
    tz : str
        Timezone for datetime parsing (default: "UTC")

    Returns
    -------
    mne.io.RawArray | None
        MNE RawArray with LFP data, or None if BrainSenseLfp not found
    """
    dataframe = import_BrainSenseLfp(filename, tz=tz)
    if dataframe is not None:
        return convert_BrainSenseLfp_to_mne(dataframe)
    else:
        return None


def import_LfpFrequencySnapshotEvents(
    filename: pathlib.Path,  # , tz: str = "UTC"
) -> pl.DataFrame | None:
    json_data = read_file(filename)
    if (
        "DiagnosticData" in json_data
        and "LfpFrequencySnapshotEvents" in json_data["DiagnosticData"]
    ):
        return (
            pl.from_dicts(json_data["DiagnosticData"]["LfpFrequencySnapshotEvents"])
            .unnest("LfpFrequencySnapshotEvents")
            .unpivot(
                on=cs.starts_with("Hemisphere"),
                index=~cs.starts_with("Hemisphere"),
                variable_name="Hemisphere",
                value_name="data",
            )
            .rename({"DateTime": "StartTime"})
            .unnest("data")
            .explode("FFTBinData", "Frequency")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test parsing DBS sensing data")
    parser.add_argument(
        "--sensitive_test_file_path",
        help="Path to json data file",
        type=pathlib.Path,
        default=pathlib.Path("tests/test_data_1.json"),
    )
    args = parser.parse_args()

    test_file_path = anonymize_file(args.sensitive_test_file_path)
    json_data = read_file(test_file_path)
    brainsense_df_data = import_BrainSenseTimeDomain_df(test_file_path)
    assert brainsense_df_data is not None
    mne_data = convert_BrainSenseTimeDomain_to_mne(brainsense_df_data)
    fig = mne_data.plot(scalings="auto")
    plt.show(block=True)
    # fig.show()

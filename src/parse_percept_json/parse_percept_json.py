#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import logging
import pathlib
from functools import cache
from typing import Collection, Optional, Set

import mne
import polars as pl
import polars.selectors as cs

logger = logging.getLogger(__name__)

"""
TODO: rename module, don't use - and _
"""


def anonymize_file(file_path: pathlib.Path, prefix: str = "Sensitive_") -> pathlib.Path:
    """Creates a copy of the file with sensitive data removed, and moves the
    original file to {prefix}{file_path.name}

    """
    new_file_path = file_path.parent / f"{prefix}{file_path.name}"
    file_path.rename(new_file_path)
    with open(new_file_path) as json_file:
        with open(file_path, "w") as anonymized_file:
            json.dump(anonymize_data(json.load(json_file)), anonymized_file)
    return file_path


def anonymize_data(data: dict) -> dict:
    """Returns a anonymized copy of the data dict with sensitive fields blanked
    out. Does not delete any sensitive data on disk.

    """
    new = copy.deepcopy(data)
    del data
    # 3 layered dict of fields that should be blanked out
    for k1, v1 in {  # k,v for layer 1
        "PatientInformation": {
            "Final": [
                "PatientLastName",
                "PatientFirstName",
                "PatientId",
                "PatientDateOfBirth",
                "PatientGender",
            ]
        },
        "DeviceInformation": {"Final": ["NeurostimulatorSerialNumber"]},
    }.items():
        for k2, v2 in v1.items():  # k,v for layer 2
            for k3 in v2:  # v for layer 3 (it's just a list of keys)
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
    dataframe: pl.DataFrame, ch_names=["LFPL02", "LFPR02"], sfreq=250, ch_types="eeg"
) -> mne.io.RawArray:

    # ms_per_sample = 1 / sfreq * 1000
    start_time = dataframe.get_column("BlockTimeInterpolatedMs").explode().min()

    missing_data_ms = (
        dataframe.filter(pl.col("TimeDomainData").is_null())
        .unique("BlockTimeInterpolatedMs")
        .explode("BlockTimeInterpolatedMs")
        .with_columns((pl.col("BlockTimeInterpolatedMs") - start_time).alias("onset"))
        .with_columns(
            (
                pl.col("BlockTimeInterpolatedMs").max()
                - pl.col("BlockTimeInterpolatedMs").min()
            )
            .over("GlobalSequences", "Channel")
            .alias("duration")
        )
        .unique("GlobalSequences", keep="first")
        .select("onset", "duration")
        # .get_column("BlockTimeInterpolatedMs")
    )

    logger.debug(missing_data_ms)

    # because mne assumpes an unbroken timeline, we set missing data to value=0
    # and then set an Annotation for the period
    data = (
        dataframe.with_columns(
            pl.when(pl.col("TimeDomainData").is_null())
            .then(
                pl.col("GlobalPacketSizesInterpolated").map_elements(lambda n: [0] * n)
            )
            .otherwise(pl.col("TimeDomainData"))
            .alias("TimeDomainData")
        )
        .select(["GlobalSequences", "Channel", "TimeDomainData"])
        .pivot("Channel", values="TimeDomainData")
        .explode(pl.exclude("GlobalSequences"))
        .select(pl.exclude("GlobalSequences"))
        # .explode("TimeDomainData")
    )

    logger.debug(data)

    annots = mne.Annotations(
        onset=missing_data_ms.get_column("onset") / 1000,
        duration=missing_data_ms.get_column("duration") / 1000,
        description=["Missing LFP packet"] * missing_data_ms.height,
    )
    logger.info(annots)
    # annotate missing data in mne

    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = mne.io.RawArray(data.transpose(), info)
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
    prev_row = None
    result = [None] * data_frame.height

    for i, row in enumerate(data_frame.iter_rows(named=True)):
        if prev_row is None:
            # first packet in block
            start_time = row[col_name_timestamp_start]
        else:
            # subsequent packets in block
            start_time = result[i - 1][-1] + int(ms_per_sample)
        result[i] = [
            int(start_time + x * ms_per_sample)
            for x in range(0, row[col_name_packet_size])
        ]
        prev_row = row  # store pointer to previous row

    return data_frame.with_columns(
        pl.Series(col_name_result, values=result, dtype=pl.List(pl.Int64))
    )


def _process_BrainSenseTimeDomainBlock(
    raw_data: dict, known_accounted_for_packets: Collection[int] | None = None
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

    # one long sequence of samples that has already been reconstructed from the packets
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
    packets_found = set(data_frame["GlobalSequences"])
    packets_implied = set(
        range(
            data_frame["GlobalSequences"].min(),  # ty: ignore[invalid-argument-type]
            1 + data_frame["GlobalSequences"].max(),  # ty: ignore[unsupported-operator]
        )
    )
    packets_missing = packets_implied - (packets_found | known_accounted_for_packets)
    if len(packets_missing) == 0:
        logger.info(
            f"Missing Packets ({len(packets_missing)}/{len(packets_implied)}): {sorted(packets_missing)}"
        )
    else:
        logger.warning(
            f"Missing Packets ({len(packets_missing)}/{len(packets_implied)}): {sorted(packets_missing)}"
        )

    # explicitly represent missing packets as null rows (polars missingness)
    # found_or_missing = list(packets_found | packets_missing)
    channels_found_this_block = (
        data_frame.unique("Channel").get_column("Channel").to_list()
    )
    # print(channels_found_this_block)
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
    data_frame = data_frame.with_columns(
        pl.when(pl.col("GlobalPacketSizes").is_null())
        .then(pl.col("GlobalPacketSizes").shift(2))
        .otherwise(pl.col("GlobalPacketSizes"))
        .alias("GlobalPacketSizesInterpolated")
    )

    data_frame = _calc_BlockTimeMs(
        data_frame,
        ms_per_sample,
        "BlockTimeInterpolatedMs",
        "TicksInMses",
        "PacketTimeMs",
        "GlobalPacketSizesInterpolated",
    )

    return data_frame


def _get_LfpData_sequences(data: dict) -> Set | None:
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
        logger.warning("LfpData not found")
        return None


def import_BrainSenseTimeDomain_df(filename: pathlib.Path) -> pl.DataFrame | None:
    """Extract the BrainSenseTimeDomain data from a percept json file"""
    json_data = read_file(filename)
    lfddata_packet_sequences = _get_LfpData_sequences(json_data)

    if "BrainSenseTimeDomain" in json_data.keys():
        return pl.concat(
            _process_BrainSenseTimeDomainBlock(block, lfddata_packet_sequences)
            for block in json_data["BrainSenseTimeDomain"]
        )

    else:
        logger.warning("BrainSenseTimeDomain not found")
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
        "sensitive_test_file_path", help="Path to json data file", type=pathlib.Path
    )
    args = parser.parse_args()

    test_file_path = anonymize_file(args.sensitive_test_file_path)
    json_data = read_file(test_file_path)
    brainsense_df_data = import_BrainSenseTimeDomain_df(test_file_path)
    print(brainsense_df_data)

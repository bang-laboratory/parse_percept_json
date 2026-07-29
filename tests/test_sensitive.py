#!/usr/bin/env python3

from pathlib import Path

import polars as pl
from src.parse_percept_json.parse_percept_json import (
    _process_BrainSenseTimeDomainBlock,
    anonymize_data,
    convert_BrainSenseTimeDomain_to_mne,
    forward_fill_nulls_shifted,
    import_BrainSenseTimeDomain,
    import_BrainSenseTimeDomain_df,
    import_LfpFrequencySnapshotEvents,
    import_LfpTrendLogs,
    read_file,
    reformat_BrainSenseTimeDomain_channelname,
)

testfile = Path("tests/Sensitive_Report_Json_Session_Report_20251028T170416.json")


def test_loading_missing():
    if not testfile.exists():
        return

    data = import_BrainSenseTimeDomain_df(testfile)
    assert data.height == 130148


def test_sample_count_equals_interpolated_packet_sizes():
    if not testfile.exists():
        return
    df = import_BrainSenseTimeDomain_df(testfile)

    expected = df["GlobalPacketSizesInterpolated"].fill_null(0).sum()

    observed = df.explode("TimeDomainData").height

    assert observed == expected


def test_no_missing_packets_no_null_samples():
    if not testfile.exists():
        return
    json_data = read_file(testfile)
    df = _process_BrainSenseTimeDomainBlock(json_data["BrainSenseTimeDomain"][0])

    assert df.get_column("TimeDomainData").is_null().sum() == 0


def test_import_BrainSenseTimeDomain_df():
    if not testfile.exists():
        return
    data = import_BrainSenseTimeDomain_df(testfile)

    assert type(data) is pl.DataFrame
    # assert data.height == 239
    assert set(data.columns) == {
        "GlobalSequences",
        "GlobalPacketSizes",
        "TicksInMses",
        "Channel",
        "Gain",
        "FirstPacketDateTime",
        "PacketStartIndex",
        "TimeDomainData",
        "PacketTimeMs",
        "BlockTimeMs",
        "GlobalPacketSizesInterpolated",
        "BlockTimeInterpolatedMs",
    }

    # assert data.explode("TimeDomainData").height == data.explode("BlockTimeMs").height
    assert (
        data.explode("TimeDomainData").height
        == data.explode("BlockTimeInterpolatedMs").height
    )
    # sample_level = data.explode("TimeDomainData", "BlockTimeInterpolatedMs")
    # assert sample_level.height == 29875
    # assert sum(sample_level[col] for col in sample_level.)

    missing = data.filter(pl.col("GlobalPacketSizes").is_null())

    assert missing.height > 0

    for row in missing.iter_rows(named=True):
        assert row["TimeDomainData"] is not None
        assert len(row["TimeDomainData"]) == row["GlobalPacketSizesInterpolated"]
        assert all(x is None for x in row["TimeDomainData"])

#!/usr/bin/env python3

import pathlib

import polars as pl
from src.parse_percept_json.parse_percept_json import (
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


def test_anonymize_data():
    d = read_file("tests/test_data_1.json", anonymize=False)

    # test data has (fake) patient info
    assert d["PatientInformation"]["Initial"]["PatientFirstName"] == "lars"

    anon_d = anonymize_data(d)
    assert anon_d["PatientInformation"]["Initial"]["PatientFirstName"] == ""


def test_import_BrainSenseTimeDomain_df():
    data = import_BrainSenseTimeDomain_df(pathlib.Path("tests/test_data_1.json"))

    assert type(data) is pl.DataFrame
    assert data.height == 239
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

    assert data.explode("TimeDomainData").height == data.explode("BlockTimeMs").height
    # assert (
    #     data.explode("TimeDomainData").height
    #     == data.explode("BlockTimeInterpolatedMs").height
    # )
    # sample_level = data.explode("TimeDomainData")
    # assert sample_level.height == 29875
    # assert not sample_level.select(pl.col("BlockTimeInterpolatedMs")) # TODO

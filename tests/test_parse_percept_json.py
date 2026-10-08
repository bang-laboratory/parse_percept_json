#!/usr/bin/env python3

import pathlib

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

testfiles = [
    pathlib.Path("tests/test_data_1.json"),
    pathlib.Path("tests/Report_Json_Session_Report_20250218T092621.json"),
    pathlib.Path("tests/Report_Json_Session_Report_20250226T084029.json"),
]


def test_anonymize_data():
    # Only test_data_1.json has fake patient info
    testfile = pathlib.Path("tests/test_data_1.json")
    d = read_file(testfile, anonymize=False)

    # test data has (fake) patient info
    assert d["PatientInformation"]["Initial"]["PatientFirstName"] == "lars"

    anon_d = anonymize_data(d)
    assert anon_d["PatientInformation"]["Initial"]["PatientFirstName"] == ""


def test_import_BrainSenseTimeDomain_df():
    for testfile in testfiles:
        data = import_BrainSenseTimeDomain_df(testfile)
        assert type(data) is pl.DataFrame
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
            "block_id",
        }

        # assert data.explode("TimeDomainData").height == data.explode("BlockTimeMs").height
        assert (
            data.explode("TimeDomainData").height
            == data.explode("BlockTimeInterpolatedMs").height
        )

        missing = data.filter(pl.col("GlobalPacketSizes").is_null())

        for row in missing.iter_rows(named=True):
            assert row["TimeDomainData"] is not None
            assert len(row["TimeDomainData"]) == row["GlobalPacketSizesInterpolated"]
            assert all(x is None for x in row["TimeDomainData"])


def test_import_BrainSenseTimeDomain_df_test_data_1():
    # Test file-specific assertions for test_data_1.json
    testfile = pathlib.Path("tests/test_data_1.json")
    data = import_BrainSenseTimeDomain_df(testfile)

    assert data.height == 239
    sample_level = data.explode("TimeDomainData", "BlockTimeInterpolatedMs")
    assert sample_level.height == 29875
    assert data.filter(pl.col("GlobalPacketSizes").is_null()).height > 0


def test_forward_fill_nulls_shifted_uses_two_back_value():
    df = pl.DataFrame({"GlobalPacketSizes": [62, 63, None, None]})

    result = forward_fill_nulls_shifted(df)

    assert result["GlobalPacketSizesInterpolated"].to_list() == [
        62,
        63,
        62,
        63,
    ]


def test_no_missing_packets_no_null_samples():
    for testfile in testfiles:
        json_data = read_file(testfile)
        df = _process_BrainSenseTimeDomainBlock(json_data["BrainSenseTimeDomain"][0])

        assert df.get_column("TimeDomainData").is_null().sum() == 0


def test_sample_count_equals_interpolated_packet_sizes():
    for testfile in testfiles:
        df = import_BrainSenseTimeDomain_df(testfile)

        expected = df["GlobalPacketSizesInterpolated"].fill_null(0).sum()

        observed = df.explode("TimeDomainData").height

        assert observed == expected


# def test_interpolated_timeline_has_no_jump():
#     df = import_BrainSenseTimeDomain_df(pathlib.Path("tests/test_data_1.json"))

#     times = (
#         df.explode("BlockTimeInterpolatedMs")
#         .sort("BlockTimeInterpolatedMs")
#         .get_column("BlockTimeInterpolatedMs")
#         .to_list()
#     )

#     diffs = [b - a for a, b in zip(times[:-1], times[1:])]


# def test_mne_length_matches_dataframe():
#     df = import_BrainSenseTimeDomain_df(pathlib.Path("tests/test_data_1.json"))
#     assert df is not None

#     raw = convert_BrainSenseTimeDomain_to_mne(df)

#     expected = df.explode("TimeDomainData").height

#     assert raw.n_times == expected


def test_missing_packets_contribute_expected_interpolated_samples():
    for testfile in testfiles:
        df = import_BrainSenseTimeDomain_df(testfile)
        assert df is not None

        missing_packets = df.filter(pl.col("GlobalPacketSizes").is_null())

        expected_missing_samples = (
            missing_packets["GlobalPacketSizesInterpolated"].fill_null(0).sum()
        )

        observed_missing_samples = (
            missing_packets.explode("TimeDomainData")
            .filter(pl.col("TimeDomainData").is_null())
            .height
        )

        assert observed_missing_samples == expected_missing_samples


def test_detected_missing_packets_match_interpolated_samples():
    for testfile in testfiles:
        df = import_BrainSenseTimeDomain_df(testfile)
        assert df is not None

        missing_rows = df.filter(pl.col("GlobalPacketSizes").is_null())

        packet_sizes = missing_rows["GlobalPacketSizesInterpolated"].fill_null(0).to_list()

        inserted_samples = (
            missing_rows.explode("TimeDomainData")
            .filter(pl.col("TimeDomainData").is_null())
            .height
        )

        assert inserted_samples == sum(packet_sizes)

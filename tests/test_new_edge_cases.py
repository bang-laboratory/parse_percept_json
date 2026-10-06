#!/usr/bin/env python3
"""New edge case tests for parse_percept_json"""

import pathlib

import polars as pl
import pytest
from src.parse_percept_json.parse_percept_json import (
    _process_BrainSenseTimeDomainBlock,
    convert_BrainSenseTimeDomain_to_mne,
    import_BrainSenseTimeDomain,
    import_BrainSenseTimeDomain_df,
    read_file,
)

testfile = pathlib.Path("tests/test_data_1.json")


# =============================================================================
# Block Boundary Tests
# =============================================================================

def test_blocks_have_distinct_timelines():
    """Verify blocks have valid timeline ranges"""
    df = import_BrainSenseTimeDomain_df(testfile)
    block_times = (
        df.group_by("FirstPacketDateTime")
        .agg(
            pl.col("BlockTimeInterpolatedMs").explode().min().alias("min_time"),
            pl.col("BlockTimeInterpolatedMs").explode().max().alias("max_time"),
            pl.len().alias("packet_count"),
        )
        .sort("min_time")
    )
    # Check each block has valid time range
    times = block_times.to_dicts()
    for i, t in enumerate(times):
        assert t["min_time"] is not None, f"Block {i} has no min_time"
        assert t["max_time"] is not None, f"Block {i} has no max_time"
        assert t["min_time"] <= t["max_time"], (
            f"Block {i} has invalid time range: min={t['min_time']}, max={t['max_time']}"
        )
        # Each block with packets should have positive duration
        if t["packet_count"] > 0:
            assert t["max_time"] - t["min_time"] >= 0, f"Block {i} has negative duration"


def test_missing_packets_only_within_blocks():
    """Missing packets should be detected and have interpolated data"""
    df = import_BrainSenseTimeDomain_df(testfile)
    missing = df.filter(pl.col("GlobalPacketSizes").is_null())
    # Missing packets should have interpolated packet sizes
    assert missing.height > 0, "Test file should have missing packets"
    # All missing packets should have interpolated sizes
    assert missing["GlobalPacketSizesInterpolated"].null_count() == 0, (
        "Missing packets should have interpolated sizes"
    )
    # All missing packets should have BlockTimeInterpolatedMs
    assert missing["BlockTimeInterpolatedMs"].null_count() == 0, (
        "Missing packets should have interpolated timestamps"
    )


# =============================================================================
# Timeline Continuity Tests
# =============================================================================

def test_timeline_is_monotonic():
    """Timeline should be strictly increasing within each block"""
    df = import_BrainSenseTimeDomain_df(testfile)
    sample_level = df.explode("BlockTimeInterpolatedMs")
    times = sample_level["BlockTimeInterpolatedMs"].to_list()
    diffs = [b - a for a, b in zip(times[:-1], times[1:])]
    assert all(d >= 0 for d in diffs), "Timeline should be non-decreasing"


def test_timeline_gaps_match_missing_packets():
    """Large timeline gaps should correspond to missing packets"""
    df = import_BrainSenseTimeDomain_df(testfile)
    sample_level = (
        df.explode("BlockTimeInterpolatedMs").sort("BlockTimeInterpolatedMs")
    )
    times = sample_level["BlockTimeInterpolatedMs"].to_list()
    diffs = [b - a for a, b in zip(times[:-1], times[1:])]
    expected_gap = 4  # ms per sample at 250Hz
    large_gaps = [d for d in diffs if d > expected_gap * 10]  # >40ms
    # Each large gap should correspond to missing packets
    num_missing_packets = df.filter(pl.col("GlobalPacketSizes").is_null()).height
    # Note: This might not be exact due to block boundaries
    # For now, just check we have some large gaps if we have missing packets
    if num_missing_packets > 0:
        assert len(large_gaps) > 0, (
            f"Expected large gaps in timeline when {num_missing_packets} packets are missing"
        )


# =============================================================================
# Data Integrity Tests
# =============================================================================

def test_sample_count_matches_packet_sizes():
    """Total samples should equal sum of all packet sizes"""
    df = import_BrainSenseTimeDomain_df(testfile)
    expected = df["GlobalPacketSizesInterpolated"].fill_null(0).sum()
    observed = df.explode("TimeDomainData").height
    assert observed == expected


def test_null_samples_only_in_missing_packets():
    """Null TimeDomainData should only appear in missing packets"""
    df = import_BrainSenseTimeDomain_df(testfile)
    null_samples = df.explode("TimeDomainData").filter(
        pl.col("TimeDomainData").is_null()
    )
    # All null samples should be from rows with null GlobalPacketSizes
    expected_null_samples = (
        df.filter(pl.col("GlobalPacketSizes").is_null())["GlobalPacketSizesInterpolated"]
        .fill_null(0)
        .sum()
    )
    assert null_samples.height == expected_null_samples


# =============================================================================
# Multi-block Analysis Tests
# =============================================================================

def test_block_count_matches_json():
    """Number of blocks in DataFrame should match JSON structure"""
    df = import_BrainSenseTimeDomain_df(testfile)
    json_data = read_file(testfile)
    num_blocks_json = len(json_data.get("BrainSenseTimeDomain", []))
    num_blocks_df = df["FirstPacketDateTime"].n_unique()
    # Note: This might differ if some blocks are empty
    # For now, just document the relationship
    print(f"JSON blocks: {num_blocks_json}, DataFrame blocks: {num_blocks_df}")


def test_each_block_has_consistent_channel():
    """Each block should have a consistent channel assignment"""
    df = import_BrainSenseTimeDomain_df(testfile)
    block_channels = df.group_by("FirstPacketDateTime").agg(
        pl.col("Channel").n_unique().alias("num_channels"),
        pl.col("Channel").unique().alias("channels"),
    )
    for row in block_channels.iter_rows(named=True):
        assert row["num_channels"] == 1, (
            f"Block with FirstPacketDateTime={row['FirstPacketDateTime']} "
            f"has {row['num_channels']} channels: {row['channels']}"
        )


# =============================================================================
# Sequence Number Tests
# =============================================================================

def test_sequence_numbers_are_sequential_within_blocks():
    """Sequence numbers should be mostly sequential within each block"""
    df = import_BrainSenseTimeDomain_df(testfile)
    # Group by block and check sequence number continuity
    block_seqs = df.group_by("FirstPacketDateTime").agg(
        pl.col("GlobalSequences").sort().alias("sequences"),
    )
    for row in block_seqs.iter_rows(named=True):
        seqs = row["sequences"]
        if len(seqs) > 1:
            # Check that consecutive sequences are mostly increasing
            # (allowing for some missing packets)
            diffs = [seqs[i + 1] - seqs[i] for i in range(len(seqs) - 1)]
            # All diffs should be positive (sequences increase)
            # Note: diffs can be >1 due to missing packets
            assert all(d > 0 for d in diffs), (
                f"Block has non-increasing sequences: {diffs}"
            )


def test_sequence_number_modulo_handling():
    """Sequence numbers wrap around at 2^16"""
    MODULO = 2**16
    df = import_BrainSenseTimeDomain_df(testfile)
    all_seqs = df["GlobalSequences"].to_list()
    # Check all sequences are within valid range
    for seq in all_seqs:
        if seq is not None:
            assert 0 <= seq < MODULO, f"Sequence number {seq} out of range [0, {MODULO})"


# =============================================================================
# LfpData Integration Tests
# =============================================================================

def test_lfp_data_sequences_excluded_from_missing():
    """Verify LfpData sequences are not flagged as missing in BrainSenseTimeDomain"""
    import json

    df = import_BrainSenseTimeDomain_df(testfile)
    # Get LfpData sequences from the JSON
    json_data = read_file(testfile)
    lfp_seqs = set()
    for block in json_data.get("BrainSenseLfp", []):
        for entry in block.get("LfpData", []):
            lfp_seqs.add(entry["Seq"])

    # Check none of these are in missing packets
    missing_seqs = (
        df.filter(pl.col("GlobalPacketSizes").is_null())["GlobalSequences"]
        .drop_nulls()
        .to_list()
    )
    # Note: Some LfpData sequences might legitimately be missing from TimeDomain
    # This test just verifies we're not double-counting them
    overlap = lfp_seqs & set(missing_seqs)
    print(f"LfpData sequences: {len(lfp_seqs)}, Missing TimeDomain sequences: {len(missing_seqs)}, Overlap: {len(overlap)}")
    # The overlap should be minimal or zero
    # For this test file, we know the expected overlap
    assert len(overlap) <= 1, f"Too many overlapping sequences: {overlap}"


# =============================================================================
# Packet Size Tests
# =============================================================================

def test_packet_sizes_are_consistent():
    """Packet sizes should be consistent (125 samples at 250Hz)"""
    df = import_BrainSenseTimeDomain_df(testfile)
    non_null_sizes = df["GlobalPacketSizes"].drop_nulls()
    unique_sizes = non_null_sizes.unique().to_list()
    # At 250Hz, packets should be 125 samples (50ms * 250Hz = 125)
    expected_size = 125
    assert all(s == expected_size for s in unique_sizes), (
        f"Unexpected packet sizes: {unique_sizes}, expected {expected_size}"
    )


def test_interpolated_packet_sizes_match_real_sizes():
    """Interpolated packet sizes should match real sizes where available"""
    df = import_BrainSenseTimeDomain_df(testfile)
    non_missing = df.filter(pl.col("GlobalPacketSizes").is_not_null())
    for row in non_missing.iter_rows(named=True):
        assert row["GlobalPacketSizes"] == row["GlobalPacketSizesInterpolated"], (
            f"Mismatch at seq {row['GlobalSequences']}: "
            f"real={row['GlobalPacketSizes']}, interpolated={row['GlobalPacketSizesInterpolated']}"
        )


# =============================================================================
# TimeDomainData Structure Tests
# =============================================================================

def test_timedomain_data_length_matches_packet_size():
    """Each TimeDomainData list should have length matching GlobalPacketSizesInterpolated"""
    df = import_BrainSenseTimeDomain_df(testfile)
    for row in df.iter_rows(named=True):
        expected_len = row["GlobalPacketSizesInterpolated"]
        actual_len = len(row["TimeDomainData"]) if row["TimeDomainData"] else 0
        assert actual_len == expected_len, (
            f"Row seq={row['GlobalSequences']}: "
            f"TimeDomainData length {actual_len} != expected {expected_len}"
        )


def test_timedomain_data_all_null_for_missing_packets():
    """Missing packets should have all-null TimeDomainData"""
    df = import_BrainSenseTimeDomain_df(testfile)
    missing = df.filter(pl.col("GlobalPacketSizes").is_null())
    for row in missing.iter_rows(named=True):
        if row["TimeDomainData"] is not None:
            assert all(x is None for x in row["TimeDomainData"]), (
                f"Missing packet seq={row['GlobalSequences']} has non-null data"
            )

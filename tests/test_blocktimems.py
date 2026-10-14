#!/usr/bin/env python3
"""Tests for BlockTimeMs and BlockTimeInterpolatedMs columns in _process_BrainSenseTimeDomainBlock.

These tests verify the behavior of the _calc_BlockTimeMs function and ensure that:
1. BlockTimeMs uses TicksInMses for the first packet in each block
2. BlockTimeMs chains from previous packet for subsequent packets
3. BlockTimeInterpolatedMs creates a continuous timeline within each block
4. BlockTimeInterpolatedMs fills in gaps for missing packets with continuous timeline
5. The timeline is continuous (no gaps or overlaps) in BlockTimeInterpolatedMs within each block
"""

import pathlib

import polars as pl
from src.parse_percept_json.parse_percept_json import (
    _process_BrainSenseTimeDomainBlock,
    read_file,
    import_BrainSenseTimeDomain_df,
)

testfiles = [
    pathlib.Path("tests/test_data_1.json"),
    pathlib.Path("tests/Report_Json_Session_Report_20250218T092621.json"),
    pathlib.Path("tests/Report_Json_Session_Report_20250226T084029.json"),
]


def test_BlockTimeMs_uses_TicksInMses_for_first_packet_in_block():
    """For the first packet in each block, BlockTimeMs should start at TicksInMses value."""
    for testfile in testfiles:
        json_data = read_file(testfile)
        for block_idx, block_data in enumerate(json_data["BrainSenseTimeDomain"]):
            df = _process_BrainSenseTimeDomainBlock(block_data, block_id=block_idx)
            
            non_missing = df.filter(pl.col("GlobalPacketSizes").is_not_null())
            
            # Only check the first packet in each block
            if non_missing.height > 0:
                first_row = non_missing.head(1)
                seq = first_row[0, "GlobalSequences"]
                ticks = first_row[0, "TicksInMses"]
                btm = first_row[0, "BlockTimeMs"]
                
                # BlockTimeMs should start at TicksInMses for the first non-missing packet
                assert btm is not None, f"BlockTimeMs is None for first packet {seq} in block {block_idx}"
                assert len(btm) > 0, f"BlockTimeMs is empty for first packet {seq} in block {block_idx}"
                assert btm[0] == ticks, f"BlockTimeMs[0] ({btm[0]}) != TicksInMses ({ticks}) for first packet {seq} in block {block_idx}"


def test_BlockTimeInterpolatedMs_starts_correctly():
    """For the first packet in each block, BlockTimeInterpolatedMs should start at TicksInMses value."""
    for testfile in testfiles:
        json_data = read_file(testfile)
        for block_idx, block_data in enumerate(json_data["BrainSenseTimeDomain"]):
            df = _process_BrainSenseTimeDomainBlock(block_data, block_id=block_idx)
            non_missing = df.filter(pl.col("GlobalPacketSizes").is_not_null())
            if non_missing.height > 0:
                first_row = non_missing.head(1)
                seq = first_row[0, "GlobalSequences"]
                ticks = first_row[0, "TicksInMses"]
                btm_interp = first_row[0, "BlockTimeInterpolatedMs"]
                assert btm_interp is not None
                assert len(btm_interp) > 0
                assert btm_interp[0] == ticks


def test_BlockTimeMs_and_BlockTimeInterpolatedMs_match_for_first_packet():
    """For the first non-missing packet in each block, BlockTimeMs and BlockTimeInterpolatedMs should be identical."""
    for testfile in testfiles:
        json_data = read_file(testfile)
        for block_idx, block_data in enumerate(json_data["BrainSenseTimeDomain"]):
            df = _process_BrainSenseTimeDomainBlock(block_data, block_id=block_idx)
            non_missing = df.filter(pl.col("GlobalPacketSizes").is_not_null())
            if non_missing.height > 0:
                first_row = non_missing.head(1)
                seq = first_row[0, "GlobalSequences"]
                btm = first_row[0, "BlockTimeMs"]
                btm_interp = first_row[0, "BlockTimeInterpolatedMs"]
                # Convert to lists for comparison
                assert list(btm) == list(btm_interp), f"BlockTimeMs != BlockTimeInterpolatedMs for first packet {seq} in block {block_idx}"


def test_BlockTimeInterpolatedMs_is_continuous():
    """BlockTimeInterpolatedMs should form a continuous timeline with no gaps or overlaps WITHIN each block."""
    for testfile in testfiles:
        df = import_BrainSenseTimeDomain_df(testfile)
        
        # Check continuity within each block
        for block_id in sorted(df['block_id'].unique().to_list()):
            block_df = df.filter(pl.col("block_id") == block_id)
            block_df_sorted = block_df.sort("GlobalSequences")
            
            # Explode BlockTimeInterpolatedMs to get sample-level timeline
            exploded = block_df_sorted.explode("BlockTimeInterpolatedMs")
            times = exploded["BlockTimeInterpolatedMs"].to_list()
            
            # Check that all times are unique within this block (no overlaps)
            assert len(times) == len(set(times)), f"BlockTimeInterpolatedMs has duplicate timestamps in block {block_id}"
            
            # Check that times are monotonically increasing within this block
            for i in range(1, len(times)):
                assert times[i] > times[i-1], f"BlockTimeInterpolatedMs not monotonically increasing at index {i} in block {block_id}: {times[i-1]} -> {times[i]}"
                gap = times[i] - times[i-1]
                assert gap == 4, f"Gap between consecutive timestamps is {gap}ms, expected 4ms at index {i} in block {block_id}"


def test_BlockTimeInterpolatedMs_fills_missing_packets():
    """Missing packets should have BlockTimeInterpolatedMs that fills the gap between surrounding packets."""
    for testfile in testfiles:
        json_data = read_file(testfile)
        for block_idx, block_data in enumerate(json_data["BrainSenseTimeDomain"]):
            df = _process_BrainSenseTimeDomainBlock(block_data, block_id=block_idx)
            
            missing = df.filter(pl.col("GlobalPacketSizes").is_null())
            
            for row in missing.iter_rows(named=True):
                seq = row["GlobalSequences"]
                btm_interp = row["BlockTimeInterpolatedMs"]
                
                assert btm_interp is not None, f"BlockTimeInterpolatedMs is None for missing packet {seq}"
                assert len(btm_interp) > 0, f"BlockTimeInterpolatedMs is empty for missing packet {seq}"
                
                gpsi = row["GlobalPacketSizesInterpolated"]
                assert gpsi is not None, f"GlobalPacketSizesInterpolated is None for missing packet {seq}"
                assert len(btm_interp) == gpsi, f"BlockTimeInterpolatedMs length != GlobalPacketSizesInterpolated for missing packet {seq}"
                
                prev_seq = seq - 1
                prev_row = df.filter(pl.col("GlobalSequences") == prev_seq)
                if prev_row.height > 0:
                    prev_btm_interp = prev_row[0, "BlockTimeInterpolatedMs"]
                    if prev_btm_interp is not None and len(prev_btm_interp) > 0:
                        gap = btm_interp[0] - prev_btm_interp[-1]
                        assert gap == 4, f"Gap from previous packet is {gap}ms, expected 4ms for missing packet {seq} in block {block_idx}"


def test_BlockTimeMs_length_matches_packet_size():
    """BlockTimeMs list length should match GlobalPacketSizes for non-missing packets."""
    for testfile in testfiles:
        json_data = read_file(testfile)
        for block_idx, block_data in enumerate(json_data["BrainSenseTimeDomain"]):
            df = _process_BrainSenseTimeDomainBlock(block_data, block_id=block_idx)
            
            non_missing = df.filter(pl.col("GlobalPacketSizes").is_not_null())
            
            for row in non_missing.iter_rows(named=True):
                seq = row["GlobalSequences"]
                btm = row["BlockTimeMs"]
                gps = row["GlobalPacketSizes"]
                
                assert len(btm) == gps, f"BlockTimeMs length != GlobalPacketSizes for packet {seq} in block {block_idx}"


def test_BlockTimeInterpolatedMs_length_matches_interpolated_packet_size():
    """BlockTimeInterpolatedMs list length should match GlobalPacketSizesInterpolated."""
    for testfile in testfiles:
        json_data = read_file(testfile)
        for block_idx, block_data in enumerate(json_data["BrainSenseTimeDomain"]):
            df = _process_BrainSenseTimeDomainBlock(block_data, block_id=block_idx)
            
            for row in df.iter_rows(named=True):
                seq = row["GlobalSequences"]
                btm_interp = row["BlockTimeInterpolatedMs"]
                gpsi = row["GlobalPacketSizesInterpolated"]
                
                assert gpsi is not None, f"GlobalPacketSizesInterpolated is None for packet {seq} in block {block_idx}"
                assert btm_interp is not None, f"BlockTimeInterpolatedMs is None for packet {seq} in block {block_idx}"
                assert len(btm_interp) == gpsi, f"BlockTimeInterpolatedMs length != GlobalPacketSizesInterpolated for packet {seq} in block {block_idx}"


def test_BlockTimeMs_step_is_correct():
    """Consecutive timestamps in BlockTimeMs should be spaced by ms_per_sample."""
    for testfile in testfiles:
        json_data = read_file(testfile)
        for block_idx, block_data in enumerate(json_data["BrainSenseTimeDomain"]):
            df = _process_BrainSenseTimeDomainBlock(block_data, block_id=block_idx)
            
            non_missing = df.filter(pl.col("GlobalPacketSizes").is_not_null())
            sample_rate = block_data["SampleRateInHz"]
            ms_per_sample = 1000 / sample_rate
            
            for row in non_missing.iter_rows(named=True):
                seq = row["GlobalSequences"]
                btm = row["BlockTimeMs"]
                
                for i in range(1, len(btm)):
                    gap = btm[i] - btm[i-1]
                    assert gap == int(ms_per_sample), f"Gap in BlockTimeMs is {gap}ms, expected {int(ms_per_sample)}ms for packet {seq} in block {block_idx}"


def test_BlockTimeInterpolatedMs_step_is_correct():
    """Consecutive timestamps in BlockTimeInterpolatedMs should be spaced by ms_per_sample."""
    for testfile in testfiles:
        json_data = read_file(testfile)
        for block_idx, block_data in enumerate(json_data["BrainSenseTimeDomain"]):
            df = _process_BrainSenseTimeDomainBlock(block_data, block_id=block_idx)
            
            sample_rate = block_data["SampleRateInHz"]
            ms_per_sample = 1000 / sample_rate
            
            for row in df.iter_rows(named=True):
                seq = row["GlobalSequences"]
                btm_interp = row["BlockTimeInterpolatedMs"]
                
                if btm_interp is not None and len(btm_interp) > 1:
                    for i in range(1, len(btm_interp)):
                        gap = btm_interp[i] - btm_interp[i-1]
                        assert gap == int(ms_per_sample), f"Gap in BlockTimeInterpolatedMs is {gap}ms, expected {int(ms_per_sample)}ms for packet {seq} in block {block_idx}"


def test_first_packet_BlockTimeMs_starts_at_TicksInMses():
    """The first packet's BlockTimeMs should start exactly at its TicksInMses value."""
    for testfile in testfiles:
        json_data = read_file(testfile)
        for block_idx, block_data in enumerate(json_data["BrainSenseTimeDomain"]):
            df = _process_BrainSenseTimeDomainBlock(block_data, block_id=block_idx)
            
            first_row = df.head(1)
            first_btm = first_row[0, "BlockTimeMs"]
            first_ticks = first_row[0, "TicksInMses"]
            
            assert first_btm[0] == first_ticks, f"First packet BlockTimeMs[0] != TicksInMses in block {block_idx}"


def test_first_packet_BlockTimeInterpolatedMs_starts_at_TicksInMses():
    """The first packet's BlockTimeInterpolatedMs should start exactly at its TicksInMses value."""
    for testfile in testfiles:
        json_data = read_file(testfile)
        for block_idx, block_data in enumerate(json_data["BrainSenseTimeDomain"]):
            df = _process_BrainSenseTimeDomainBlock(block_data, block_id=block_idx)
            
            first_row = df.head(1)
            first_btm_interp = first_row[0, "BlockTimeInterpolatedMs"]
            first_ticks = first_row[0, "TicksInMses"]
            
            assert first_btm_interp[0] == first_ticks, f"First packet BlockTimeInterpolatedMs[0] != TicksInMses in block {block_idx}"

#!/usr/bin/env python3


from pathlib import Path

import mne
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

testfile = Path("tests/test_data_1.json")


def test_import_BrainSenseTimeDomain():
    data = import_BrainSenseTimeDomain(testfile)
    assert type(data) is mne.io.RawArray

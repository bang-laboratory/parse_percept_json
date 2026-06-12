# Parse percept json

Parse data from the Medtronic Percept implants in a way that uses the
information about a) timing and b) packet missingness that is available in the data
format.

Uses polars under the hood for fast columnar data storage, and supports
transforming to mne.RawArray, using Annotation to mark missing data from dropped
packets or from periods of no data collection during a data collection session.

Inspired by https://github.com/neuromodulation/icn icn_perceive.py

## Examples

``` python
from pathlib import Path
import parse_percept_json

data_path = Path("file.json")

# For BrainSenseTimeDomain data
df = import_BrainSenseTimeDomain_df(data_path)
raw = convert_BrainSenseTimeDomain_to_mne(df)

# Also supports other data streams available in the json format:
df2 = import_LfpTrendLogs(data_path)
df3 = import_LfpFrequencySnapshotEvents(data_path)
```

`

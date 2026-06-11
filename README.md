# Read sensing

Parse data from the Medtronic Percept implants in a way that uses the
information about timing and packet missingness that is available in the data
format.

Uses polars under the hood for fast columnar data storage, and supports
transforming to mne.RawArray, using Annotation to mark missing data from dropped
packets or from periods of no data collection during a data collection session.

Inspired by https://github.com/neuromodulation/icn icn_perceive.py

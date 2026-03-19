# OrbitalChain

Privacy-Preserving Data Streaming for Satellite Networks.

## Project Structure

```
OrbitalChain/
├── data/
│   ├── Active_satellites_in_orbit_July_2016.csv   # 1420 satellite catalog
│   ├── LAN-SMA-AP-25462-SupGP.csv                 # GEO satellite time-series
│   ├── objects-by-apogee.csv                       # Apogee distribution histogram
│   └── satellite_ccsds_dataset/                    # CCSDS operational telemetry
│       ├── 01_tm_housekeeping.bin                  # Raw CCSDS TM packets (binary)
│       ├── 02_tc_command_sequence.bin              # TC command frames (binary)
│       ├── 03_aos_sar_frames.bin                   # AOS SAR payload frames (binary)
│       ├── 04_telemetry_decoded.csv                # Decoded housekeeping TM, 500 rows
│       ├── 05_isl_stream_log.json                  # ISL crosslink log, 5 sats x 40 epochs
│       ├── 06_gnss_ephemeris.csv                   # GPS SV Keplerian elements, 24 SVs
│       ├── 07_attitude_quaternions.csv             # ADCS quaternion time-series, 1000 rows
│       ├── 08_power_subsystem.csv                  # Power telemetry, 480 epochs
│       ├── 09_ccsds_parser.py                      # CCSDS binary packet parser
│       └── 10_dataset_metadata.json                # Dataset schema reference
├── config/
│   └── default_config.yaml
├── src/
│   ├── data/
│   │   ├── data_loader.py        # Loads Active_satellites catalog CSVs
│   │   └── ccsds_adapter.py      # Maps CCSDS dataset files to module inputs
│   ├── satellite/
│   │   ├── channel_model.py      # Default: 08_power_subsystem + 05_isl_stream_log
│   │   └── orbital_mechanics.py  # Default: 06_gnss_ephemeris (24 GPS SVs)
│   ├── clustering/
│   │   └── d_stream.py           # Default: 07_attitude_quaternions
│   ├── consensus/
│   │   └── sa_sbft.py
│   ├── crypto/
│   │   ├── secret_sharing.py     # Default: battery_soc_pct from 08_power_subsystem.csv
│   │   ├── beaver_triples.py     # Default: soc_pct * reliability_weight
│   │   └── garbled_circuits.py   # Default: weighted quorum score and trust weights
│   └── truth_discovery/
│       └── streaming_truth.py    # Default: 05_isl_stream_log (5 ISL satellites)
└── requirements.txt
```



## Real data used by each module

| Module | Dataset file | Field | Why |
|---|---|---|---|
| `secret_sharing.py` | `08_power_subsystem.csv` 
| `beaver_triples.py` | `08_power_subsystem.csv` + `05_isl_stream_log.json` 
| `garbled_circuits.py` |  `08_power_subsystem.csv` + `05_isl_stream_log.json` | 
| `orbital_mechanics.py` | `06_gnss_ephemeris.csv` | 
| `channel_model.py` | `08_power_subsystem.csv` + `05_isl_stream_log.json` | 
| `d_stream.py` | `07_attitude_quaternions.csv` | 
| `streaming_truth.py` | `05_isl_stream_log.json` | 
| `sa_sbft.py` | `08_power_subsystem.csv` + `05_isl_stream_log.json` | 


## 

The secret sharing module splits one scalar value across N satellite nodes such that
no single node learns the value — only all N together reconstruct it.
`battery_soc_pct` fits because:

- A satellite must prove it meets the SA-SBFT energy threshold (≥50% active, ≥20% semi)
  for quorum participation without revealing the exact charge level to routing adversaries.
- It is a float in [0, 100] with no negative values, encoding cleanly as
  `secret_integer = int(soc_pct × 100)` (e.g. 60.0% → 6000).
- The multiplication in `beaver_triples.py` computes `weight_k × soc_k` where
  `weight_k` is the satellite's reliability score derived from fault alerts in the ISL log.
- The division in `garbled_circuits.py` then computes the weighted mean SoC as the
  quorum score without any party learning the individual SoC values.


### Data loaders

```bash
# Satellite catalog summary (Active_satellites_in_orbit_July_2016.csv)
python -m src.data.data_loader

# CCSDS dataset mapping — shows which file feeds which module
python -m src.data.ccsds_adapter
```

### Cryptography

```bash
# Secret sharing — real battery_soc_pct from 08_power_subsystem.csv
python -m src.crypto.secret_sharing

# Beaver triples — soc_pct * reliability_weight (quorum score)
python -m src.crypto.beaver_triples

# Garbled circuits — weighted quorum score and trust weights
python -m src.crypto.garbled_circuits
```

### Satellite

```bash
# Orbital mechanics — real GNSS ephemeris (06_gnss_ephemeris.csv, 24 GPS SVs)
python -m src.satellite.orbital_mechanics

# Orbital mechanics — synthetic test orbit
python -m src.satellite.orbital_mechanics --random

# Channel model — real power subsystem + ISL log (08_power_subsystem.csv + 05_isl_stream_log.json)
python -m src.satellite.channel_model

# Channel model — fixed elevation angles
python -m src.satellite.channel_model --random
```

### Clustering

```bash
# D-Stream — real attitude quaternions (07_attitude_quaternions.csv, 1000 points at 10 Hz)
python -m src.clustering.d_stream

# D-Stream — synthetic Gaussian clusters
python -m src.clustering.d_stream --random
```

### Truth Discovery

```bash
# Real CCSDS ISL telemetry (05_isl_stream_log.json, 5 satellites x 40 epochs)
python -m src.truth_discovery.streaming_truth

# With custom MPC satellite count and decay factor
python -m src.truth_discovery.streaming_truth --num-satellites 4 --decay 0.85

# Tighter suspicion threshold (flag fewer suspects)
python -m src.truth_discovery.streaming_truth --iqr-threshold 1.5

# Synthetic random mode — configurable providers and malicious count
python -m src.truth_discovery.streaming_truth --random
python -m src.truth_discovery.streaming_truth --random --num-providers 10 --num-satellites 4 --num-malicious 3 --num-epochs 20
python -m src.truth_discovery.streaming_truth --random --num-providers 6 --num-malicious 2 --malicious-bias 0.6
python -m src.truth_discovery.streaming_truth --random --seed 42
```


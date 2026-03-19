# CCSDS dataset adapter for OrbitalChain
# Bridges satellite_ccsds_dataset files to OrbitalChain module inputs.
#
# Each loader returns data in the exact shape the target module expects,
# replacing the random/fixed values used during development.

import csv
import json
import numpy as np
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

_HERE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "data", "satellite_ccsds_dataset"))


# ── tiny helpers ──────────────────────────────────────────────────────────────

def _csv(filename: str) -> List[Dict]:
    with open(os.path.join(DATA_DIR, filename), newline="") as f:
        return list(csv.DictReader(f))

def _json(filename: str):
    with open(os.path.join(DATA_DIR, filename)) as f:
        return json.load(f)

def _norm(arr: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return np.full_like(arr, (lo + hi) / 2)
    return lo + (arr - mn) / (mx - mn) * (hi - lo)


# ─────────────────────────────────────────────────────────────────────────────
# target: src/satellite/orbital_mechanics.py  →  KeplerianOrbit.from_catalog_row()
# source: 06_gnss_ephemeris.csv
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KeplerianParams:
    """One satellite's orbital elements ready for KeplerianOrbit.from_catalog_row()"""
    name:             str
    altitude_km:      float   # mean orbit altitude above Earth surface
    eccentricity:     float
    inclination_deg:  float
    raan_deg:         float
    arg_perigee_deg:  float
    mean_anomaly_deg: float
    prn:              str

def load_gnss_orbits(epoch_index: int = 0) -> List[KeplerianParams]:
    """
    Load GPS SV orbital elements from 06_gnss_ephemeris.csv.
    Returns one KeplerianParams per SV at the requested epoch (0, 1, or 2).
    Altitude is computed as SMA − R_EARTH (6371 km).
    All angles converted from radians to degrees.
    """
    R_EARTH = 6371.0
    rows    = _csv("06_gnss_ephemeris.csv")
    epochs  = sorted(set(r["epoch_utc"] for r in rows))
    if epoch_index >= len(epochs):
        epoch_index = 0
    target  = epochs[epoch_index]
    subset  = [r for r in rows if r["epoch_utc"] == target]

    result = []
    for r in subset:
        sma_m = float(r["semi_major_axis_m"])
        result.append(KeplerianParams(
            name            = r["prn"],
            prn             = r["prn"],
            altitude_km     = sma_m / 1000.0 - R_EARTH,
            eccentricity    = float(r["eccentricity"]),
            inclination_deg = float(r["inclination_rad"]) * 180.0 / np.pi,
            raan_deg        = float(r["raan_rad"])         * 180.0 / np.pi,
            arg_perigee_deg = float(r["arg_perigee_rad"]) * 180.0 / np.pi,
            mean_anomaly_deg= float(r["mean_anomaly_rad"])* 180.0 / np.pi,
        ))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# target: src/satellite/channel_model.py  →  compute_link_budget(), SatelliteLink
# source: 08_power_subsystem.csv  (orbit_phase_deg → elevation)
#         05_isl_stream_log.json  (signal_strength_dBm → measured SNR)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LinkObservation:
    """One link measurement epoch for channel_model inputs"""
    timestamp:          str
    orbit_phase_deg:    float
    elevation_deg:      float   # derived from orbit_phase
    eclipse:            bool
    solar_W:            float
    measured_rssi_dBm:  float   # from ISL log or telemetry
    battery_V:          float


def _phase_to_elevation(phase_deg: float) -> float:
    """
    Approximate elevation from orbit phase.
    Phase=0 → overhead (90°), Phase=90 → horizon (0°), wraps every 180°.
    """
    return max(0.0, 90.0 - abs(phase_deg % 180.0 - 90.0))


def load_link_observations() -> List[LinkObservation]:
    """
    Load 08_power_subsystem.csv and derive per-epoch link budget inputs.
    Returns 480 observations (one per minute over 8 hours / ~5 orbits).
    """
    rows = _csv("08_power_subsystem.csv")
    obs  = []
    for r in rows:
        phase = float(r["orbit_phase_deg"])
        obs.append(LinkObservation(
            timestamp       = r["timestamp_utc"],
            orbit_phase_deg = phase,
            elevation_deg   = _phase_to_elevation(phase),
            eclipse         = r["eclipse"] == "1",
            solar_W         = float(r["solar_W"]),
            measured_rssi_dBm = float(r.get("bus_V", -100)) * -10,  # proxy; replace with real RSSI
            battery_V       = float(r["battery_V"]),
        ))
    return obs


def load_isl_link_quality() -> Dict[str, List[Dict]]:
    """
    Load 05_isl_stream_log.json.
    Returns dict keyed by satellite ID, each value a time-ordered list of:
      { timestamp, signal_dBm, ber, apid, link_direction, crc_valid }
    These replace the fixed link parameters in SatelliteLink.compute_link_quality().
    """
    data = _json("05_isl_stream_log.json")
    by_sat: Dict[str, List[Dict]] = {}
    for p in data["packets"]:
        sat = p.get("source_sat_id", "unknown")
        if sat not in by_sat:
            by_sat[sat] = []
        by_sat[sat].append({
            "timestamp":   p.get("timestamp_utc"),
            "signal_dBm":  p.get("signal_strength_dBm"),
            "ber":         p.get("bit_error_rate", 0.0),
            "apid":        p.get("apid"),
            "link":        p.get("link"),
            "crc_valid":   p.get("crc_valid", True),
        })
    return by_sat


# ─────────────────────────────────────────────────────────────────────────────
# target: src/truth_discovery/streaming_truth.py  →  sensing values per epoch
# source: 05_isl_stream_log.json  (5 satellites × 40 epochs of signal_strength_dBm)
#         04_telemetry_decoded.csv (fault_flag labels for ground-truth verification)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TruthDiscoveryEpoch:
    epoch_index:        int
    sensing_values:     List[float]    # one per satellite provider, normalised [0,1]
    satellite_ids:      List[str]
    raw_signal_dBm:     List[float]    # original values for debugging
    has_fault:          List[bool]     # ground-truth label (hidden from engine)
    timestamps:         List[str]


def load_isl_truth_epochs(
    normalise: bool = True,
    signal_lo: float = -130.0,
    signal_hi: float = -90.0,
) -> List[TruthDiscoveryEpoch]:
    """
    Build truth-discovery epochs from 05_isl_stream_log.json.

    The 5 ISL satellites (SAT-10 … SAT-14) each report signal_strength_dBm
    per packet.  Grouping by packet-index // 5 gives 40 epochs where all
    5 satellites report their observation of the shared channel quality.
    This is the exact multi-provider sensing model the truth discovery engine
    expects.

    Fault alerts (apid=0x7FE) and CRC failures flag a provider as anomalous;
    these labels are stored in has_fault but are NEVER passed to the engine.

    normalise=True maps dBm to [0, 1] using signal_lo / signal_hi.
    """
    data   = _json("05_isl_stream_log.json")
    pkts   = data["packets"]

    # Group into epochs of 5 (one reading per satellite per epoch)
    from collections import defaultdict
    raw_epochs: Dict[int, List[Dict]] = defaultdict(list)
    for p in pkts:
        raw_epochs[p["index"] // 5].append(p)

    # Fault-alert satellite IDs across the whole session
    fault_sats = {p["source_sat_id"] for p in pkts if p.get("apid") == "0x7FE"}

    sat_order = sorted({p["source_sat_id"] for p in pkts})  # SAT-10..14

    epochs: List[TruthDiscoveryEpoch] = []
    for ep_idx in sorted(raw_epochs.keys()):
        bucket = {p["source_sat_id"]: p for p in raw_epochs[ep_idx]}
        raw_vals = []
        has_fault = []
        timestamps = []
        for sat in sat_order:
            p = bucket.get(sat)
            if p is None:
                raw_vals.append(float(signal_lo))
                has_fault.append(False)
                timestamps.append("")
            else:
                raw_vals.append(float(p.get("signal_strength_dBm") or signal_lo))
                # CRC failure OR fault-alert APID → anomalous provider
                is_fault = (not p.get("crc_valid", True)) or (sat in fault_sats)
                has_fault.append(is_fault)
                timestamps.append(p.get("timestamp_utc", ""))

        if normalise:
            sensing = [
                float(np.clip((v - signal_lo) / (signal_hi - signal_lo), 0.0, 1.0))
                for v in raw_vals
            ]
        else:
            sensing = list(raw_vals)

        epochs.append(TruthDiscoveryEpoch(
            epoch_index     = ep_idx,
            sensing_values  = sensing,
            satellite_ids   = sat_order,
            raw_signal_dBm  = raw_vals,
            has_fault       = has_fault,
            timestamps      = timestamps,
        ))
    return epochs


# ─────────────────────────────────────────────────────────────────────────────
# target: src/truth_discovery/streaming_truth.py  →  alternative: power-based sensing
# source: 04_telemetry_decoded.csv
# ─────────────────────────────────────────────────────────────────────────────

def load_telemetry_sensing_windows(
    field:       str  = "rssi_dBm",
    window_size: int  = 5,
    normalise:   bool = True,
) -> List[TruthDiscoveryEpoch]:
    """
    Treat the 500-row telemetry CSV as a single-satellite time series.
    Slide a window of `window_size` consecutive rows to form one epoch,
    treating each row within the window as a separate 'provider' report.

    This models the case where a ground station receives the same measurement
    from multiple redundant sensors on-board (or multiple ground stations
    receiving the same downlink telemetry).

    fault_flag=1 marks anomalous rows (injected every 97th packet per metadata).
    """
    rows = _csv("04_telemetry_decoded.csv")

    # field bounds for normalisation
    vals_all = [float(r[field]) for r in rows]
    lo, hi   = min(vals_all), max(vals_all)

    def norm(v):
        if abs(hi - lo) < 1e-9:
            return 0.5
        return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))

    epochs = []
    for i in range(0, len(rows) - window_size + 1, window_size):
        window = rows[i: i + window_size]
        sensing = [norm(float(r[field])) if normalise else float(r[field])
                   for r in window]
        faults  = [r["fault_flag"] == "1" for r in window]
        ts      = [r["timestamp_utc"] for r in window]
        epochs.append(TruthDiscoveryEpoch(
            epoch_index     = i // window_size,
            sensing_values  = sensing,
            satellite_ids   = [f"row_{i+j}" for j in range(window_size)],
            raw_signal_dBm  = [float(r[field]) for r in window],
            has_fault       = faults,
            timestamps      = ts,
        ))
    return epochs


# ─────────────────────────────────────────────────────────────────────────────
# target: src/clustering/d_stream.py  →  DataPoint stream
# source: 07_attitude_quaternions.csv  (q0..q3, pointing_error_arcsec)
#         04_telemetry_decoded.csv     (rssi, battery, solar → 3D data points)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClusteringPoint:
    coordinates: List[float]   # 3D: (x, y, z) or (lat, lon, truth_score)
    weight:      float
    timestamp:   float         # seconds since start
    source:      str


def load_attitude_cluster_stream(max_rows: int = 500) -> List[ClusteringPoint]:
    """
    Load 07_attitude_quaternions.csv as a D-Stream clustering input.
    Coordinates: (euler_roll_deg, euler_pitch_deg, pointing_error_arcsec normalised).
    Weight: 1.0 for Fine Point, 0.5 for Slewing (less reliable).
    Timestamp: time_s column directly.

    The OrbitalChainDStream subclass expects dimensionality=3 and
    coordinates in (latitude, longitude, truth_score) form; this maps
    (roll, pitch, pointing_error_norm) to that same 3D space.
    """
    rows = _csv("07_attitude_quaternions.csv")[:max_rows]
    pe_vals = [float(r["pointing_error_arcsec"]) for r in rows]
    pe_lo, pe_hi = min(pe_vals), max(pe_vals)

    points = []
    for r in rows:
        pe_norm = (float(r["pointing_error_arcsec"]) - pe_lo) / max(pe_hi - pe_lo, 1e-9)
        points.append(ClusteringPoint(
            coordinates = [
                float(r["euler_roll_deg"]),
                float(r["euler_pitch_deg"]),
                pe_norm,
            ],
            weight    = 1.0 if r["adcs_mode"] == "Fine Point" else 0.5,
            timestamp = float(r["time_s"]),
            source    = r["adcs_mode"],
        ))
    return points


def load_telemetry_cluster_stream(max_rows: int = 500) -> List[ClusteringPoint]:
    """
    Load 04_telemetry_decoded.csv as a D-Stream clustering input.
    Coordinates: (battery_V_norm, panel_temp_norm, rssi_norm).
    Weight: 0.5 if fault_flag=1, else 1.0.
    Timestamp: sequential index × 0.4 s (400 ms sample interval).
    """
    rows = _csv("04_telemetry_decoded.csv")[:max_rows]
    def _field_norm(field):
        v = [float(r[field]) for r in rows]
        lo, hi = min(v), max(v)
        return [(x - lo) / max(hi - lo, 1e-9) for x in v]

    bv   = _field_norm("battery_V")
    temp = _field_norm("panel_temp_C")
    rssi = _field_norm("rssi_dBm")

    points = []
    for i, r in enumerate(rows):
        points.append(ClusteringPoint(
            coordinates = [bv[i], temp[i], rssi[i]],
            weight      = 0.5 if r["fault_flag"] == "1" else 1.0,
            timestamp   = i * 0.4,
            source      = r["adcs_mode"],
        ))
    return points


# ─────────────────────────────────────────────────────────────────────────────
# target: src/consensus/sa_sbft.py  →  satellite energy / reliability state
# source: 08_power_subsystem.csv   (SoC, eclipse flag → energy level)
#         05_isl_stream_log.json   (per-satellite reliability from CRC + fault rate)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SatelliteEnergyState:
    satellite_id:    str
    timestamp:       str
    soc_pct:         float    # state of charge [0, 100]
    solar_W:         float
    eclipse:         bool
    energy_level:    float    # normalised [0, 1] for SA-SBFT threshold checks
    orbit_phase_deg: float
    crc_fail_rate:   float    # from ISL log, 0..1 reliability proxy
    fault_count:     int


def load_consensus_energy_states() -> Tuple[List[SatelliteEnergyState], Dict[str, float]]:
    """
    Load energy and reliability state for each satellite, combining:
      - 08_power_subsystem.csv  →  per-epoch SoC, eclipse, solar power
      - 05_isl_stream_log.json →  per-satellite CRC fail rate and fault count

    Returns:
      (list_of_energy_states_over_time,  reliability_scores_by_sat_id)

    energy_level = soc_pct / 100 * (1 - eclipse * 0.2)
    Maps directly to the energy_threshold_active / energy_threshold_semi
    parameters in default_config.yaml.
    """
    pwr_rows = _csv("08_power_subsystem.csv")
    isl      = _json("05_isl_stream_log.json")["packets"]

    # Per-satellite reliability from ISL log
    from collections import defaultdict
    sat_stats: Dict[str, Dict] = defaultdict(lambda: {"total": 0, "crc_fail": 0, "faults": 0})
    for p in isl:
        sat = p.get("source_sat_id", "unknown")
        sat_stats[sat]["total"] += 1
        if not p.get("crc_valid", True):
            sat_stats[sat]["crc_fail"] += 1
        if p.get("apid") == "0x7FE":
            sat_stats[sat]["faults"] += 1

    reliability: Dict[str, float] = {}
    for sat, s in sat_stats.items():
        fail_rate = s["crc_fail"] / max(s["total"], 1)
        reliability[sat] = 1.0 - fail_rate - 0.05 * s["faults"]

    # Map power rows to energy states (assume single-satellite power data)
    states = []
    for r in pwr_rows:
        soc    = float(r["battery_soc_pct"])
        eclipse = r["eclipse"] == "1"
        energy = (soc / 100.0) * (0.8 if eclipse else 1.0)
        states.append(SatelliteEnergyState(
            satellite_id    = "SAT-MAIN",
            timestamp       = r["timestamp_utc"],
            soc_pct         = soc,
            solar_W         = float(r["solar_W"]),
            eclipse         = eclipse,
            energy_level    = energy,
            orbit_phase_deg = float(r["orbit_phase_deg"]),
            crc_fail_rate   = 0.0,
            fault_count     = 0,
        ))
    return states, reliability


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: full dataset summary
# ─────────────────────────────────────────────────────────────────────────────

def dataset_summary() -> None:
    print("CCSDS Dataset → OrbitalChain mapping summary\n")

    gnss = load_gnss_orbits()
    print(f"  06_gnss_ephemeris     : {len(gnss)} GPS SVs → orbital_mechanics.KeplerianOrbit")

    link_obs = load_link_observations()
    print(f"  08_power_subsystem    : {len(link_obs)} epochs → channel_model link budget (elevation, eclipse)")

    isl_qual = load_isl_link_quality()
    total_isl = sum(len(v) for v in isl_qual.values())
    print(f"  05_isl_stream_log     : {total_isl} packets / {len(isl_qual)} satellites → channel_model + consensus reliability")

    td_epochs = load_isl_truth_epochs()
    fault_providers = sum(1 for ep in td_epochs for f in ep.has_fault if f)
    print(f"  05_isl (truth disc.)  : {len(td_epochs)} epochs × {len(td_epochs[0].sensing_values)} providers → streaming_truth sensing values")
    print(f"                          {fault_providers} anomalous provider-epochs (hidden labels for verification)")

    att_pts = load_attitude_cluster_stream()
    tm_pts  = load_telemetry_cluster_stream()
    print(f"  07_attitude_quats     : {len(att_pts)} DataPoints → d_stream clustering (roll, pitch, pointing_error)")
    print(f"  04_telemetry_decoded  : {len(tm_pts)} DataPoints → d_stream clustering (battery, temp, rssi)")

    energy_states, reliability = load_consensus_energy_states()
    print(f"  08_power+05_isl       : {len(energy_states)} energy states + {len(reliability)} satellite reliability scores → sa_sbft consensus")

    print("\n  Sample GNSS orbit (G01):")
    g = gnss[0]
    print(f"    altitude={g.altitude_km:.1f} km  ecc={g.eccentricity:.6f}  inc={g.inclination_deg:.2f}°")

    print("\n  Sample truth-discovery epoch 0:")
    ep0 = td_epochs[0]
    for sat, val, raw, fault in zip(ep0.satellite_ids, ep0.sensing_values, ep0.raw_signal_dBm, ep0.has_fault):
        print(f"    {sat}: signal={raw:.1f} dBm  normalised={val:.4f}  fault={fault}")


if __name__ == "__main__":
    dataset_summary()

"""
Data Loader for OrbitalChain
=============================
Loads and parses the three CSV datasets:

  1. Active_satellites_in_orbit_July_2016.csv
     Full catalog of 1420 active satellites with orbital parameters.

  2. LAN-SMA-AP-25462-SupGP.csv
     Time-series of a single GEO satellite (NORAD 25462):
     Date/Time, LAN (°), SMA (km), GEO altitude (km).

  3. objects-by-apogee.csv
     Histogram of catalogued space objects by apogee band.
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, List

# Resolve the data directory relative to this file
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "data"))


# ---------------------------------------------------------------------------
# 1. Active Satellites Catalog
# ---------------------------------------------------------------------------

def load_active_satellites(
    orbit_class: Optional[str] = None,
    purpose: Optional[str] = None,
    min_perigee: Optional[float] = None,
    max_apogee: Optional[float] = None,
) -> pd.DataFrame:
    """
    Load Active_satellites_in_orbit_July_2016.csv.

    Parameters
    ----------
    orbit_class : str, optional
        Filter by Class of Orbit: 'LEO', 'MEO', 'GEO', 'Elliptical'.
    purpose : str, optional
        Filter by Purpose (case-insensitive substring match).
    min_perigee : float, optional
        Minimum perigee altitude in km.
    max_apogee : float, optional
        Maximum apogee altitude in km.

    Returns
    -------
    pd.DataFrame with columns:
        name, orbit_class, orbit_type, perigee_km, apogee_km,
        altitude_km (mean), eccentricity, inclination_deg, period_min,
        purpose, operator, country, launch_date, norad
    """
    path = os.path.join(DATA_DIR, "Active_satellites_in_orbit_July_2016.csv")
    raw = pd.read_csv(path, encoding="utf-8-sig")

    # Rename to short, clean column names
    rename = {
        "Official Name of Satellite": "name",
        "Class of Orbit": "orbit_class",
        "Type of Orbit": "orbit_type",
        "Perigee (Kilometers)": "perigee_km",
        "Apogee (Kilometers)": "apogee_km",
        "Eccentricity": "eccentricity",
        "Inclination (Degrees)": "inclination_deg",
        "Period (Minutes)": "period_min",
        "Purpose": "purpose",
        "Operator/Owner": "operator",
        "Country of Operator/Owner": "country",
        "Date of Launch": "launch_date",
        "NORAD Number": "norad",
        "Launch Mass (Kilograms)": "launch_mass_kg",
        "Power (Watts)": "power_w",
    }
    df = raw.rename(columns=rename)

    # Keep only the renamed columns that exist
    keep = [c for c in rename.values() if c in df.columns]
    df = df[keep].copy()

    # Coerce numeric columns
    for col in ["perigee_km", "apogee_km", "eccentricity",
                "inclination_deg", "period_min", "launch_mass_kg", "power_w"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived mean altitude
    df["altitude_km"] = (df["perigee_km"] + df["apogee_km"]) / 2.0

    # Apply filters
    if orbit_class:
        df = df[df["orbit_class"].str.upper() == orbit_class.upper()]
    if purpose:
        mask = df["purpose"].str.contains(purpose, case=False, na=False)
        df = df[mask]
    if min_perigee is not None:
        df = df[df["perigee_km"] >= min_perigee]
    if max_apogee is not None:
        df = df[df["apogee_km"] <= max_apogee]

    return df.reset_index(drop=True)


def get_leo_satellites(max_altitude: float = 2000.0) -> pd.DataFrame:
    """Return LEO satellites with complete orbital parameters."""
    df = load_active_satellites(orbit_class="LEO")
    df = df.dropna(subset=["perigee_km", "apogee_km", "inclination_deg"])
    df = df[df["altitude_km"] <= max_altitude]
    return df.reset_index(drop=True)


def get_geo_satellites() -> pd.DataFrame:
    """Return GEO satellites."""
    return load_active_satellites(orbit_class="GEO")


def sample_satellites(
    n: int = 10,
    orbit_class: str = "LEO",
    seed: int = 42
) -> pd.DataFrame:
    """Return a reproducible random sample of satellites."""
    df = load_active_satellites(orbit_class=orbit_class)
    df = df.dropna(subset=["perigee_km", "apogee_km", "inclination_deg"])
    n = min(n, len(df))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. GEO Satellite Time-Series (NORAD 25462)
# ---------------------------------------------------------------------------

def load_geo_timeseries() -> pd.DataFrame:
    """
    Load LAN-SMA-AP-25462-SupGP.csv.

    Returns
    -------
    pd.DataFrame with columns:
        datetime (pd.Timestamp), lan_deg, sma_km, geo_alt_km
    """
    path = os.path.join(DATA_DIR, "LAN-SMA-AP-25462-SupGP.csv")
    raw = pd.read_csv(path, encoding="utf-8-sig")

    # Normalise the BOM / quoted header
    raw.columns = [c.strip().strip('"').strip() for c in raw.columns]

    rename = {
        "Date/Time (UTC)": "datetime",
        "LAN": "lan_deg",
        "SMA": "sma_km",
        "GEO": "geo_alt_km",
    }
    df = raw.rename(columns=rename)

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for col in ["lan_deg", "sma_km", "geo_alt_km"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["datetime"]).reset_index(drop=True)


def get_geo_altitude_stats() -> dict:
    """Summary statistics for the GEO satellite altitude time-series."""
    df = load_geo_timeseries()
    return {
        "mean_sma_km":    df["sma_km"].mean(),
        "std_sma_km":     df["sma_km"].std(),
        "mean_alt_km":    df["geo_alt_km"].mean(),
        "min_alt_km":     df["geo_alt_km"].min(),
        "max_alt_km":     df["geo_alt_km"].max(),
        "mean_lan_deg":   df["lan_deg"].mean(),
        "num_samples":    len(df),
        "time_start":     str(df["datetime"].min()),
        "time_end":       str(df["datetime"].max()),
    }


# ---------------------------------------------------------------------------
# 3. Objects-by-Apogee Distribution
# ---------------------------------------------------------------------------

def load_apogee_distribution() -> pd.DataFrame:
    """
    Load objects-by-apogee.csv (pipe-separated).

    Returns
    -------
    pd.DataFrame with columns: apogee_band (str), count (int),
        apogee_lower_km (float), apogee_upper_km (float, NaN for '<100')
    """
    path = os.path.join(DATA_DIR, "objects-by-apogee.csv")
    df = pd.read_csv(path, sep="|", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Apogee": "apogee_band", "Count": "count"})
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)

    # Parse numeric bounds from band strings  (e.g. "100-200" → 100, 200)
    lowers, uppers = [], []
    for band in df["apogee_band"]:
        band = str(band).strip()
        if band.startswith("<"):
            lowers.append(0.0)
            uppers.append(float(band[1:]))
        elif "-" in band:
            lo, hi = band.split("-")
            lowers.append(float(lo))
            uppers.append(float(hi))
        elif band.endswith("+"):
            lowers.append(float(band[:-1]))
            uppers.append(np.nan)
        else:
            try:
                v = float(band)
                lowers.append(v)
                uppers.append(v)
            except ValueError:
                lowers.append(np.nan)
                uppers.append(np.nan)

    df["apogee_lower_km"] = lowers
    df["apogee_upper_km"] = uppers
    return df.reset_index(drop=True)


def get_leo_object_count() -> int:
    """Total catalogued objects with apogee ≤ 2000 km."""
    df = load_apogee_distribution()
    leo = df[df["apogee_upper_km"] <= 2000]
    return int(leo["count"].sum())


# ---------------------------------------------------------------------------
# Convenience: orbital parameters dict for a single satellite row
# ---------------------------------------------------------------------------

def satellite_to_orbit_params(row: pd.Series) -> dict:
    """
    Convert a row from the active-satellites DataFrame into a dict
    of orbital parameters ready for KeplerianOrbit / channel model.

    Returns
    -------
    dict with keys: name, altitude_km, perigee_km, apogee_km,
                    eccentricity, inclination_rad, inclination_deg,
                    period_s, semi_major_axis_km
    """
    R_EARTH = 6371.0
    alt = float(row.get("altitude_km", 550))
    perigee = float(row.get("perigee_km", alt))
    apogee = float(row.get("apogee_km", alt))
    ecc = float(row.get("eccentricity", 0.0)) if not pd.isna(row.get("eccentricity", np.nan)) else 0.0
    inc_deg = float(row.get("inclination_deg", 0.0)) if not pd.isna(row.get("inclination_deg", np.nan)) else 0.0
    period_min = row.get("period_min", np.nan)
    period_s = float(period_min) * 60 if not pd.isna(period_min) else None

    a = R_EARTH + alt  # semi-major axis (approx for near-circular)

    return {
        "name": str(row.get("name", "Unknown")),
        "altitude_km": alt,
        "perigee_km": perigee,
        "apogee_km": apogee,
        "eccentricity": ecc,
        "inclination_deg": inc_deg,
        "inclination_rad": np.radians(inc_deg),
        "period_s": period_s,
        "semi_major_axis_km": a,
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== OrbitalChain Data Loader ===\n")

    # Active satellites
    all_sats = load_active_satellites()
    print(f"Total active satellites loaded : {len(all_sats)}")
    for cls in ["LEO", "MEO", "GEO", "Elliptical"]:
        n = len(load_active_satellites(orbit_class=cls))
        print(f"  {cls:12s}: {n}")

    print()
    leo = get_leo_satellites()
    print(f"LEO satellites with complete params: {len(leo)}")
    print("Sample LEO entries:")
    print(leo[["name", "altitude_km", "inclination_deg", "eccentricity"]].head(5).to_string(index=False))

    # GEO time-series
    print("\n--- GEO Time-Series (NORAD 25462) ---")
    stats = get_geo_altitude_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Apogee distribution
    print("\n--- Objects by Apogee ---")
    dist = load_apogee_distribution()
    print(dist[["apogee_band", "count"]].head(10).to_string(index=False))
    print(f"\nTotal LEO objects (apogee ≤ 2000 km): {get_leo_object_count():,}")

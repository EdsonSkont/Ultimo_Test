import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import math

MU_EARTH = 398600.4418
R_EARTH  = 6371.0
J2       = 1.08263e-3


@dataclass
class KeplerianOrbit:
    a: float
    e: float
    i: float
    omega: float
    Omega: float
    nu: float
    epoch: float

    @classmethod
    def from_catalog_row(
        cls,
        altitude_km:      float,
        eccentricity:     float = 0.001,
        inclination_deg:  float = 53.0,
        raan_deg:         float = 0.0,
        arg_perigee_deg:  float = 0.0,
        true_anomaly_deg: float = 0.0,
        epoch_jd:         float = 2460000.5,
    ) -> 'KeplerianOrbit':
        return cls(
            a     = R_EARTH + altitude_km,
            e     = max(eccentricity, 1e-6),
            i     = np.radians(inclination_deg),
            omega = np.radians(arg_perigee_deg),
            Omega = np.radians(raan_deg),
            nu    = np.radians(true_anomaly_deg),
            epoch = epoch_jd,
        )

    @classmethod
    def from_tle(cls, tle_line1: str, tle_line2: str) -> 'KeplerianOrbit':
        i     = float(tle_line2[8:16])  * np.pi / 180
        Omega = float(tle_line2[17:25]) * np.pi / 180
        e     = float('0.' + tle_line2[26:33])
        omega = float(tle_line2[34:42]) * np.pi / 180
        M     = float(tle_line2[43:51]) * np.pi / 180
        n     = float(tle_line2[52:63])
        n_rad = n * 2 * np.pi / 86400
        a     = (MU_EARTH / n_rad**2) ** (1/3)
        E     = cls._solve_kepler(M, e)
        nu    = 2 * np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
        year  = int(tle_line1[18:20])
        year  = 2000+year if year < 57 else 1900+year
        day   = float(tle_line1[20:32])
        epoch = cls._ymd_to_jd(year,1,1) + day - 1
        return cls(a=a, e=e, i=i, omega=omega, Omega=Omega, nu=nu, epoch=epoch)

    @staticmethod
    def _solve_kepler(M: float, e: float, tol: float = 1e-10) -> float:
        E = M
        for _ in range(100):
            E_new = M + e * np.sin(E)
            if abs(E_new - E) < tol: return E_new
            E = E_new
        return E

    @staticmethod
    def _ymd_to_jd(year, month, day):
        if month <= 2: year -= 1; month += 12
        A = int(year/100); B = 2 - A + int(A/4)
        return int(365.25*(year+4716)) + int(30.6001*(month+1)) + day + B - 1524.5

    def to_state_vector(self) -> Tuple[np.ndarray, np.ndarray]:
        p     = self.a * (1 - self.e**2)
        r_mag = p / (1 + self.e * np.cos(self.nu))
        r_pqw = np.array([r_mag*np.cos(self.nu), r_mag*np.sin(self.nu), 0])
        v_pqw = np.sqrt(MU_EARTH/p) * np.array([-np.sin(self.nu), self.e+np.cos(self.nu), 0])
        Q = self._rotation_matrix(self.Omega,3) @ self._rotation_matrix(self.i,1) @ self._rotation_matrix(self.omega,3)
        return Q @ r_pqw, Q @ v_pqw

    @staticmethod
    def _rotation_matrix(angle, axis):
        c, s = np.cos(angle), np.sin(angle)
        if axis == 1: return np.array([[1,0,0],[0,c,-s],[0,s,c]])
        if axis == 2: return np.array([[c,0,s],[0,1,0],[-s,0,c]])
        return np.array([[c,-s,0],[s,c,0],[0,0,1]])

    @property
    def period(self): return 2*np.pi*np.sqrt(self.a**3/MU_EARTH)

    @property
    def altitude(self):
        r, _ = self.to_state_vector()
        return np.linalg.norm(r) - R_EARTH


class OrbitalPropagator:
    def __init__(self, include_j2: bool = True):
        self.include_j2 = include_j2

    def propagate(self, orbit: KeplerianOrbit, dt: float) -> KeplerianOrbit:
        n = np.sqrt(MU_EARTH / orbit.a**3)
        if self.include_j2:
            p         = orbit.a * (1 - orbit.e**2)
            dOmega_dt = -1.5*n*J2*(R_EARTH/p)**2*np.cos(orbit.i)
            domega_dt =  0.75*n*J2*(R_EARTH/p)**2*(5*np.cos(orbit.i)**2-1)
        else:
            dOmega_dt = domega_dt = 0
        M     = self._true_to_mean(orbit.nu, orbit.e)
        M_new = M + n*dt
        E_new = KeplerianOrbit._solve_kepler(M_new % (2*np.pi), orbit.e)
        nu_new = 2*np.arctan2(np.sqrt(1+orbit.e)*np.sin(E_new/2), np.sqrt(1-orbit.e)*np.cos(E_new/2))
        return KeplerianOrbit(a=orbit.a, e=orbit.e, i=orbit.i,
                              omega=orbit.omega+domega_dt*dt, Omega=orbit.Omega+dOmega_dt*dt,
                              nu=nu_new, epoch=orbit.epoch+dt/86400.0)

    @staticmethod
    def _true_to_mean(nu, e):
        E = 2*np.arctan2(np.sqrt(1-e)*np.sin(nu/2), np.sqrt(1+e)*np.cos(nu/2))
        return E - e*np.sin(E)


class VisibilityCalculator:
    def __init__(self, min_elevation: float = 10.0):
        self.min_elevation = min_elevation

    def compute_elevation(self, sat_pos: np.ndarray, gs_pos: np.ndarray) -> float:
        r = sat_pos - gs_pos
        local_up = gs_pos / np.linalg.norm(gs_pos)
        cos_z = np.dot(r, local_up) / np.linalg.norm(r)
        return 90 - np.degrees(np.arccos(np.clip(cos_z, -1, 1)))

    def is_visible(self, sat_pos, gs_pos):
        return self.compute_elevation(sat_pos, gs_pos) >= self.min_elevation

    def compute_visibility_window(self, orbit, ground_position, start_time,
                                   duration, time_step=10.0):
        propagator = OrbitalPropagator()
        windows, current_orbit = [], orbit
        in_vis, window_start = False, None
        for t in np.arange(0, duration, time_step):
            sat_pos, _ = current_orbit.to_state_vector()
            visible = self.is_visible(sat_pos, ground_position)
            if visible and not in_vis:
                window_start = start_time + t/86400.0; in_vis = True
            elif not visible and in_vis:
                windows.append((window_start, start_time+t/86400.0)); in_vis = False
            current_orbit = propagator.propagate(current_orbit, time_step)
        if in_vis: windows.append((window_start, start_time+duration/86400.0))
        return windows


def compute_orbital_period(altitude: float) -> float:
    return 2*np.pi*np.sqrt((R_EARTH+altitude)**3/MU_EARTH)


def compute_visibility_window(altitude: float, latitude: float, min_elevation: float = 10.0) -> float:
    Re, theta_min = R_EARTH, np.radians(min_elevation)
    rho = np.arcsin(Re/(Re+altitude)*np.cos(theta_min))
    lambda_arc = np.pi - 2*(theta_min+rho)
    v = np.sqrt(MU_EARTH/(Re+altitude))
    return lambda_arc*(Re+altitude)/v


def eci_to_geodetic(position: np.ndarray, time_jd: float) -> Tuple[float, float, float]:
    theta_g = (280.46061837 + 360.98564736629*(time_jd-2451545.0)) % 360.0
    theta   = np.radians(theta_g)
    R = np.array([[np.cos(theta),np.sin(theta),0],[-np.sin(theta),np.cos(theta),0],[0,0,1]])
    pos = R @ position
    x, y, z = pos
    return np.degrees(np.arctan2(z, np.sqrt(x**2+y**2))), np.degrees(np.arctan2(y,x)), np.linalg.norm(pos)-R_EARTH


# Default data source: 06_gnss_ephemeris.csv (best fit — full Keplerian elements)
# Fallback:            Active_satellites_in_orbit_July_2016.csv (perigee/apogee/inc only)

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    use_random = "--random" in sys.argv

    propagator = OrbitalPropagator(include_j2=True)

    if use_random:
        # Fallback: single hardcoded orbit for quick testing
        print("\nOrbital Mechanics  —  test orbit (--random mode)\n")
        orbit = KeplerianOrbit.from_catalog_row(altitude_km=550, eccentricity=0.0001,
                                                  inclination_deg=53.0)
        print(f"  Period  : {orbit.period/60:.2f} min")
        print(f"  Altitude: {orbit.altitude:.2f} km")
        o1h = propagator.propagate(orbit, 3600)
        print(f"  After 1h true anomaly: {np.degrees(o1h.nu):.2f} deg")

    else:
        # Default: real GNSS ephemeris — 24 GPS SVs, full Keplerian elements
        from src.data.ccsds_adapter import load_gnss_orbits
        svs = load_gnss_orbits(epoch_index=0)

        print(f"\nOrbital Mechanics  —  06_gnss_ephemeris.csv  ({len(svs)} GPS SVs)\n")
        print(f"  Data: full Keplerian elements (SMA, ecc, inc, RAAN, arg_perigee, mean_anomaly)")
        print(f"  Propagator: J2-perturbed  |  Ground station: Chengdu 30.7N 104.1E\n")

        hdr = f"{'SV':<5} {'Alt(km)':>8} {'Inc(deg)':>9} {'Ecc':>9} {'Period(min)':>12} {'dOmega/orbit(deg)':>18} {'Vis(min)':>9}"
        print(hdr)
        print("-" * len(hdr))

        lat_gs, lon_gs = 30.7, 104.1
        gs = R_EARTH * np.array([
            np.cos(np.radians(lat_gs))*np.cos(np.radians(lon_gs)),
            np.cos(np.radians(lat_gs))*np.sin(np.radians(lon_gs)),
            np.sin(np.radians(lat_gs)),
        ])

        for sv in svs:
            orbit = KeplerianOrbit.from_catalog_row(
                altitude_km=sv.altitude_km, eccentricity=sv.eccentricity,
                inclination_deg=sv.inclination_deg, raan_deg=sv.raan_deg,
                arg_perigee_deg=sv.arg_perigee_deg, true_anomaly_deg=sv.mean_anomaly_deg,
            )
            T = orbit.period
            dOmega = np.degrees(-1.5*np.sqrt(MU_EARTH/orbit.a**3)*J2
                                *(R_EARTH/(orbit.a*(1-orbit.e**2)))**2*np.cos(orbit.i)*T)
            vis = compute_visibility_window(sv.altitude_km, lat_gs) / 60
            print(f"  {sv.name:<5} {sv.altitude_km:>8.1f} {sv.inclination_deg:>9.2f} "
                  f"{sv.eccentricity:>9.5f} {T/60:>12.2f} {dOmega:>18.5f} {vis:>9.1f}")

        print(f"\n  Note: GPS SVs at ~20,200 km have 12-hour periods and wide visibility windows.")
        print(f"  J2 precession is ~0.019 deg/orbit — much smaller than for LEO satellites.")

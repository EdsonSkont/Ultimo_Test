import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

SPEED_OF_LIGHT = 299792.458  # km/s
BOLTZMANN      = 1.38e-23    # J/K


@dataclass
class LinkParameters:
    frequency:        float
    transmit_power:   float
    transmit_gain:    float
    receive_gain:     float
    system_noise_temp: float
    bandwidth:        float

    @property
    def wavelength(self):
        return SPEED_OF_LIGHT / self.frequency * 1e3


class RicianChannel:
    def __init__(self, k_factor: float = 10.0, doppler_shift: float = 0.0):
        self.k_factor     = k_factor
        self.doppler_shift = doppler_shift

    def generate_coefficient(self, num_samples: int = 1) -> np.ndarray:
        h_los  = np.sqrt(self.k_factor / (self.k_factor + 1))
        h_nlos = ((np.random.randn(num_samples) + 1j*np.random.randn(num_samples))
                  / np.sqrt(2) * np.sqrt(1/(self.k_factor+1)))
        return h_los + h_nlos

    def get_fading_loss(self, num_samples: int = 1) -> np.ndarray:
        return 10 * np.log10(np.abs(self.generate_coefficient(num_samples))**2)

    @staticmethod
    def compute_k_factor(elevation: float) -> float:
        if elevation < 10:  return 1.0
        if elevation < 30:  return 5.0 + (elevation-10)*0.5
        if elevation < 60:  return 15.0 + (elevation-30)*0.3
        return 24.0 + (elevation-60)*0.1


class ISLChannel:
    def __init__(self, wavelength_nm: float = 1550.0, pointing_error_urad: float = 1.0):
        self.wavelength      = wavelength_nm * 1e-9
        self.pointing_error  = pointing_error_urad * 1e-6

    def compute_geometric_loss(self, distance: float) -> float:
        return 20 * np.log10(4 * np.pi * distance*1000 / self.wavelength)

    def compute_pointing_loss(self, beam_divergence: float = 10e-6) -> float:
        return 4.343 * (self.pointing_error/beam_divergence)**2

    def compute_total_loss(self, distance: float) -> float:
        return self.compute_geometric_loss(distance) + self.compute_pointing_loss()


def compute_path_loss(distance: float, frequency: float, elevation: float = 90.0) -> float:
    wavelength = SPEED_OF_LIGHT*1e3 / frequency
    fspl = 20*np.log10(4*np.pi*distance*1000/wavelength)
    atm  = 2.0 if elevation < 10 else (0.5 if elevation < 30 else 0.2)
    return fspl + atm


def compute_link_budget(params: LinkParameters, distance: float, elevation: float,
                        include_fading: bool = True) -> dict:
    path_loss      = compute_path_loss(distance, params.frequency, elevation)
    eirp           = params.transmit_power + params.transmit_gain
    received_power = eirp - path_loss + params.receive_gain
    noise_power    = 10*np.log10(BOLTZMANN*params.system_noise_temp*params.bandwidth)
    snr            = received_power - noise_power
    fading_margin  = -np.percentile(RicianChannel(RicianChannel.compute_k_factor(elevation))
                                    .get_fading_loss(10000), 1) if include_fading else 0
    effective_snr  = snr - fading_margin
    return dict(eirp_dBW=eirp, path_loss_dB=path_loss, received_power_dBW=received_power,
                noise_power_dBW=noise_power, snr_dB=snr, fading_margin_dB=fading_margin,
                effective_snr_dB=effective_snr)


def compute_data_rate(snr: float, bandwidth: float, efficiency: float = 0.8) -> float:
    return bandwidth * np.log2(1 + snr) * efficiency


def compute_slant_range(altitude_km: float, elevation_deg: float) -> float:
    Re = 6371.0
    theta = np.radians(elevation_deg)
    return max(np.sqrt((Re+altitude_km)**2 - (Re*np.cos(theta))**2) - Re*np.sin(theta), altitude_km)


class SatelliteLink:
    def __init__(self, link_params: LinkParameters, channel_type: str = 'rician'):
        self.params       = link_params
        self.channel_type = channel_type

    def compute_link_quality(self, distance: float, elevation: float = 90.0) -> dict:
        budget      = compute_link_budget(self.params, distance, elevation,
                                          include_fading=(self.channel_type=='rician'))
        snr_linear  = 10**(budget['effective_snr_dB']/10)
        data_rate   = compute_data_rate(snr_linear, self.params.bandwidth)
        latency     = distance / SPEED_OF_LIGHT * 1000
        ber         = 0.5*np.exp(-snr_linear/2) if snr_linear > 0 else 0.5
        return {**budget, 'data_rate_bps': data_rate, 'latency_ms': latency,
                'ber': ber, 'link_margin_dB': budget['effective_snr_dB']-10}


# Default data source: 08_power_subsystem.csv  (orbit_phase → elevation, eclipse, 480 epochs)
#                    + 05_isl_stream_log.json  (measured signal_strength_dBm per satellite)
# Fallback:          --random  for quick testing with fixed parameters

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    use_random = "--random" in sys.argv

    # Ka-band terminal parameters (constant in both modes)
    params = LinkParameters(frequency=26.5e9, transmit_power=10, transmit_gain=35,
                            receive_gain=40, system_noise_temp=300, bandwidth=500e6)
    link = SatelliteLink(params, 'rician')

    if use_random:
        print("\nChannel Model  —  fixed parameters (--random mode)\n")
        alt_km = 550.0
        print(f"  {'Elevation':>10} {'PathLoss(dB)':>13} {'SNR(dB)':>8} {'Rate(Gbps)':>11} {'Lat(ms)':>8}")
        print("-" * 55)
        for elev in [10, 30, 60, 90]:
            dist = compute_slant_range(alt_km, elev)
            q    = link.compute_link_quality(dist, elev)
            print(f"  {elev:>10}° {q['path_loss_dB']:>13.1f} {q['effective_snr_dB']:>8.1f} "
                  f"{q['data_rate_bps']/1e9:>11.3f} {q['latency_ms']:>8.2f}")

    else:
        # Real data: orbit phases from power subsystem + measured RSSI from ISL log
        from src.data.ccsds_adapter import load_link_observations, load_isl_link_quality
        obs      = load_link_observations()
        isl_qual = load_isl_link_quality()

        alt_km = 780.0  # from 10_dataset_metadata.json
        print(f"\nChannel Model  —  real satellite data\n")
        print(f"  Elevation source : 08_power_subsystem.csv  ({len(obs)} epochs, 1-min, 5 orbits)")
        print(f"  RSSI source      : 05_isl_stream_log.json  ({sum(len(v) for v in isl_qual.values())} packets, {len(isl_qual)} satellites)")
        print(f"  Orbit altitude   : {alt_km} km  (from dataset metadata)\n")

        # Section 1: link budget driven by real orbit phase
        print("  Link budget over one orbit (sampled every 9 minutes)\n")
        print(f"  {'Time':>8} {'Phase(°)':>9} {'Elev(°)':>8} {'Eclipse':>8} "
              f"{'Solar(W)':>9} {'PathLoss':>9} {'SNR(dB)':>8} {'Rate(Gbps)':>11} {'Lat(ms)':>8}")
        print("  " + "-" * 82)
        # One orbit ≈ 96 min at 780 km → sample every 9 min = ~10 points per orbit
        for o in obs[:96:9]:
            elev = max(o.elevation_deg, 5.0)
            dist = compute_slant_range(alt_km, elev)
            q    = link.compute_link_quality(dist, elev)
            ecl  = "YES" if o.eclipse else "no"
            print(f"  {o.timestamp[11:19]:>8} {o.orbit_phase_deg:>9.1f} {o.elevation_deg:>8.1f} "
                  f"{ecl:>8} {o.solar_W:>9.1f} {q['path_loss_dB']:>9.1f} "
                  f"{q['effective_snr_dB']:>8.1f} {q['data_rate_bps']/1e9:>11.3f} {q['latency_ms']:>8.2f}")

        # Section 2: per-satellite ISL measured vs computed SNR
        print(f"\n  ISL per-satellite link quality (measured vs computed)\n")
        print(f"  {'Satellite':>10} {'Pkts':>5} {'Measured mean(dBm)':>19} {'Measured std':>13} "
              f"{'Computed SNR(dB)':>17} {'CRC fails':>10}")
        print("  " + "-" * 80)
        for sat, pkts in sorted(isl_qual.items()):
            sigs  = [p["signal_dBm"] for p in pkts if p["signal_dBm"] is not None]
            fails = sum(1 for p in pkts if not p["crc_valid"])
            # Compute expected SNR at mean ISL distance (5 satellites in 780-km LEO → ~1200 km apart)
            isl_dist = 1200.0
            isl_ch   = ISLChannel()
            isl_loss = isl_ch.compute_total_loss(isl_dist)
            comp_snr = params.transmit_power + params.transmit_gain - isl_loss + params.receive_gain \
                       - 10*np.log10(BOLTZMANN*params.system_noise_temp*params.bandwidth)
            print(f"  {sat:>10} {len(pkts):>5} {np.mean(sigs):>19.2f} {np.std(sigs):>13.2f} "
                  f"{comp_snr:>17.2f} {fails:>10}")

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

SPEED_OF_LIGHT = 299792.458  # km/s
BOLTZMANN = 1.38e-23  # J/K


# Parameters for a communication link
@dataclass
class LinkParameters:
    frequency: float  # Hz
    transmit_power: float  # dBW
    transmit_gain: float  # dBi
    receive_gain: float  # dBi
    system_noise_temp: float  # K
    bandwidth: float  # Hz
    
    @property
    def wavelength(self) -> float:
        """Wavelength in km."""
        return SPEED_OF_LIGHT / self.frequency * 1e3  # Convert to km

# Rician fading channel model for satellite links
class RicianChannel:    
    def __init__(
        self,
        k_factor: float = 10.0,
        doppler_shift: float = 0.0
    ):
        self.k_factor = k_factor
        self.doppler_shift = doppler_shift
    
    def generate_coefficient(self, num_samples: int = 1) -> np.ndarray:
        # LoS component
        h_los = np.sqrt(self.k_factor / (self.k_factor + 1))
        
        # NLoS component (Rayleigh)
        h_nlos_real = np.random.randn(num_samples) / np.sqrt(2)
        h_nlos_imag = np.random.randn(num_samples) / np.sqrt(2)
        h_nlos = (h_nlos_real + 1j * h_nlos_imag) * np.sqrt(1 / (self.k_factor + 1))
        
        # Combined channel
        h = h_los + h_nlos
        
        return h
    
    # Get fading loss in dB.
    def get_fading_loss(self, num_samples: int = 1) -> np.ndarray:
        h = self.generate_coefficient(num_samples)
        power = np.abs(h) ** 2
        return 10 * np.log10(power)
    
    """
    Estimate K-factor based on elevation angle.
    
    Higher elevation = stronger LoS = higher K-factor.
    """
    @staticmethod
    def compute_k_factor(elevation: float) -> float:
        # Empirical model based on elevation
        if elevation < 10:
            return 1.0  # Low elevation, multipath dominant
        elif elevation < 30:
            return 5.0 + (elevation - 10) * 0.5
        elif elevation < 60:
            return 15.0 + (elevation - 30) * 0.3
        else:
            return 24.0 + (elevation - 60) * 0.1


# Inter-Satellite Link channel model.
class ISLChannel:    
    def __init__(
        self,
        wavelength_nm: float = 1550.0,  # Optical ISL
        pointing_error_urad: float = 1.0
    ):
        self.wavelength = wavelength_nm * 1e-9  # Convert to meters
        self.pointing_error = pointing_error_urad * 1e-6  # Convert to radians
    
    #Compute geometric spreading loss
    def compute_geometric_loss(self, distance: float) -> float:
        # Free space path loss for optical
        distance_m = distance * 1000
        loss = 20 * np.log10(4 * np.pi * distance_m / self.wavelength)
        return loss
    
    # Compute pointing loss due to misalignment
    def compute_pointing_loss(self, beam_divergence: float = 10e-6) -> float:
        # Gaussian beam model
        theta_ratio = self.pointing_error / beam_divergence
        loss = 4.343 * theta_ratio ** 2  # dB
        return loss
    
    # Compute total ISL loss
    def compute_total_loss(self, distance: float) -> float:
        geom_loss = self.compute_geometric_loss(distance)
        point_loss = self.compute_pointing_loss()
        return geom_loss + point_loss


# Compute path loss for satellite link
def compute_path_loss(
    distance: float,
    frequency: float,
    elevation: float = 90.0
) -> float:
    wavelength = SPEED_OF_LIGHT * 1e3 / frequency  # meters
    distance_m = distance * 1000
    fspl = 20 * np.log10(4 * np.pi * distance_m / wavelength)
    
    # Atmospheric loss (based on elevation)
    if elevation < 10:
        atm_loss = 2.0  # dB
    elif elevation < 30:
        atm_loss = 0.5
    else:
        atm_loss = 0.2
    
    return fspl + atm_loss


# Compute link budget for satellite link
def compute_link_budget(
    params: LinkParameters,
    distance: float,
    elevation: float,
    include_fading: bool = True
) -> dict:
    # Path loss
    path_loss = compute_path_loss(distance, params.frequency, elevation)
    
    eirp = params.transmit_power + params.transmit_gain
    
    # Received power
    received_power = eirp - path_loss + params.receive_gain
    
    # Noise power
    noise_power = 10 * np.log10(BOLTZMANN * params.system_noise_temp * params.bandwidth)
    
    snr = received_power - noise_power
    
    # Fading margin
    if include_fading:
        k_factor = RicianChannel.compute_k_factor(elevation)
        channel = RicianChannel(k_factor)
        fading_samples = channel.get_fading_loss(10000)
        fading_margin = -np.percentile(fading_samples, 1)
    else:
        fading_margin = 0
    
    effective_snr = snr - fading_margin
    
    return {
        'eirp_dBW': eirp,
        'path_loss_dB': path_loss,
        'received_power_dBW': received_power,
        'noise_power_dBW': noise_power,
        'snr_dB': snr,
        'fading_margin_dB': fading_margin,
        'effective_snr_dB': effective_snr
    }


# Estimate achievable data rate
def compute_data_rate(
    snr: float,
    bandwidth: float,
    efficiency: float = 0.8
) -> float:
    capacity = bandwidth * np.log2(1 + snr) * efficiency
    return capacity

# Initialize satellite link
class SatelliteLink:    
    def __init__(
        self,
        link_params: LinkParameters,
        channel_type: str = 'rician'
    ):
        self.params = link_params
        self.channel_type = channel_type
    
    def compute_link_quality(
        self,
        distance: float,
        elevation: float = 90.0
    ) -> dict:
        budget = compute_link_budget(
            self.params, distance, elevation,
            include_fading=(self.channel_type == 'rician')
        )
        
        # Convert SNR to linear
        snr_linear = 10 ** (budget['effective_snr_dB'] / 10)
        
        # Data rate
        data_rate = compute_data_rate(
            snr_linear, self.params.bandwidth
        )
        
        # Latency
        latency = distance / SPEED_OF_LIGHT * 1000  # ms
        
        ber = 0.5 * np.exp(-snr_linear / 2) if snr_linear > 0 else 0.5
        
        return {
            **budget,
            'data_rate_bps': data_rate,
            'latency_ms': latency,
            'ber': ber,
            'link_margin_dB': budget['effective_snr_dB'] - 10 
        }


if __name__ == "__main__":
    print("=== Channel Model Test ===\n")
    
    # Define link parameters
    params = LinkParameters(
        frequency=26.5e9,  # 26.5 GHz
        transmit_power=10,  # 10 dBW
        transmit_gain=35,  # 35 dBi
        receive_gain=40,  # 40 dBi
        system_noise_temp=300,  # 300 K
        bandwidth=500e6  # 500 MHz
    )
    
    # Test at different elevations
    print("Link budget analysis for LEO satellite at 550 km:\n")
    print(f"{'Elevation':<12} {'Path Loss':<12} {'SNR':<10} {'Data Rate':<15} {'Latency':<10}")
    print("-" * 60)
    
    link = SatelliteLink(params, 'rician')
    
    for elevation in [10, 30, 60, 90]:
        # Approximate slant range
        h = 550  # km
        Re = 6371  # km
        theta = np.radians(elevation)
        distance = np.sqrt((Re + h)**2 - (Re * np.cos(theta))**2) - Re * np.sin(theta)
        
        quality = link.compute_link_quality(distance, elevation)
        
        print(f"{elevation}°{'':<9} {quality['path_loss_dB']:<12.1f} "
              f"{quality['effective_snr_dB']:<10.1f} "
              f"{quality['data_rate_bps']/1e9:<15.2f} Gbps "
              f"{quality['latency_ms']:<10.2f} ms")
    
    print("\n\nISL between satellites 1000 km apart:")
    isl = ISLChannel()
    isl_loss = isl.compute_total_loss(1000)
    print(f"Total ISL loss: {isl_loss:.1f} dB")

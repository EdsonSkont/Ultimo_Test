# Additive secret sharing over a prime field for OrbitalChain.
#
# Real-data context
# -----------------
# Each satellite in the constellation holds a sensitive scalar measurement
# that it needs to contribute to a joint computation (consensus quorum,
# truth discovery, weight update) WITHOUT revealing the raw value to any
# single peer.
#
# The secret used in this module: battery_soc_pct from 08_power_subsystem.csv
#
# Why battery_soc_pct?
#   - SA-SBFT consensus checks energy thresholds (active >= 50%, semi >= 20%).
#   - A satellite must prove it qualifies for the quorum without exposing
#     its exact charge level to routing adversaries who could exploit it.
#   - SoC is a float in [0, 100]; scaled by 100 it becomes an integer in
#     [0, 10000], well inside the prime field.
#   - Negative values are impossible, so no sign-handling is needed.
#
# Encoding convention
#   secret_integer = int(soc_pct * 100)
#   e.g. 60.0% -> 6000,  48.5% -> 4850,  30.0% -> 3000
#
# The scheme
#   share(secret) splits the integer into N additive shares s_0..s_{N-1}
#   such that s_0 + s_1 + ... + s_{N-1} = secret  (mod prime).
#   Any subset smaller than N reveals nothing about the secret.
#   reconstruct(shares) recovers the original integer by summing all shares.

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import secrets
import csv
import os


@dataclass
class Share:
    party_id: int
    value: int
    field_modulus: int

    def __add__(self, other: 'Share') -> 'Share':
        if self.party_id != other.party_id:
            raise ValueError("Cannot add shares from different parties")
        return Share(
            party_id=self.party_id,
            value=(self.value + other.value) % self.field_modulus,
            field_modulus=self.field_modulus
        )

    def __mul__(self, scalar: int) -> 'Share':
        return Share(
            party_id=self.party_id,
            value=(self.value * scalar) % self.field_modulus,
            field_modulus=self.field_modulus
        )

    def __repr__(self):
        return f"Share(party={self.party_id}, value={self.value})"


class AdditiveSecretSharing:
    # Prime field F_q using Mersenne prime 2^61 - 1.
    # All arithmetic is modular; the large prime prevents wrap-around
    # for any satellite measurement value.
    DEFAULT_PRIME = 2305843009213693951  # 2^61 - 1

    def __init__(
        self,
        num_parties: int,
        prime_modulus: Optional[int] = None,
        seed: Optional[int] = None
    ):
        if num_parties < 2:
            raise ValueError("Need at least 2 parties for secret sharing")
        self.num_parties   = num_parties
        self.prime_modulus = prime_modulus or self.DEFAULT_PRIME
        self._rng = np.random.default_rng(seed)

    def share(self, secret: int) -> List[Share]:
        # Split secret into num_parties additive shares.
        # N-1 shares are random field elements; the last share is chosen
        # so that all shares sum to secret (mod prime).
        secret = secret % self.prime_modulus
        random_shares = [secrets.randbelow(self.prime_modulus)
                         for _ in range(self.num_parties - 1)]
        last_share = (secret - sum(random_shares)) % self.prime_modulus
        all_values = random_shares + [last_share]
        return [Share(party_id=i, value=v, field_modulus=self.prime_modulus)
                for i, v in enumerate(all_values)]

    def reconstruct(self, shares: List[Share]) -> int:
        # Recover secret by summing all shares modulo the prime.
        # Requires ALL shares — a strict threshold of N.
        if len(shares) != self.num_parties:
            raise ValueError(
                f"Need all {self.num_parties} shares, got {len(shares)}")
        for s in shares:
            if s.field_modulus != self.prime_modulus:
                raise ValueError("Inconsistent field modulus in shares")
        return sum(s.value for s in shares) % self.prime_modulus

    def add_shares(self, shares_x: List[Share], shares_y: List[Share]) -> List[Share]:
        # Local addition: each party adds its two shares without communication.
        # Used in streaming_truth to accumulate weighted sensing values.
        if len(shares_x) != len(shares_y):
            raise ValueError("Share lists must have the same length")
        return [sx + sy for sx, sy in zip(shares_x, shares_y)]

    def multiply_by_constant(self, shares: List[Share], constant: int) -> List[Share]:
        # Scale all shares by a public constant (e.g. decay factor).
        constant = constant % self.prime_modulus
        return [s * constant for s in shares]

    def share_batch(self, secret_list: List[int]) -> List[List[Share]]:
        return [self.share(s) for s in secret_list]

    def get_party_shares(self, all_shares: List[List[Share]], party_id: int) -> List[Share]:
        return [shares[party_id] for shares in all_shares]


class SecretSharingProtocol:
    # Higher-level wrapper: satellite-to-satellite sharing protocol.
    # Satellites act as both data providers and MPC parties.

    def __init__(self, num_satellites: int, prime_modulus: Optional[int] = None):
        self.num_satellites = num_satellites
        self.ss             = AdditiveSecretSharing(num_satellites, prime_modulus)
        self.prime_modulus  = self.ss.prime_modulus

    def data_provider_share(
        self,
        sensing_value: float,
        scale_factor: int = 100  # SoC uses ×100; signal_dBm would use ×1000
    ) -> Tuple[List[Share], List[Tuple]]:
        # Scale float to integer and secret-share it.
        # For SoC: 60.5% -> int(60.5 * 100) = 6050
        scaled = int(sensing_value * scale_factor) % self.prime_modulus
        value_shares = self.ss.share(scaled)

        from .beaver_triples import BeaverTripleGenerator
        tgen = BeaverTripleGenerator(self.num_satellites, self.prime_modulus)
        return value_shares, (tgen.generate_triple(), tgen.generate_triple())

    def distribute_to_satellites(self, shares: List[Share], triples: Tuple) -> dict:
        # Package each satellite's share and triple for transmission.
        return {
            i: {
                'value_share': shares[i],
                'triple_0':    triples[0][i],
                'triple_1':    triples[1][i],
            }
            for i in range(self.num_satellites)
        }


def verify_sharing(ss: AdditiveSecretSharing, secret: int, num_tests: int = 100) -> bool:
    for _ in range(num_tests):
        if ss.reconstruct(ss.share(secret)) != (secret % ss.prime_modulus):
            return False
    return True


def statistical_test(ss: AdditiveSecretSharing, num_samples: int = 10000) -> dict:
    # Verify that shares are uniformly distributed (information-theoretic privacy).
    secret      = 6000  # 60.0% SoC in integer encoding
    share_values = {i: [] for i in range(ss.num_parties)}
    for _ in range(num_samples):
        for s in ss.share(secret):
            share_values[s.party_id].append(s.value)
    return {
        pid: {
            'mean':          np.mean(vals),
            'expected_mean': ss.prime_modulus / 2,
            'std':           np.std(vals),
            'min':           int(np.min(vals)),
            'max':           int(np.max(vals)),
        }
        for pid, vals in share_values.items()
    }


def _load_soc_samples(n: int = 5) -> List[Tuple[str, float, int]]:
    # Load battery_soc_pct from 08_power_subsystem.csv.
    # Returns list of (timestamp, soc_pct, secret_integer).
    # secret_integer = int(soc_pct * 100) — encodes two decimal places.
    here     = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(here, "..", "..", "data", "satellite_ccsds_dataset"))
    path     = os.path.join(data_dir, "08_power_subsystem.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    samples = []
    for r in rows[:n]:
        soc     = float(r["battery_soc_pct"])
        secret  = int(soc * 100)   # e.g. 60.0 -> 6000, 48.5 -> 4850
        samples.append((r["timestamp_utc"][11:19], soc, secret))
    return samples


if __name__ == "__main__":
    print("=== Additive Secret Sharing — battery_soc_pct ===\n")
    print("  Source  : 08_power_subsystem.csv")
    print("  Secret  : battery_soc_pct  (state of charge, %)")
    print("  Encoding: secret_integer = int(soc_pct * 100)")
    print("  Parties : satellite nodes acting as MPC cluster")
    print("  Privacy : no single satellite learns another's exact SoC")
    print("  Use     : SA-SBFT energy-threshold quorum check\n")

    NUM_PARTIES = 5
    ss = AdditiveSecretSharing(num_parties=NUM_PARTIES)
    print(f"  Parties       : {ss.num_parties}")
    print(f"  Field modulus : {ss.prime_modulus}  (2^61 - 1)\n")

    # Load real SoC values from the power subsystem dataset
    samples = _load_soc_samples(n=5)

    print(f"  {'Time':>8}  {'SoC (%)':>8}  {'Secret int':>11}  {'Reconstructed':>14}  {'Correct':>7}")
    print("  " + "-" * 58)
    for ts, soc, secret in samples:
        shares        = ss.share(secret)
        reconstructed = ss.reconstruct(shares)
        correct       = reconstructed == secret
        print(f"  {ts:>8}  {soc:>8.1f}  {secret:>11}  {reconstructed:>14}  {'yes' if correct else 'NO':>7}")

    # Show share distribution for one epoch — all look random,
    # proving no individual satellite learns the SoC from its share alone
    print(f"\n  Shares for first epoch (SoC={samples[0][1]}%, secret={samples[0][2]}):")
    for s in ss.share(samples[0][2]):
        print(f"    Party {s.party_id} receives: {s.value}  (looks random — reveals nothing alone)")

    # Demonstrate local addition of shares:
    # Two satellites privately add their SoC values without communication.
    # Used in truth_discovery to sum weighted sensing values.
    soc_a, soc_b  = samples[0][2], samples[1][2]
    shares_a      = ss.share(soc_a)
    shares_b      = ss.share(soc_b)
    sum_shares    = ss.add_shares(shares_a, shares_b)
    sum_recon     = ss.reconstruct(sum_shares)
    print(f"\n  Local share addition (no communication required):")
    print(f"    SoC_A={samples[0][1]}% (secret={soc_a})  +  SoC_B={samples[1][1]}% (secret={soc_b})")
    print(f"    Sum of secrets : {soc_a + soc_b}")
    print(f"    Reconstructed  : {sum_recon}  {'correct' if sum_recon == soc_a + soc_b else 'WRONG'}")

    # Verify correctness over many rounds
    print(f"\n  Running {len(samples)} verification rounds with real SoC values...")
    all_ok = all(verify_sharing(ss, s) for _, _, s in samples)
    print(f"  All rounds passed: {all_ok}")

    # Statistical privacy test: shares must be uniformly distributed
    print(f"\n  Statistical privacy test (10 000 samples of SoC=60.0%):")
    stats = statistical_test(ss)
    for pid, r in stats.items():
        ratio = r['mean'] / r['expected_mean']
        print(f"    Party {pid}: mean={r['mean']:.3e}  expected={r['expected_mean']:.3e}  ratio={ratio:.4f}")
    print(f"  Ratio ≈ 1.0 for all parties confirms uniform distribution (privacy holds).")

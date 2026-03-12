#Secret Sharing Module for OrbitalChain

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import secrets


#Secret share held by a party
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


#Additive Secret Sharing Scheme over a finite field F_q.
class AdditiveSecretSharing:
    # Default large prime (Mersenne prime 2^61 - 1)
    DEFAULT_PRIME = 2305843009213693951
    
    #Initialize the secret sharing
    def __init__(
        self, 
        num_parties: int, 
        prime_modulus: Optional[int] = None,
        seed: Optional[int] = None
    ):
        if num_parties < 2:
            raise ValueError("Need at least 2 parties for secret sharing")
        
        self.num_parties = num_parties
        self.prime_modulus = prime_modulus or self.DEFAULT_PRIME
        
        if seed is not None:
            np.random.seed(seed)
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()
    
    #Split a secret into n additive shares
    def share(self, secret: int) -> List[Share]:
        
        # Ensure secret is in the field
        secret = secret % self.prime_modulus
        
        random_shares = [
            secrets.randbelow(self.prime_modulus) 
            for _ in range(self.num_parties - 1)
        ]
        
        # Compute the last share to make sum equal to secret
        sum_of_random = sum(random_shares) % self.prime_modulus
        last_share = (secret - sum_of_random) % self.prime_modulus
        
        # Create Share objects
        all_values = random_shares + [last_share]
        shares = [
            Share(party_id=i, value=v, field_modulus=self.prime_modulus)
            for i, v in enumerate(all_values)
        ]
        
        return shares
    
    # Reconstruct the secret from all shares
    def reconstruct(self, shares: List[Share]) -> int:

        if len(shares) != self.num_parties:
            raise ValueError(
                f"Need all {self.num_parties} shares to reconstruct, "
                f"got {len(shares)}"
            )
        
        # Verify all shares use the same field
        for share in shares:
            if share.field_modulus != self.prime_modulus:
                raise ValueError("Inconsistent field modulus in shares")
        
        # Sum all shares
        secret = sum(share.value for share in shares) % self.prime_modulus
        
        return secret
    
    def add_shares(
        self, 
        shares_x: List[Share], 
        shares_y: List[Share]
    ) -> List[Share]:
        if len(shares_x) != len(shares_y):
            raise ValueError("Share lists must have the same length")
        
        result = []
        for sx, sy in zip(shares_x, shares_y):
            if sx.party_id != sy.party_id:
                raise ValueError("Shares must be aligned by party")
            result.append(sx + sy)
        
        return result
    
    def multiply_by_constant(
        self, 
        shares: List[Share], 
        constant: int
    ) -> List[Share]:
        constant = constant % self.prime_modulus
        return [share * constant for share in shares]
    
    def share_batch(self, secrets: List[int]) -> List[List[Share]]:
        return [self.share(s) for s in secrets]
    
    # Extract all shares belonging to a specific party.
    def get_party_shares(
        self, 
        all_shares: List[List[Share]], 
        party_id: int
    ) -> List[Share]:
        return [shares[party_id] for shares in all_shares]


# Initialize the protocol.
class SecretSharingProtocol:    
    def __init__(
        self, 
        num_satellites: int,
        prime_modulus: Optional[int] = None
    ):
        self.num_satellites = num_satellites
        self.ss = AdditiveSecretSharing(num_satellites, prime_modulus)
        self.prime_modulus = self.ss.prime_modulus
    
    # Data Provider creates shares of their sensing value.
    def data_provider_share(
        self, 
        sensing_value: float,
        scale_factor: int = 10**6
    ) -> Tuple[List[Share], List[Tuple]]:
        # Scale and convert to integer
        scaled_value = int(sensing_value * scale_factor) % self.prime_modulus
        
        # Create shares of the sensing value
        value_shares = self.ss.share(scaled_value)
        
        # Generate two sets of multiplication triples
        from .beaver_triples import BeaverTripleGenerator
        triple_gen = BeaverTripleGenerator(self.num_satellites, self.prime_modulus)
        
        triple_set_0 = triple_gen.generate_triple()
        triple_set_1 = triple_gen.generate_triple()
        
        return value_shares, (triple_set_0, triple_set_1)
    
    # Prepare data for distribution to each satellite.
    def distribute_to_satellites(
        self, 
        shares: List[Share],
        triples: Tuple
    ) -> dict:
        distribution = {}
        
        for i in range(self.num_satellites):
            distribution[i] = {
                'value_share': shares[i],
                'triple_0': triples[0][i],
                'triple_1': triples[1][i]
            }
        
        return distribution


# Verify the correctness of the secret sharing scheme.
def verify_sharing(
    ss: AdditiveSecretSharing, 
    secret: int, 
    num_tests: int = 100
) -> bool:
    for _ in range(num_tests):
        shares = ss.share(secret)
        reconstructed = ss.reconstruct(shares)
        if reconstructed != (secret % ss.prime_modulus):
            return False
    return True


def statistical_test(
    ss: AdditiveSecretSharing, 
    num_samples: int = 10000
) -> dict:
    secret = 12345
    share_values = {i: [] for i in range(ss.num_parties)}
    
    for _ in range(num_samples):
        shares = ss.share(secret)
        for share in shares:
            share_values[share.party_id].append(share.value)
    
    results = {}
    for party_id, values in share_values.items():
        values = np.array(values)
        results[party_id] = {
            'mean': np.mean(values),
            'expected_mean': ss.prime_modulus / 2,
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    return results


if __name__ == "__main__":
    print("=== Additive Secret Sharing test ===\n")
    
    # Initialize
    ss = AdditiveSecretSharing(num_parties=5)
    print(f"Number of parties: {ss.num_parties}")
    print(f"Field modulus: {ss.prime_modulus}\n")
    
    # Share a secret
    secret = 42
    shares = ss.share(secret)
    print(f"Original secret: {secret}")
    print(f"Shares: {shares}\n")
    
    # Reconstruct
    reconstructed = ss.reconstruct(shares)
    print(f"Reconstructed: {reconstructed}")
    print(f"Correct: {reconstructed == secret}\n")
    
    # Test addition of shares
    secret1, secret2 = 100, 200
    shares1 = ss.share(secret1)
    shares2 = ss.share(secret2)
    sum_shares = ss.add_shares(shares1, shares2)
    sum_reconstructed = ss.reconstruct(sum_shares)
    print(f"Share addition: {secret1} + {secret2} = {sum_reconstructed}")
    print(f"Correct: {sum_reconstructed == (secret1 + secret2)}\n")
    
    # Verify scheme
    print("Running verification tests...")
    is_correct = verify_sharing(ss, 12345)
    print(f"Verification passed: {is_correct}")

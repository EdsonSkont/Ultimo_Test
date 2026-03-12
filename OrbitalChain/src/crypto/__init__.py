"""
Cryptographic modules for privacy-preserving computation.

    - secret_sharing: Additive secret sharing over finite fields
    - beaver_triples: Beaver multiplication triples for secure MPC
    - garbled_circuits: Yao's garbled circuits for division/logarithm
"""

from .secret_sharing import AdditiveSecretSharing, Share
from .beaver_triples import BeaverTripleGenerator, SecureMultiplication, SharedTriple
from .garbled_circuits import GarbledCircuitProtocol

__all__ = [
    'AdditiveSecretSharing',
    'Share',
    'BeaverTripleGenerator', 
    'SecureMultiplication',
    'SharedTriple',
    'GarbledCircuitProtocol'
]

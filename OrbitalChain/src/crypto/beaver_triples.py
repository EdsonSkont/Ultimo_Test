import secrets
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from .secret_sharing import AdditiveSecretSharing, Share


@dataclass
class BeaverTriple:
    a: int
    b: int
    c: int
    field_modulus: int
    
    def verify(self) -> bool:
        return (self.a * self.b) % self.field_modulus == self.c
    
    def __repr__(self):
        return f"BeaverTriple(a={self.a}, b={self.b}, c={self.c})"


# Shares of a Beaver triple held by a single party
@dataclass 
class SharedTriple:
    party_id: int
    a_share: Share
    b_share: Share
    c_share: Share
    
    def __repr__(self):
        return f"SharedTriple(party={self.party_id})"


# Generator for Beaver multiplication triples
class BeaverTripleGenerator:

    
    def __init__(
        self, 
        num_parties: int, 
        prime_modulus: Optional[int] = None
    ):
        self.num_parties = num_parties
        self.ss = AdditiveSecretSharing(num_parties, prime_modulus)
        self.prime_modulus = self.ss.prime_modulus
   
    def generate_triple(self) -> List[SharedTriple]:
        a = secrets.randbelow(self.prime_modulus)
        b = secrets.randbelow(self.prime_modulus)
        
        c = (a * b) % self.prime_modulus
        
        triple = BeaverTriple(a, b, c, self.prime_modulus)
        assert triple.verify(), "Triple verification failed"
        
        # Share each component
        a_shares = self.ss.share(a)
        b_shares = self.ss.share(b)
        c_shares = self.ss.share(c)
        
        # Combine into SharedTriple objects
        shared_triples = [
            SharedTriple(
                party_id=i,
                a_share=a_shares[i],
                b_share=b_shares[i],
                c_share=c_shares[i]
            )
            for i in range(self.num_parties)
        ]
        
        return shared_triples
    
    def generate_batch(self, count: int) -> List[List[SharedTriple]]:
        return [self.generate_triple() for _ in range(count)]


# Secure multiplication protocol using Beaver triples
class SecureMultiplication:
    def __init__(
        self, 
        num_parties: int,
        prime_modulus: Optional[int] = None
    ):
        self.num_parties = num_parties
        self.ss = AdditiveSecretSharing(num_parties, prime_modulus)
        self.prime_modulus = self.ss.prime_modulus
    
    # Securely multiply two secret-shared values
    def multiply(
        self,
        x_shares: List[Share],
        y_shares: List[Share],
        triple: List[SharedTriple]
    ) -> List[Share]:
        n = self.num_parties
        q = self.prime_modulus
        
        # Compute masked differences for each party
        u_shares = []  # u = x - a
        v_shares = []  # v = y - b
        
        for i in range(n):
            u_i = (x_shares[i].value - triple[i].a_share.value) % q
            v_i = (y_shares[i].value - triple[i].b_share.value) % q
            u_shares.append(u_i)
            v_shares.append(v_i)
        
        # Broadcast - all parties learn u and v
        u = sum(u_shares) % q
        v = sum(v_shares) % q
        
        # Each party computes their share of x*y        
        result_shares = []
        uv_term = (u * v) % q
        
        for i in range(n):
            if i == 0:
                uv_component = uv_term
            else:
                uv_component = 0
            
            share_value = (
                uv_component +
                (u * triple[i].b_share.value) % q +
                (v * triple[i].a_share.value) % q +
                triple[i].c_share.value
            ) % q
            
            result_shares.append(Share(
                party_id=i,
                value=share_value,
                field_modulus=q
            ))
        
        return result_shares
    
    def multiply_by_constant(
        self,
        shares: List[Share],
        constant: int
    ) -> List[Share]:

        return self.ss.multiply_by_constant(shares, constant)
    
    def square(
        self,
        x_shares: List[Share],
        triple: List[SharedTriple]
    ) -> List[Share]:

        return self.multiply(x_shares, x_shares, triple)


class BatchMultiplication:
    
    def __init__(
        self,
        num_parties: int,
        prime_modulus: Optional[int] = None
    ):
        self.num_parties = num_parties
        self.prime_modulus = prime_modulus or AdditiveSecretSharing.DEFAULT_PRIME
        self.mult = SecureMultiplication(num_parties, prime_modulus)
        self.triple_gen = BeaverTripleGenerator(num_parties, prime_modulus)
    
    def batch_multiply(
        self,
        x_shares_list: List[List[Share]],
        y_shares_list: List[List[Share]],
        triples: List[List[SharedTriple]]
    ) -> List[List[Share]]:
        if len(x_shares_list) != len(y_shares_list):
            raise ValueError("Input lists must have same length")
        if len(x_shares_list) != len(triples):
            raise ValueError("Need one triple per multiplication")
        
        results = []
        for x_shares, y_shares, triple in zip(x_shares_list, y_shares_list, triples):
            product = self.mult.multiply(x_shares, y_shares, triple)
            results.append(product)
        
        return results
    
    def prepare_triples(self, count: int) -> List[List[SharedTriple]]:
        return self.triple_gen.generate_batch(count)


# Verify correctness of secure multiplication
def verify_secure_multiplication(
    num_parties: int = 3,
    num_tests: int = 100
) -> bool:
    ss = AdditiveSecretSharing(num_parties)
    mult = SecureMultiplication(num_parties, ss.prime_modulus)
    triple_gen = BeaverTripleGenerator(num_parties, ss.prime_modulus)
    
    for _ in range(num_tests):
        x = secrets.randbelow(ss.prime_modulus)
        y = secrets.randbelow(ss.prime_modulus)
        
        expected = (x * y) % ss.prime_modulus
        
        # Share x and y
        x_shares = ss.share(x)
        y_shares = ss.share(y)
        
        # Generate triple
        triple = triple_gen.generate_triple()
        
        # Secure multiplication
        result_shares = mult.multiply(x_shares, y_shares, triple)
        
        # Reconstruct and verify
        result = ss.reconstruct(result_shares)
        
        if result != expected:
            print(f"FAILED: {x} * {y} = {expected}, got {result}")
            return False
    
    return True


if __name__ == "__main__":
    print("=== Beaver Multiplication Triples Test ===\n")
    
    # Setup
    num_parties = 5
    ss = AdditiveSecretSharing(num_parties)
    mult = SecureMultiplication(num_parties, ss.prime_modulus)
    triple_gen = BeaverTripleGenerator(num_parties, ss.prime_modulus)
    
    # Test values
    x = 7
    y = 6
    expected = x * y
    
    print(f"Computing {x} * {y} securely among {num_parties} parties\n")
    
    # Share values
    x_shares = ss.share(x)
    y_shares = ss.share(y)
    print(f"Shares of x={x}: {[s.value for s in x_shares]}")
    print(f"Shares of y={y}: {[s.value for s in y_shares]}\n")
    
    # Generate Beaver triple
    triple = triple_gen.generate_triple()
    print("Beaver triple generated and shared\n")
    
    # Secure multiplication
    result_shares = mult.multiply(x_shares, y_shares, triple)
    result = ss.reconstruct(result_shares)
    
    print(f"Result shares: {[s.value for s in result_shares]}")
    print(f"Reconstructed result: {result}")
    print(f"Expected: {expected}")
    print(f"Correct: {result == expected}\n")
    
    # Run verification tests
    print("Running 100 verification tests...")
    success = verify_secure_multiplication(num_parties, 100)
    print(f"All tests passed: {success}")

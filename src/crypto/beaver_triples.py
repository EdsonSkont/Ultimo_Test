# Beaver multiplication triples for secure MPC in OrbitalChain.
#
# Real-data context
# -----------------
# The SA-SBFT consensus module computes a weighted quorum score:
#   quorum_score = sum(weight_k * soc_k) / sum(weight_k)
#
# where soc_k is battery_soc_pct for satellite k and weight_k reflects
# its orbital reliability (derived from CRC fail rate in 05_isl_stream_log.json).
#
# Both soc_k and weight_k are secret-shared across the MPC cluster.
# Multiplying two secret-shared values requires a Beaver triple (a, b, c=a*b).
# The triple is generated offline (trusted setup or distributed generation)
# and consumed once per multiplication — it hides the operands during computation.
#
# Encoding convention (same as secret_sharing.py)
#   soc_integer    = int(soc_pct * 100)        e.g. 60.0% -> 6000
#   weight_integer = int(reliability * 10000)  e.g. 0.97  -> 9700
#
# The product soc * weight in field arithmetic is verified by reconstructing
# both operands and comparing to the direct product.

import secrets as _secrets
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import csv
import os

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


@dataclass
class SharedTriple:
    # Shares of one Beaver triple held by a single satellite node
    party_id: int
    a_share: Share
    b_share: Share
    c_share: Share

    def __repr__(self):
        return f"SharedTriple(party={self.party_id})"


class BeaverTripleGenerator:
    # Generates and distributes Beaver triples to all satellite parties.
    # In a real deployment this runs in a trusted offline phase or via
    # a distributed protocol (e.g. MASCOT). Here it is centralised for simulation.

    def __init__(self, num_parties: int, prime_modulus: Optional[int] = None):
        self.num_parties = num_parties
        self.ss          = AdditiveSecretSharing(num_parties, prime_modulus)
        self.prime_modulus = self.ss.prime_modulus

    def generate_triple(self) -> List[SharedTriple]:
        # Draw random a, b; compute c = a*b mod prime; share all three.
        a = _secrets.randbelow(self.prime_modulus)
        b = _secrets.randbelow(self.prime_modulus)
        c = (a * b) % self.prime_modulus
        triple = BeaverTriple(a, b, c, self.prime_modulus)
        assert triple.verify(), "Triple verification failed"
        a_shares = self.ss.share(a)
        b_shares = self.ss.share(b)
        c_shares = self.ss.share(c)
        return [
            SharedTriple(party_id=i,
                         a_share=a_shares[i],
                         b_share=b_shares[i],
                         c_share=c_shares[i])
            for i in range(self.num_parties)
        ]

    def generate_batch(self, count: int) -> List[List[SharedTriple]]:
        return [self.generate_triple() for _ in range(count)]


class SecureMultiplication:
    # Multiplies two secret-shared values using a Beaver triple.
    #
    # Protocol (Beaver, 1992):
    #   1. Each party computes its share of  u = x - a  and  v = y - b  locally.
    #   2. All parties broadcast their u/v shares; everyone reconstructs u and v.
    #      (u and v are masked by random a/b — they reveal nothing about x or y.)
    #   3. Each party computes its share of  x*y = uv + ub_i + va_i + c_i.
    # Result: shares of x*y without any party learning x or y individually.

    def __init__(self, num_parties: int, prime_modulus: Optional[int] = None):
        self.num_parties = num_parties
        self.ss          = AdditiveSecretSharing(num_parties, prime_modulus)
        self.prime_modulus = self.ss.prime_modulus

    def multiply(
        self,
        x_shares: List[Share],
        y_shares: List[Share],
        triple:   List[SharedTriple],
    ) -> List[Share]:
        q = self.prime_modulus
        n = self.num_parties

        # Step 1: compute masked differences locally (no communication)
        u_shares = [(x_shares[i].value - triple[i].a_share.value) % q for i in range(n)]
        v_shares = [(y_shares[i].value - triple[i].b_share.value) % q for i in range(n)]

        # Step 2: reconstruct u and v by broadcasting shares (u, v are masked)
        u = sum(u_shares) % q
        v = sum(v_shares) % q

        # Step 3: each party computes its share of x*y
        uv = (u * v) % q
        result = []
        for i in range(n):
            share_val = (
                (uv if i == 0 else 0) +          # uv term added only by party 0
                (u * triple[i].b_share.value) % q +
                (v * triple[i].a_share.value) % q +
                triple[i].c_share.value
            ) % q
            result.append(Share(party_id=i, value=share_val, field_modulus=q))
        return result

    def multiply_by_constant(self, shares: List[Share], constant: int) -> List[Share]:
        return self.ss.multiply_by_constant(shares, constant)

    def square(self, x_shares: List[Share], triple: List[SharedTriple]) -> List[Share]:
        return self.multiply(x_shares, x_shares, triple)


class BatchMultiplication:
    # Batch secure multiplication for multiple operand pairs in one call.
    # Used in truth_discovery to compute w_k * x_k for all providers at once.

    def __init__(self, num_parties: int, prime_modulus: Optional[int] = None):
        self.num_parties = num_parties
        self.prime_modulus = prime_modulus or AdditiveSecretSharing.DEFAULT_PRIME
        self.mult       = SecureMultiplication(num_parties, prime_modulus)
        self.triple_gen = BeaverTripleGenerator(num_parties, prime_modulus)

    def batch_multiply(
        self,
        x_shares_list: List[List[Share]],
        y_shares_list: List[List[Share]],
        triples:       List[List[SharedTriple]],
    ) -> List[List[Share]]:
        if len(x_shares_list) != len(y_shares_list):
            raise ValueError("Input lists must have same length")
        if len(x_shares_list) != len(triples):
            raise ValueError("Need one triple per multiplication")
        return [
            self.mult.multiply(x, y, t)
            for x, y, t in zip(x_shares_list, y_shares_list, triples)
        ]

    def prepare_triples(self, count: int) -> List[List[SharedTriple]]:
        return self.triple_gen.generate_batch(count)


def verify_secure_multiplication(num_parties: int = 3, num_tests: int = 100) -> bool:
    ss         = AdditiveSecretSharing(num_parties)
    mult       = SecureMultiplication(num_parties, ss.prime_modulus)
    triple_gen = BeaverTripleGenerator(num_parties, ss.prime_modulus)
    for _ in range(num_tests):
        x = _secrets.randbelow(ss.prime_modulus)
        y = _secrets.randbelow(ss.prime_modulus)
        expected     = (x * y) % ss.prime_modulus
        result_shares = mult.multiply(ss.share(x), ss.share(y), triple_gen.generate_triple())
        if ss.reconstruct(result_shares) != expected:
            return False
    return True


def _load_soc_and_weights(n: int = 5) -> List[Tuple[str, float, float]]:
    # Load (timestamp, soc_pct, reliability_weight) from CCSDS dataset.
    # soc_pct       from 08_power_subsystem.csv  — the secret to be multiplied
    # reliability   derived from 05_isl_stream_log.json fault counts
    #   reliable satellite (0 faults)  -> weight = 1.00
    #   SAT-14 (2 fault alerts)        -> weight = 0.90 (penalised)
    import json
    here     = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(here, "..", "..", "data", "satellite_ccsds_dataset"))

    with open(os.path.join(data_dir, "08_power_subsystem.csv"), newline="") as f:
        pwr_rows = list(csv.DictReader(f))

    with open(os.path.join(data_dir, "05_isl_stream_log.json")) as f:
        isl = json.load(f)

    from collections import Counter
    fault_counts = Counter(
        p["source_sat_id"] for p in isl["packets"] if p.get("apid") == "0x7FE"
    )
    # Map row index to satellite weight: 5 power rows -> 5 satellite nodes
    sat_names  = [f"SAT-{10+i}" for i in range(min(n, 5))]
    weights    = [max(1.0 - fault_counts.get(s, 0) * 0.05, 0.80) for s in sat_names]

    result = []
    for i, row in enumerate(pwr_rows[:n]):
        soc    = float(row["battery_soc_pct"])
        weight = weights[i % len(weights)]
        result.append((row["timestamp_utc"][11:19], soc, weight))
    return result


if __name__ == "__main__":
    print("=== Beaver Multiplication Triples — soc_pct * reliability_weight ===\n")
    print("  Source  : 08_power_subsystem.csv  (soc_pct)")
    print("            05_isl_stream_log.json  (fault count -> reliability weight)")
    print("  Purpose : compute weighted quorum score without revealing individual SoC")
    print("  Formula : quorum_score = Σ(weight_k * soc_k) / Σ(weight_k)\n")

    NUM_PARTIES = 5
    ss         = AdditiveSecretSharing(NUM_PARTIES)
    mult       = SecureMultiplication(NUM_PARTIES, ss.prime_modulus)
    triple_gen = BeaverTripleGenerator(NUM_PARTIES, ss.prime_modulus)

    samples = _load_soc_and_weights(n=NUM_PARTIES)

    print(f"  {'Time':>8}  {'SoC(%)':>7}  {'Weight':>7}  {'Product':>9}  {'MPC result':>11}  {'Match':>6}")
    print("  " + "-" * 58)

    total_weighted = 0.0
    total_weight   = 0.0

    for ts, soc, weight in samples:
        # Encode both values as field integers
        # SoC:    int(soc * 100)     -> e.g. 60.0% = 6000
        # Weight: int(weight * 1000) -> e.g. 1.00  = 1000
        # After MPC multiplication the product is soc_int * weight_int,
        # decode by dividing by 100 * 1000 = 100000 to recover soc * weight
        soc_int    = int(soc * 100)
        weight_int = int(weight * 1000)
        expected   = (soc_int * weight_int) % ss.prime_modulus

        # Secret-share both values across the MPC cluster
        x_shares = ss.share(soc_int)
        y_shares = ss.share(weight_int)

        # Generate Beaver triple and compute secure product
        triple        = triple_gen.generate_triple()
        result_shares = mult.multiply(x_shares, y_shares, triple)
        result        = ss.reconstruct(result_shares)

        match = result == expected
        # Decode: divide by scale factor to get soc * weight
        decoded = result / (100 * 1000)
        print(f"  {ts:>8}  {soc:>7.1f}  {weight:>7.3f}  {soc*weight:>9.2f}  "
              f"{decoded:>11.2f}  {'yes' if match else 'NO':>6}")

        total_weighted += soc * weight
        total_weight   += weight

    quorum_score = total_weighted / total_weight
    print(f"\n  Quorum score (mean weighted SoC) = {quorum_score:.2f}%")
    print(f"  Threshold active >= 50.0%  -> {'PASS' if quorum_score >= 50.0 else 'FAIL'}")
    print(f"  Threshold semi   >= 20.0%  -> {'PASS' if quorum_score >= 20.0 else 'FAIL'}")

    print(f"\n  Running 100 verification tests with random field values...")
    ok = verify_secure_multiplication(NUM_PARTIES, 100)
    print(f"  All tests passed: {ok}")

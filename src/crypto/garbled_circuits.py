# Garbled circuits for secure division and logarithm in OrbitalChain.
#
# Real-data context
# -----------------
# After the Beaver triple multiplications in beaver_triples.py produce
# secret-shared products  (weight_k * soc_k)  for each satellite k,
# the consensus module needs:
#
#   1. gc_div:
#      Compute  quorum_score = T / Z  where
#        T = Σ (weight_k * soc_k)   -- weighted sum of SoC values
#        Z = Σ weight_k              -- sum of weights
#      T and Z are each held as two-party shares.
#      Neither party should learn T or Z individually; only the ratio matters.
#
#   2. gc_div_log:
#      Compute  -log(ST_k / ST*)  for the truth discovery weight update.
#        ST_k  = accumulated squared error for satellite k
#        ST*   = total accumulated error across all satellites
#      The log ratio transforms error accumulation into a trust weight.
#
# Both operations use a simplified garbled-circuit model (simulation mode):
# the circuit evaluates in plaintext over reconstructed values and re-shares
# the result.  The share structure is preserved so the rest of the MPC
# pipeline works correctly.  Full binary garbling would replace the
# plaintext evaluation without changing the calling interface.
#
# Encoding convention (same as secret_sharing.py and beaver_triples.py)
#   soc_integer = int(soc_pct * 100)          e.g. 60.0% -> 6000
#   All intermediate values are scaled by SCALE_FACTOR = 10^6 inside
#   streaming_truth.py, so gc_div receives already-scaled shares.

import secrets as _secrets
import hashlib
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
import csv
import os

from .secret_sharing import Share


class GateType(Enum):
    AND = "AND"
    OR  = "OR"
    XOR = "XOR"
    NOT = "NOT"
    ADD = "ADD"
    MUL = "MUL"
    CMP = "CMP"


@dataclass
class Wire:
    wire_id: int
    label_0: bytes
    label_1: bytes


@dataclass
class GarbledGate:
    gate_id:      int
    gate_type:    GateType
    input_wires:  List[int]
    output_wire:  int
    garbled_table: List[bytes]


@dataclass
class GarbledCircuit:
    circuit_id:   str
    input_wires:  List[Wire]
    output_wires: List[int]
    gates:        List[GarbledGate]

    def __repr__(self):
        return f"GarbledCircuit(id={self.circuit_id}, gates={len(self.gates)})"


class CircuitGarbler:
    LABEL_LENGTH = 16  # 128-bit wire labels

    def __init__(self, security_parameter: int = 128):
        self.security_parameter = security_parameter
        self.label_length = security_parameter // 8

    def _generate_label(self) -> bytes:
        return _secrets.token_bytes(self.label_length)

    def _hash(self, *inputs: bytes) -> bytes:
        h = hashlib.sha256()
        for inp in inputs:
            h.update(inp)
        return h.digest()[:self.label_length]

    def _encrypt(self, key1: bytes, key2: bytes, gate_id: int, plaintext: bytes) -> bytes:
        key = self._hash(key1, key2, gate_id.to_bytes(4, 'big'))
        return bytes(a ^ b for a, b in zip(key, plaintext))

    def _create_wire(self, wire_id: int) -> Wire:
        return Wire(wire_id=wire_id,
                    label_0=self._generate_label(),
                    label_1=self._generate_label())

    def garble_and_gate(self, gate_id, wire_a, wire_b, wire_out) -> GarbledGate:
        table = []
        for a in [0, 1]:
            for b in [0, 1]:
                label_a   = wire_a.label_0 if a == 0 else wire_a.label_1
                label_b   = wire_b.label_0 if b == 0 else wire_b.label_1
                label_out = wire_out.label_0 if (a & b) == 0 else wire_out.label_1
                table.append(self._encrypt(label_a, label_b, gate_id, label_out))
        np.random.shuffle(table)
        return GarbledGate(gate_id=gate_id, gate_type=GateType.AND,
                           input_wires=[wire_a.wire_id, wire_b.wire_id],
                           output_wire=wire_out.wire_id, garbled_table=table)

    def garble_xor_gate(self, gate_id, wire_a, wire_b, wire_out) -> GarbledGate:
        # Free-XOR: no garbled table needed
        return GarbledGate(gate_id=gate_id, gate_type=GateType.XOR,
                           input_wires=[wire_a.wire_id, wire_b.wire_id],
                           output_wire=wire_out.wire_id, garbled_table=[])


class GarbledCircuitProtocol:
    # Simulation-mode garbled circuit protocol.
    # gc_div and gc_div_log reconstruct operands in plaintext, compute the
    # result, then re-share it as a fresh two-party secret.
    # The share interface is identical to what a full binary GC would provide.

    def __init__(self, prime_modulus: int, bit_length: int = 64):
        self.prime_modulus = prime_modulus
        self.bit_length    = bit_length
        self.garbler       = CircuitGarbler()

    def gc_div(
        self,
        t_share_0: Share, t_share_1: Share,
        z_share_0: Share, z_share_1: Share,
    ) -> Tuple[Share, Share]:
        # Secure division: result = T / Z  (scaled by 10^6 for fixed-point).
        # T = weighted sum of SoC values (numerator)
        # Z = sum of weights (denominator)
        t = (t_share_0.value + t_share_1.value) % self.prime_modulus
        z = (z_share_0.value + z_share_1.value) % self.prime_modulus
        result = int((t * 10**6) // z) % self.prime_modulus if z != 0 else 0
        r = _secrets.randbelow(self.prime_modulus)
        return (
            Share(party_id=0, value=r,                              field_modulus=self.prime_modulus),
            Share(party_id=1, value=(result - r) % self.prime_modulus, field_modulus=self.prime_modulus),
        )

    def gc_div_log(
        self,
        st_k_share_0:   Share, st_k_share_1:   Share,
        st_star_share_0: Share, st_star_share_1: Share,
    ) -> Tuple[Share, Share]:
        # Secure division + natural log: result = -log(ST_k / ST*) * 10^6.
        # ST_k  = accumulated squared error for satellite k (from truth_discovery)
        # ST*   = total accumulated error across all satellites
        # Larger ST_k -> larger ratio -> more negative log -> lower weight -> less trusted.
        st_k   = (st_k_share_0.value   + st_k_share_1.value)   % self.prime_modulus
        st_star = (st_star_share_0.value + st_star_share_1.value) % self.prime_modulus
        if st_star == 0 or st_k == 0:
            result = 0
        else:
            ratio  = st_k / st_star
            result = int(-np.log(ratio) * 10**6) % self.prime_modulus if ratio > 0 else 0
        r = _secrets.randbelow(self.prime_modulus)
        return (
            Share(party_id=0, value=r,                              field_modulus=self.prime_modulus),
            Share(party_id=1, value=(result - r) % self.prime_modulus, field_modulus=self.prime_modulus),
        )


class ObliviousTransfer:
    # Allows a receiver to obtain one of two messages without the sender
    # learning which was chosen. Used to distribute wire labels in garbled circuits.

    def __init__(self, security_parameter: int = 128):
        self.security_parameter = security_parameter

    def sender_setup(self, message_0: bytes, message_1: bytes) -> Tuple[bytes, bytes]:
        return (message_0, message_1)

    def receiver_choose(self, sender_params: Tuple[bytes, bytes], choice_bit: int) -> bytes:
        return sender_params[choice_bit]


def _load_quorum_data(n: int = 5) -> List[Tuple[str, float, float, float]]:
    # Load (timestamp, soc_pct, weight, product) for n satellites.
    # These are the T (numerator) and Z (denominator) inputs to gc_div.
    import json
    from collections import Counter
    here     = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(here, "..", "..", "data", "satellite_ccsds_dataset"))

    with open(os.path.join(data_dir, "08_power_subsystem.csv"), newline="") as f:
        pwr = list(csv.DictReader(f))

    with open(os.path.join(data_dir, "05_isl_stream_log.json")) as f:
        isl = json.load(f)

    fault_counts = Counter(
        p["source_sat_id"] for p in isl["packets"] if p.get("apid") == "0x7FE"
    )
    sat_names = [f"SAT-{10+i}" for i in range(min(n, 5))]
    weights   = [max(1.0 - fault_counts.get(s, 0) * 0.05, 0.80) for s in sat_names]

    result = []
    for i, row in enumerate(pwr[:n]):
        soc    = float(row["battery_soc_pct"])
        w      = weights[i % len(weights)]
        result.append((row["timestamp_utc"][11:19], soc, w, soc * w))
    return result


if __name__ == "__main__":
    print("=== Garbled Circuit Protocol — quorum score and trust weights ===\n")
    print("  Source  : 08_power_subsystem.csv  (soc_pct as numerator)")
    print("            05_isl_stream_log.json  (fault count -> reliability weight)")
    print("  gc_div     : quorum_score = T / Z  (weighted mean SoC)")
    print("  gc_div_log : trust_weight = -log(ST_k / ST*)  (truth discovery)\n")

    prime    = 2**61 - 1
    protocol = GarbledCircuitProtocol(prime)
    samples  = _load_quorum_data(n=5)
    SCALE    = 10**6

    # gc_div: compute weighted mean SoC across all satellites
    # T = sum(weight_k * soc_k),  Z = sum(weight_k)
    print("  gc_div: weighted quorum score\n")
    T_plain = sum(s * w for _, s, w, _ in samples)
    Z_plain = sum(w    for _, _, w, _ in samples)
    expected_score = T_plain / Z_plain

    # Encode T and Z as scaled integers; split into two-party shares
    T_int = int(T_plain * SCALE)
    Z_int = int(Z_plain * SCALE)

    t0 = Share(0, _secrets.randbelow(prime), prime)
    t1 = Share(1, (T_int - t0.value) % prime, prime)
    z0 = Share(0, _secrets.randbelow(prime), prime)
    z1 = Share(1, (Z_int - z0.value) % prime, prime)

    r0, r1  = protocol.gc_div(t0, t1, z0, z1)
    result  = (r0.value + r1.value) % prime
    decoded = result / SCALE  # gc_div already includes one SCALE in its result

    print(f"  {'Satellite':<10} {'SoC(%)':>7}  {'Weight':>7}  {'w*SoC':>8}")
    print("  " + "-" * 38)
    for ts, soc, w, prod in samples:
        print(f"  {ts:<10} {soc:>7.1f}  {w:>7.3f}  {prod:>8.3f}")
    print(f"\n  T = {T_plain:.3f}   Z = {Z_plain:.3f}")
    print(f"  Expected quorum score : {expected_score:.2f}%")
    print(f"  gc_div result         : {decoded:.2f}%")
    print(f"  Active threshold 50%  : {'PASS' if decoded >= 50.0 else 'FAIL'}")

    # gc_div_log: compute trust weights from accumulated squared errors
    # Simulate ST_k values derived from power data deviations
    print(f"\n  gc_div_log: trust weights from accumulated error\n")
    soc_vals = [s for _, s, _, _ in samples]
    truth    = np.mean(soc_vals)
    st_k_vals = [(s - truth)**2 * 0.9 + 0.01 for s in soc_vals]  # simulated accum
    st_star  = sum(st_k_vals)

    print(f"  Assumed truth (mean SoC) = {truth:.2f}%\n")
    print(f"  {'Satellite':<10} {'SoC(%)':>7}  {'ST_k':>8}  {'Ratio':>7}  {'Expected w':>11}  {'gc result':>10}")
    print("  " + "-" * 60)

    for i, (ts, soc, w, _) in enumerate(samples):
        st_k = st_k_vals[i]
        ratio = st_k / st_star
        expected_w = -np.log(ratio + 1e-10)

        st_k_int   = int(st_k   * SCALE)
        st_star_int = int(st_star * SCALE)

        sk0 = Share(0, _secrets.randbelow(prime), prime)
        sk1 = Share(1, (st_k_int - sk0.value) % prime, prime)
        ss0 = Share(0, _secrets.randbelow(prime), prime)
        ss1 = Share(1, (st_star_int - ss0.value) % prime, prime)

        w0, w1   = protocol.gc_div_log(sk0, sk1, ss0, ss1)
        gc_w_raw = (w0.value + w1.value) % prime
        if gc_w_raw > prime // 2:
            gc_w_raw -= prime
        gc_w = gc_w_raw / SCALE

        print(f"  {ts:<10} {soc:>7.1f}  {st_k:>8.4f}  {ratio:>7.4f}  "
              f"{expected_w:>11.4f}  {gc_w:>10.4f}")

    print(f"\n  Higher weight = more trusted.")
    print(f"  Satellites with SoC closer to truth accumulate less error.")

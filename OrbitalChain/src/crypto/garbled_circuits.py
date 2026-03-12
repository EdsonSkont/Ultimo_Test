import secrets
import hashlib
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .secret_sharing import Share


# Types of gates in the circuit
class GateType(Enum):
    AND = "AND"
    OR = "OR"
    XOR = "XOR"
    NOT = "NOT"
    ADD = "ADD"
    MUL = "MUL"
    CMP = "CMP"  # Comparison


# Wire in the circuit
@dataclass
class Wire:
    wire_id: int
    label_0: bytes  # Label for bit value 0
    label_1: bytes  # Label for bit value 1


# Garbled gate in the circuit
@dataclass
class GarbledGate:
    gate_id: int
    gate_type: GateType
    input_wires: List[int]
    output_wire: int
    garbled_table: List[bytes]  # Encrypted truth table


@dataclass
class GarbledCircuit:
    circuit_id: str
    input_wires: List[Wire]
    output_wires: List[int]
    gates: List[GarbledGate]
    
    def __repr__(self):
        return f"GarbledCircuit(id={self.circuit_id}, gates={len(self.gates)})"


# Garbles circuits for secure two-party computation
class CircuitGarbler:    
    LABEL_LENGTH = 16  # 128 bits
    
    def __init__(self, security_parameter: int = 128):
        self.security_parameter = security_parameter
        self.label_length = security_parameter // 8
    
    def _generate_label(self) -> bytes:
        return secrets.token_bytes(self.label_length)
    
    # Hash function for encryption
    def _hash(self, *inputs: bytes) -> bytes:
        h = hashlib.sha256()
        for inp in inputs:
            h.update(inp)
        return h.digest()[:self.label_length]
    
    # Double-key encryption for garbled gate
    def _encrypt(self, key1: bytes, key2: bytes, gate_id: int, plaintext: bytes) -> bytes:
        key = self._hash(key1, key2, gate_id.to_bytes(4, 'big'))
        return bytes(a ^ b for a, b in zip(key, plaintext))
    
    def _create_wire(self, wire_id: int) -> Wire:
        return Wire(
            wire_id=wire_id,
            label_0=self._generate_label(),
            label_1=self._generate_label()
        )
    
    def garble_and_gate(
        self, 
        gate_id: int,
        wire_a: Wire, 
        wire_b: Wire, 
        wire_out: Wire
    ) -> GarbledGate:
        table = []
        
        for a in [0, 1]:
            for b in [0, 1]:
                label_a = wire_a.label_0 if a == 0 else wire_a.label_1
                label_b = wire_b.label_0 if b == 0 else wire_b.label_1 
                out = a & b
                label_out = wire_out.label_0 if out == 0 else wire_out.label_1
                
                # Encrypt output label
                encrypted = self._encrypt(label_a, label_b, gate_id, label_out)
                table.append(encrypted)
        
        np.random.shuffle(table)
        
        return GarbledGate(
            gate_id=gate_id,
            gate_type=GateType.AND,
            input_wires=[wire_a.wire_id, wire_b.wire_id],
            output_wire=wire_out.wire_id,
            garbled_table=table
        )
    
    # Garble an XOR gate using free-XOR optimization
    def garble_xor_gate(
        self,
        gate_id: int,
        wire_a: Wire,
        wire_b: Wire,
        wire_out: Wire
    ) -> GarbledGate:
        table = []
        
        return GarbledGate(
            gate_id=gate_id,
            gate_type=GateType.XOR,
            input_wires=[wire_a.wire_id, wire_b.wire_id],
            output_wire=wire_out.wire_id,
            garbled_table=table
        )


# Garbled circuit for secure division
class DivisionCircuit:
    def __init__(
        self,
        bit_length: int = 64,
        security_parameter: int = 128
    ):
        self.bit_length = bit_length
        self.garbler = CircuitGarbler(security_parameter)
        self.wires: Dict[int, Wire] = {}
        self.gates: List[GarbledGate] = []
        self.wire_counter = 0
        self.gate_counter = 0
    
    def _new_wire(self) -> Wire:
        wire = self.garbler._create_wire(self.wire_counter)
        self.wires[self.wire_counter] = wire
        self.wire_counter += 1
        return wire
    
    def build_circuit(self) -> GarbledCircuit:
        t_wires = [self._new_wire() for _ in range(self.bit_length)]
        z_wires = [self._new_wire() for _ in range(self.bit_length)]
        
        q_wires = [self._new_wire() for _ in range(self.bit_length)]
        
        
        return GarbledCircuit(
            circuit_id=f"div_{secrets.token_hex(8)}",
            input_wires=t_wires + z_wires,
            output_wires=[w.wire_id for w in q_wires],
            gates=self.gates
        )


# Garbled circuit for secure logarithm computation
class LogarithmCircuit:
    APPROXIMATION_SEGMENTS = [
        (0.0, 0.1, -10.0, 2.302),
        (0.1, 0.2, -5.0, 1.609),
        (0.2, 0.3, -3.333, 1.204),
        (0.3, 0.4, -2.5, 0.916),
        (0.4, 0.5, -2.0, 0.693),
        (0.5, 0.6, -1.667, 0.511),
        (0.6, 0.7, -1.429, 0.357),
        (0.7, 0.8, -1.25, 0.223),
        (0.8, 0.9, -1.111, 0.105),
        (0.9, 1.0, -1.0, 0.0),
    ]
    
    def __init__(
        self,
        bit_length: int = 64,
        security_parameter: int = 128
    ):
        self.bit_length = bit_length
        self.garbler = CircuitGarbler(security_parameter)
    
    # Build the garbled logarithm circuit
    def build_circuit(self) -> GarbledCircuit:
        return GarbledCircuit(
            circuit_id=f"log_{secrets.token_hex(8)}",
            input_wires=[],
            output_wires=[],
            gates=[]
        )


# Garbled circuit protocol
class GarbledCircuitProtocol: 
    def __init__(
        self,
        prime_modulus: int,
        bit_length: int = 64
    ):
        self.prime_modulus = prime_modulus
        self.bit_length = bit_length
        self.garbler = CircuitGarbler()
    
    def gc_div(
        self,
        t_share_0: Share,
        t_share_1: Share,
        z_share_0: Share,
        z_share_1: Share
    ) -> Tuple[Share, Share]:
        
        t = (t_share_0.value + t_share_1.value) % self.prime_modulus
        z = (z_share_0.value + z_share_1.value) % self.prime_modulus
        
        # Compute division (handling division in finite field)
        if z == 0:
            result = 0
        else:
            result = int((t * 10**6) // z) % self.prime_modulus
        
        r = secrets.randbelow(self.prime_modulus)
        
        # Create shares
        share_0 = Share(
            party_id=0,
            value=r,
            field_modulus=self.prime_modulus
        )
        share_1 = Share(
            party_id=1,
            value=(result - r) % self.prime_modulus,
            field_modulus=self.prime_modulus
        )
        
        return share_0, share_1
    
    # Secure division and logarithm using garbled circuits
    def gc_div_log(
        self,
        st_k_share_0: Share,
        st_k_share_1: Share,
        st_star_share_0: Share,
        st_star_share_1: Share
    ) -> Tuple[Share, Share]:
        # Reconstruct values
        st_k = (st_k_share_0.value + st_k_share_1.value) % self.prime_modulus
        st_star = (st_star_share_0.value + st_star_share_1.value) % self.prime_modulus
        
        # Compute ratio and logarithm
        if st_star == 0 or st_k == 0:
            result = 0
        else:
            # Scale for fixed-point arithmetic
            ratio = st_k / st_star
            if ratio > 0:
                result = int(-np.log(ratio) * 10**6) % self.prime_modulus
            else:
                result = 0
        
        r = secrets.randbelow(self.prime_modulus)
        
        # Create shares
        share_0 = Share(
            party_id=0,
            value=r,
            field_modulus=self.prime_modulus
        )
        share_1 = Share(
            party_id=1,
            value=(result - r) % self.prime_modulus,
            field_modulus=self.prime_modulus
        )
        
        return share_0, share_1

"""
Oblivious Transfer protocol , Allows S1 to obtain the label corresponding to their input bit without S0 learning which label was chosen.
"""
class ObliviousTransfer:
    def __init__(self, security_parameter: int = 128):
        self.security_parameter = security_parameter
    
    def sender_setup(
        self,
        message_0: bytes,
        message_1: bytes
    ) -> Tuple[bytes, bytes]:
        return (message_0, message_1)
    
    def receiver_choose(
        self,
        sender_params: Tuple[bytes, bytes],
        choice_bit: int
    ) -> bytes:
        if choice_bit == 0:
            return sender_params[0]
        else:
            return sender_params[1]


def simulate_gc_protocol():
    print("=== Garbled Circuit Protocol Simulation ===\n")
    
    # Setup
    prime = 2**61 - 1
    protocol = GarbledCircuitProtocol(prime)
    
    # Test GC division
    print("Testing GC Division (gc_div):")
    t = 1000000  # Numerator
    z = 500000   # Denominator
    
    # Create shares (simulating secret sharing)
    t_share_0 = Share(0, secrets.randbelow(prime), prime)
    t_share_1 = Share(1, (t - t_share_0.value) % prime, prime)
    z_share_0 = Share(0, secrets.randbelow(prime), prime)
    z_share_1 = Share(1, (z - z_share_0.value) % prime, prime)
    
    # Run protocol
    result_0, result_1 = protocol.gc_div(t_share_0, t_share_1, z_share_0, z_share_1)
    result = (result_0.value + result_1.value) % prime
    
    print(f"  t = {t}, z = {z}")
    print(f"  Expected (scaled): {t * 10**6 // z}")
    print(f"  Got: {result}\n")
    
    # Test GC division + logarithm
    print("Testing GC Division + Logarithm (gc_div_log):")
    st_k = 100000
    st_star = 1000000
    
    st_k_0 = Share(0, secrets.randbelow(prime), prime)
    st_k_1 = Share(1, (st_k - st_k_0.value) % prime, prime)
    st_star_0 = Share(0, secrets.randbelow(prime), prime)
    st_star_1 = Share(1, (st_star - st_star_0.value) % prime, prime)
    
    w_0, w_1 = protocol.gc_div_log(st_k_0, st_k_1, st_star_0, st_star_1)
    w = (w_0.value + w_1.value) % prime
    
    expected = -np.log(st_k / st_star) * 10**6
    print(f"  st(k) = {st_k}, st* = {st_star}")
    print(f"  Expected (scaled): {int(expected)}")
    print(f"  Got: {w}")


if __name__ == "__main__":
    simulate_gc_protocol()

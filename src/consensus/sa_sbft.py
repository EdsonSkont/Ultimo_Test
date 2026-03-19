# Satellite-Adapted Consensus

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import hashlib
import time
from abc import ABC, abstractmethod

# Role of satellite in consensus
class SatelliteRole(Enum):
    ACTIVE = "active"
    SEMI_ACTIVE = "semi_active"
    DORMANT = "dormant"


# Phases of SA-SBFT consensus
class ConsensusPhase(Enum):
    IDLE = "idle"
    REQUEST = "request"
    PRE_PREPARE = "pre_prepare"
    PREPARE = "prepare"
    COMMIT = "commit"
    REPLY = "reply"


# Types of consensus messages
class MessageType(Enum):
    REQUEST = "REQUEST"
    PRE_PREPARE = "PRE_PREPARE"
    PREPARE = "PREPARE"
    COMMIT = "COMMIT"
    REPLY = "REPLY"
    VIEW_CHANGE = "VIEW_CHANGE"
    NEW_VIEW = "NEW_VIEW"
    CHECKPOINT = "CHECKPOINT"


# Orbital state of a satellite
@dataclass
class OrbitalState:
    position: np.ndarray  # ECI coordinates [x, y, z] in km
    velocity: np.ndarray  # ECI velocity [vx, vy, vz] in km/s
    epoch: float  # Julian date
    
    def propagate(self, dt: float) -> 'OrbitalState':
        new_pos = self.position + self.velocity * dt
        return OrbitalState(
            position=new_pos,
            velocity=self.velocity,
            epoch=self.epoch + dt / 86400.0
        )
    
    # Compute distance to another satellite
    def distance_to(self, other: 'OrbitalState') -> float:
        return np.linalg.norm(self.position - other.position)
    
    # Compute elevation angle from ground point
    def elevation_angle(self, ground_point: np.ndarray) -> float:
        # Vector from ground to satellite
        r = self.position - ground_point
        # Assume ground_point normal is radially outward
        ground_normal = ground_point / np.linalg.norm(ground_point)
        # Elevation angle
        cos_zenith = np.dot(r, ground_normal) / (np.linalg.norm(r) * np.linalg.norm(ground_normal))
        zenith = np.arccos(np.clip(cos_zenith, -1, 1))
        elevation = np.pi/2 - zenith
        return np.degrees(elevation)

# Classification of a satellite node in the network
@dataclass
class Satellite:
    sat_id: int
    reputation: float = 1.0
    energy: float = 1.0  # Normalized [0, 1]
    orbital_state: OrbitalState = None
    role: SatelliteRole = SatelliteRole.DORMANT
    is_primary: bool = False
    
    # Consensus state
    current_view: int = 0
    last_checkpoint: int = 0
    message_log: List = field(default_factory=list)
    
    def __post_init__(self):
        if self.orbital_state is None:
            # Default LEO orbit at ~550 km
            angle = np.random.uniform(0, 2*np.pi)
            r = 6371 + 550  # Earth radius + altitude
            self.orbital_state = OrbitalState(
                position=np.array([r * np.cos(angle), r * np.sin(angle), 0]),
                velocity=np.array([-7.6 * np.sin(angle), 7.6 * np.cos(angle), 0]),
                epoch=time.time() / 86400.0
            )


# Message exchanged during consensus
@dataclass
class ConsensusMessage:
    msg_type: MessageType
    view: int
    sequence: int
    digest: str
    sender_id: int
    payload: Dict = field(default_factory=dict)
    mac: str = ""
    timestamp: float = field(default_factory=time.time)
    
    # Compute MAC for message authentication
    def compute_mac(self, key: bytes) -> str:
        data = f"{self.msg_type.value}:{self.view}:{self.sequence}:{self.digest}:{self.sender_id}"
        self.mac = hashlib.sha256(data.encode() + key).hexdigest()[:16]
        return self.mac
    
    # Verify MAC
    def verify_mac(self, key: bytes) -> bool:
        expected = f"{self.msg_type.value}:{self.view}:{self.sequence}:{self.digest}:{self.sender_id}"
        expected_mac = hashlib.sha256(expected.encode() + key).hexdigest()[:16]
        return self.mac == expected_mac


# Block to be committed
@dataclass
class Block:
    height: int
    transactions: List[Dict]
    prev_hash: str
    timestamp: float = field(default_factory=time.time)
    
    @property
    def hash(self) -> str:
        data = f"{self.height}:{self.prev_hash}:{self.timestamp}:{len(self.transactions)}"
        return hashlib.sha256(data.encode()).hexdigest()

# Calculates orbital reliability score for satellites
class OrbitalReliabilityCalculator:    
    def __init__(
        self,
        epoch_duration: float = 60.0,  # seconds
        distance_scale: float = 1000.0,  # km
        energy_threshold_active: float = 0.5,
        energy_threshold_semi: float = 0.2,
        elevation_min: float = 10.0  # degrees
    ):
        self.epoch_duration = epoch_duration
        self.distance_scale = distance_scale
        self.energy_threshold_active = energy_threshold_active
        self.energy_threshold_semi = energy_threshold_semi
        self.elevation_min = elevation_min
    
    # Predict how long satellite will be visible from shard
    def compute_visibility_duration(
        self,
        satellite: Satellite,
        shard_center: np.ndarray,
        current_time: float,
        prediction_horizon: float = 60.0
    ) -> float:
        visible_time = 0.0
        dt = 1.0  # Check every second
        
        state = satellite.orbital_state
        for t in np.arange(0, prediction_horizon, dt):
            future_state = state.propagate(t)
            elevation = future_state.elevation_angle(shard_center)
            if elevation >= self.elevation_min:
                visible_time += dt
        
        return visible_time
    
    def compute_average_distance(
        self,
        satellite: Satellite,
        shard_satellites: List[Satellite],
        current_time: float
    ) -> float:
        if len(shard_satellites) <= 1:
            return 0.0
        
        distances = []
        for other in shard_satellites:
            if other.sat_id != satellite.sat_id:
                d = satellite.orbital_state.distance_to(other.orbital_state)
                distances.append(d)
        
        return np.mean(distances) if distances else 0.0
    

    # Compute orbital reliability score
    def compute_orbital_reliability(
        self,
        satellite: Satellite,
        shard_satellites: List[Satellite],
        shard_center: np.ndarray,
        current_time: float
    ) -> float:
        # Visibility factor
        t_vis = self.compute_visibility_duration(
            satellite, shard_center, current_time
        )
        vis_factor = t_vis / self.epoch_duration
        
        # Distance factor (closer is better)
        d_avg = self.compute_average_distance(
            satellite, shard_satellites, current_time
        )
        dist_factor = np.exp(-d_avg / self.distance_scale)
        
        # Energy factor
        energy_factor = satellite.energy
        
        # Combined reliability
        R_orb = satellite.reputation * vis_factor * dist_factor * energy_factor
        
        return R_orb
    

    # Assign role based on orbital reliability and energy
    def assign_role(
        self,
        satellite: Satellite,
        orbital_reliability: float,
        threshold_active: float
    ) -> SatelliteRole:
        if (orbital_reliability >= threshold_active and 
            satellite.energy >= self.energy_threshold_active):
            return SatelliteRole.ACTIVE
        elif (orbital_reliability >= 0.1 and 
              satellite.energy >= self.energy_threshold_semi):
            return SatelliteRole.SEMI_ACTIVE
        else:
            return SatelliteRole.DORMANT


# Inter-Satellite Link routing optimizer
class ISLRouter:
    
    def __init__(
        self,
        max_isl_distance: float = 5000.0,  # km
        latency_weight: float = 0.3,
        bandwidth_weight: float = 0.5,
        reliability_weight: float = 0.2
    ):
        self.max_isl_distance = max_isl_distance
        self.latency_weight = latency_weight
        self.bandwidth_weight = bandwidth_weight
        self.reliability_weight = reliability_weight
    
    def has_isl_link(
        self,
        sat_i: Satellite,
        sat_j: Satellite,
        current_time: float
    ) -> bool:
        distance = sat_i.orbital_state.distance_to(sat_j.orbital_state)
        return distance <= self.max_isl_distance
    
    def compute_isl_cost(
        self,
        sat_i: Satellite,
        sat_j: Satellite,
        current_time: float
    ) -> float:

        distance = sat_i.orbital_state.distance_to(sat_j.orbital_state)
        
        # Latency (proportional to distance, speed of light)
        latency = distance / 299792.458  # seconds
        
        # Bandwidth (inverse of distance, simplified model)
        bandwidth_factor = 1.0 / (1.0 + distance / 1000.0)
        
        # Reliability (based on both satellites' energy)
        reliability = (sat_i.energy + sat_j.energy) / 2.0
        
        # Combined cost
        cost = (
            self.latency_weight * latency +
            self.bandwidth_weight * (1.0 - bandwidth_factor) +
            self.reliability_weight * (1.0 - reliability)
        )
        
        return cost
    
    def build_routing_tree(
        self,
        primary: Satellite,
        satellites: List[Satellite],
        current_time: float
    ) -> Dict[int, List[int]]:

        # Build graph
        n = len(satellites)
        sat_ids = [s.sat_id for s in satellites]
        id_to_idx = {sid: i for i, sid in enumerate(sat_ids)}
        
        # Prim's algorithm for MST
        in_tree = [False] * n
        cost = [float('inf')] * n
        parent = [-1] * n
        
        root_idx = id_to_idx[primary.sat_id]
        cost[root_idx] = 0
        
        for _ in range(n):
            # Find minimum cost node not in tree
            min_cost = float('inf')
            u = -1
            for i in range(n):
                if not in_tree[i] and cost[i] < min_cost:
                    min_cost = cost[i]
                    u = i
            
            if u == -1:
                break
            
            in_tree[u] = True
            sat_u = satellites[u]
            
            # Update costs of adjacent nodes
            for v in range(n):
                if not in_tree[v]:
                    sat_v = satellites[v]
                    if self.has_isl_link(sat_u, sat_v, current_time):
                        edge_cost = self.compute_isl_cost(sat_u, sat_v, current_time)
                        if edge_cost < cost[v]:
                            cost[v] = edge_cost
                            parent[v] = u
        
        # Build adjacency list
        tree = defaultdict(list)
        for i in range(n):
            if parent[i] != -1:
                tree[sat_ids[parent[i]]].append(sat_ids[i])
        
        return dict(tree)


# Consensus
class SASBFTConsensus:    
    def __init__(
        self,
        satellites: List[Satellite],
        shard_center: np.ndarray,
        epsilon: float = 1.2,  # Threshold adjustment parameter
        checkpoint_interval: int = 100,
        view_change_timeout: float = 30.0  # seconds
    ):
        self.satellites = {s.sat_id: s for s in satellites}
        self.shard_center = shard_center
        self.epsilon = epsilon
        self.checkpoint_interval = checkpoint_interval
        self.view_change_timeout = view_change_timeout
        
        # Consensus state
        self.current_view = 0
        self.sequence_number = 0
        self.primary_id: Optional[int] = None
        self.backup_list: List[int] = []
        
        # Node classification
        self.active_nodes: Set[int] = set()
        self.semi_active_nodes: Set[int] = set()
        self.dormant_nodes: Set[int] = set()
        
        # Message logs
        self.pre_prepare_log: Dict[int, List[ConsensusMessage]] = defaultdict(list)
        self.prepare_log: Dict[int, List[ConsensusMessage]] = defaultdict(list)
        self.commit_log: Dict[int, List[ConsensusMessage]] = defaultdict(list)
        
        # Committed blocks
        self.committed_blocks: List[Block] = []
        
        # Helpers
        self.orbital_calc = OrbitalReliabilityCalculator()
        self.isl_router = ISLRouter()
        
        # Shared key (simplified; use proper key exchange in production)
        self.shared_key = b"orbitalchain_consensus_key"
    
    # Total number of participating nodes
    @property
    def n(self) -> int:
        return len(self.active_nodes) + len(self.semi_active_nodes)
    

    # Maximum number of Byzantine faults tolerated
    @property
    def f(self) -> int:
        return (self.n - 1) // 3
    

    # Classify satellites into Active/Semi-Active/Dormant
    def classify_nodes(self, current_time: float):
        sat_list = list(self.satellites.values())
        
        # Compute orbital reliability for all satellites
        reliabilities = {}
        for sat in sat_list:
            r_orb = self.orbital_calc.compute_orbital_reliability(
                sat, sat_list, self.shard_center, current_time
            )
            reliabilities[sat.sat_id] = r_orb
        
        # Compute threshold
        mean_reliability = np.mean(list(reliabilities.values()))
        threshold = self.epsilon * mean_reliability
        
        # Assign roles
        self.active_nodes.clear()
        self.semi_active_nodes.clear()
        self.dormant_nodes.clear()
        
        for sat in sat_list:
            role = self.orbital_calc.assign_role(
                sat, reliabilities[sat.sat_id], threshold
            )
            sat.role = role
            
            if role == SatelliteRole.ACTIVE:
                self.active_nodes.add(sat.sat_id)
            elif role == SatelliteRole.SEMI_ACTIVE:
                self.semi_active_nodes.add(sat.sat_id)
            else:
                self.dormant_nodes.add(sat.sat_id)
        
        return reliabilities
    
    # Select primary node with highest orbital reliability
    def select_primary(
        self,
        reliabilities: Dict[int, float],
        current_time: float
    ):
        if not self.active_nodes:
            raise ValueError("No active nodes available for primary selection")
        
        # Sort active nodes by reliability
        active_reliabilities = [
            (sat_id, reliabilities[sat_id])
            for sat_id in self.active_nodes
        ]
        active_reliabilities.sort(key=lambda x: x[1], reverse=True)
        
        # Primary is highest reliability
        self.primary_id = active_reliabilities[0][0]
        self.satellites[self.primary_id].is_primary = True
        
        # Backup list is rest of active nodes
        self.backup_list = [sat_id for sat_id, _ in active_reliabilities[1:]]
        
        return self.primary_id
    
    def verify_bft_requirement(self) -> bool:
        return self.n >= 3 * self.f + 1
    
    def run_consensus(
        self,
        transactions: List[Dict],
        current_time: float
    ) -> Tuple[bool, Optional[Block]]:

        # Node classification
        reliabilities = self.classify_nodes(current_time)
        
        # Verify BFT requirement
        if not self.verify_bft_requirement():
            print(f"Insufficient nodes: {self.n} < {3*self.f + 1}")
            return False, None
        
        # Primary selection
        primary_id = self.select_primary(reliabilities, current_time)
        primary = self.satellites[primary_id]
        print(f"Primary selected: Satellite {primary_id} with R_orb={reliabilities[primary_id]:.4f}")
        
        # Build ISL routing tree
        participating = [
            self.satellites[sid] 
            for sid in self.active_nodes | self.semi_active_nodes
        ]
        routing_tree = self.isl_router.build_routing_tree(
            primary, participating, current_time
        )
        
        # Request
        self.sequence_number += 1
        tx_digest = hashlib.sha256(str(transactions).encode()).hexdigest()[:16]
        
        request_msg = ConsensusMessage(
            msg_type=MessageType.REQUEST,
            view=self.current_view,
            sequence=self.sequence_number,
            digest=tx_digest,
            sender_id=primary_id,
            payload={'transactions': transactions}
        )
        request_msg.compute_mac(self.shared_key)
        
        # Pre-Prepare (collect votes)
        votes = {}
        for sat_id in self.active_nodes | self.semi_active_nodes:
            # Each satellite evaluates and votes
            decision = self._evaluate_transactions(transactions)
            votes[sat_id] = decision
            
            pp_msg = ConsensusMessage(
                msg_type=MessageType.PRE_PREPARE,
                view=self.current_view,
                sequence=self.sequence_number,
                digest=tx_digest,
                sender_id=sat_id,
                payload={'decision': decision}
            )
            pp_msg.compute_mac(self.shared_key)
            self.pre_prepare_log[self.sequence_number].append(pp_msg)
        
        # Prepare (determine majority)
        vote_counts = defaultdict(int)
        for decision in votes.values():
            vote_counts[decision] += 1
        
        majority_decision = max(vote_counts.keys(), key=lambda k: vote_counts[k])
        majority_count = vote_counts[majority_decision]
        
        if majority_count < 2 * self.f + 1:
            print(f"Consensus failed: {majority_count} votes < {2*self.f + 1} required")
            return False, None
        
        print(f"Majority decision: {majority_decision} with {majority_count} votes")
        
        # Correct dissenting active nodes
        for sat_id in self.active_nodes:
            if votes[sat_id] != majority_decision:
                self.satellites[sat_id].reputation *= 0.9  # Penalty
                votes[sat_id] = majority_decision
        
        # Commit
        commit_msg = ConsensusMessage(
            msg_type=MessageType.COMMIT,
            view=self.current_view,
            sequence=self.sequence_number,
            digest=tx_digest,
            sender_id=primary_id,
            payload={'decision': majority_decision}
        )
        commit_msg.compute_mac(self.shared_key)
        
        # Reply
        reply_count = 0
        for sat_id in self.active_nodes | self.semi_active_nodes:
            reply_msg = ConsensusMessage(
                msg_type=MessageType.REPLY,
                view=self.current_view,
                sequence=self.sequence_number,
                digest=tx_digest,
                sender_id=sat_id,
                payload={'decision': majority_decision}
            )
            reply_count += 1
        

        sample_size = max(1, self.n // 3)

        
        # Commit block
        prev_hash = self.committed_blocks[-1].hash if self.committed_blocks else "genesis"
        block = Block(
            height=len(self.committed_blocks),
            transactions=transactions,
            prev_hash=prev_hash
        )
        self.committed_blocks.append(block)
        
        # Update reputations
        self._update_reputations(votes, majority_decision)
        
        print(f"Block {block.height} committed with {len(transactions)} transactions")
        
        return True, block
    
    # Evaluate transactions and return decision
    def _evaluate_transactions(self, transactions: List[Dict]) -> str:        
        for tx in transactions:
            if 'data' not in tx:
                return "REJECT"
        return "ACCEPT"
    
    # Update satellite reputations based on voting alignment
    def _update_reputations(
        self,
        votes: Dict[int, str],
        majority_decision: str,
        a: float = 0.05,
        b: float = 0.1
    ):
        for sat_id, decision in votes.items():
            sat = self.satellites[sat_id]
            if decision == majority_decision:
                # Increase reputation
                sat.reputation = 0.5 * (sat.reputation * (1 - a) + a) + 0.5
            else:
                # Decrease reputation
                sat.reputation = sat.reputation * (1 - b)
            
            sat.reputation = max(0.1, min(1.0, sat.reputation))
    
    def predictive_view_change(
        self,
        current_time: float,
        prediction_horizon: float = 30.0,
        elevation_threshold: float = 5.0
    ) -> bool:
        if self.primary_id is None:
            return False
        
        primary = self.satellites[self.primary_id]
        
        # Predict primary position
        future_state = primary.orbital_state.propagate(prediction_horizon)
        future_elevation = future_state.elevation_angle(self.shard_center)
        
        if future_elevation < elevation_threshold:
            # Primary will lose visibility, initiate handover
            print(f"Predictive view change: Primary {self.primary_id} elevation={future_elevation:.1f}°")
            
            # Select new primary from backup list
            if self.backup_list:
                old_primary = self.primary_id
                self.satellites[old_primary].is_primary = False
                
                self.primary_id = self.backup_list.pop(0)
                self.satellites[self.primary_id].is_primary = True
                self.backup_list.append(old_primary)
                
                self.current_view += 1
                print(f"New primary: Satellite {self.primary_id}, view={self.current_view}")
                return True
        
        return False
    
    # Create checkpoint for recovery
    def create_checkpoint(self) -> Dict:
        if not self.committed_blocks:
            return {}
        
        latest_block = self.committed_blocks[-1]
        checkpoint = {
            'block_height': latest_block.height,
            'block_hash': latest_block.hash,
            'view': self.current_view,
            'sequence': self.sequence_number,
            'primary_id': self.primary_id,
            'reputations': {
                sat_id: sat.reputation 
                for sat_id, sat in self.satellites.items()
            }
        }
        
        return checkpoint
    
    # Recover state from checkpoint
    def recover_from_checkpoint(self, checkpoint: Dict):
        if not checkpoint:
            return
        
        self.current_view = checkpoint['view']
        self.sequence_number = checkpoint['sequence']
        self.primary_id = checkpoint['primary_id']
        
        for sat_id, reputation in checkpoint['reputations'].items():
            if sat_id in self.satellites:
                self.satellites[sat_id].reputation = reputation


def demonstrate_sa_sbft():
    print("=" * 60)
    print("SA-SBFT Consensus Demonstration")
    print("=" * 60)
    
    # Create satellite constellation
    np.random.seed(42)
    num_satellites = 20
    
    satellites = []
    for i in range(num_satellites):
        # Random orbital parameters
        angle = 2 * np.pi * i / num_satellites
        r = 6371 + 550  # km
        
        sat = Satellite(
            sat_id=i,
            reputation=np.random.uniform(0.5, 1.0),
            energy=np.random.uniform(0.3, 1.0),
            orbital_state=OrbitalState(
                position=np.array([
                    r * np.cos(angle),
                    r * np.sin(angle),
                    np.random.uniform(-100, 100)
                ]),
                velocity=np.array([
                    -7.6 * np.sin(angle),
                    7.6 * np.cos(angle),
                    0
                ]),
                epoch=time.time() / 86400.0
            )
        )
        satellites.append(sat)
    
    # ground station location
    shard_center = np.array([6371 + 10, 0, 0])  # 10 km altitude reference
    
    # Create consensus engine
    consensus = SASBFTConsensus(
        satellites=satellites,
        shard_center=shard_center,
        epsilon=1.2
    )
    
    # Run consensus rounds
    for round_num in range(5):
        print(f"\n--- Consensus Round {round_num + 1} ---")
        
        current_time = time.time()
        
        # Check for predictive view change
        consensus.predictive_view_change(current_time)
        
        # Create test transactions
        transactions = [
            {'tx_id': f'tx_{round_num}_{i}', 'data': f'value_{i}'}
            for i in range(10)
        ]
        
        # Run consensus
        success, block = consensus.run_consensus(transactions, current_time)
        
        if success:
            print(f"Round {round_num + 1}: SUCCESS")
            print(f"  Active nodes: {len(consensus.active_nodes)}")
            print(f"  Semi-active nodes: {len(consensus.semi_active_nodes)}")
            print(f"  Dormant nodes: {len(consensus.dormant_nodes)}")
        else:
            print(f"Round {round_num + 1}: FAILED")
        
        # Create checkpoint periodically
        if (round_num + 1) % 3 == 0:
            checkpoint = consensus.create_checkpoint()
            print(f"  Checkpoint created at height {checkpoint.get('block_height', 0)}")
    
    print("\n" + "=" * 60)
    print("Final Statistics")
    print("=" * 60)
    print(f"Total blocks committed: {len(consensus.committed_blocks)}")
    print(f"Current view: {consensus.current_view}")
    print(f"Current primary: Satellite {consensus.primary_id}")
    
    # Print reputation changes
    print("\nReputation Summary:")
    for sat_id in sorted(consensus.satellites.keys())[:5]:
        sat = consensus.satellites[sat_id]
        print(f"  Satellite {sat_id}: reputation={sat.reputation:.4f}, role={sat.role.value}")


if __name__ == "__main__":
    demonstrate_sa_sbft()

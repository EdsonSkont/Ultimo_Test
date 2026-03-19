import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time


# Data point in the stream
@dataclass
class DataPoint:
    coordinates: np.ndarray
    weight: float
    timestamp: float
    
    def __post_init__(self):
        self.coordinates = np.array(self.coordinates)
    
    @property
    def dimensionality(self) -> int:
        return len(self.coordinates)


# Density grid cell
@dataclass
class DensityGrid:
    grid_id: Tuple[int, ...]
    density: float = 0.0
    sum_coordinates: np.ndarray = None
    sum_squares: np.ndarray = None
    last_update_time: float = 0.0
    cluster_id: Optional[int] = None
    
    def __post_init__(self):
        if self.sum_coordinates is None:
            self.sum_coordinates = np.zeros(len(self.grid_id))
        if self.sum_squares is None:
            self.sum_squares = np.zeros(len(self.grid_id))
    
    # Compute the center of mass of the grid
    @property
    def center(self) -> np.ndarray:
        if self.density > 0:
            return self.sum_coordinates / self.density
        return np.array(self.grid_id) + 0.5  # Grid center
    
    # Update grid statistics with a new data point
    def update(self, point: DataPoint):
        self.density += point.weight
        self.sum_coordinates += point.coordinates * point.weight
        self.sum_squares += (point.coordinates ** 2) * point.weight
        self.last_update_time = point.timestamp


# Represents a cluster of density grids
@dataclass
class Cluster:
    cluster_id: int
    grids: Set[Tuple[int, ...]] = field(default_factory=set)
    
    @property
    def size(self) -> int:
        return len(self.grids)
    
    def add_grid(self, grid_id: Tuple[int, ...]):
        self.grids.add(grid_id)
    
    def remove_grid(self, grid_id: Tuple[int, ...]):
        self.grids.discard(grid_id)


# Initialize the D-Stream clustering
class DStreamClustering:
  
    def __init__(
        self,
        grid_size: float = 0.1,
        density_threshold: float = 1.0,
        decay_factor: float = 0.998,
        gap_time: float = 1.0,
        dimensionality: int = 2
    ):
        self.grid_size = grid_size
        self.density_threshold = density_threshold
        self.decay_factor = decay_factor
        self.gap_time = gap_time
        self.dimensionality = dimensionality
        
        # Grid storage
        self.grid_list: Dict[Tuple[int, ...], DensityGrid] = {}
        
        # Cluster storage
        self.clusters: Dict[int, Cluster] = {}
        self.next_cluster_id = 0
        
        # Time tracking
        self.current_time = 0.0
        self.last_adjustment_time = 0.0
        
        # Statistics
        self.total_points_processed = 0
        self.sporadic_grids_removed = 0
    
    # Determine which grid cell a point belongs to
    def _get_grid_id(self, point: DataPoint) -> Tuple[int, ...]:
        indices = tuple(
            int(np.floor(coord / self.grid_size))
            for coord in point.coordinates
        )
        return indices
    
    # Apply density decay to a grid based on elapsed time
    def _apply_density_decay(self, grid: DensityGrid, current_time: float):
        if grid.last_update_time < current_time:
            time_elapsed = current_time - grid.last_update_time
            decay = self.decay_factor ** time_elapsed
            grid.density *= decay
            grid.sum_coordinates *= decay
            grid.sum_squares *= decay
            grid.last_update_time = current_time
    
    # Get neighboring grid IDs
    def _get_neighbors(self, grid_id: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        neighbors = []
        d = len(grid_id)
        
        # Generate all combinations of -1, 0, 1 for each dimension
        def generate_offsets(dim):
            if dim == 0:
                yield ()
            else:
                for offset in [-1, 0, 1]:
                    for rest in generate_offsets(dim - 1):
                        yield (offset,) + rest
        
        for offset in generate_offsets(d):
            if offset != tuple([0] * d):  # Exclude self
                neighbor = tuple(g + o for g, o in zip(grid_id, offset))
                neighbors.append(neighbor)
        
        return neighbors
    
    # Process a single data point from the stream
    def process_point(self, point: DataPoint) -> Optional[Tuple[int, ...]]:
        self.current_time = point.timestamp
        self.total_points_processed += 1
        
        # Step 6: Determine the density grid
        grid_id = self._get_grid_id(point)
        
        # Step 7-9: Create grid if not exists
        if grid_id not in self.grid_list:
            self.grid_list[grid_id] = DensityGrid(
                grid_id=grid_id,
                sum_coordinates=np.zeros(self.dimensionality),
                sum_squares=np.zeros(self.dimensionality)
            )
        
        grid = self.grid_list[grid_id]
        
        # Apply decay before update
        self._apply_density_decay(grid, self.current_time)
        
        # Step 11-16: Update grid statistics
        grid.update(point)
        
        # Check if it's time for clustering adjustment
        if self.current_time - self.last_adjustment_time >= self.gap_time:
            self._adjust_clustering()
            self.last_adjustment_time = self.current_time
        
        return grid_id
    
    # Assigns grids to clusters based on density threshold
    def _initial_clustering(self):
        for grid_id, grid in self.grid_list.items():
            self._apply_density_decay(grid, self.current_time)
            
            if grid.density >= self.density_threshold:
                # Find or create cluster
                self._assign_to_cluster(grid_id, grid)
            else:
                # Mark as noise
                grid.cluster_id = None
    
    # Assign a grid to a cluster based on neighboring grids
    def _assign_to_cluster(self, grid_id: Tuple[int, ...], grid: DensityGrid):
        # Check neighbors for existing clusters
        neighbor_clusters = set()
        for neighbor_id in self._get_neighbors(grid_id):
            if neighbor_id in self.grid_list:
                neighbor_grid = self.grid_list[neighbor_id]
                if (neighbor_grid.cluster_id is not None and 
                    neighbor_grid.density >= self.density_threshold):
                    neighbor_clusters.add(neighbor_grid.cluster_id)
        
        if not neighbor_clusters:
            # Create new cluster
            cluster_id = self.next_cluster_id
            self.next_cluster_id += 1
            self.clusters[cluster_id] = Cluster(cluster_id=cluster_id)
            self.clusters[cluster_id].add_grid(grid_id)
            grid.cluster_id = cluster_id
        elif len(neighbor_clusters) == 1:
            # Join existing cluster
            cluster_id = neighbor_clusters.pop()
            self.clusters[cluster_id].add_grid(grid_id)
            grid.cluster_id = cluster_id
        else:
            # Merge clusters (join to the smallest ID)
            cluster_id = min(neighbor_clusters)
            for other_id in neighbor_clusters:
                if other_id != cluster_id:
                    self._merge_clusters(cluster_id, other_id)
            self.clusters[cluster_id].add_grid(grid_id)
            grid.cluster_id = cluster_id
    
    # Merge source cluster into target cluster
    def _merge_clusters(self, target_id: int, source_id: int):
        if source_id not in self.clusters:
            return
        
        source_cluster = self.clusters[source_id]
        target_cluster = self.clusters[target_id]
        
        # Move all grids
        for grid_id in source_cluster.grids:
            target_cluster.add_grid(grid_id)
            if grid_id in self.grid_list:
                self.grid_list[grid_id].cluster_id = target_id
        
        # Remove source cluster
        del self.clusters[source_id]
    
    # Adjust clustering periodically
    def _adjust_clustering(self):
        grids_to_remove = []
        
        for grid_id, grid in self.grid_list.items():
            # Apply decay
            self._apply_density_decay(grid, self.current_time)
            
            # Line 32-33: Check if density falls below threshold
            if grid.density < self.density_threshold * 0.1:  # Sporadic threshold
                grids_to_remove.append(grid_id)
            elif grid.density < self.density_threshold:
                # Remove from cluster and mark as noise
                if grid.cluster_id is not None:
                    if grid.cluster_id in self.clusters:
                        self.clusters[grid.cluster_id].remove_grid(grid_id)
                    grid.cluster_id = None
            else:
                # Line 35: Reassign to appropriate cluster
                self._assign_to_cluster(grid_id, grid)
        
        # Line 28: Remove sporadic grids
        for grid_id in grids_to_remove:
            if grid_id in self.grid_list:
                grid = self.grid_list[grid_id]
                if grid.cluster_id is not None and grid.cluster_id in self.clusters:
                    self.clusters[grid.cluster_id].remove_grid(grid_id)
                del self.grid_list[grid_id]
                self.sporadic_grids_removed += 1
        
        # Remove empty clusters
        empty_clusters = [
            cid for cid, cluster in self.clusters.items() 
            if cluster.size == 0
        ]
        for cid in empty_clusters:
            del self.clusters[cid]
    
    # Process a batch of data points
    def process_batch(self, points: List[DataPoint]):
        for point in points:
            self.process_point(point)
    
    def get_clusters(self) -> Dict[int, List[Tuple[int, ...]]]:

        return {
            cid: list(cluster.grids) 
            for cid, cluster in self.clusters.items()
        }
    
    # Get the center of mass for each cluster
    def get_cluster_centers(self) -> Dict[int, np.ndarray]:
        centers = {}
        for cid, cluster in self.clusters.items():
            if cluster.size > 0:
                total_density = 0
                weighted_center = np.zeros(self.dimensionality)
                
                for grid_id in cluster.grids:
                    if grid_id in self.grid_list:
                        grid = self.grid_list[grid_id]
                        weighted_center += grid.center * grid.density
                        total_density += grid.density
                
                if total_density > 0:
                    centers[cid] = weighted_center / total_density
        
        return centers
    
    def get_statistics(self) -> Dict:
        return {
            'total_points': self.total_points_processed,
            'active_grids': len(self.grid_list),
            'num_clusters': len(self.clusters),
            'sporadic_removed': self.sporadic_grids_removed,
            'current_time': self.current_time
        }
    
    def predict_cluster(self, point: DataPoint) -> Optional[int]:
        grid_id = self._get_grid_id(point)
        
        if grid_id in self.grid_list:
            return self.grid_list[grid_id].cluster_id
        
        # Check neighbors
        for neighbor_id in self._get_neighbors(grid_id):
            if neighbor_id in self.grid_list:
                neighbor = self.grid_list[neighbor_id]
                if (neighbor.cluster_id is not None and 
                    neighbor.density >= self.density_threshold):
                    return neighbor.cluster_id
        
        return None


class OrbitalChainDStream(DStreamClustering):
    
    def __init__(
        self,
        grid_size: float = 0.1,
        density_threshold: float = 1.0,
        decay_factor: float = 0.998,
        gap_time: float = 10.0  # Longer gap for satellite data
    ):
        super().__init__(
            grid_size=grid_size,
            density_threshold=density_threshold,
            decay_factor=decay_factor,
            gap_time=gap_time,
            dimensionality=3  # Latitude, Longitude, Truth Score
        )
        
        # Satellite-specific tracking
        self.satellite_contributions: Dict[int, int] = defaultdict(int)
    
    # Process data from a satellite.
    def process_satellite_data(
        self,
        satellite_id: int,
        latitude: float,
        longitude: float,
        truth_score: float,
        weight: float,
        timestamp: float
    ) -> Optional[int]:
        # Create data point
        point = DataPoint(
            coordinates=np.array([latitude, longitude, truth_score]),
            weight=weight,
            timestamp=timestamp
        )
        
        # Process point
        grid_id = self.process_point(point)
        
        # Track satellite contribution
        self.satellite_contributions[satellite_id] += 1
        
        # Return cluster assignment
        return self.predict_cluster(point)


def demonstrate_dstream(use_random=False):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    if use_random:
        # Synthetic Gaussian clusters for quick algorithm testing
        print("\nD-Stream Clustering  —  synthetic data (--random mode)\n")
        clustering = DStreamClustering(grid_size=0.5, density_threshold=2.0,
                                        decay_factor=0.99, gap_time=5.0, dimensionality=2)
        rng = np.random.default_rng(42)
        centers = [np.array([2.0,2.0]), np.array([8.0,2.0]), np.array([5.0,8.0])]
        for t in range(100):
            c = centers[rng.integers(0,3)]
            clustering.process_point(DataPoint(
                coordinates=c + rng.standard_normal(2)*0.5,
                weight=float(rng.uniform(0.5,1.5)),
                timestamp=float(t)
            ))
    else:
        # Default: real ADCS attitude data — natural clusters from Fine Point vs Slewing modes
        # Coordinates: (euler_roll_deg, euler_pitch_deg, pointing_error_norm)
        # Weight: 1.0 for Fine Point (reliable), 0.5 for Slewing (transitional)
        from src.data.ccsds_adapter import load_attitude_cluster_stream
        stream = load_attitude_cluster_stream(max_rows=1000)

        print(f"\nD-Stream Clustering  —  07_attitude_quaternions.csv\n")
        print(f"  Points     : {len(stream)}  (10 Hz, 100 seconds of ADCS data)")
        print(f"  Coordinates: (euler_roll_deg, euler_pitch_deg, pointing_error_norm)")
        print(f"  Weights    : 1.0 = Fine Point  |  0.5 = Slewing (less reliable)")
        modes = {}
        for pt in stream:
            modes[pt.source] = modes.get(pt.source, 0) + 1
        for mode, cnt in modes.items():
            print(f"  ADCS mode  : {mode}  ({cnt} points)")
        print()

        clustering = DStreamClustering(grid_size=2.0, density_threshold=3.0,
                                        decay_factor=0.99, gap_time=1.0, dimensionality=3)
        for pt in stream:
            clustering.process_point(DataPoint(
                coordinates=pt.coordinates,
                weight=pt.weight,
                timestamp=pt.timestamp,
            ))

    stats    = clustering.get_statistics()
    clusters = clustering.get_clusters()
    centers_out = clustering.get_cluster_centers()

    print(f"  Total points processed : {stats['total_points']}")
    print(f"  Active density grids   : {stats['active_grids']}")
    print(f"  Clusters found         : {stats['num_clusters']}")
    print(f"  Sporadic grids removed : {stats['sporadic_removed']}")
    print()
    for cid, grids in sorted(clusters.items(), key=lambda x: -len(x[1])):
        c = centers_out.get(cid, np.zeros(clustering.dimensionality))
        coord_str = "  ".join(f"{v:.3f}" for v in c)
        print(f"  Cluster {cid}: {len(grids)} grids   center = ({coord_str})")


if __name__ == "__main__":
    import sys
    demonstrate_dstream(use_random="--random" in sys.argv)

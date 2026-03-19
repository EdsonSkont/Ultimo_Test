# Privacy-preserving streaming truth discovery for OrbitalChain
# Default: real CCSDS ISL telemetry  |  --random: synthetic providers

import numpy as np
import argparse
import os
from enum import Enum
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field

from ..crypto.secret_sharing import AdditiveSecretSharing, Share
from ..crypto.beaver_triples import BeaverTripleGenerator, SecureMultiplication, SharedTriple


# Malicious behavior profiles (random mode only)

class MaliciousProfile(Enum):
    CONSTANT_BIAS   = "constant_bias"
    RANDOM_SPIKE    = "random_spike"
    GRADUAL_DRIFT   = "gradual_drift"
    SMART_ADVERSARY = "smart_adversary"


@dataclass
class ProviderConfig:
    # Behavioral spec; never seen by detection engines
    provider_id:   int
    is_malicious:  bool
    profile:       Optional[MaliciousProfile] = None
    base_bias:     float = 0.0
    drift_rate:    float = 0.0
    spike_prob:    float = 0.0
    spike_scale:   float = 1.0
    honest_epochs: int   = 0
    noise_std:     float = 0.05

    def sense(self, true_value: float, epoch: int, rng: np.random.Generator) -> float:
        noise = float(rng.normal(0, self.noise_std))
        if not self.is_malicious:
            return true_value + noise
        if self.profile == MaliciousProfile.CONSTANT_BIAS:
            return true_value + self.base_bias + noise
        if self.profile == MaliciousProfile.RANDOM_SPIKE:
            if rng.random() < self.spike_prob:
                return true_value + rng.choice([-1, 1]) * self.spike_scale * abs(float(rng.normal(1, 0.3)))
            return true_value + noise
        if self.profile == MaliciousProfile.GRADUAL_DRIFT:
            return true_value + self.base_bias + self.drift_rate * epoch + noise
        if self.profile == MaliciousProfile.SMART_ADVERSARY:
            if epoch <= self.honest_epochs:
                return true_value + noise
            return true_value + self.base_bias * (1 + 0.1 * (epoch - self.honest_epochs)) + noise
        return true_value + noise


def assign_malicious_configs(
    num_providers: int, num_malicious: int, malicious_ids: Set[int],
    base_bias: float, honest_noise: float, rng: np.random.Generator,
) -> Dict[int, ProviderConfig]:
    # Assign a different randomised profile to each malicious provider
    profiles = list(MaliciousProfile)
    rng.shuffle(profiles)
    configs: Dict[int, ProviderConfig] = {}
    for idx, k in enumerate(sorted(malicious_ids)):
        profile = profiles[idx % len(profiles)]
        if profile == MaliciousProfile.CONSTANT_BIAS:
            cfg = ProviderConfig(k, True, profile,
                                 base_bias=base_bias * float(rng.uniform(0.7, 1.4)),
                                 noise_std=honest_noise)
        elif profile == MaliciousProfile.RANDOM_SPIKE:
            cfg = ProviderConfig(k, True, profile,
                                 spike_prob=float(rng.uniform(0.3, 0.7)),
                                 spike_scale=base_bias * float(rng.uniform(1.5, 3.0)),
                                 noise_std=honest_noise)
        elif profile == MaliciousProfile.GRADUAL_DRIFT:
            cfg = ProviderConfig(k, True, profile,
                                 base_bias=base_bias * float(rng.uniform(0.1, 0.3)),
                                 drift_rate=base_bias * float(rng.uniform(0.03, 0.08)),
                                 noise_std=honest_noise)
        elif profile == MaliciousProfile.SMART_ADVERSARY:
            cfg = ProviderConfig(k, True, profile,
                                 base_bias=base_bias * float(rng.uniform(0.8, 1.2)),
                                 honest_epochs=int(rng.integers(2, 6)),
                                 noise_std=honest_noise)
        else:
            cfg = ProviderConfig(k, True, profile, base_bias=base_bias, noise_std=honest_noise)
        configs[k] = cfg
    for k in range(num_providers):
        if k not in malicious_ids:
            configs[k] = ProviderConfig(k, False, noise_std=honest_noise)
    return configs


# MPC truth discovery engine

@dataclass
class DataProviderState:
    provider_id:          int
    weight:               float = 1.0
    accumulated_distance: float = 0.0
    weight_shares:        List[Share] = field(default_factory=list)
    distance_shares:      List[Share] = field(default_factory=list)


@dataclass
class EpochResult:
    epoch:                 int
    truth_value:           float
    weights:               Dict[int, float]
    accumulated_distances: Dict[int, float]
    truth_shares:          List[Share]
    sensing_values:        List[float]
    processing_time_ms:    float


class StreamingTruthDiscovery:
    # w_k = max(-log(ST_k / ST* + eps), 0.01),  ST_k <- decay*ST_k + (x_k - truth)^2

    SCALE = 10 ** 6
    EPS   = 1e-10

    def __init__(self, num_satellites: int, num_providers: int,
                 decay: float = 0.9, prime: Optional[int] = None):
        if num_satellites < 2: raise ValueError("num_satellites >= 2")
        if num_providers  < 2: raise ValueError("num_providers >= 2")
        if not 0 < decay <= 1: raise ValueError("decay in (0,1]")
        self.num_satellites = num_satellites
        self.num_providers  = num_providers
        self.decay          = decay
        self.ss    = AdditiveSecretSharing(num_satellites, prime)
        self.prime = self.ss.prime_modulus
        self.mult  = SecureMultiplication(num_satellites, self.prime)
        self.tgen  = BeaverTripleGenerator(num_satellites, self.prime)
        self.providers: Dict[int, DataProviderState] = {
            k: DataProviderState(provider_id=k) for k in range(num_providers)
        }
        self.epoch = 0

    def _enc(self, v: float) -> int:
        return int(round(v * self.SCALE)) % self.prime

    def _dec(self, x: int) -> float:
        if x > self.prime // 2: x -= self.prime
        return x / self.SCALE

    def _share(self, v: float) -> List[Share]:
        return self.ss.share(self._enc(v))

    def _recon(self, shares: List[Share]) -> float:
        return self._dec(self.ss.reconstruct(shares))

    def _truth_epoch1(self, sensing: Dict[int, List[Share]]) -> List[Share]:
        # Unweighted mean for epoch 1
        mean = sum(self._recon(s) for s in sensing.values()) / len(sensing)
        return self._share(mean)

    def _truth_weighted(self, sensing: Dict[int, List[Share]],
                        weights: Dict[int, List[Share]],
                        triples: Dict[int, List[SharedTriple]]) -> List[Share]:
        # Weighted mean via Beaver triple products
        num = den = 0.0
        for k in sensing:
            prod = self.mult.multiply(weights[k], sensing[k], triples[k])
            num += self._dec(self.ss.reconstruct(prod))
            den += self._dec(self.ss.reconstruct(weights[k]))
        if abs(den) < self.EPS: den = self.EPS
        return self._share(num / den / self.SCALE)

    def _update_weights(self, truth: List[Share],
                        sensing: Dict[int, List[Share]],
                        triples: Dict[int, List[SharedTriple]]) -> None:
        # MPC squared errors -> update ST_k -> derive weights
        neg = self.prime - 1
        for k, x in sensing.items():
            diff  = self.ss.add_shares(x, self.ss.multiply_by_constant(truth, neg))
            d_sq  = self.mult.square(diff, triples[k])
            raw   = abs(self._dec(self.ss.reconstruct(d_sq))) / self.SCALE
            self.providers[k].accumulated_distance = (
                self.decay * self.providers[k].accumulated_distance + raw
            )
        total = sum(p.accumulated_distance for p in self.providers.values())
        if total <= 0: total = self.EPS
        for k, p in self.providers.items():
            w = max(-np.log(p.accumulated_distance / total + self.EPS), 0.01)
            p.weight        = w
            p.weight_shares = self._share(w)

    def run_epoch(self, values: List[float]) -> EpochResult:
        import time
        assert len(values) == self.num_providers
        self.epoch += 1
        t0 = time.time()
        sensing   = {k: self._share(v)             for k, v in enumerate(values)}
        t_triples = {k: self.tgen.generate_triple() for k in range(self.num_providers)}
        if self.epoch == 1:
            truth = self._truth_epoch1(sensing)
        else:
            truth = self._truth_weighted(
                sensing,
                {k: p.weight_shares for k, p in self.providers.items()},
                t_triples
            )
        sq_tri = {k: self.tgen.generate_triple() for k in range(self.num_providers)}
        self._update_weights(truth, sensing, sq_tri)
        return EpochResult(
            epoch                 = self.epoch,
            truth_value           = self._recon(truth),
            weights               = {k: p.weight for k, p in self.providers.items()},
            accumulated_distances = {k: p.accumulated_distance for k, p in self.providers.items()},
            truth_shares          = truth,
            sensing_values        = list(values),
            processing_time_ms    = (time.time() - t0) * 1000,
        )

    def trust_ranking(self) -> List[Tuple[int, float, float]]:
        return sorted(
            [(k, p.weight, p.accumulated_distance) for k, p in self.providers.items()],
            key=lambda x: x[2]
        )

    def reset(self):
        self.epoch = 0
        for p in self.providers.values():
            p.weight = 1.0; p.accumulated_distance = 0.0
            p.weight_shares = []; p.distance_shares = []


# Multi-signal suspicion engine

@dataclass
class SuspicionMetrics:
    provider_id:        int
    accum_sq_error:     float
    accum_majority_dev: float
    report_variance:    float
    composite_score:    float
    is_flagged:         bool
    confidence:         float


class SuspicionEngine:
    # Signals: ST_k (MPC error), MD_k (majority deviation), VAR_k (self-variance)
    # Adaptive IQR threshold — num_malicious is never used

    def __init__(self, num_providers: int, decay: float = 0.9,
                 trim_fraction: float = 0.20,
                 w_accum: float = 0.50, w_majority: float = 0.35, w_variance: float = 0.15,
                 iqr_threshold: float = 1.0):
        self.num_providers  = num_providers
        self.decay          = decay
        self.trim           = trim_fraction
        self.w              = np.array([w_accum, w_majority, w_variance])
        self.iqr_k          = iqr_threshold
        self.accum_sq_error = np.zeros(num_providers)
        self.accum_maj_dev  = np.zeros(num_providers)
        self._report_history: Dict[int, List[float]] = {k: [] for k in range(num_providers)}
        self.epoch          = 0

    def _robust_consensus(self, values: np.ndarray) -> float:
        # Trimmed mean — resistant to outliers
        n  = len(values)
        k  = min(max(1, int(np.floor(self.trim * n))), n // 3)
        sv = np.sort(values)
        return float(np.mean(sv[k: n - k])) if n - 2 * k > 0 else float(np.mean(values))

    def update(self, result: EpochResult) -> None:
        self.epoch += 1
        values = np.array(result.sensing_values)
        robust = self._robust_consensus(values)
        for k in range(self.num_providers):
            self.accum_sq_error[k] = result.accumulated_distances[k]
            self.accum_maj_dev[k]  = self.decay * self.accum_maj_dev[k] + abs(values[k] - robust)
            self._report_history[k].append(values[k])

    def _report_variances(self) -> np.ndarray:
        return np.array([
            float(np.var(self._report_history[k])) if len(self._report_history[k]) > 1 else 0.0
            for k in range(self.num_providers)
        ])

    def _normalise(self, arr: np.ndarray) -> np.ndarray:
        r = arr.max() - arr.min()
        return (arr - arr.min()) / r if r > 1e-12 else np.zeros_like(arr)

    def composite_scores(self) -> np.ndarray:
        s1 = self._normalise(self.accum_sq_error)
        s2 = self._normalise(self.accum_maj_dev)
        s3 = self._normalise(self._report_variances())
        return self.w[0] * s1 + self.w[1] * s2 + self.w[2] * s3

    def get_suspects(self) -> Set[int]:
        # Flag providers above median + iqr_k * IQR with a minimum floor of 0.3
        scores   = self.composite_scores()
        q25, q75 = np.percentile(scores, 25), np.percentile(scores, 75)
        thresh   = max(np.median(scores) + self.iqr_k * (q75 - q25), 0.3)
        return {k for k in range(self.num_providers) if scores[k] > thresh}

    def get_metrics(self) -> Dict[int, SuspicionMetrics]:
        scores    = self.composite_scores()
        suspects  = self.get_suspects()
        variances = self._report_variances()
        q25, q75  = np.percentile(scores, 25), np.percentile(scores, 75)
        thresh    = max(np.median(scores) + self.iqr_k * (q75 - q25), 0.3)
        max_s     = scores.max() if scores.max() > thresh else thresh + 1e-9
        return {
            k: SuspicionMetrics(
                provider_id        = k,
                accum_sq_error     = float(self.accum_sq_error[k]),
                accum_majority_dev = float(self.accum_maj_dev[k]),
                report_variance    = float(variances[k]),
                composite_score    = float(scores[k]),
                is_flagged         = k in suspects,
                confidence         = float(np.clip(
                    (scores[k] - thresh) / (max_s - thresh + 1e-9), 0, 1)),
            )
            for k in range(self.num_providers)
        }


# Plain-text reference engine (no MPC, same weight formula)

class SimplifiedTruthDiscovery:
    EPS = 1e-10

    def __init__(self, num_providers: int, decay: float = 0.9):
        self.num_providers = num_providers
        self.decay         = decay
        self.weights       = np.ones(num_providers)
        self.accum         = np.zeros(num_providers)
        self.epoch         = 0

    def run_epoch(self, values: np.ndarray) -> Tuple[float, np.ndarray]:
        self.epoch += 1
        truth      = np.sum(self.weights * values) / np.sum(self.weights)
        self.accum = self.decay * self.accum + (values - truth) ** 2
        total      = np.sum(self.accum)
        if total > 0:
            self.weights = np.maximum(-np.log(self.accum / total + self.EPS), 0.01)
        return truth, self.weights.copy()

    def reset(self):
        self.weights = np.ones(self.num_providers)
        self.accum   = np.zeros(self.num_providers)
        self.epoch   = 0


# Load sensing values from CCSDS ISL log

def _load_real_epochs(fault_threshold: int = 2) -> Tuple[List[List[float]], List[List[bool]], List[str]]:
    # Returns (sensing_epochs, fault_labels, satellite_ids)
    # fault_threshold: min fault-alert count to classify a satellite as malicious
    # Default=2 keeps honest > malicious (only persistent faulters are flagged)
    import json
    _HERE    = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(_HERE, "..", "..", "data", "satellite_ccsds_dataset"))

    with open(os.path.join(data_dir, "05_isl_stream_log.json")) as f:
        isl = json.load(f)

    pkts     = isl["packets"]
    sat_ids  = sorted({p["source_sat_id"] for p in pkts})

    # Count fault alerts per satellite; only persistent faulters are malicious
    from collections import defaultdict
    fault_count: Dict[str, int] = defaultdict(int)
    for p in pkts:
        if p.get("apid") == "0x7FE":
            fault_count[p["source_sat_id"]] += 1
    fault_sats = {sat for sat, cnt in fault_count.items() if cnt >= fault_threshold}

    raw: Dict[int, Dict[str, float]] = defaultdict(dict)
    flt: Dict[int, Dict[str, bool]]  = defaultdict(dict)
    for p in pkts:
        ep  = p["index"] // len(sat_ids)
        sat = p["source_sat_id"]
        raw[ep][sat] = float(p.get("signal_strength_dBm") or -130.0)
        flt[ep][sat] = sat in fault_sats

    LO, HI = -130.0, -90.0
    sensing_epochs, fault_labels = [], []
    for ep in sorted(raw.keys()):
        sensing_epochs.append([
            float(np.clip((raw[ep].get(s, LO) - LO) / (HI - LO), 0.0, 1.0))
            for s in sat_ids
        ])
        fault_labels.append([flt[ep].get(s, False) for s in sat_ids])

    return sensing_epochs, fault_labels, sat_ids


# Shared output helpers

def _print_config(mode, num_providers, num_satellites, num_epochs,
                  decay, iqr_threshold, extra_lines=()):
    print(f"\n  Mode             : {mode}")
    print(f"  Providers        : {num_providers}")
    print(f"  Satellites (MPC) : {num_satellites}")
    print(f"  Epochs           : {num_epochs}")
    print(f"  Decay            : {decay}")
    print(f"  IQR threshold    : {iqr_threshold}")
    for line in extra_lines:
        print(f"  {line}")
    print()


def _run_engines(sensing_epochs, num_satellites, num_providers, decay, iqr_threshold,
                 provider_labels=None):
    # Feed all epochs through MPC engine, reference engine and suspicion detector
    mpc_td  = StreamingTruthDiscovery(num_satellites, num_providers, decay)
    ref_td  = SimplifiedTruthDiscovery(num_providers, decay)
    sus_eng = SuspicionEngine(num_providers, decay, iqr_threshold=iqr_threshold)

    labels = provider_labels or [f"p{k:02d}" for k in range(num_providers)]
    w_hdr  = " ".join(f"{lb[:4]:>5}" for lb in labels)
    hdr    = f"{'Ep':>3}  {'MPC truth':>9}  {'Ref truth':>9}  {'ms':>5}  {w_hdr}"
    print(hdr)
    print("-" * len(hdr))

    for ep_idx, values in enumerate(sensing_epochs):
        result       = mpc_td.run_epoch(values)
        ref_truth, _ = ref_td.run_epoch(np.array(values))
        sus_eng.update(result)
        w_str = " ".join(f"{result.weights[k]:5.2f}" for k in range(num_providers))
        print(f"{ep_idx+1:>3}  {result.truth_value:>9.4f}  {ref_truth:>9.4f}  "
              f"{result.processing_time_ms:>5.1f}  {w_str}")

    return mpc_td, ref_td, sus_eng


def _print_results(mpc_td, ref_td, sus_eng, num_providers,
                   true_malicious: Set[int], provider_labels=None):
    # Print suspicion scores, verdict table, error chart, and detection metrics
    metrics  = sus_eng.get_metrics()
    suspects = sus_eng.get_suspects()
    labels   = provider_labels or [f"p{k:02d}" for k in range(num_providers)]

    print(f"\n  Suspicion scores\n")
    print(f"  {'Prov':<10} {'ST_k':>10} {'MajDev':>9} {'Var':>8} "
          f"{'Score':>7} {'Verdict':<10} {'Conf%':>6}")
    print("-" * 65)
    for k in sorted(metrics, key=lambda x: metrics[x].composite_score, reverse=True):
        m   = metrics[k]
        flg = "SUSPECT" if m.is_flagged else "trusted"
        print(f"  {labels[k]:<10} {m.accum_sq_error:>10.5f} {m.accum_majority_dev:>9.5f} "
              f"{m.report_variance:>8.5f} {m.composite_score:>7.4f} {flg:<10} {m.confidence*100:>5.1f}%")

    honest_ids = sorted(set(range(num_providers)) - true_malicious)
    print(f"\n  Ground truth revealed")
    print(f"  Honest     ({len(honest_ids)}) : {honest_ids} -> {[labels[k] for k in honest_ids]}")
    print(f"  Malicious  ({len(true_malicious)}) : {sorted(true_malicious)} -> "
          f"{[labels[k] for k in sorted(true_malicious)]}")
    print(f"  Flagged    : {sorted(suspects)}")

    tp  = suspects & true_malicious
    fp  = suspects - true_malicious
    fn  = true_malicious - suspects
    pre = len(tp) / len(suspects)       if suspects       else 0.0
    rec = len(tp) / len(true_malicious) if true_malicious else 0.0
    f1  = 2*pre*rec / (pre+rec)         if (pre+rec) > 0  else 0.0
    print(f"  TP {sorted(tp)}  FP {sorted(fp)}  FN {sorted(fn)}")
    print(f"  Precision {pre:.2f}   Recall {rec:.2f}   F1 {f1:.2f}")

    print(f"\n  Verdict\n")
    print(f"  {'Prov':<10} {'True role':<12} {'Verdict':<10} {'ST_k':>9} {'Score':>7}  match")
    print("-" * 60)
    for k in sorted(metrics, key=lambda x: metrics[x].composite_score, reverse=True):
        m       = metrics[k]
        role    = "MALICIOUS" if k in true_malicious else "honest"
        verdict = "SUSPECT"   if m.is_flagged        else "trusted"
        match   = "✓" if (k in true_malicious) == m.is_flagged else "✗"
        print(f"  {labels[k]:<10} {role:<12} {verdict:<10} {m.accum_sq_error:>9.5f} "
              f"{m.composite_score:>7.4f}  {match}")

    print(f"\n  Accumulated error chart\n")
    max_st = max(m.accum_sq_error for m in metrics.values()) or 1.0
    for k in range(num_providers):
        m    = metrics[k]
        role = "MALICIOUS" if k in true_malicious else "honest"
        bar  = "█" * max(1, int(34 * m.accum_sq_error / max_st))
        tag  = "  <- SUSPECT" if m.is_flagged else ""
        print(f"  {labels[k]:<10} ({role:<9}) {m.accum_sq_error:8.5f}  {bar}{tag}")
    print()


# Real-data simulation

def run_real_data(num_satellites: int = 3, decay: float = 0.9, iqr_threshold: float = 1.0):
    # Runs on 05_isl_stream_log.json — 5 ISL satellites, 40 epochs of signal_strength_dBm
    sensing_epochs, fault_labels, sat_ids = _load_real_epochs()

    num_providers  = len(sat_ids)
    num_epochs     = len(sensing_epochs)
    true_malicious: Set[int] = {
        k for k in range(num_providers)
        if any(fault_labels[ep][k] for ep in range(num_epochs))
    }
    honest_count   = num_providers - len(true_malicious)

    _print_config(
        "REAL CCSDS DATA  (05_isl_stream_log.json)",
        num_providers, num_satellites, num_epochs, decay, iqr_threshold,
        extra_lines=[
            f"Signal           : signal_strength_dBm normalised [0, 1]",
            f"Providers        : {sat_ids}",
            f"Honest / Malicious : {honest_count} / {len(true_malicious)}  "
            f"(malicious = persistent fault-alert sats, hidden from engine)",
        ]
    )

    labels = [s.replace("SAT-", "S") for s in sat_ids]
    mpc_td, ref_td, sus_eng = _run_engines(
        sensing_epochs, num_satellites, num_providers, decay, iqr_threshold,
        provider_labels=labels
    )
    _print_results(mpc_td, ref_td, sus_eng, num_providers, true_malicious,
                   provider_labels=labels)


# Random-data simulation

def run_random(
    num_providers:  int   = 8,
    num_satellites: int   = 3,
    num_epochs:     int   = 15,
    num_malicious:  int   = 2,
    malicious_bias: float = 0.5,
    honest_noise:   float = 0.05,
    true_value:     float = 0.5,
    decay:          float = 0.9,
    iqr_threshold:  float = 1.0,
    seed:           Optional[int] = None,
):
    if num_malicious >= num_providers:
        raise ValueError("num_malicious must be < num_providers")

    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big")
        seed_label = f"random  (this run: {seed})"
    else:
        seed_label = str(seed)
    rng = np.random.default_rng(seed)

    malicious_ids: Set[int] = set(
        int(x) for x in rng.choice(list(range(num_providers)), size=num_malicious, replace=False)
    )
    honest_ids = [k for k in range(num_providers) if k not in malicious_ids]
    configs    = assign_malicious_configs(
        num_providers, num_malicious, malicious_ids, malicious_bias, honest_noise, rng
    )

    _print_config(
        "RANDOM (synthetic)",
        num_providers, num_satellites, num_epochs, decay, iqr_threshold,
        extra_lines=[
            f"Honest / Malicious : {len(honest_ids)} / {num_malicious}  (randomly assigned)",
            f"Seed             : {seed_label}",
            f"True value       : {true_value}",
            f"Base bias        : {malicious_bias}",
            "Ground-truth labels hidden from all engines.",
        ]
    )

    sensing_epochs = [
        [configs[k].sense(true_value, ep, rng) for k in range(num_providers)]
        for ep in range(1, num_epochs + 1)
    ]

    labels = [f"p{k:02d}" for k in range(num_providers)]
    mpc_td, ref_td, sus_eng = _run_engines(
        sensing_epochs, num_satellites, num_providers, decay, iqr_threshold,
        provider_labels=labels
    )
    _print_results(mpc_td, ref_td, sus_eng, num_providers, malicious_ids,
                   provider_labels=labels)


# CLI

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Privacy-preserving streaming truth discovery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--random",         action="store_true",
                   help="Use synthetic random data instead of real CCSDS telemetry")
    p.add_argument("--num-providers",  type=int,   default=8)
    p.add_argument("--num-satellites", type=int,   default=3)
    p.add_argument("--num-epochs",     type=int,   default=15)
    p.add_argument("--num-malicious",  type=int,   default=2)
    p.add_argument("--malicious-bias", type=float, default=0.5)
    p.add_argument("--honest-noise",   type=float, default=0.05)
    p.add_argument("--true-value",     type=float, default=0.5)
    p.add_argument("--decay",          type=float, default=0.9)
    p.add_argument("--iqr-threshold",  type=float, default=1.0)
    p.add_argument("--seed",           type=int,   default=None)
    return p


if __name__ == "__main__":
    args = _parser().parse_args()
    if args.random:
        run_random(
            num_providers  = args.num_providers,
            num_satellites = args.num_satellites,
            num_epochs     = args.num_epochs,
            num_malicious  = args.num_malicious,
            malicious_bias = args.malicious_bias,
            honest_noise   = args.honest_noise,
            true_value     = args.true_value,
            decay          = args.decay,
            iqr_threshold  = args.iqr_threshold,
            seed           = args.seed,
        )
    else:
        run_real_data(
            num_satellites = args.num_satellites,
            decay          = args.decay,
            iqr_threshold  = args.iqr_threshold,
        )

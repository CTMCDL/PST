"""
PST / GS-ST coherent GPU + CPU sampler-decomposition timing worker
==================================================================

This worker is launched by the parent notebook in a fresh Python process for
each device/profile so TensorFlow device visibility and CPU thread pools are
fixed before runtime initialization.

Timed methods:
* PST-cat: original TensorFlow categorical hard draw + PST backward.
* PST-SSA: direct inverse-CDF SSA hard draw from raw propensities + analytic
  affine PST VJP; no reaction-selection logits or softmax.
* PST-SSA-XLA: exactly the same PST-SSA estimator, but ONLY the detached hard
  inverse-CDF sampler is XLA-compiled. The outer simulator, loss, GradientTape,
  optimizer update, and PST custom VJP remain non-XLA.
* PST-GM: explicit Gumbel-Max hard draw + PST backward.
* GSST-GM: the same explicit Gumbel-Max hard draw + Gumbel-Softmax backward.
* SSA-fwd: exact forward simulation + loss only; no backward or optimizer.

The primary estimator-cost control is PST-GM/GSST-GM because the hard Gumbel-Max
forward construction is matched. PST-SSA and PST-SSA-XLA measure practical
direct-propensity implementations. XLA is isolated as an implementation
optimization rather than changing the estimator.

All timed training rows include forward simulation, readout/loss, reverse-mode
backpropagation, gradient clipping/checks, optimizer update, and the clock
parameter projection. Tracing/warm-up, parameter/optimizer resets, clock batch
selection, and reference generation are outside timed intervals.
"""

from __future__ import annotations

import os

# -----------------------------------------------------------------------------
# DEVICE / PROCESS SETTINGS -- supplied by the parent notebook before TF import.
# -----------------------------------------------------------------------------
BENCHMARK_PROFILE = os.environ.get("PST_BENCH_LABEL", "gpu_primary")
BENCHMARK_DEVICE = os.environ.get("PST_BENCH_DEVICE", "gpu").strip().lower()
GPU_ID = os.environ.get("PST_BENCH_GPU_ID", "0").strip()
CPU_THREADS = os.environ.get("PST_BENCH_CPU_THREADS", "auto").strip().lower()

if BENCHMARK_DEVICE not in {"gpu", "cpu"}:
    raise ValueError("PST_BENCH_DEVICE must be 'gpu' or 'cpu'")

if BENCHMARK_DEVICE == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if CPU_THREADS != "auto":
        n_threads = int(CPU_THREADS)
        if n_threads < 1:
            raise ValueError("CPU thread count must be positive")
        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        os.environ["TF_NUM_INTRAOP_THREADS"] = str(n_threads)
        os.environ["TF_NUM_INTEROP_THREADS"] = "1"
else:
    # Empty/'auto' leaves CUDA visibility unchanged. Otherwise expose one GPU.
    if GPU_ID and GPU_ID.lower() != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import time
import json
import math
import random
import platform
import statistics
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import tensorflow as tf

# Explicit CPU thread control must occur before any TensorFlow computation.
if BENCHMARK_DEVICE == "cpu" and CPU_THREADS != "auto":
    _n_threads = int(CPU_THREADS)
    tf.config.threading.set_intra_op_parallelism_threads(_n_threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)

_physical_gpus = tf.config.list_physical_devices("GPU")
if BENCHMARK_DEVICE == "gpu" and not _physical_gpus:
    raise RuntimeError(
        "GPU benchmark requested, but TensorFlow sees no physical GPU. "
        "Check GPU_ID/CUDA_VISIBLE_DEVICES before interpreting this run."
    )
if BENCHMARK_DEVICE == "cpu" and _physical_gpus:
    raise RuntimeError(
        "CPU benchmark requested, but a GPU is still visible to TensorFlow."
    )


# =============================================================================
# USER SETTINGS -- supplied by parent notebook via environment variables
# =============================================================================

RUN_CLOCK = os.environ.get("PST_RUN_CLOCK", "1") == "1"
RUN_ION = os.environ.get("PST_RUN_ION", "1") == "1"

ION_DATA_FILE = Path(os.environ.get(
    "PST_ION_DATA_FILE",
    "HEK293_Cell01_40mV_100s_raw_DET_IDEl.xlsx",
))

ION_NUM_STEPS = int(os.environ.get("PST_ION_NUM_STEPS", "20"))
ION_EPOCHS = 400
CLOCK_SEGMENT_STEPS = int(os.environ.get("PST_CLOCK_SEGMENT_STEPS", "150"))
CLOCK_M_TRAIN = int(os.environ.get("PST_CLOCK_M_TRAIN", "25"))

ION_M_SIMS = int(os.environ["PST_ION_M_SIMS"])
CLOCK_NUM_START_POINTS = int(os.environ["PST_CLOCK_NUM_START_POINTS"])
CLOCK_REFERENCE_STEPS = int(os.environ["PST_CLOCK_REFERENCE_STEPS"])
CLOCK_BURN_IN = int(os.environ["PST_CLOCK_BURN_IN"])
CLOCK_BATCH_SIZE = int(os.environ["PST_CLOCK_BATCH_SIZE"])
WARMUP = int(os.environ["PST_BENCH_WARMUP"])
REPEATS = int(os.environ["PST_BENCH_REPEATS"])
SSA_DIAGNOSTIC_DRAWS = int(os.environ["PST_SSA_DIAGNOSTIC_DRAWS"])
CLOCK_REFERENCE_FILE = Path(os.environ["PST_CLOCK_REFERENCE_FILE"])

# Reference generation is outside timing. A common ion-readout XLA setting may
# be enabled for the GPU profile, but it is identical across all timed methods.
# PST-SSA-XLA remains the only METHOD-SPECIFIC XLA difference.
REFERENCE_JIT_COMPILE = os.environ.get("PST_REFERENCE_JIT", "0") == "1"
ION_READOUT_JIT = os.environ.get("PST_ION_READOUT_JIT", "0") == "1"

BASE_SEED = 43
SHOW_PROGRESS = os.environ.get("PST_SHOW_PROGRESS", "1") == "1"
ION_TIMING_EPOCH = 200

PST_CAT_NAME = "PST-cat"
PST_GM_NAME = "PST-GM"
GSST_GM_NAME = "GSST-GM"
PST_DIRECT_NAME = "PST-SSA"
PST_XLA_NAME = "PST-SSA-XLA"
SSA_METHOD_NAME = "SSA-fwd"

INCLUDE_PST_DIRECT = True
INCLUDE_PST_XLA = True
INCLUDE_SSA_BASELINE = True

OUTPUT_CSV = Path(f"pst_gsst_timing_{BENCHMARK_PROFILE}.csv")
OUTPUT_JSON = Path(f"pst_gsst_timing_{BENCHMARK_PROFILE}.json")


# =============================================================================
# ENVIRONMENT / HELPERS
# =============================================================================

def configure_devices():
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except (RuntimeError, ValueError):
            pass

configure_devices()


def environment_metadata():
    build = {}
    try:
        build = tf.sysconfig.get_build_info()
    except Exception:
        pass
    return {
        "benchmark_profile": BENCHMARK_PROFILE,
        "benchmark_device": BENCHMARK_DEVICE,
        "gpu_id_requested": GPU_ID,
        "cpu_threads_requested": CPU_THREADS,
        "tf_intra_op_threads": tf.config.threading.get_intra_op_parallelism_threads(),
        "tf_inter_op_threads": tf.config.threading.get_inter_op_parallelism_threads(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "os_cpu_count": os.cpu_count(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "physical_cpus": [x.name for x in tf.config.list_physical_devices("CPU")],
        "physical_gpus": [x.name for x in tf.config.list_physical_devices("GPU")],
        "logical_gpus": [x.name for x in tf.config.list_logical_devices("GPU")],
        "cuda_version": build.get("cuda_version"),
        "cudnn_version": build.get("cudnn_version"),
        "is_cuda_build": build.get("is_cuda_build"),
    }


def print_environment():
    print("\nEnvironment")
    print("-----------")
    for k, v in environment_metadata().items():
        print(f"{k:>16}: {v}")


def sync_tensor(x):
    """Materialize returned values and therefore synchronize device work.

    TensorFlow tensors/variables are copied to host with .numpy(). Some
    Python-side training steps (notably the ion-channel step) already return a
    NumPy array after performing their own .numpy() synchronization; those are
    simply passed through.
    """
    if isinstance(x, (tuple, list)):
        return [sync_tensor(v) for v in x]
    if isinstance(x, dict):
        return {k: sync_tensor(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, np.generic):
        return np.asarray(x)
    if tf.is_tensor(x) or isinstance(x, tf.Variable):
        return np.asarray(x.numpy())
    return np.asarray(x)


def snapshot_variables(variables):
    return [np.array(v.numpy(), copy=True) for v in variables]


def restore_variables(variables, values):
    for v, value in zip(variables, values):
        v.assign(value)


def optimizer_snapshot(optimizer):
    return snapshot_variables(list(optimizer.variables))


def optimizer_restore(optimizer, values):
    restore_variables(list(optimizer.variables), values)


@dataclass
class TimingResult:
    model: str
    method: str
    trajectories: int
    ssa_steps: int
    trajectory_events: int
    repeats: int
    median_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    ns_per_trajectory_event: float
    million_trajectory_events_per_s: float
    samples_ms: list[float]


@dataclass
class PairedStats:
    model: str
    numerator: str
    denominator: str
    repeats: int
    median_ratio: float
    ratio_ci_low: float
    ratio_ci_high: float
    mean_diff_ms: float
    sd_diff_ms: float
    t_statistic: float
    degrees_of_freedom: int
    numerator_slower_count: int


def summarize(model, method, trajectories, ssa_steps, samples):
    samples = [float(x) for x in samples]
    med = statistics.median(samples)
    events = trajectories * ssa_steps
    return TimingResult(
        model=model,
        method=method,
        trajectories=trajectories,
        ssa_steps=ssa_steps,
        trajectory_events=events,
        repeats=len(samples),
        median_ms=med,
        mean_ms=statistics.mean(samples),
        std_ms=statistics.stdev(samples) if len(samples) > 1 else 0.0,
        min_ms=min(samples),
        max_ms=max(samples),
        ns_per_trajectory_event=(med * 1e6) / events,
        million_trajectory_events_per_s=events / (med * 1e3),
        samples_ms=samples,
    )


def _format_seconds(seconds):
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:04.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes):02d}m {sec:04.1f}s"


def time_paired(methods, resetters, callers, warmup, repeats, base_seed):
    """
    methods: ordered method labels to time within each repeat
    resetters[name](): restore identical parameters and optimizer state
    callers[name](seed_tensor): execute one full stochastic gradient update

    Progress output is outside the timed interval and reports completion of
    each warm-up and measured call.
    """
    samples = {m: [] for m in methods}
    total_calls = warmup * len(methods) + repeats * len(methods)
    completed_calls = 0
    overall_start = time.perf_counter()

    print(f"  warm-up={warmup}, timed repeats={repeats}", flush=True)
    print(f"  total expensive calls={total_calls}", flush=True)

    # -------------------------
    # Warm-up / graph tracing
    # -------------------------
    for w in range(warmup):
        for m in methods:
            resetters[m]()
            seed = tf.constant([base_seed, 10_000 + w], tf.int32)

            if SHOW_PROGRESS:
                print(
                    f"  [warm-up {w + 1:02d}/{warmup:02d}] "
                    f"{m:<7} ... ",
                    end="",
                    flush=True,
                )

            call_start = time.perf_counter()
            sync_tensor(callers[m](seed))
            call_elapsed = time.perf_counter() - call_start

            completed_calls += 1
            if SHOW_PROGRESS:
                elapsed = time.perf_counter() - overall_start
                avg_call = elapsed / completed_calls
                remaining = (total_calls - completed_calls) * avg_call
                print(
                    f"done in {_format_seconds(call_elapsed)} "
                    f"| overall {completed_calls}/{total_calls} "
                    f"| rough ETA {_format_seconds(remaining)}",
                    flush=True,
                )

    if SHOW_PROGRESS:
        print("  Warm-up complete. Starting timed repetitions.", flush=True)

    # -------------------------
    # Timed paired repetitions
    # -------------------------
    timed_start = time.perf_counter()

    for r in range(repeats):
        # Balanced cyclic order. Over every len(methods) repeats, each
        # method occupies every ordinal position exactly once. Alternate the
        # direction in successive blocks to reduce carry-over/thermal bias.
        n_methods = len(methods)
        shift = r % n_methods
        order = list(methods[shift:]) + list(methods[:shift])
        if (r // n_methods) % 2 == 1:
            order = list(reversed(order))
        seed = tf.constant([base_seed, 100_000 + r], tf.int32)
        pair_values = {}

        for m in order:
            resetters[m]()

            if SHOW_PROGRESS:
                print(
                    f"  [repeat {r + 1:02d}/{repeats:02d}] "
                    f"{m:<7} ... ",
                    end="",
                    flush=True,
                )

            # Only this interval contributes to the reported timing.
            t0 = time.perf_counter_ns()
            out = callers[m](seed)
            sync_tensor(out)
            t1 = time.perf_counter_ns()

            ms = (t1 - t0) / 1e6
            samples[m].append(ms)
            pair_values[m] = ms
            completed_calls += 1

            if SHOW_PROGRESS:
                print(f"{ms:,.1f} ms", flush=True)

        if SHOW_PROGRESS:
            timed_elapsed = time.perf_counter() - timed_start
            repeats_done = r + 1
            mean_repeat_time = timed_elapsed / repeats_done
            eta = (repeats - repeats_done) * mean_repeat_time

            pair_text = " | ".join(
                f"{m}={pair_values[m]:,.1f} ms"
                for m in methods
                if m in pair_values
            )
            median_text = ", ".join(
                f"{m}={statistics.median(samples[m]):,.1f} ms"
                for m in methods
                if samples[m]
            )
            print(
                f"    completed set {r + 1:02d}/{repeats:02d}: {pair_text}\n"
                f"    running medians: {median_text}"
                f" | ETA {_format_seconds(eta)}",
                flush=True,
            )

    if SHOW_PROGRESS:
        print(
            f"  Timing complete in "
            f"{_format_seconds(time.perf_counter() - timed_start)}.",
            flush=True,
        )

    return samples


# =============================================================================
# CLOCK / VILAR OSCILLATOR — COPIED FROM THE TRAINING PROTOCOL
# =============================================================================

CLOCK_K_TRUE = np.array([
    1.0, 50.0, 1.0, 100.0,
    50.0, 500.0, 0.01, 50.0,
    50.0, 5.0, 10.0, 0.5,
    1.0, 0.2, 2.0, 1.0,
], dtype=np.float32)

CLOCK_S = tf.constant([
    [-1., +1.,  0.,  0.,  0.,  0., -1.,  0.,  0.],
    [+1., -1.,  0.,  0.,  0.,  0., +1.,  0.,  0.],
    [ 0.,  0., -1., +1.,  0.,  0., -1.,  0.,  0.],
    [ 0.,  0., +1., -1.,  0.,  0., +1.,  0.,  0.],
    [ 0.,  0.,  0.,  0., +1.,  0.,  0.,  0.,  0.],
    [ 0.,  0.,  0.,  0., +1.,  0.,  0.,  0.,  0.],
    [ 0.,  0.,  0.,  0.,  0., +1.,  0.,  0.,  0.],
    [ 0.,  0.,  0.,  0.,  0., +1.,  0.,  0.,  0.],
    [ 0.,  0.,  0.,  0.,  0.,  0., +1.,  0.,  0.],
    [ 0.,  0.,  0.,  0.,  0.,  0.,  0., +1.,  0.],
    [ 0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.],
    [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.],
    [ 0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.],
    [ 0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.],
    [ 0.,  0.,  0.,  0.,  0.,  0., -1., -1., +1.],
    [ 0.,  0.,  0.,  0.,  0.,  0.,  0., +1., -1.],
], tf.float32)


@tf.function(jit_compile=False, reduce_retracing=True)
def clock_propensity(states, k):
    X = tf.maximum(states, 1e-10)
    DA, DAp, DR, DRp = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    MA, MR, A, R, C = X[:, 4], X[:, 5], X[:, 6], X[:, 7], X[:, 8]
    return tf.stack([
        k[0] * DA * A,   k[1] * DAp,
        k[2] * DR * A,   k[3] * DRp,
        k[4] * DA,       k[5] * DAp,
        k[6] * DR,       k[7] * DRp,
        k[8] * MA,       k[9] * MR,
        k[10] * MA,      k[11] * MR,
        k[12] * A,       k[13] * R,
        k[14] * A * R,   k[15] * C,
    ], axis=1)


def _clock_gumbel_max(logits, seed):
    """Exact Gumbel-Max hard categorical draw shared by PST-GM/GSST-GM."""
    u = tf.random.stateless_uniform(
        tf.shape(logits),
        seed=seed,
        minval=1e-20,
        maxval=1.0,
        dtype=logits.dtype,
    )
    g = -tf.math.log(-tf.math.log(u))
    hard_idx = tf.argmax(logits + g, axis=-1)
    y_hard = tf.one_hot(
        hard_idx,
        depth=tf.shape(logits)[-1],
        dtype=logits.dtype,
    )
    return y_hard, g


def clock_pst_categorical_estimator(logits, seed):
    """Original oscillator PST implementation.

    Forward: exact hard draw via tf.random.stateless_categorical.
    Backward: derivative of pi = softmax(logits).

    The flatten/reshape is intentionally retained because it is part of the
    TensorFlow categorical API path being measured.
    """
    logits_shape = tf.shape(logits)
    num_reactions = logits_shape[-1]
    flat_logits = tf.reshape(logits, [-1, num_reactions])
    hard_idx_flat = tf.random.stateless_categorical(
        flat_logits,
        num_samples=1,
        seed=seed,
    )
    hard_idx = tf.reshape(hard_idx_flat, logits_shape[:-1])
    y_hard = tf.one_hot(
        hard_idx,
        depth=num_reactions,
        dtype=logits.dtype,
    )
    probability = tf.nn.softmax(logits, axis=-1)
    return probability + tf.stop_gradient(y_hard - probability)


def clock_pst_gumbel_estimator(logits, seed):
    """Exact Gumbel-Max hard forward; PST backward.

    Gumbel noise is used only for the hard categorical realization. The
    backward derivative is d pi, with no temperature or Gumbel-dependent
    relaxation.
    """
    y_hard, _ = _clock_gumbel_max(logits, seed)
    probability = tf.nn.softmax(logits, axis=-1)
    return probability + tf.stop_gradient(y_hard - probability)


def clock_gsst_gumbel_estimator(logits, tau, seed):
    """Exact Gumbel-Max hard forward; Gumbel-Softmax ST backward."""
    y_hard, g = _clock_gumbel_max(logits, seed)
    y_soft = tf.nn.softmax((logits + g) / tau, axis=-1)
    return y_soft + tf.stop_gradient(y_hard - y_soft)


def make_clock_simulator(method):
    """One oscillator loop, shared by both estimators.

    Only the reaction-selection call differs.  Propensities, absorbing-state
    masking, waiting-time sampling, the state update and the tensor-array
    bookkeeping are identical between the two branches, so a timing
    difference between the traced graphs is attributable to the estimator and
    not to how the surrounding loop happens to be written.

    The absorbing-state guard is retained for BOTH methods.  It is needed
    whenever a row can reach a0 = 0: a softmax over an all -inf logit row is
    NaN and the sampled index is arbitrary, so the fallback keeps one entry
    positive and the trailing mask zeroes the resulting weights.  The same guard is applied to both estimators.
    """
    assert method in {PST_CAT_NAME, PST_GM_NAME, PST_DIRECT_NAME, PST_XLA_NAME, GSST_GM_NAME}

    @tf.function(jit_compile=False, reduce_retracing=True)
    def simulate(batch_initial_states, num_steps, k, temperature, S_matrix, seed):
        num_simulations = tf.shape(batch_initial_states)[0]
        time_t = tf.zeros([num_simulations, 1], tf.float32)
        states_ta = tf.TensorArray(tf.float32, size=num_steps + 1)
        times_ta = tf.TensorArray(tf.float32, size=num_steps + 1)
        states_ta = states_ta.write(0, batch_initial_states)
        times_ta = times_ta.write(0, time_t)

        t = tf.constant(0)
        states = batch_initial_states

        def cond(t, states, time_t, states_ta, times_ta, seed):
            return t < num_steps

        def body(t, states, time_t, states_ta, times_ta, seed):
            seeds = tf.random.experimental.stateless_split(seed, 3)
            next_seed, seed_time, seed_reaction = seeds[0], seeds[1], seeds[2]

            prop = clock_propensity(states, k)
            a0 = tf.reduce_sum(prop, axis=1, keepdims=True)
            active = a0 > 0.0
            safe_a0 = tf.where(active, a0, tf.ones_like(a0))

            u_time = tf.random.stateless_uniform(
                [num_simulations, 1],
                seed=seed_time,
                minval=1e-20,
                maxval=1.0,
                dtype=prop.dtype,
            )
            dt = tf.where(
                active,
                -tf.math.log(u_time) / safe_a0,
                tf.zeros_like(u_time),
            )

            fallback = tf.one_hot(
                tf.zeros([num_simulations], dtype=tf.int32),
                depth=tf.shape(prop)[-1],
                dtype=prop.dtype,
            )
            sampling_prop = tf.where(active, prop, fallback)

            if method in {PST_DIRECT_NAME, PST_XLA_NAME}:
                # Both PST-SSA variants use the same exact direct-method
                # forward and the same analytic PST affine VJP.
                #
                # PST-SSA-XLA differs ONLY by routing the detached
                # CDF/compare/reduce/gather hard sampler through an
                # XLA-compiled subfunction.
                delta = clock_pst_delta_ssa(
                    sampling_prop,
                    safe_a0,
                    S_matrix,
                    seed_reaction,
                    use_xla_sampler=(method == PST_XLA_NAME),
                )
                delta = tf.where(active, delta, tf.zeros_like(delta))
            else:
                # Logits are needed only by the categorical/Gumbel variants.
                positive = sampling_prop > 0.0
                finite_log = tf.math.log(
                    tf.where(
                        positive,
                        sampling_prop,
                        tf.ones_like(sampling_prop),
                    )
                )
                logits = tf.where(
                    positive,
                    finite_log,
                    tf.constant(float("-inf"), dtype=prop.dtype),
                )

                if method == PST_CAT_NAME:
                    weights = clock_pst_categorical_estimator(
                        logits, seed_reaction
                    )
                elif method == PST_GM_NAME:
                    weights = clock_pst_gumbel_estimator(
                        logits, seed_reaction
                    )
                else:
                    weights = clock_gsst_gumbel_estimator(
                        logits, temperature, seed_reaction
                    )

                weights = tf.where(active, weights, tf.zeros_like(weights))
                delta = tf.matmul(weights, S_matrix)

            new_states = tf.maximum(states + delta, 0.0)
            new_time = time_t + dt

            states_ta = states_ta.write(t + 1, new_states)
            times_ta = times_ta.write(t + 1, new_time)
            return t + 1, new_states, new_time, states_ta, times_ta, next_seed

        _, _, _, states_ta, times_ta, _ = tf.while_loop(
            cond, body,
            [t, states, time_t, states_ta, times_ta, seed],
            shape_invariants=[
                t.shape,
                tf.TensorShape([None, 9]),
                tf.TensorShape([None, 1]),
                tf.TensorShape([]),
                tf.TensorShape([]),
                tf.TensorShape([2]),
            ],
        )
        states = tf.transpose(states_ta.stack(), [1, 0, 2])
        times = tf.transpose(times_ta.stack(), [1, 0, 2])
        return states, times

    return simulate



def verify_clock_matched_gumbel_forward():
    """Check PST-GM and GSST-GM have the same hard-forward trajectory."""
    print("\nMatched Gumbel-Max forward check")
    print("--------------------------------")

    sim_pst = make_clock_simulator(PST_GM_NAME)
    sim_gs = make_clock_simulator(GSST_GM_NAME)

    x0 = tf.repeat(
        tf.constant([[1., 0., 1., 0., 0., 0., 0., 0., 0.]], tf.float32),
        repeats=32,
        axis=0,
    )
    k = tf.constant(CLOCK_K_TRUE, tf.float32)
    seed = tf.constant([123, 456], tf.int32)
    steps = tf.constant(20, tf.int32)
    tau = tf.constant(3e-6, tf.float32)

    s_pst, t_pst = sim_pst(x0, steps, k, tau, CLOCK_S, seed)
    s_gs, t_gs = sim_gs(x0, steps, k, tau, CLOCK_S, seed)

    state_err = float(tf.reduce_max(tf.abs(s_pst - s_gs)).numpy())
    time_err = float(tf.reduce_max(tf.abs(t_pst - t_gs)).numpy())

    print(f"  max |state PST-GM - GSST-GM| = {state_err:.3e}")
    print(f"  max |time  PST-GM - GSST-GM| = {time_err:.3e}")

    if state_err > 1e-5 or time_err > 1e-5:
        raise AssertionError(
            "Matched Gumbel-Max methods did not produce the same forward "
            f"trajectory: state_err={state_err}, time_err={time_err}"
        )
    print("  matched hard-forward construction confirmed.")


def clock_reference_core(initial_states, num_steps, k, seed):
    """
    Fast exact-forward reference trajectory core.

    This deliberately follows the ORIGINAL oscillator notebook design:
      * reference/target generation: XLA JIT compiled (no gradients needed)
      * differentiable PST/GS-ST training: jit_compile=False

    Although the PST straight-through tensor is constructed, its numerical
    forward value is exactly the hard categorical one-hot, so the generated
    reference path is an exact SSA path.
    """
    num_simulations = tf.shape(initial_states)[0]
    time_t = tf.zeros([num_simulations, 1], tf.float32)

    states_ta = tf.TensorArray(tf.float32, size=num_steps + 1)
    times_ta = tf.TensorArray(tf.float32, size=num_steps + 1)
    states_ta = states_ta.write(0, initial_states)
    times_ta = times_ta.write(0, time_t)

    t = tf.constant(0)
    states = initial_states

    def cond(t, states, time_t, states_ta, times_ta, seed):
        return t < num_steps

    def body(t, states, time_t, states_ta, times_ta, seed):
        seeds = tf.random.experimental.stateless_split(seed, 3)
        next_seed = seeds[0]
        seed_time = seeds[1]
        seed_reaction = seeds[2]

        prop = clock_propensity(states, k)
        a0 = tf.reduce_sum(prop, axis=1, keepdims=True)
        active = a0 > 0.0
        safe_a0 = tf.where(active, a0, tf.ones_like(a0))

        u_time = tf.random.stateless_uniform(
            [num_simulations, 1],
            seed=seed_time,
            minval=1e-20,
            maxval=1.0,
            dtype=prop.dtype,
        )
        dt = tf.where(
            active,
            -tf.math.log(u_time) / safe_a0,
            tf.zeros_like(u_time),
        )

        # Same exact-forward reaction-selection construction as the original
        # PST oscillator notebook.
        fallback = tf.one_hot(
            tf.zeros([num_simulations], dtype=tf.int32),
            depth=tf.shape(prop)[-1],
            dtype=prop.dtype,
        )
        sampling_prop = tf.where(active, prop, fallback)
        positive = sampling_prop > 0.0
        finite_log = tf.math.log(
            tf.where(positive, sampling_prop, tf.ones_like(sampling_prop))
        )
        logits = tf.where(
            positive,
            finite_log,
            tf.constant(float("-inf"), dtype=prop.dtype),
        )

        weights = clock_pst_categorical_estimator(logits, seed_reaction)
        weights = tf.where(active, weights, tf.zeros_like(weights))

        delta = tf.matmul(weights, CLOCK_S)
        new_states = tf.maximum(states + delta, 0.0)
        new_time = time_t + dt

        states_ta = states_ta.write(t + 1, new_states)
        times_ta = times_ta.write(t + 1, new_time)
        return (
            t + 1,
            new_states,
            new_time,
            states_ta,
            times_ta,
            next_seed,
        )

    _, _, _, states_ta, times_ta, _ = tf.while_loop(
        cond,
        body,
        [t, states, time_t, states_ta, times_ta, seed],
        shape_invariants=[
            t.shape,
            tf.TensorShape([None, 9]),
            tf.TensorShape([None, 1]),
            tf.TensorShape([]),
            tf.TensorShape([]),
            tf.TensorShape([2]),
        ],
    )

    return (
        tf.transpose(states_ta.stack(), [1, 0, 2]),
        tf.transpose(times_ta.stack(), [1, 0, 2]),
    )


# Reference generation is outside timed intervals.  The GPU-full profile uses
# the original XLA path; the CPU profile disables this unrelated JIT for
# portability and lower compilation overhead.
clock_reference_fast = tf.function(
    clock_reference_core,
    jit_compile=REFERENCE_JIT_COMPILE,
    reduce_retracing=True,
)


def load_or_generate_clock_reference(path=CLOCK_REFERENCE_FILE):
    if path.exists():
        print(f"Loading clock reference trajectory: {Path(path).name}")
        with np.load(path) as z:
            states = np.asarray(z["states"], dtype=np.float32)
            times = np.asarray(z["times"], dtype=np.float32)
        if states.ndim == 3:
            states = states[0]
        if times.ndim == 3:
            times = times[0]
        return tf.constant(states), tf.constant(times)

    print(
        f"{Path(path).name} not found. Generating {CLOCK_REFERENCE_STEPS:,} exact "
        "reference events once (not timed).",
        flush=True,
    )
    print(
        f"  Reference-generator XLA JIT: {REFERENCE_JIT_COMPILE}.",
        flush=True,
    )

    init = tf.constant([[1., 0., 1., 0., 0., 0., 0., 0., 0.]], tf.float32)
    k_ref = tf.constant(CLOCK_K_TRUE, tf.float32)

    # Compile/warm the XLA graph with the same input signatures first.
    print("  Compiling/warming fast reference simulator ... ", end="", flush=True)
    warm_start = time.perf_counter()
    warm_states, warm_times = clock_reference_fast(
        init,
        tf.constant(10, tf.int32),
        k_ref,
        tf.constant([BASE_SEED, 10], tf.int32),
    )
    sync_tensor((warm_states[:, -1, :], warm_times[:, -1, :]))
    print(
        f"done in {_format_seconds(time.perf_counter() - warm_start)}.",
        flush=True,
    )

    print(
        f"  Generating full {CLOCK_REFERENCE_STEPS:,}-event reference ... ",
        end="",
        flush=True,
    )
    ref_start = time.perf_counter()
    states, times = clock_reference_fast(
        init,
        tf.constant(CLOCK_REFERENCE_STEPS, tf.int32),
        k_ref,
        tf.constant([BASE_SEED, 11], tf.int32),
    )
    sync_tensor((states[:, -1, :], times[:, -1, :]))
    print(
        f"done in {_format_seconds(time.perf_counter() - ref_start)}.",
        flush=True,
    )
    states_np = states.numpy()
    times_np = times.numpy()
    np.savez(path, states=states_np, times=times_np)
    return tf.constant(states_np[0]), tf.constant(times_np[0])


def prepare_clock_batch():
    """
    Reproduce the protocol's stratified target-map construction, but materialize
    only the start-state batch configured for the selected benchmark profile.
    """
    reference_states, reference_times = load_or_generate_clock_reference()

    needed = CLOCK_REFERENCE_STEPS + 1
    if int(reference_states.shape[0]) < needed:
        raise ValueError(
            f"Reference trajectory has {reference_states.shape[0]} stored states; "
            f"need at least {needed}."
        )

    tf.random.set_seed(BASE_SEED)

    burn_in = CLOCK_BURN_IN
    max_start_excl = CLOCK_REFERENCE_STEPS - CLOCK_SEGMENT_STEPS + 1
    N = tf.constant(CLOCK_NUM_START_POINTS, tf.int32)

    edges = tf.linspace(
        tf.cast(burn_in, tf.float32),
        tf.cast(max_start_excl, tf.float32),
        N + 1,
    )
    left, right = edges[:-1], edges[1:]
    u = tf.random.uniform([N], 0.0, 1.0)
    map_start_indices = tf.cast(tf.floor(left + u * (right - left)), tf.int32)
    map_start_indices = tf.clip_by_value(
        map_start_indices, burn_in, max_start_excl - 1
    )

    all_map_indices = tf.range(CLOCK_NUM_START_POINTS, dtype=tf.int32)
    chosen = tf.random.shuffle(all_map_indices)[:CLOCK_BATCH_SIZE]
    start_indices = tf.gather(map_start_indices, chosen)

    offsets = tf.range(CLOCK_SEGMENT_STEPS + 1, dtype=tf.int32)[None, :]
    trajectory_indices = start_indices[:, None] + offsets

    target_states = tf.gather(reference_states, trajectory_indices)
    target_times_raw = tf.gather(reference_times, trajectory_indices)
    target_times = target_times_raw - target_times_raw[:, :1, :]

    initial_states = tf.gather(reference_states, start_indices)
    initial_states = tf.repeat(
        initial_states[:, None, :], repeats=CLOCK_M_TRAIN, axis=1
    )

    initial_states = tf.ensure_shape(
        initial_states, [CLOCK_BATCH_SIZE, CLOCK_M_TRAIN, 9]
    )
    target_states = tf.ensure_shape(
        target_states, [CLOCK_BATCH_SIZE, CLOCK_SEGMENT_STEPS + 1, 9]
    )
    target_times = tf.ensure_shape(
        target_times, [CLOCK_BATCH_SIZE, CLOCK_SEGMENT_STEPS + 1, 1]
    )

    print("Clock timing batch prepared:")
    print(f"  starting states: {CLOCK_BATCH_SIZE:,}")
    print(f"  trajectories/start: {CLOCK_M_TRAIN}")
    print(f"  total trajectories: {CLOCK_BATCH_SIZE * CLOCK_M_TRAIN:,}")
    print(f"  events/trajectory: {CLOCK_SEGMENT_STEPS}")
    return initial_states, target_states, target_times


def clock_compute_loss(sim_states, target_states, sim_times, target_times):
    epsilon = 1e-9

    def rate(states, times, feature_idx):
        mean_final_state = tf.reduce_mean(
            states[:, -100:, feature_idx], axis=1, keepdims=True
        )
        mean_final_time = tf.reduce_mean(
            times[:, -100:, 0], axis=1, keepdims=True
        )
        delta_state = mean_final_state - states[:, 0:1, feature_idx]
        delta_time = mean_final_time - times[:, 0:1, 0]
        return delta_state / (delta_time + epsilon)

    loss_A = tf.reduce_mean(tf.square(
        rate(sim_states, sim_times, 6) - rate(target_states, target_times, 6)
    ))
    loss_R = tf.reduce_mean(tf.square(
        rate(sim_states, sim_times, 7) - rate(target_states, target_times, 7)
    ))
    loss_C = tf.reduce_mean(tf.square(
        rate(sim_states, sim_times, 8) - rate(target_states, target_times, 8)
    ))
    return loss_A + loss_R + loss_C


def make_clock_trainer(method):
    """
    Actual clock train_step workload: simulation + rate loss + reverse-mode
    gradient + clipping + finite check + RMSprop + true-centered projection.
    """
    assert method in {PST_CAT_NAME, PST_GM_NAME, PST_DIRECT_NAME, PST_XLA_NAME, GSST_GM_NAME}

    true_params = tf.constant([50., 5., 1., 0.2, 2.], tf.float32)
    # Same parameters for both methods during timing.
    initial_params = true_params

    variables = [
        tf.Variable(tf.math.log(initial_params[0]), name=f"{method}_log_beta_A"),
        tf.Variable(tf.math.log(initial_params[1]), name=f"{method}_log_beta_R"),
        tf.Variable(tf.math.log(initial_params[2]), name=f"{method}_log_delta_A"),
        tf.Variable(tf.math.log(initial_params[3]), name=f"{method}_log_delta_R"),
        tf.Variable(tf.math.log(initial_params[4]), name=f"{method}_log_gamma_C"),
    ]
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.1)
    optimizer.build(variables)

    simulate = make_clock_simulator(method)

    initial_var_state = snapshot_variables(variables)
    initial_opt_state = optimizer_snapshot(optimizer)

    true_log = tf.math.log(true_params)
    true_log_list = tf.unstack(true_log)
    log10 = tf.math.log(tf.constant(10.0, tf.float32))
    k_const = tf.constant(CLOCK_K_TRUE, tf.float32)
    tau = tf.constant(3e-6, tf.float32)

    @tf.function(
        jit_compile=False,
        reduce_retracing=True,
        input_signature=[
            tf.TensorSpec([CLOCK_BATCH_SIZE, CLOCK_M_TRAIN, 9], tf.float32),
            tf.TensorSpec([CLOCK_BATCH_SIZE, CLOCK_SEGMENT_STEPS + 1, 9], tf.float32),
            tf.TensorSpec([CLOCK_BATCH_SIZE, CLOCK_SEGMENT_STEPS + 1, 1], tf.float32),
            tf.TensorSpec([2], tf.int32),
        ],
    )
    def train_step(batch_initial_states, batch_target_states, batch_target_times, seed2):
        with tf.GradientTape() as tape:
            p = [tf.exp(v) for v in variables]
            k_learn = tf.concat([
                k_const[:8],
                [p[0]], [p[1]],
                k_const[10:12],
                [p[2]], [p[3]], [p[4]], [p[2]],
            ], axis=0)

            super_batch_initial_states = tf.repeat(
                batch_initial_states[:, 0, :],
                repeats=CLOCK_M_TRAIN,
                axis=0,
            )

            super_states, super_times = simulate(
                super_batch_initial_states,
                tf.constant(CLOCK_SEGMENT_STEPS, tf.int32),
                k_learn,
                tau,
                CLOCK_S,
                seed2,
            )

            sim_states = tf.reshape(
                super_states,
                [CLOCK_BATCH_SIZE, CLOCK_M_TRAIN, CLOCK_SEGMENT_STEPS + 1, 9],
            )
            sim_times = tf.reshape(
                super_times,
                [CLOCK_BATCH_SIZE, CLOCK_M_TRAIN, CLOCK_SEGMENT_STEPS + 1, 1],
            )
            mean_states = tf.reduce_mean(sim_states, axis=1)
            mean_times = tf.reduce_mean(sim_times, axis=1)

            loss = clock_compute_loss(
                mean_states, batch_target_states, mean_times, batch_target_times
            )

        grads = tape.gradient(loss, variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)

        # Original pre-update metrics.
        p_pre = [tf.exp(v) for v in variables]
        rpd = tf.norm(tf.stack(variables) - true_log)
        mape = tf.reduce_mean(
            tf.abs((tf.stack(p_pre) - true_params) / (true_params + 1e-9)) * 100
        )

        grads_finite = tf.reduce_all(
            [tf.reduce_all(tf.math.is_finite(g)) for g in grads]
        )
        ok = tf.math.is_finite(loss) & grads_finite

        def apply_update():
            optimizer.apply_gradients(zip(grads, variables))
            for v, truth in zip(variables, true_log_list):
                v.assign(tf.clip_by_value(v, truth - log10, truth + log10))
            return tf.constant(True)

        did_update = tf.cond(ok, apply_update, lambda: tf.constant(False))

        # Force the returned value to depend on the optimizer update/projection.
        with tf.control_dependencies([did_update]):
            post_checksum = tf.add_n([tf.reduce_sum(v.read_value()) for v in variables])

        return tf.stack([
            loss,
            rpd,
            mape,
            post_checksum,
            tf.cast(did_update, tf.float32),
        ])

    def reset():
        restore_variables(variables, initial_var_state)
        optimizer_restore(optimizer, initial_opt_state)

    return train_step, reset



def benchmark_clock():
    print("\n" + "=" * 78)
    print("CLOCK: SAMPLER / BACKWARD-RULE DECOMPOSITION")
    print("=" * 78)

    print("Preparing the real oscillator timing batch ...", flush=True)
    batch_initial, batch_target_states, batch_target_times = prepare_clock_batch()

    print("Building oscillator training graphs ...", flush=True)
    cat_step, cat_reset = make_clock_trainer(PST_CAT_NAME)
    gm_step, gm_reset = make_clock_trainer(PST_GM_NAME)
    gs_step, gs_reset = make_clock_trainer(GSST_GM_NAME)

    check_clock_graph_parity(gm_step, gs_step)

    def clock_caller(step_fn):
        return lambda seed: step_fn(
            batch_initial, batch_target_states, batch_target_times, seed
        )

    methods = [PST_CAT_NAME]
    resetters = {PST_CAT_NAME: cat_reset}
    callers = {PST_CAT_NAME: clock_caller(cat_step)}

    if INCLUDE_PST_DIRECT:
        print(
            "Building PST-SSA oscillator graph "
            "(unfused direct SSA forward + analytic affine PST VJP) ...",
            flush=True,
        )
        ssa_pst_step, ssa_pst_reset = make_clock_trainer(PST_DIRECT_NAME)
        methods.append(PST_DIRECT_NAME)
        resetters[PST_DIRECT_NAME] = ssa_pst_reset
        callers[PST_DIRECT_NAME] = clock_caller(ssa_pst_step)

    if INCLUDE_PST_XLA:
        print(
            "Building PST-SSA-XLA oscillator graph "
            "(XLA hard sampler only; outer training remains non-XLA) ...",
            flush=True,
        )
        ssa_xla_step, ssa_xla_reset = make_clock_trainer(PST_XLA_NAME)
        methods.append(PST_XLA_NAME)
        resetters[PST_XLA_NAME] = ssa_xla_reset
        callers[PST_XLA_NAME] = clock_caller(ssa_xla_step)

    methods.extend([PST_GM_NAME, GSST_GM_NAME])
    resetters.update({
        PST_GM_NAME: gm_reset,
        GSST_GM_NAME: gs_reset,
    })
    callers.update({
        PST_GM_NAME: clock_caller(gm_step),
        GSST_GM_NAME: clock_caller(gs_step),
    })

    if INCLUDE_SSA_BASELINE:
        print("Building exact SSA forward-only oscillator graph ...", flush=True)
        ssa_step, ssa_reset = make_clock_ssa_forward()
        methods.append(SSA_METHOD_NAME)
        resetters[SSA_METHOD_NAME] = ssa_reset
        callers[SSA_METHOD_NAME] = clock_caller(ssa_step)

    samples = time_paired(
        methods, resetters, callers, WARMUP, REPEATS, BASE_SEED
    )

    ntraj = CLOCK_BATCH_SIZE * CLOCK_M_TRAIN
    return [
        summarize("clock", m, ntraj, CLOCK_SEGMENT_STEPS, samples[m])
        for m in methods
    ]


# =============================================================================
# ION CHANNEL — COPIED FROM THE TRAINING PROTOCOL
# =============================================================================

ION_S = tf.constant([
    [-1.0, +1.0, 0.0],
    [+1.0, -1.0, 0.0],
    [0.0, -1.0, +1.0],
], tf.float32)

ION_N_CHANNELS = 2


@tf.function(jit_compile=False, reduce_retracing=True)
def ion_propensity(states, k_open, k_close, k_inact):
    states = tf.maximum(states, 0.0)
    n_closed = states[:, 0]
    n_open = states[:, 1]
    return tf.stack([
        k_open * n_closed,
        k_close * n_open,
        k_inact * n_open,
    ], axis=1)


def _ion_gumbel_max(logits):
    """Exact stateful Gumbel-Max hard categorical draw."""
    u = tf.random.uniform(
        tf.shape(logits),
        minval=1e-20,
        maxval=1.0,
        dtype=logits.dtype,
    )
    g = -tf.math.log(-tf.math.log(u))
    hard_idx = tf.argmax(logits + g, axis=-1)
    y_hard = tf.one_hot(
        hard_idx,
        depth=tf.shape(logits)[-1],
        dtype=logits.dtype,
    )
    return y_hard, g


def ion_pst_categorical_estimator(logits):
    """Original ion-channel PST: categorical hard forward, d pi backward."""
    probabilities = tf.nn.softmax(logits, axis=-1)
    logits_shape = tf.shape(logits)
    flat_logits = tf.reshape(logits, [-1, logits_shape[-1]])
    hard_idx = tf.random.categorical(flat_logits, 1)
    hard_idx = tf.reshape(hard_idx, logits_shape[:-1])
    y_hard = tf.one_hot(
        hard_idx,
        depth=logits_shape[-1],
        dtype=logits.dtype,
    )
    return probabilities + tf.stop_gradient(y_hard - probabilities)


def ion_pst_gumbel_estimator(logits):
    """Exact Gumbel-Max hard forward; PST backward."""
    y_hard, _ = _ion_gumbel_max(logits)
    probabilities = tf.nn.softmax(logits, axis=-1)
    return probabilities + tf.stop_gradient(y_hard - probabilities)


def ion_gsst_gumbel_estimator(logits, tau):
    """Exact Gumbel-Max hard forward; Gumbel-Softmax ST backward."""
    y_hard, g = _ion_gumbel_max(logits)
    y_soft = tf.nn.softmax((logits + g) / tau, axis=-1)
    return y_soft + tf.stop_gradient(y_hard - y_soft)

def make_ion_simulator(method):
    assert method in {PST_CAT_NAME, PST_GM_NAME, PST_DIRECT_NAME, PST_XLA_NAME, GSST_GM_NAME}

    @tf.function(jit_compile=False, reduce_retracing=True)
    def simulate(batch_initial_states, k_open, k_close, k_inact, temperature):
        num_sims = tf.shape(batch_initial_states)[0]
        time_t = tf.zeros([num_sims, 1], tf.float32)
        states_ta = tf.TensorArray(tf.float32, size=ION_NUM_STEPS + 1)
        times_ta = tf.TensorArray(tf.float32, size=ION_NUM_STEPS + 1)
        states_ta = states_ta.write(0, batch_initial_states)
        times_ta = times_ta.write(0, time_t)

        states = batch_initial_states
        step = tf.constant(0)

        def cond(step, states, time_t, states_ta, times_ta):
            return step < ION_NUM_STEPS

        def body(step, states, time_t, states_ta, times_ta):
            prop = ion_propensity(states, k_open, k_close, k_inact)
            a0 = tf.reduce_sum(prop, axis=1, keepdims=True)

            absorbing = a0 < 1e-10
            safe_a0 = tf.where(absorbing, tf.ones_like(a0), a0)

            u_time = tf.random.uniform(
                [num_sims, 1],
                minval=1e-20,
                maxval=1.0,
                dtype=tf.float32,
            )
            dt = -tf.math.log(u_time) / safe_a0
            dt = tf.where(absorbing, tf.ones_like(dt) * 100000.0, dt)

            # Preserve exact zero-propensity support. Absorbing rows get
            # one dummy positive channel solely to keep sampling well defined;
            # the update is masked to zero below.
            fallback = tf.one_hot(
                tf.zeros([num_sims], dtype=tf.int32),
                depth=tf.shape(prop)[-1],
                dtype=prop.dtype,
            )
            sampling_prop = tf.where(absorbing, fallback, prop)

            if method in {PST_DIRECT_NAME, PST_XLA_NAME}:
                # Same PST estimator in both cases.  PST-SSA-XLA only
                # XLA-compiles the detached hard direct-method sampler.
                delta = ion_pst_delta_ssa(
                    sampling_prop,
                    safe_a0,
                    ION_S,
                    use_xla_sampler=(method == PST_XLA_NAME),
                )
            else:
                positive = sampling_prop > 0.0
                finite_log = tf.math.log(
                    tf.where(
                        positive,
                        sampling_prop,
                        tf.ones_like(sampling_prop),
                    )
                )
                logits = tf.where(
                    positive,
                    finite_log,
                    tf.constant(float("-inf"), dtype=prop.dtype),
                )

                if method == PST_CAT_NAME:
                    weights = ion_pst_categorical_estimator(logits)
                elif method == PST_GM_NAME:
                    weights = ion_pst_gumbel_estimator(logits)
                else:
                    weights = ion_gsst_gumbel_estimator(
                        logits, temperature
                    )

                delta = tf.matmul(weights, ION_S)

            delta = tf.where(absorbing, tf.zeros_like(delta), delta)

            states = tf.maximum(states + delta, 0.0)
            time_t = time_t + dt

            states_ta = states_ta.write(step + 1, states)
            times_ta = times_ta.write(step + 1, time_t)
            return step + 1, states, time_t, states_ta, times_ta

        _, _, _, states_ta, times_ta = tf.while_loop(
            cond,
            body,
            [step, states, time_t, states_ta, times_ta],
            shape_invariants=[
                step.shape,
                tf.TensorShape([None, 3]),
                tf.TensorShape([None, 1]),
                tf.TensorShape([]),
                tf.TensorShape([]),
            ],
        )
        states = tf.transpose(states_ta.stack(), [1, 0, 2])
        times = tf.transpose(times_ta.stack(), [1, 0, 2])
        return states, times

    return simulate


@tf.function(jit_compile=ION_READOUT_JIT, reduce_retracing=True)
def ion_bin_integrated_readout(sim_t, sim_v, tgt_t):
    sim_t = tf.squeeze(sim_t, axis=-1)
    n_sims = tf.shape(sim_t)[0]
    n_steps = tf.shape(sim_t)[1]
    dtype = sim_t.dtype

    ramp = tf.cast(tf.range(n_steps)[None, :], dtype) * tf.cast(1e-8, dtype)
    sim_t = sim_t + ramp

    dt = tf.maximum(sim_t[:, 1:] - sim_t[:, :-1], tf.cast(0.0, dtype))
    step_areas = sim_v[:, :-1] * dt
    cumulative_area = tf.concat(
        [tf.zeros([n_sims, 1], dtype), tf.cumsum(step_areas, axis=1)],
        axis=1,
    )

    first_t = sim_t[:, :1]
    last_t = sim_t[:, -1:]
    last_area = cumulative_area[:, -1:]
    last_value = sim_v[:, -1:]

    def cumulative_at(query_t):
        n_query = tf.shape(query_t)[0]
        q = tf.broadcast_to(query_t[None, :], [n_sims, n_query])
        q = tf.cast(q, dtype)

        idx = tf.searchsorted(sim_t, q, side="right") - 1
        idx = tf.clip_by_value(idx, 0, n_steps - 2)

        area0 = tf.gather(cumulative_area, idx, batch_dims=1)
        time0 = tf.gather(sim_t, idx, batch_dims=1)
        value0 = tf.gather(sim_v, idx, batch_dims=1)
        inside = area0 + value0 * (q - time0)

        before = tf.zeros_like(q)
        after = last_area + last_value * (q - last_t)
        return tf.where(q < first_t, before, tf.where(q >= last_t, after, inside))

    area_start = cumulative_at(tgt_t[:-1])
    area_end = cumulative_at(tgt_t[1:])
    widths = tf.cast(tgt_t[1:] - tgt_t[:-1], dtype)
    return (area_end - area_start) / widths[None, :]


def make_ion_loss(method):
    simulate = make_ion_simulator(method)

    @tf.function(jit_compile=False, reduce_retracing=True)
    def loss_fn(initial_state, t_target, n_open_target,
                k_open, k_close, k_inact, temperature):
        super_batch = tf.repeat(initial_state, repeats=ION_M_SIMS, axis=0)
        sim_states, sim_times = simulate(
            super_batch, k_open, k_close, k_inact, temperature
        )

        n_open_sim = sim_states[:, :, 1]
        model_bin_avg = ion_bin_integrated_readout(
            sim_times, n_open_sim, t_target
        )

        model_bin_avg = tf.reshape(
            model_bin_avg,
            [1, ION_M_SIMS, tf.shape(t_target)[0] - 1],
        )
        model_mean = tf.reduce_mean(model_bin_avg, axis=[0, 1])
        experimental_bin_avg = 0.5 * (
            n_open_target[:-1] + n_open_target[1:]
        )
        return tf.reduce_mean(tf.square(model_mean - experimental_bin_avg))

    return loss_fn


def load_ion_data(path=ION_DATA_FILE):
    if not path.exists():
        raise FileNotFoundError(
            f"Ion-channel data file not found: {Path(path).name}\n"
            "Edit ION_DATA_FILE at the top of the notebook to point to the "
            "original HEK293 XLSX file. A synthetic replacement is intentionally "
            "not used in this faithful benchmark."
        )
    df = pd.read_excel(path, header=None, skiprows=5)
    time_ms = df.iloc[:, 0].values.astype(np.float32)
    sweeps = df.iloc[:, 1:].values.astype(np.float32)
    mean_open = np.nanmean(sweeps, axis=1).astype(np.float32)
    time_shifted = (time_ms - time_ms[0]).astype(np.float32)
    print("Ion-channel data loaded:")
    print(f"  sweeps={sweeps.shape[1]}, timepoints={len(time_ms)}")
    return tf.constant(time_shifted), tf.constant(mean_open)


def make_ion_trainer(method, t_target, n_open_target):
    """
    Mirrors the original ion training loop: GradientTape in Python, loss graph,
    gradient materialization for NaN checking, then RMSprop update.
    """
    assert method in {PST_CAT_NAME, PST_GM_NAME, PST_DIRECT_NAME, PST_XLA_NAME, GSST_GM_NAME}
    loss_fn = make_ion_loss(method)

    log_k_open = tf.Variable(tf.math.log(0.25), name=f"{method}_ion_log_open")
    log_k_close = tf.Variable(tf.math.log(0.25), name=f"{method}_ion_log_close")
    log_k_inact = tf.Variable(tf.math.log(0.25), name=f"{method}_ion_log_inact")
    variables = [log_k_open, log_k_close, log_k_inact]

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.1)
    optimizer.build(variables)

    initial_var_state = snapshot_variables(variables)
    initial_opt_state = optimizer_snapshot(optimizer)

    initial_state = tf.constant([[2.0, 0.0, 0.0]], tf.float32)

    schedule_value = 0.1 * (0.001 / 0.1) ** (
        ION_TIMING_EPOCH / max(1, ION_EPOCHS - 1)
    )
    temperature = tf.constant(schedule_value, tf.float32)
    learning_rate = 0.05 * (schedule_value / 0.1)

    def reset():
        restore_variables(variables, initial_var_state)
        optimizer_restore(optimizer, initial_opt_state)

    def step(_seed_unused):
        # The ion-channel training notebooks use stateful TensorFlow RNG, so the seed
        # argument is intentionally unused here.
        optimizer.learning_rate.assign(learning_rate)

        with tf.GradientTape() as tape:
            k_open = tf.exp(log_k_open)
            k_close = tf.exp(log_k_close)
            k_inact = tf.exp(log_k_inact)
            loss = loss_fn(
                initial_state,
                t_target,
                n_open_target,
                k_open,
                k_close,
                k_inact,
                temperature,
            )

        gradients = tape.gradient(loss, variables)

        # Match the original Python-side NaN check; .numpy() synchronizes the
        # gradient before the optimizer update, exactly as the training loop did.
        invalid_gradient = any(
            g is None or bool(tf.reduce_any(tf.math.is_nan(g)).numpy())
            for g in gradients
        )
        if not invalid_gradient:
            optimizer.apply_gradients(zip(gradients, variables))

        # Force completion of the update and match original scalar history reads.
        values = np.array([
            float(loss.numpy()),
            float(tf.exp(log_k_open).numpy()),
            float(tf.exp(log_k_close).numpy()),
            float(tf.exp(log_k_inact).numpy()),
            float(optimizer.iterations.numpy()),
        ], dtype=np.float64)
        return values

    return step, reset



def benchmark_ion():
    print("\n" + "=" * 78)
    print("ION CHANNEL: SAMPLER / BACKWARD-RULE DECOMPOSITION")
    print("=" * 78)

    t_target, n_open_target = load_ion_data()

    print("Building ion-channel training graphs ...", flush=True)
    cat_step, cat_reset = make_ion_trainer(
        PST_CAT_NAME, t_target, n_open_target
    )
    gm_step, gm_reset = make_ion_trainer(
        PST_GM_NAME, t_target, n_open_target
    )
    gs_step, gs_reset = make_ion_trainer(
        GSST_GM_NAME, t_target, n_open_target
    )

    methods = [PST_CAT_NAME]
    resetters = {PST_CAT_NAME: cat_reset}
    callers = {PST_CAT_NAME: cat_step}

    if INCLUDE_PST_DIRECT:
        print(
            "Building PST-SSA ion-channel graph "
            "(unfused direct SSA forward + analytic affine PST VJP) ...",
            flush=True,
        )
        ssa_pst_step, ssa_pst_reset = make_ion_trainer(
            PST_DIRECT_NAME, t_target, n_open_target
        )
        methods.append(PST_DIRECT_NAME)
        resetters[PST_DIRECT_NAME] = ssa_pst_reset
        callers[PST_DIRECT_NAME] = ssa_pst_step

    if INCLUDE_PST_XLA:
        print(
            "Building PST-SSA-XLA ion-channel graph "
            "(XLA hard sampler only; outer training remains non-XLA) ...",
            flush=True,
        )
        ssa_xla_step, ssa_xla_reset = make_ion_trainer(
            PST_XLA_NAME, t_target, n_open_target
        )
        methods.append(PST_XLA_NAME)
        resetters[PST_XLA_NAME] = ssa_xla_reset
        callers[PST_XLA_NAME] = ssa_xla_step

    methods.extend([PST_GM_NAME, GSST_GM_NAME])
    resetters.update({
        PST_GM_NAME: gm_reset,
        GSST_GM_NAME: gs_reset,
    })
    callers.update({
        PST_GM_NAME: gm_step,
        GSST_GM_NAME: gs_step,
    })

    if INCLUDE_SSA_BASELINE:
        print("Building exact SSA forward-only ion graph ...", flush=True)
        ssa_step, ssa_reset = make_ion_ssa_forward(
            t_target, n_open_target
        )
        methods.append(SSA_METHOD_NAME)
        resetters[SSA_METHOD_NAME] = ssa_reset
        callers[SSA_METHOD_NAME] = ssa_step

    samples = time_paired(
        methods, resetters, callers, WARMUP, REPEATS, BASE_SEED
    )

    return [
        summarize("ion", m, ION_M_SIMS, ION_NUM_STEPS, samples[m])
        for m in methods
    ]


# =============================================================================
# STANDARD GILLESPIE SSA (DIRECT METHOD)
# =============================================================================
#
# Reaction selection by inverse-transform sampling on the cumulative
# propensity (Gillespie 1977): draw u ~ U[0,1), take the smallest j with
# cumsum(a)[j] > u * a0.  No categorical kernel, no Gumbel noise.
#
# This serves two distinct purposes and they must not be conflated:
#
#   (1) A non-differentiable exact-SSA forward (clock_simulate_ssa /
#       ion_simulate_ssa) timed WITHOUT a tape or optimizer, reported as
#       "SSA-fwd".  It is a lower bound on simulation cost, not a training
#       step, and is not comparable to the PST/GS-ST rows.
#
#   (2) A direct-method forward inside the PST estimator ("PST-dm").  The
#       backward branch is bit-identical to PST, so this remains a full
#       training step and stays comparable to GS-ST.  GS-ST has no such
#       variant: its backward relaxation needs the Gumbel variates, so it
#       cannot avoid drawing them.  That asymmetry is a property of the
#       estimators, not of the implementation.

# Graph-level assertions on the sampler.  MUST stay False for timed runs: the
# assert op sits inside the inner while_loop body and is not free.  Read at
# trace time, so changing it requires rebuilding the graphs.
SSA_DEBUG_CHECKS = False

# False -> state update is tf.gather(S, index), what a real SSA does.
# True  -> state update is one_hot(index) @ S, matching the PST/GS-ST loop body
#          op-for-op so the timing difference isolates the sampler alone.
SSA_ONEHOT_STATE_UPDATE = False

# Method labels and inclusion flags are defined with the main settings.


def _apply_reaction(reaction_index, S_matrix, num_reactions, dtype):
    """State delta for a sampled reaction index; see SSA_ONEHOT_STATE_UPDATE."""
    if SSA_ONEHOT_STATE_UPDATE:
        weights = tf.one_hot(reaction_index, depth=num_reactions, dtype=dtype)
        return tf.matmul(weights, S_matrix)
    return tf.gather(S_matrix, reaction_index)


def direct_method_index(propensities, u):
    """Smallest j with cumsum(propensities)[j] > u * a0, per row.

    propensities : [B, R] float, non-negative, row sum STRICTLY POSITIVE.
    u            : [B, 1] float in [0, 1).
    returns      : [B] int32 in [0, R-1].

    Precondition: every row must have a positive sum.  Absorbing rows must be
    masked by the CALLER before this call -- see the `fallback` construction in
    the simulators.  A dead row reaching here returns R, which is out of range
    for tf.gather, and SSA_DEBUG_CHECKS detects this invalid input.

    a0 is taken as the last cumsum element rather than a separate reduce_sum so
    that `threshold <= a0 = cdf[R-1]` is exact.  Recomputing the total with
    reduce_sum can let threshold exceed cdf[R-1] by an ulp through rounding
    and push the index to R.

    On the returned index being in range, which matters because tf.gather with
    an out-of-range index raises on CPU but silently yields zeros on GPU: the
    bound needs u < 1 strictly AND `u * a0` never rounding up to a0.  Both
    hold.  tf.random.uniform returns [minval, maxval), so the largest float32
    draw is 1 - 2^-24; and for a0 in [2^e, 2^(e+1)) the exact product
    a0 - a0*2^-24 sits strictly between a0 - ulp and a0 - ulp/2, so
    round-to-nearest gives a0 - ulp, not a0.  Verified over 1.2e7 draws
    spanning a0 from 1e-6 to 1e6: zero occurrences.  Feeding u == 1.0 by hand
    breaks this; the RNG cannot produce it.
    """
    cdf = tf.cumsum(propensities, axis=-1)
    threshold = u * cdf[..., -1:]
    # #{j : cdf[j] <= threshold} == smallest j with cdf[j] > threshold.
    # Zero-propensity reactions leave cdf flat and are therefore unreachable,
    # which is the correct direct-method behaviour.
    index = tf.reduce_sum(tf.cast(cdf <= threshold, tf.int32), axis=-1)

    if SSA_DEBUG_CHECKS:
        num_reactions = tf.shape(propensities)[-1]
        tf.debugging.assert_less(
            index,
            num_reactions,
            message="direct_method_index: index out of range; either an "
                    "all-zero propensity row reached the sampler (caller "
                    "failed to mask absorbing states) or u == 1.0 was passed",
        )
        # The postcondition that actually matters, and which subsumes the
        # range check: a zero-propensity reaction must never be selected.
        tf.debugging.assert_positive(
            tf.gather(
                propensities,
                tf.minimum(index, num_reactions - 1),
                batch_dims=tf.rank(propensities) - 1,
            ),
            message="direct_method_index: selected a zero-propensity reaction",
        )
    return index


def direct_method_index_searchsorted(propensities, u):
    """Identical semantics to direct_method_index; O(log R) instead of O(R).

    For R <= ~32 the compare-and-count form above beats this on GPU (one fused
    elementwise+reduce against a binary-search kernel launch).  Kept for large
    reaction networks.  side="right" gives #{j : cdf[j] <= threshold}, matching
    direct_method_index exactly.  Requires a rank-2 propensity tensor.
    """
    cdf = tf.cumsum(propensities, axis=-1)
    threshold = u * cdf[..., -1:]
    index = tf.squeeze(tf.searchsorted(cdf, threshold, side="right"), axis=-1)

    if SSA_DEBUG_CHECKS:
        tf.debugging.assert_less(
            index,
            tf.shape(propensities)[-1],
            message="direct_method_index_searchsorted: index out of range",
        )
    return index


# ------------------------------------------- direct-method PST estimators


def _hard_ssa_delta_core(propensities, S_matrix, u_reaction):
    """Forward-only direct SSA reaction selection and stoichiometric gather.

    Inputs are already detached by the caller.  This function is intentionally
    gradient-agnostic: its sole job is

        cdf -> threshold comparison -> integer reaction index -> S[index].

    It is used in two forms below:
      * ordinary TensorFlow graph execution (`PST-SSA`);
      * an XLA-compiled subfunction (`PST-SSA-XLA`).

    The outer differentiable simulator remains jit_compile=False in both cases.
    """
    hard_index = direct_method_index(propensities, u_reaction)
    return tf.gather(S_matrix, hard_index)


# Only this detached hard sampler is XLA-compiled.  The surrounding simulator,
# loss, GradientTape, optimizer update, and PST custom VJP remain non-XLA.
hard_ssa_delta_xla = tf.function(
    _hard_ssa_delta_core,
    jit_compile=True,
    reduce_retracing=True,
)


def _pst_affine_vjp(
    propensities,
    total_propensity,
    S_matrix,
    d_delta,
):
    """Analytic PST VJP of (a/a0) @ S with respect to raw propensities."""
    # q[b,j] = <d_delta[b,:], S[j,:]>
    q = tf.linalg.matmul(d_delta, S_matrix, transpose_b=True)

    inv_a0 = tf.math.reciprocal(total_propensity)
    probability = propensities * inv_a0
    mean_q = tf.reduce_sum(
        q * probability,
        axis=-1,
        keepdims=True,
    )

    return (q - mean_q) * inv_a0


@tf.custom_gradient
def pst_ssa_delta_from_uniform(
    propensities,
    total_propensity,
    S_matrix,
    u_reaction,
):
    """Unfused direct-SSA hard jump + analytic PST affine VJP.

    This is retained as the control arm for the XLA sampler experiment.
    """
    hard_prop = tf.stop_gradient(propensities)
    hard_S = tf.stop_gradient(S_matrix)
    hard_u = tf.stop_gradient(u_reaction)

    delta_hard = _hard_ssa_delta_core(
        hard_prop,
        hard_S,
        hard_u,
    )

    def grad(d_delta):
        d_propensities = _pst_affine_vjp(
            propensities,
            total_propensity,
            S_matrix,
            d_delta,
        )
        return (
            d_propensities,
            tf.zeros_like(total_propensity),
            None,
            None,
        )

    return delta_hard, grad


@tf.custom_gradient
def pst_ssa_delta_xla_from_uniform(
    propensities,
    total_propensity,
    S_matrix,
    u_reaction,
):
    """XLA-fused detached direct-SSA hard jump + the SAME analytic PST VJP.

    The estimator is unchanged relative to PST-SSA.  Only the forward hard
    sampler implementation changes:

        PST-SSA:
            _hard_ssa_delta_core(...)

        PST-SSA-XLA:
            hard_ssa_delta_xla(...)

    This isolates whether fusing Cumsum/compare/reduce/gather removes the
    TensorFlow direct-method sampler overhead seen for the oscillator.
    """
    hard_prop = tf.stop_gradient(propensities)
    hard_S = tf.stop_gradient(S_matrix)
    hard_u = tf.stop_gradient(u_reaction)

    # Partitioned call to an XLA-compiled FORWARD-ONLY subgraph.
    delta_hard = hard_ssa_delta_xla(
        hard_prop,
        hard_S,
        hard_u,
    )

    def grad(d_delta):
        d_propensities = _pst_affine_vjp(
            propensities,
            total_propensity,
            S_matrix,
            d_delta,
        )
        return (
            d_propensities,
            tf.zeros_like(total_propensity),
            None,
            None,
        )

    return delta_hard, grad


def clock_pst_delta_ssa(
    propensities,
    total_propensity,
    S_matrix,
    seed,
    use_xla_sampler=False,
):
    """Stateless PST-SSA reaction jump, with optional XLA hard sampler."""
    u = tf.random.stateless_uniform(
        tf.concat([tf.shape(propensities)[:-1], [1]], axis=0),
        seed=seed,
        minval=0.0,
        maxval=1.0,
        dtype=propensities.dtype,
    )

    if use_xla_sampler:
        return pst_ssa_delta_xla_from_uniform(
            propensities,
            total_propensity,
            S_matrix,
            u,
        )

    return pst_ssa_delta_from_uniform(
        propensities,
        total_propensity,
        S_matrix,
        u,
    )


def ion_pst_delta_ssa(
    propensities,
    total_propensity,
    S_matrix,
    use_xla_sampler=False,
):
    """Stateful PST-SSA reaction jump, with optional XLA hard sampler."""
    u = tf.random.uniform(
        tf.concat([tf.shape(propensities)[:-1], [1]], axis=0),
        minval=0.0,
        maxval=1.0,
        dtype=propensities.dtype,
    )

    if use_xla_sampler:
        return pst_ssa_delta_xla_from_uniform(
            propensities,
            total_propensity,
            S_matrix,
            u,
        )

    return pst_ssa_delta_from_uniform(
        propensities,
        total_propensity,
        S_matrix,
        u,
    )


def clock_simulate_ssa(batch_initial_states, num_steps, k, S_matrix, seed):
    """Standard Gillespie direct method, same loop shape as the PST simulator.

    Differences from the PST/GS-ST simulators, all intrinsic to the SSA rather
    than incidental: the reaction is an integer index, not a one-hot row; the
    state update is tf.gather(S, index), not one_hot @ S; and there is no
    softmax, Gumbel, or stop_gradient, hence no gradient path from `k` to the
    trajectory.

    jit_compile stays False.  The reference generator elsewhere in this cell
    uses jit_compile=True because it is never compared against a training
    step; a JIT-compiled row here would not be comparable to the non-JIT
    PST/GS-ST rows.
    """
    num_simulations = tf.shape(batch_initial_states)[0]
    num_reactions = tf.shape(S_matrix)[0]
    time_t = tf.zeros([num_simulations, 1], tf.float32)
    states_ta = tf.TensorArray(tf.float32, size=num_steps + 1)
    times_ta = tf.TensorArray(tf.float32, size=num_steps + 1)
    states_ta = states_ta.write(0, batch_initial_states)
    times_ta = times_ta.write(0, time_t)

    t = tf.constant(0)
    states = batch_initial_states

    def cond(t, states, time_t, states_ta, times_ta, seed):
        return t < num_steps

    def body(t, states, time_t, states_ta, times_ta, seed):
        seeds = tf.random.experimental.stateless_split(seed, 3)
        next_seed, seed_time, seed_reaction = seeds[0], seeds[1], seeds[2]

        prop = clock_propensity(states, k)
        a0 = tf.reduce_sum(prop, axis=1, keepdims=True)
        active = a0 > 0.0
        safe_a0 = tf.where(active, a0, tf.ones_like(a0))

        u_time = tf.random.stateless_uniform(
            [num_simulations, 1],
            seed=seed_time,
            minval=1e-20,
            maxval=1.0,
            dtype=prop.dtype,
        )
        dt = tf.where(
            active,
            -tf.math.log(u_time) / safe_a0,
            tf.zeros_like(u_time),
        )

        # Absorbing rows get a dummy positive propensity so the sampler's
        # precondition holds; their state delta is zeroed below.
        fallback = tf.one_hot(
            tf.zeros([num_simulations], dtype=tf.int32),
            depth=num_reactions,
            dtype=prop.dtype,
        )
        sampling_prop = tf.where(active, prop, fallback)

        u_reaction = tf.random.stateless_uniform(
            [num_simulations, 1],
            seed=seed_reaction,
            minval=0.0,
            maxval=1.0,
            dtype=prop.dtype,
        )
        reaction_index = direct_method_index(sampling_prop, u_reaction)

        delta = _apply_reaction(
            reaction_index, S_matrix, num_reactions, prop.dtype
        )
        delta = tf.where(active, delta, tf.zeros_like(delta))

        new_states = tf.maximum(states + delta, 0.0)
        new_time = time_t + dt

        states_ta = states_ta.write(t + 1, new_states)
        times_ta = times_ta.write(t + 1, new_time)
        return t + 1, new_states, new_time, states_ta, times_ta, next_seed

    _, _, _, states_ta, times_ta, _ = tf.while_loop(
        cond, body,
        [t, states, time_t, states_ta, times_ta, seed],
        shape_invariants=[
            t.shape,
            tf.TensorShape([None, 9]),
            tf.TensorShape([None, 1]),
            tf.TensorShape([]),
            tf.TensorShape([]),
            tf.TensorShape([2]),
        ],
    )
    states = tf.transpose(states_ta.stack(), [1, 0, 2])
    times = tf.transpose(times_ta.stack(), [1, 0, 2])
    return states, times


@tf.function(jit_compile=False, reduce_retracing=True)
def ion_simulate_ssa(batch_initial_states, k_open, k_close, k_inact):
    """Direct method with the ion protocol's absorbing-state convention."""
    num_sims = tf.shape(batch_initial_states)[0]
    num_reactions = tf.shape(ION_S)[0]
    time_t = tf.zeros([num_sims, 1], tf.float32)
    states_ta = tf.TensorArray(tf.float32, size=ION_NUM_STEPS + 1)
    times_ta = tf.TensorArray(tf.float32, size=ION_NUM_STEPS + 1)
    states_ta = states_ta.write(0, batch_initial_states)
    times_ta = times_ta.write(0, time_t)

    states = batch_initial_states
    step = tf.constant(0)

    def cond(step, states, time_t, states_ta, times_ta):
        return step < ION_NUM_STEPS

    def body(step, states, time_t, states_ta, times_ta):
        prop = ion_propensity(states, k_open, k_close, k_inact)
        a0 = tf.reduce_sum(prop, axis=1, keepdims=True)

        absorbing = a0 < 1e-10
        safe_a0 = tf.where(absorbing, tf.ones_like(a0), a0)

        u_time = tf.random.uniform(
            [num_sims, 1], minval=1e-20, maxval=1.0, dtype=tf.float32
        )
        dt = -tf.math.log(u_time) / safe_a0
        dt = tf.where(absorbing, tf.ones_like(dt) * 100000.0, dt)

        # Direct SSA uses the true propensities. Zero-propensity reactions
        # therefore have probability exactly zero.
        fallback = tf.one_hot(
            tf.zeros([num_sims], dtype=tf.int32),
            depth=num_reactions,
            dtype=prop.dtype,
        )
        sampling_prop = tf.where(absorbing, fallback, prop)

        u_reaction = tf.random.uniform(
            [num_sims, 1], minval=0.0, maxval=1.0, dtype=prop.dtype
        )
        reaction_index = direct_method_index(sampling_prop, u_reaction)

        delta = _apply_reaction(
            reaction_index, ION_S, num_reactions, prop.dtype
        )
        delta = tf.where(absorbing, tf.zeros_like(delta), delta)

        states = tf.maximum(states + delta, 0.0)
        time_t = time_t + dt

        states_ta = states_ta.write(step + 1, states)
        times_ta = times_ta.write(step + 1, time_t)
        return step + 1, states, time_t, states_ta, times_ta

    _, _, _, states_ta, times_ta = tf.while_loop(
        cond, body,
        [step, states, time_t, states_ta, times_ta],
        shape_invariants=[
            step.shape,
            tf.TensorShape([None, 3]),
            tf.TensorShape([None, 1]),
            tf.TensorShape([]),
            tf.TensorShape([]),
        ],
    )
    states = tf.transpose(states_ta.stack(), [1, 0, 2])
    times = tf.transpose(times_ta.stack(), [1, 0, 2])
    return states, times


# ------------------------------------------------ forward-only timing entries

def make_clock_ssa_forward():
    """Simulation + rate loss.  No tape, no gradient, no optimizer.

    There is deliberately no GradientTape here, and the reason is sharper than
    "the SSA is non-differentiable".

    The reaction-choice channel is genuinely dead: tf.gather by a sampled
    integer index has no derivative, so d(states)/dk is exactly zero.  But the
    waiting-time channel is NOT dead -- dt = -log(u)/a0 is an analytic
    function of the propensity sum, so d(times)/dk is finite and nonzero.  The
    clock loss is a rate, delta_state / delta_time, so it touches both.

    A tape wrapped around the SSA therefore does not raise and does not return
    None.  It returns a large, finite, entirely plausible-looking number that
    is missing the whole reaction-selection term.  Timing that as a "training
    step", or worse training on it, is a silent failure.  The only safe
    options are to time the forward pass alone, as here, or to use one of the
    straight-through estimators.  verify_ssa_gradient_channels() asserts both
    halves of this.
    """
    k_const = tf.constant(CLOCK_K_TRUE, tf.float32)

    @tf.function(
        jit_compile=False,
        reduce_retracing=True,
        input_signature=[
            tf.TensorSpec([CLOCK_BATCH_SIZE, CLOCK_M_TRAIN, 9], tf.float32),
            tf.TensorSpec([CLOCK_BATCH_SIZE, CLOCK_SEGMENT_STEPS + 1, 9], tf.float32),
            tf.TensorSpec([CLOCK_BATCH_SIZE, CLOCK_SEGMENT_STEPS + 1, 1], tf.float32),
            tf.TensorSpec([2], tf.int32),
        ],
    )
    def forward_step(batch_initial_states, batch_target_states,
                     batch_target_times, seed2):
        super_batch_initial_states = tf.repeat(
            batch_initial_states[:, 0, :], repeats=CLOCK_M_TRAIN, axis=0
        )
        super_states, super_times = clock_simulate_ssa(
            super_batch_initial_states,
            tf.constant(CLOCK_SEGMENT_STEPS, tf.int32),
            k_const,
            CLOCK_S,
            seed2,
        )
        sim_states = tf.reshape(
            super_states,
            [CLOCK_BATCH_SIZE, CLOCK_M_TRAIN, CLOCK_SEGMENT_STEPS + 1, 9],
        )
        sim_times = tf.reshape(
            super_times,
            [CLOCK_BATCH_SIZE, CLOCK_M_TRAIN, CLOCK_SEGMENT_STEPS + 1, 1],
        )
        mean_states = tf.reduce_mean(sim_states, axis=1)
        mean_times = tf.reduce_mean(sim_times, axis=1)
        loss = clock_compute_loss(
            mean_states, batch_target_states, mean_times, batch_target_times
        )
        return tf.stack([loss])

    def reset():
        # No trainable variables and no optimizer slots exist for this entry.
        return None

    return forward_step, reset


def make_ion_ssa_forward(t_target, n_open_target):
    """Ion-channel simulation + loss, forward only.  See make_clock_ssa_forward."""
    initial_state = tf.constant([[2.0, 0.0, 0.0]], tf.float32)
    k_open = tf.constant(0.25, tf.float32)
    k_close = tf.constant(0.25, tf.float32)
    k_inact = tf.constant(0.25, tf.float32)

    @tf.function(jit_compile=False, reduce_retracing=True)
    def forward_step_graph():
        super_batch = tf.repeat(initial_state, repeats=ION_M_SIMS, axis=0)
        sim_states, sim_times = ion_simulate_ssa(
            super_batch, k_open, k_close, k_inact
        )
        n_open_sim = sim_states[:, :, 1]
        model_bin_avg = ion_bin_integrated_readout(
            sim_times, n_open_sim, t_target
        )
        model_bin_avg = tf.reshape(
            model_bin_avg, [1, ION_M_SIMS, tf.shape(t_target)[0] - 1]
        )
        model_mean = tf.reduce_mean(model_bin_avg, axis=[0, 1])
        experimental_bin_avg = 0.5 * (n_open_target[:-1] + n_open_target[1:])
        return tf.reduce_mean(tf.square(model_mean - experimental_bin_avg))

    def forward_step(_seed_unused):
        # Stateful RNG, matching the ion protocol; the seed argument is unused
        # for the same reason it is unused in make_ion_trainer.
        return np.array([float(forward_step_graph().numpy())], dtype=np.float64)

    def reset():
        return None

    return forward_step, reset


# --------------------------------------------------------------- diagnostics

def verify_ssa_sampler(num_draws=400_000, seed=(7, 11)):
    """Assert the direct-method sampler reproduces prop / a0, including zeros."""
    propensity_row = tf.constant(
        [[3.0, 0.0, 1.0, 6.0, 0.0, 10.0]], tf.float32
    )
    num_reactions = int(propensity_row.shape[-1])
    batch = tf.tile(propensity_row, [num_draws, 1])
    u = tf.random.stateless_uniform([num_draws, 1], seed=seed, dtype=tf.float32)
    index = direct_method_index(batch, u)

    assert int(tf.reduce_max(index)) < num_reactions, "index out of range"
    assert int(tf.reduce_min(index)) >= 0, "negative index"

    counts = tf.math.bincount(index, minlength=num_reactions)
    empirical = counts.numpy() / num_draws
    expected = (propensity_row / tf.reduce_sum(propensity_row)).numpy()[0]

    zero_mask = expected == 0.0
    assert (empirical[zero_mask] == 0.0).all(), (
        "zero-propensity reaction was selected"
    )
    max_error = float(np.max(np.abs(empirical - expected)))
    tolerance = 4.0 / np.sqrt(num_draws)
    assert max_error < tolerance, (
        f"sampler deviates from prop/a0: max |error| = {max_error:.5f} "
        f"> {tolerance:.5f}"
    )
    print(f"  verify_ssa_sampler: OK (max |error| = {max_error:.5f})")


def verify_ssa_gradient_channels():
    """Pin down which SSA channel carries a gradient and which does not.

    Asserts the two facts that make the forward-only design mandatory:
      * d(states)/dk == 0  -- the reaction choice is a hard integer index;
      * d(times)/dk  != 0  -- the waiting time is reparameterised in a0.

    The second is the trap: a GradientTape over the SSA returns a finite,
    plausible, WRONG gradient for any loss that reads the clock, rather than
    failing.
    """
    log_k = tf.Variable(tf.math.log(tf.constant(CLOCK_K_TRUE, tf.float32)))
    initial = tf.constant([[1., 0., 1., 0., 0., 0., 0., 0., 0.]], tf.float32)
    seed = tf.constant([1, 2], tf.int32)

    with tf.GradientTape() as tape:
        states, _ = clock_simulate_ssa(
            initial, tf.constant(8, tf.int32), tf.exp(log_k), CLOCK_S, seed
        )
        state_objective = tf.reduce_sum(states)
    state_gradient = tape.gradient(state_objective, log_k)
    # TensorFlow materialises structural zeros rather than None through
    # while_loop backprop, so accept either but require exact zero.
    if state_gradient is not None:
        leaked = float(tf.reduce_max(tf.abs(state_gradient)))
        assert leaked == 0.0, (
            f"the reaction-selection path leaked a gradient; max |g| = {leaked}"
        )

    with tf.GradientTape() as tape:
        _, times = clock_simulate_ssa(
            initial, tf.constant(8, tf.int32), tf.exp(log_k), CLOCK_S, seed
        )
        time_objective = tf.reduce_sum(times)
    time_gradient = tape.gradient(time_objective, log_k)
    assert time_gradient is not None, "the waiting-time path lost its gradient"
    time_magnitude = float(tf.reduce_max(tf.abs(time_gradient)))
    assert time_magnitude > 0.0, (
        "the waiting-time path lost its gradient; the partial-gradient hazard "
        "documented in make_clock_ssa_forward no longer applies and that "
        "docstring must be corrected"
    )

    print(
        "  verify_ssa_gradient_channels: OK "
        f"(d states/dk == 0, d times/dk = {time_magnitude:.3e} != 0)"
    )




def verify_pst_direct_backward_matches():
    """Verify optimized affine custom VJP equals the standard PST VJP."""
    prop_ref = tf.constant([[3.0, 1.0, 6.0, 10.0]], tf.float32)
    prop_opt = tf.constant([[3.0, 1.0, 6.0, 10.0]], tf.float32)

    S_test = tf.constant(
        [
            [1.0, 0.0, -1.0],
            [0.0, 1.0,  1.0],
            [-1.0, 0.0, 2.0],
            [0.0, -1.0, 0.5],
        ],
        tf.float32,
    )
    probe = tf.constant([[0.7, -1.3, 2.1]], tf.float32)

    with tf.GradientTape() as tape_ref:
        tape_ref.watch(prop_ref)
        pi_ref = tf.nn.softmax(tf.math.log(prop_ref), axis=-1)
        delta_ref = tf.matmul(pi_ref, S_test)
        scalar_ref = tf.reduce_sum(delta_ref * probe)
    grad_ref = tape_ref.gradient(scalar_ref, prop_ref)

    with tf.GradientTape() as tape_opt:
        tape_opt.watch(prop_opt)
        a0 = tf.reduce_sum(prop_opt, axis=-1, keepdims=True)
        delta_opt = pst_ssa_delta_from_uniform(
            prop_opt,
            a0,
            S_test,
            tf.constant([[0.37]], tf.float32),
        )
        scalar_opt = tf.reduce_sum(delta_opt * probe)
    grad_opt = tape_opt.gradient(scalar_opt, prop_opt)

    np.testing.assert_allclose(
        grad_ref.numpy(),
        grad_opt.numpy(),
        rtol=2e-6,
        atol=2e-7,
        err_msg=(
            "Optimized PST-SSA affine custom VJP does not match "
            "standard PST d/d(a)"
        ),
    )

    print(
        "  verify_pst_direct_backward_matches: OK "
        "(optimized affine VJP == standard PST d/d(a))"
    )




def verify_xla_hard_sampler_matches_unfused():
    """Bitwise-equivalent hard SSA jump for identical propensities and U."""
    print("\nPST-SSA XLA hard-sampler equivalence check")
    print("------------------------------------------")

    prop = tf.constant(
        [
            [3.0, 1.0, 6.0, 10.0],
            [2.0, 5.0, 1.0,  2.0],
            [0.0, 4.0, 0.0,  6.0],
        ],
        tf.float32,
    )
    S_test = tf.constant(
        [
            [1.0, 0.0, -1.0],
            [0.0, 1.0,  1.0],
            [-1.0, 0.0, 2.0],
            [0.0, -1.0, 0.5],
        ],
        tf.float32,
    )
    u = tf.constant([[0.13], [0.71], [0.39]], tf.float32)

    unfused = _hard_ssa_delta_core(prop, S_test, u)
    fused = hard_ssa_delta_xla(prop, S_test, u)

    tf.debugging.assert_equal(unfused, fused)

    print("  hard SSA delta: exact match")
    print("  XLA hard-sampler call executed successfully.")


def verify_xla_pst_vjp_matches_unfused():
    """XLA and unfused PST-SSA variants must expose the same custom VJP."""
    print("\nPST-SSA XLA custom-VJP equivalence check")
    print("----------------------------------------")

    prop_a = tf.constant([[3.0, 1.0, 6.0, 10.0]], tf.float32)
    prop_b = tf.constant([[3.0, 1.0, 6.0, 10.0]], tf.float32)

    S_test = tf.constant(
        [
            [1.0, 0.0, -1.0],
            [0.0, 1.0,  1.0],
            [-1.0, 0.0, 2.0],
            [0.0, -1.0, 0.5],
        ],
        tf.float32,
    )
    u = tf.constant([[0.37]], tf.float32)
    probe = tf.constant([[0.7, -1.3, 2.1]], tf.float32)

    with tf.GradientTape() as tape_a:
        tape_a.watch(prop_a)
        a0 = tf.reduce_sum(prop_a, axis=-1, keepdims=True)
        delta_a = pst_ssa_delta_from_uniform(
            prop_a, a0, S_test, u
        )
        scalar_a = tf.reduce_sum(delta_a * probe)
    grad_a = tape_a.gradient(scalar_a, prop_a)

    with tf.GradientTape() as tape_b:
        tape_b.watch(prop_b)
        a0 = tf.reduce_sum(prop_b, axis=-1, keepdims=True)
        delta_b = pst_ssa_delta_xla_from_uniform(
            prop_b, a0, S_test, u
        )
        scalar_b = tf.reduce_sum(delta_b * probe)
    grad_b = tape_b.gradient(scalar_b, prop_b)

    np.testing.assert_allclose(
        grad_a.numpy(),
        grad_b.numpy(),
        rtol=1e-6,
        atol=1e-7,
    )

    print("  custom PST VJP: exact numerical match within tolerance.")


def verify_streamlined_pst_ssa_estimator_graph():
    """Confirm optimized PST-SSA avoids unnecessary reaction-selection ops."""
    print("\nOptimized PST-SSA affine-op graph check")
    print("---------------------------------------")

    prop = tf.constant(
        [[3.0, 1.0, 6.0, 10.0],
         [2.0, 5.0, 1.0, 2.0]],
        tf.float32,
    )
    a0 = tf.reduce_sum(prop, axis=-1, keepdims=True)
    S_test = tf.constant(
        [
            [1.0, 0.0, -1.0],
            [0.0, 1.0,  1.0],
            [-1.0, 0.0, 2.0],
            [0.0, -1.0, 0.5],
        ],
        tf.float32,
    )
    u = tf.constant([[0.2], [0.8]], tf.float32)

    @tf.function(jit_compile=False)
    def pst_ssa_forward(a, total, S, draw):
        return pst_ssa_delta_from_uniform(a, total, S, draw)

    concrete = pst_ssa_forward.get_concrete_function(prop, a0, S_test, u)
    counts = graph_op_counts(concrete)

    forbidden = {
        "Softmax": counts.get("Softmax", 0),
        "Log": counts.get("Log", 0),
        "OneHot": counts.get("OneHot", 0),
    }

    for name, count in forbidden.items():
        print(f"  {name:<8} ops: {count}")
    print(f"  Cumsum   ops: {counts.get('Cumsum', 0)}")
    print(f"  GatherV2 ops: {counts.get('GatherV2', 0)}")

    if any(forbidden.values()):
        raise AssertionError(
            "Optimized PST-SSA unexpectedly contains forbidden forward ops: "
            f"{forbidden}"
        )

    print(
        "  optimized forward confirmed: raw-propensity CDF + hard S[j] gather."
    )



def verify_optimized_pst_ssa_gradient_graph():
    """Inspect a forward+backward wrapper around the custom affine PST op."""
    print("\nOptimized PST-SSA forward+backward graph check")
    print("-----------------------------------------------")

    prop = tf.constant(
        [[3.0, 1.0, 6.0, 10.0],
         [2.0, 5.0, 1.0, 2.0]],
        tf.float32,
    )
    S_test = tf.constant(
        [
            [1.0, 0.0, -1.0],
            [0.0, 1.0,  1.0],
            [-1.0, 0.0, 2.0],
            [0.0, -1.0, 0.5],
        ],
        tf.float32,
    )
    u = tf.constant([[0.2], [0.8]], tf.float32)
    probe = tf.constant(
        [[0.7, -1.3, 2.1], [1.1, 0.4, -0.8]],
        tf.float32,
    )

    @tf.function(jit_compile=False)
    def value_and_grad(a, S, draw, adjoint):
        with tf.GradientTape() as tape:
            tape.watch(a)
            total = tf.reduce_sum(a, axis=-1, keepdims=True)
            delta = pst_ssa_delta_from_uniform(a, total, S, draw)
            scalar = tf.reduce_sum(delta * adjoint)
        return scalar, tape.gradient(scalar, a)

    concrete = value_and_grad.get_concrete_function(
        prop, S_test, u, probe
    )
    counts = graph_op_counts(concrete)

    print(f"  Softmax ops: {counts.get('Softmax', 0)}")
    print(f"  Log ops:     {counts.get('Log', 0)}")
    print(f"  Cumsum ops:  {counts.get('Cumsum', 0)}")
    print(f"  MatMul ops:  {counts.get('MatMul', 0)}")

    if counts.get("Softmax", 0) or counts.get("Log", 0):
        raise AssertionError(
            "Optimized PST-SSA gradient graph contains Softmax/Log."
        )

    print(
        "  custom analytic propensity-space VJP is active; "
        "the hard CDF sampler is forward-only."
    )


def run_ssa_diagnostics():
    """SSA/direct-method correctness checks; never inside timed intervals."""
    print("\nSSA diagnostics")
    print("---------------")
    verify_ssa_sampler(num_draws=SSA_DIAGNOSTIC_DRAWS)
    verify_ssa_gradient_channels()
    if INCLUDE_PST_DIRECT:
        verify_streamlined_pst_ssa_estimator_graph()
        verify_optimized_pst_ssa_gradient_graph()
        verify_pst_direct_backward_matches()
    if INCLUDE_PST_XLA:
        verify_xla_hard_sampler_matches_unfused()
        verify_xla_pst_vjp_matches_unfused()


# =============================================================================
# COST-PARITY VERIFICATION
# =============================================================================
#
# Timing alone cannot distinguish an estimator-level cost difference from an
# implementation asymmetry in the surrounding loop.  These checks trace the
# two reaction-selection rules and compare their op-type counts directly, so
# the cost claim is verified structurally before any clock is read.

def graph_op_counts(concrete_fn):
    """Op-type histogram of a traced graph, including nested function bodies.

    A tf.while_loop places its body in a separate FunctionDef, so counting
    only the top-level nodes would miss the per-step work entirely.  The
    serialized GraphDef carries those bodies in its `library` field.
    """
    graph_def = concrete_fn.graph.as_graph_def()
    counts = {}
    for node in graph_def.node:
        counts[node.op] = counts.get(node.op, 0) + 1
    for func in graph_def.library.function:
        for node in func.node_def:
            counts[node.op] = counts.get(node.op, 0) + 1
    return counts


def diff_op_counts(counts_a, counts_b):
    """Op types whose counts differ, as {op_type: (count_a, count_b)}."""
    keys = set(counts_a) | set(counts_b)
    return {
        k: (counts_a.get(k, 0), counts_b.get(k, 0))
        for k in sorted(keys)
        if counts_a.get(k, 0) != counts_b.get(k, 0)
    }



def check_estimator_parity(num_rows=1024, num_reactions=16, tolerance=6):
    """Structural check for the matched PST-GM / GSST-GM estimator pair.

    Op counts are used only to catch accidental graph asymmetry. They are not
    interpreted as runtime cost.
    """
    print("\nMatched-sampler estimator graph structure")
    print("----------------------------------------")

    logits = tf.random.uniform(
        [num_rows, num_reactions], -3.0, 3.0, tf.float32
    )
    seed = tf.constant([1, 2], tf.int32)
    tau = tf.constant(3e-6, tf.float32)

    @tf.function(jit_compile=False)
    def pst_fn(x, s):
        return clock_pst_gumbel_estimator(x, s)

    @tf.function(jit_compile=False)
    def gsst_fn(x, t, s):
        return clock_gsst_gumbel_estimator(x, t, s)

    pst_counts = graph_op_counts(
        pst_fn.get_concrete_function(logits, seed)
    )
    gsst_counts = graph_op_counts(
        gsst_fn.get_concrete_function(logits, tau, seed)
    )

    n_pst = sum(pst_counts.values())
    n_gsst = sum(gsst_counts.values())
    print(f"  {PST_GM_NAME:<8} total ops: {n_pst}")
    print(f"  {GSST_GM_NAME:<8} total ops: {n_gsst}")

    differences = diff_op_counts(pst_counts, gsst_counts)
    if differences:
        print(f"  differing op types ({PST_GM_NAME}, {GSST_GM_NAME}):")
        for op_type, (a, b) in differences.items():
            print(f"    {op_type:<24} {a:>4}  {b:>4}")
    else:
        print("  op-type histograms are identical")

    if abs(n_gsst - n_pst) > tolerance:
        raise AssertionError(
            "Matched-sampler estimator graphs differ by more than expected. "
            f"{PST_GM_NAME}={n_pst}, {GSST_GM_NAME}={n_gsst}; "
            f"differences={differences}"
        )

    print("  structural path check passed; timing determines cost.")
    return pst_counts, gsst_counts



def check_clock_graph_parity(pst_gm_step, gsst_gm_step):
    """Structural check of complete matched-sampler clock training graphs.

    This checks implementation-path similarity only; op counts are not a
    performance measurement.
    """
    print("\nMatched-sampler oscillator training-step graph structure")
    print("--------------------------------------------------------")

    pst_counts = graph_op_counts(pst_gm_step.get_concrete_function())
    gsst_counts = graph_op_counts(gsst_gm_step.get_concrete_function())

    n_pst = sum(pst_counts.values())
    n_gsst = sum(gsst_counts.values())
    print(f"  {PST_GM_NAME:<8} total ops: {n_pst}")
    print(f"  {GSST_GM_NAME:<8} total ops: {n_gsst}")

    differences = diff_op_counts(pst_counts, gsst_counts)
    if differences:
        print(f"  differing op types ({PST_GM_NAME}, {GSST_GM_NAME}):")
        for op_type, (a, b) in differences.items():
            print(f"    {op_type:<24} {a:>4}  {b:>4}")
    else:
        print("  op-type histograms are identical")
    return pst_counts, gsst_counts


# =============================================================================
# PAIRED TIMING STATISTICS
# =============================================================================

def paired_timing_stats(model, num_name, num_samples, den_name, den_samples,
                        n_boot=20000, seed=0):
    """Paired summary of the per-repeat timings for one pair of methods.

    `time_paired` alternates execution order within a repeat but stores by
    method, so entry r of each list comes from the same repeat and the lists
    are index-aligned. Paired bootstrap resampling preserves this alignment
    when estimating uncertainty in the ratio of median times.
    """
    numerator_samples = num_samples
    denominator_samples = den_samples
    if len(numerator_samples) != len(denominator_samples):
        raise ValueError(
            f"{model} {num_name}/{den_name}: unpaired timing samples, "
            f"{len(numerator_samples)} vs {len(denominator_samples)}"
        )
    n = len(numerator_samples)
    if n < 2:
        raise ValueError(f"{model}: paired statistics need >= 2 repeats, got {n}")

    diffs = [a - b for a, b in zip(numerator_samples, denominator_samples)]
    mean_diff = statistics.mean(diffs)
    sd_diff = statistics.stdev(diffs)
    if sd_diff <= 0.0:
        raise ValueError(
            f"{model}: zero variance in paired differences. Identical timings "
            "on every repeat means the timer is not resolving the calls."
        )
    t_stat = mean_diff / (sd_diff / math.sqrt(n))
    numerator_slower = sum(1 for d in diffs if d > 0.0)

    denominator_median = statistics.median(denominator_samples)
    if denominator_median <= 0.0:
        raise ValueError(
            f"{model}: non-positive {den_name} median {denominator_median}"
        )
    median_ratio = (
        statistics.median(numerator_samples) / denominator_median
    )

    rng = random.Random(seed)
    ratios = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        den = statistics.median([denominator_samples[i] for i in idx])
        if den <= 0.0:
            raise ValueError(f"{model}: non-positive median in bootstrap resample")
        ratios.append(statistics.median([numerator_samples[i] for i in idx]) / den)
    ratios.sort()
    lo = ratios[int(0.025 * n_boot)]
    hi = ratios[int(0.975 * n_boot) - 1]
    assert lo <= hi, f"{model}: bootstrap interval inverted ({lo}, {hi})"

    return PairedStats(
        model=model,
        numerator=num_name,
        denominator=den_name,
        repeats=n,
        median_ratio=median_ratio,
        ratio_ci_low=lo,
        ratio_ci_high=hi,
        mean_diff_ms=mean_diff,
        sd_diff_ms=sd_diff,
        t_statistic=t_stat,
        degrees_of_freedom=n - 1,
        numerator_slower_count=numerator_slower,
    )


# Paired comparisons answer distinct questions:
#   PST-SSA-XLA/PST-SSA: direct measurement of hard-sampler XLA fusion.
#   PST-SSA-XLA/PST-GM: fused direct SSA vs explicit Gumbel-Max, same PST rule.
#   PST-GM/GSST-GM: backward-rule cost with hard Gumbel-Max sampler matched.
#   PST-cat/GSST-GM and PST-SSA/GSST-GM: end-to-end comparisons.
# SSA-fwd is deliberately absent because it is not a training step.
PAIRED_COMPARISONS = [
    # Hard-sampler implementation comparison: estimator and direct-method
    # semantics are identical; only the detached hard sampler is XLA-fused.
    (PST_XLA_NAME, PST_DIRECT_NAME),

    # Compare the XLA direct sampler against the other PST implementations.
    (PST_XLA_NAME, PST_CAT_NAME),
    (PST_XLA_NAME, PST_GM_NAME),

    # Existing controls.
    (PST_CAT_NAME, PST_DIRECT_NAME),
    (PST_CAT_NAME, PST_GM_NAME),
    (PST_DIRECT_NAME, PST_GM_NAME),

    # Same explicit Gumbel-Max hard sampler, different backward rules.
    (PST_GM_NAME, GSST_GM_NAME),

    # End-to-end comparisons against GS-ST.
    (PST_CAT_NAME, GSST_GM_NAME),
    (PST_DIRECT_NAME, GSST_GM_NAME),
    (PST_XLA_NAME, GSST_GM_NAME),
]



def paired_stats_for(results):
    """Paired statistics for every configured comparison present in results."""
    stats = []
    for model in sorted({r.model for r in results}):
        group = {r.method: r for r in results if r.model == model}
        required = {PST_CAT_NAME, PST_GM_NAME, GSST_GM_NAME}
        if INCLUDE_PST_DIRECT:
            required.add(PST_DIRECT_NAME)
        if INCLUDE_PST_XLA:
            required.add(PST_XLA_NAME)
        if not required.issubset(group):
            raise ValueError(
                f"{model}: expected {sorted(required)}, got {sorted(group)}"
            )
        for num_name, den_name in PAIRED_COMPARISONS:
            if num_name not in group or den_name not in group:
                continue
            stats.append(
                paired_timing_stats(
                    model,
                    num_name,
                    group[num_name].samples_ms,
                    den_name,
                    group[den_name].samples_ms,
                )
            )
    return stats


# =============================================================================
# REPORTING
# =============================================================================

def print_results(results):
    print("\nTiming results")
    print("--------------")
    header = (
        f"{'model':<8} {'method':<9} {'median ms':>12} "
        f"{'mean±sd ms':>20} {'ns/event-eq':>14} {'M events/s':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.model:<8} {r.method:<9} {r.median_ms:12.3f} "
            f"{r.mean_ms:9.3f}±{r.std_ms:<9.3f} "
            f"{r.ns_per_trajectory_event:14.3f} "
            f"{r.million_trajectory_events_per_s:12.3f}"
        )

    print("\nPaired per-step cost, training steps only")
    print("----------------------------------------")
    print(
        "Ratio is numerator/denominator; below 1 favours the numerator.  The\n"
        "bootstrap interval is over paired repeats; an interval containing 1.0\n"
        "means the two are not separated at this repeat count."
    )
    for stats in paired_stats_for(results):
        label = f"{stats.model} {stats.numerator}/{stats.denominator}"
        print(
            f"\n{label}: median ratio {stats.median_ratio:.4f}x "
            f"[95% CI {stats.ratio_ci_low:.3f}-{stats.ratio_ci_high:.3f}], "
            f"n={stats.repeats}"
        )
        print(
            f"  paired diff {stats.mean_diff_ms:+.2f} ms "
            f"(sd {stats.sd_diff_ms:.2f}), t={stats.t_statistic:+.2f} "
            f"on {stats.degrees_of_freedom} df, {stats.numerator} slower in "
            f"{stats.numerator_slower_count}/{stats.repeats} repeats"
        )
        if stats.ratio_ci_low <= 1.0 <= stats.ratio_ci_high:
            print("  -> consistent with equal per-step cost.")
        else:
            slower = (
                stats.numerator if stats.median_ratio > 1.0
                else stats.denominator
            )
            print(
                f"  -> {slower} is measurably more expensive per step. Check "
                "graph parity above before\n     reading this as an "
                "estimator-level rather than an implementation difference."
            )

    ssa_models = sorted({r.model for r in results if r.method == SSA_METHOD_NAME})
    if ssa_models:
        print("\nDifferentiable-step cost relative to the exact SSA forward")
        print("---------------------------------------------------------")
        print(
            f"'{SSA_METHOD_NAME}' is forward simulation + loss ONLY: no tape, no\n"
            "backward pass, no optimizer update.  It is a lower bound on the\n"
            "step, not a comparable training step, so these are ratios without\n"
            "paired statistics."
        )
        for model in ssa_models:
            group = {r.method: r for r in results if r.model == model}
            base = group[SSA_METHOD_NAME].median_ms
            if base <= 0.0:
                raise ValueError(f"{model}: non-positive SSA forward median {base}")
            for m in (
                PST_CAT_NAME,
                PST_DIRECT_NAME,
                PST_XLA_NAME,
                PST_GM_NAME,
                GSST_GM_NAME,
            ):
                if m in group:
                    print(
                        f"  {model:<6} {m:<8}: "
                        f"{group[m].median_ms / base:6.3f}x SSA forward"
                    )


def save_results(results):
    import csv

    fields = [
        "model", "method", "trajectories", "ssa_steps", "trajectory_events",
        "repeats", "median_ms", "mean_ms", "std_ms", "min_ms", "max_ms",
        "ns_per_trajectory_event", "million_trajectory_events_per_s",
    ]
    with OUTPUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for result in results:
            row = asdict(result)
            row.pop("samples_ms")
            w.writerow(row)

    payload = {
        "environment": environment_metadata(),
        "configuration": {
                "benchmark_profile": BENCHMARK_PROFILE,
            "benchmark_device": BENCHMARK_DEVICE,
            "gpu_id_requested": GPU_ID,
            "cpu_threads_requested": CPU_THREADS,
            "jit_compile_training": False,
            "reference_jit_compile": REFERENCE_JIT_COMPILE,
            "ion_readout_jit_compile": ION_READOUT_JIT,
            "warmup": WARMUP,
            "repeats": REPEATS,
            "ion_M_sims": ION_M_SIMS,
            "ion_steps": ION_NUM_STEPS,
            "ion_timing_epoch": ION_TIMING_EPOCH,
            "clock_batch_size": CLOCK_BATCH_SIZE,
            "clock_M_train": CLOCK_M_TRAIN,
            "clock_steps": CLOCK_SEGMENT_STEPS,
            "clock_reference_file": CLOCK_REFERENCE_FILE.name,
            "clock_reference_steps": CLOCK_REFERENCE_STEPS,
            "clock_burn_in": CLOCK_BURN_IN,
            "ssa_diagnostic_draws": SSA_DIAGNOSTIC_DRAWS,
            "ion_data_file": ION_DATA_FILE.name,
            "clock_batch_selection_timed": False,
            "optimizer_update_timed": True,
            "primary_methods": [
                PST_CAT_NAME,
                PST_DIRECT_NAME,
                PST_XLA_NAME,
                PST_GM_NAME,
                GSST_GM_NAME,
            ],
            "include_ssa_baseline": INCLUDE_SSA_BASELINE,
            "include_pst_direct": INCLUDE_PST_DIRECT,
            "include_pst_xla": INCLUDE_PST_XLA,
            "ssa_onehot_state_update": SSA_ONEHOT_STATE_UPDATE,
            "ssa_debug_checks": SSA_DEBUG_CHECKS,
        },
        "results": [asdict(r) for r in results],
        "paired_stats": [asdict(s) for s in paired_stats_for(results)],
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {OUTPUT_CSV}")
    print(f"Saved {OUTPUT_JSON.name}")


def run_benchmark():
    print_environment()
    print("\nBenchmark workload")
    print("------------------")
    print(f"profile: {BENCHMARK_PROFILE}")
    print(f"ion:   {ION_M_SIMS:,} trajectories x {ION_NUM_STEPS} event slots")
    print(
        f"clock: {CLOCK_BATCH_SIZE:,} starts x {CLOCK_M_TRAIN} trajectories/start "
        f"x {CLOCK_SEGMENT_STEPS} events"
    )
    print(f"warm-up calls/method: {WARMUP}; timed repeats/method: {REPEATS}")

    check_estimator_parity()
    verify_clock_matched_gumbel_forward()

    if INCLUDE_SSA_BASELINE or INCLUDE_PST_DIRECT or INCLUDE_PST_XLA:
        run_ssa_diagnostics()

    results = []

    if RUN_ION:
        results.extend(benchmark_ion())

    if RUN_CLOCK:
        results.extend(benchmark_clock())

    print_results(results)
    save_results(results)
    return results

if __name__ == "__main__":
    run_benchmark()

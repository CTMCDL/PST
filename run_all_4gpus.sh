#!/usr/bin/env bash
set -euo pipefail

# Activate the PST environment before launching this script.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

CUDA_VISIBLE_DEVICES=0,1,2,3 TF_CPP_MIN_LOG_LEVEL=2 python - <<'PY'
import tensorflow as tf

if not tf.test.is_built_with_cuda():
    raise SystemExit("The active Python environment does not have a CUDA-enabled TensorFlow build.")
gpus = tf.config.list_physical_devices("GPU")
if len(gpus) != 4:
    raise SystemExit(f"This launcher requires GPUs 0–3; TensorFlow detected {len(gpus)} of them.")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_soft_device_placement(False)
for index in range(4):
    with tf.device(f"/GPU:{index}"):
        matrix = tf.ones((16, 16), dtype=tf.float32)
        total = tf.reduce_sum(tf.linalg.matmul(matrix, matrix)).numpy()
    if total != 4096.0:
        raise SystemExit(f"GPU {index} failed the TensorFlow matrix-product check.")
print("TensorFlow completed a matrix product on each of the four selected GPUs.")
PY

# Bind notebook execution to the currently activated Python environment.
python -m ipykernel install --user --name pst --display-name "PST (Python 3.11)"

log_dir="${PST_LOG_DIR:-logs/4gpu_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
mkdir -p "$log_dir"
printf 'Logs: %s\n' "$log_dir"

run_nb() (
    gpu="$1"
    notebook="$2"
    printf 'Starting %s on GPU %s\n' "$notebook" "$gpu"
    cd -- "$(dirname -- "$notebook")"
    CUDA_VISIBLE_DEVICES="$gpu" PST_BENCH_GPU_ID="$gpu" \
        python -m jupyter nbconvert --to notebook --execute \
        "$(basename -- "$notebook")" --inplace \
        --ExecutePreprocessor.kernel_name=pst \
        --ExecutePreprocessor.timeout=-1
)

(
    run_nb 0 MNIST/MNIST-PST.ipynb &&
    run_nb 0 Dimerization/Dimerization-PST.ipynb &&
    run_nb 0 IonChannel/IonChannel-PST.ipynb
) >"$log_dir/gpu0.log" 2>&1 &
p0=$!

(
    run_nb 1 Oscillator/Oscillator-run.ipynb &&
    run_nb 1 Oscillator/Oscillator-analysis.ipynb
) >"$log_dir/gpu1.log" 2>&1 &
p1=$!

(
    run_nb 2 RepressilatorPST/RepressilatorPST.ipynb &&
    run_nb 2 IonChannelMR/IonChannel_PST_GSST_multi_initialization.ipynb
) >"$log_dir/gpu2.log" 2>&1 &
p2=$!

run_nb 3 RepressilatorBurgerLoss/RepressilatorPST_BL.ipynb \
    >"$log_dir/gpu3.log" 2>&1 &
p3=$!

pids=("$p0" "$p1" "$p2" "$p3")
status=0
for gpu in 0 1 2 3; do
    if wait "${pids[$gpu]}"; then
        printf 'GPU %s queue completed.\n' "$gpu"
    else
        exit_code=$?
        printf 'GPU %s queue failed (exit %s); see %s/gpu%s.log\n' \
            "$gpu" "$exit_code" "$log_dir" "$gpu" >&2
        status=1
    fi
done
if (( status != 0 )); then
    exit "$status"
fi

# Keep benchmark measurements separate from the training workloads.
# These two notebooks also share a reference-trajectory file.
printf 'Running the coherent benchmark on GPU 0.\n'
run_nb 0 Timing/PST_GSST_coherent_GPU_CPU_benchmark.ipynb \
    >"$log_dir/timing_coherent.log" 2>&1
printf 'Running the sampler benchmark on GPU 0.\n'
run_nb 0 Timing/PST_GSST_benchmark_PST_SSA_XLA_sampler.ipynb \
    >"$log_dir/timing_sampler.log" 2>&1
printf 'All notebooks completed. Logs: %s\n' "$log_dir"

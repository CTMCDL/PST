# Software environment

The notebooks target **Linux x86_64 with NVIDIA GPUs**. `environment.yml` creates the `pst` Conda environment with Python 3.11.15 and pip 26.0.1, then installs the pinned dependencies in `requirements.txt`. TensorFlow's `and-cuda` extra supplies the NVIDIA libraries; the host GPU driver is installed separately.

## Dependencies and reference system

| Component | Version or specification |
|---|---|
| Operating system | Ubuntu 22.04.5 LTS, x86_64 |
| Python | 3.11.15 |
| TensorFlow | 2.20.0, CUDA-enabled build |
| Keras | 3.15.1 |
| NumPy / SciPy | 2.4.6 / 1.17.1 |
| pandas / Matplotlib | 3.0.2 / 3.10.9 |
| GPUs | 4 × NVIDIA RTX 6000 Ada Generation |
| Memory per GPU | 49,140 MiB, as reported by `nvidia-smi` |
| NVIDIA driver | 580.178.04 |
| TensorFlow build CUDA / cuDNN | 12.5.1 / 9 |
| NVIDIA CUDA runtime package | `nvidia-cuda-runtime-cu12==12.6.77` |
| NVIDIA cuDNN package | `nvidia-cudnn-cu12==9.10.2.21` |
| JupyterLab / ipykernel / nbconvert | 4.6.3 / 7.2.0 / 7.17.1 |

TensorFlow build versions identify its compilation environment; the NVIDIA package versions identify the libraries installed by pip. The dependency files specify the Python environment. The operating system and host driver are configured separately.

JupyterLab provides the notebook interface, `ipykernel` provides the Python kernel, and `nbconvert` supports headless execution. The `openpyxl` dependency reads the ion-channel XLSX data. The full package list is in [`requirements.txt`](../requirements.txt).

## Installation

Install Conda and an NVIDIA driver compatible with the GPU and CUDA libraries. A separate system CUDA toolkit is not required by this setup. Run the following commands from the repository root:

```bash
nvidia-smi
conda env create -f environment.yml
conda activate pst
python -m pip check
python -m ipykernel install --user --name pst --display-name "PST (Python 3.11)"
python -m jupyter lab
```

Keep `requirements.txt` beside `environment.yml`, and create a dedicated environment for these dependencies. Package installation and the first MNIST download require internet access.

## Verify GPU access

Run this check in the activated `pst` environment:

```bash
python - <<'PY'
import tensorflow as tf

assert tf.test.is_built_with_cuda(), "TensorFlow is not a CUDA build"
gpus = tf.config.list_physical_devices("GPU")
assert gpus, "No NVIDIA GPU is visible to TensorFlow"
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_soft_device_placement(False)
with tf.device("/GPU:0"):
    result = tf.linalg.matmul(tf.ones((64, 64)), tf.ones((64, 64)))
print("TensorFlow:", tf.__version__)
print("Visible GPUs:", len(gpus))
print("Operation device:", result.device)
print("Check value:", float(result[0, 0].numpy()))
assert float(result[0, 0].numpy()) == 64.0
PY
```

The check executes a matrix product on the GPU with CPU fallback disabled. The four-GPU launcher performs a similar check on each of GPUs 0–3. If TensorFlow cannot initialize the GPU, consult the official [TensorFlow pip installation guide](https://www.tensorflow.org/install/pip) for troubleshooting.

## Kernel and device selection

Select the `PST (Python 3.11)` kernel and run each notebook from its containing directory so that relative input and output paths resolve correctly.

Set `CUDA_VISIBLE_DEVICES` before importing TensorFlow. The notebooks respect an existing value and otherwise default to device `0` where GPU selection is configured. Restart the kernel after changing the selection. For example, to start JupyterLab with physical GPU 1 visible:

```bash
CUDA_VISIBLE_DEVICES=1 python -m jupyter lab
```

The coherent timing notebook also accepts `PST_BENCH_GPU_ID` for its worker processes. See [Running the notebooks](RUNNING.md) for benchmark settings and the four-GPU launcher.

## Reproducibility

Use the pinned environment and the documented experiment settings when reproducing the results. Record the hardware and software versions alongside new measurements. Identical seeds do not guarantee identical stochastic trajectories or timings across software versions, GPU models, or execution conditions.

Saved notebook outputs can be viewed without running the experiments. Headless execution with a separate output filename preserves them; execution with `--inplace`, including the four-GPU launcher, replaces them with the new outputs.

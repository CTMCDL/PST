# Data sources and related methods

## Patch-clamp recordings

The ion-channel experiments use the processed patch-clamp recording `HEK293_Cell01_40mV_100s_raw_DET_IDEl.xlsx`. A copy is provided in each of `IonChannel/`, `IonChannelMR/`, and `Timing/`.

- **Dataset:** Selimi, Z., Rougier, J.-S., Abriel, H., and Kucera, J. P. (2023), [Zenodo record 7817601](https://doi.org/10.5281/zenodo.7817601).
- **Associated study:** *The Journal of Physiology* **601**, 3847–3868 (2023), [doi:10.1113/JP284861](https://doi.org/10.1113/JP284861).
- **Processed data:** The [codeExactDL repository](https://github.com/CTMCDL/codeExactDL) distributes this spreadsheet under CC BY 4.0 and provides the dataset attribution.

## Repressilator task parameters

The 50-task repressilator benchmark uses target and initial parameter pairs from Burger, Kofler, Heinrich, and Gerland's [DifferentiableGillespie repository](https://github.com/gerland-group/DifferentiableGillespie). See [arXiv:2604.02121](https://arxiv.org/abs/2604.02121) for the associated work. The task array is provided as `params__truth_init.npy` in both repressilator directories.

The TensorFlow experiments use two objectives:

| Directory | Objective | Setting |
|---|---|---|
| `RepressilatorPST/` | Relative mean-squared error | `LOSS_MODE = "relative"` |
| `RepressilatorBurgerLoss/` | Log-of-ensemble-mean loss | `LOSS_MODE = "log"` |

In the relative-MSE experiment, cache filenames contain the legacy token `target_scaled_log`. The active objective is determined by `LOSS_MODE`; see [Reruns and reproducibility](docs/RUNNING.md#reruns-and-reproducibility).

## MNIST

The stochastic gene-regulatory classifier uses the MNIST handwritten-digit dataset through `tf.keras.datasets.mnist`. The loader downloads the training and test sets on first use and reuses the local Keras dataset cache thereafter. Training writes the model weights, history, and test-data files used for subsequent evaluation.

Please cite the original MNIST dataset and its authors when using these data.

## Related GS-ST work

The [codeExactDL repository](https://github.com/CTMCDL/codeExactDL) accompanies Vilar and Saiz's GS-ST work in *Advanced Science*, e76297 (2026). It provides related implementations and datasets, including the genetic-oscillator baseline used in the PST/GS-ST comparison.

## Licensing

A repository-wide code license has not been specified. Third-party data and code are subject to their respective upstream licenses; see the source repositories linked above.

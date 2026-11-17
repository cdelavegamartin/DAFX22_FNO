# Physical Modeling using Recurrent Neural Networks with Fast Convolutional Layers

This is the accompanying repository to the paper submission for DAFx20in22, Vienna, Austria by Julian D. Parker, Sebastian J. Schlecht, Rudolf Rabenstein, Maximilian Schäfer. The paper is currently in review.


## Abstract
Discrete-time modeling of acoustic, mechanical and electrical systems is a prominent topic in the musical signal processing literature. Such models are mostly derived by discretizing a mathematical model, given in terms of ordinary or partial differential equations, using established techniques. Recent work has applied the techniques of machine-learning to construct such models automatically from data for the case of systems which have lumped states described by scalar values, such as electrical circuits. In this work, we examine how similar techniques are able to construct models of systems which have spatially distributed rather than lumped states. We describe several novel recurrent neural network structures, and show how they can be thought of as an extension of modal techniques. As a proof of concept, we generate synthetic data for three physical systems and show that the proposed network structures can be trained with this data to reproduce the behavior of these systems. 

## Environment

The project uses [uv](https://docs.astral.sh/uv/); an extra selects the PyTorch backend:

```bash
uv sync --extra cpu     # CPU-only PyTorch
uv sync --extra cu126   # CUDA 12.6 PyTorch (NVIDIA driver R525 or newer)
uv run python train_2d_wave.py
```

`uv run` keeps the backend installed by the last `uv sync`.

## Datasets

`generate_dataset.py` simulates the datasets of the paper with the settings of the training scripts at commit `e08b4b0` and saves each as one raw `.npy` array (trajectories, time steps, x, [y,] channels), with a `.json` holding the settings, seed, channel names and training-split statistics:

```bash
uv run python generate_dataset.py string             # linear lossy string, (1024, 120, 65, 2)
uv run python generate_dataset.py nonlinear_string   # tension-modulated string, (1024, 120, 65, 2)
uv run python generate_dataset.py wave2d             # 2D wave equation, (1024, 34, 40, 38, 3)
```

Files go to `output/datasets/<system>-n_<variations>-seed_<seed>.npy` (`--variations`, `--seed`, `--output` change this).
The original scripts did not seed numpy, so their exact datasets cannot be regenerated; the solvers and the order of random draws are the same.

## Spectral truncation analysis

`analysis/spectral_truncation/` compares the spectral layers of FGRU/FRNN (`FourierConv2d`) and REF (`SpectralConv2d`).
`FourierConv2d` drops the negative-kx half of the 2D spectrum; see `analysis/spectral_truncation/report.md`.

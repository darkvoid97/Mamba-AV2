<div align="center">

# Mamba for Autonomous Driving Trajectory Prediction

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

A PyTorch script for training, inference, and evaluation of a **Multimodal Mamba Forecaster**. This model predicts the future trajectories of vehicles in autonomous driving scenarios. 

This project was developed as part of my Bachelor's Degree Thesis in Computer Engineering, titled: *"Future state prediction in autonomous driving through study and implementation of the Mamba architecture"*.

## About the Project

Trajectory prediction is a critical module in Autonomous Vehicles (AVs). While Transformer models have become the standard for this task due to their self-attention mechanisms, they suffer from quadratic computational complexity $O(N^2)$ relative to the sequence length. This makes them heavily resource-intensive for long historical sequences in real-time edge environments.

This project tackles this bottleneck by implementing **Selective State Space Models (SSMs)**, specifically the **Mamba** architecture. Mamba achieves linear time complexity $O(N)$ and bounded memory footprint while matching or exceeding the context-reasoning capabilities of Transformers.

### Key Features
* **Mamba-SSM Backbone:** Replaces traditional Self-Attention with Selective State Spaces for highly efficient, parallelizable training and fast auto-regressive inference.
* **Multimodal Forecasting:** Predicts $K=6$ possible future trajectories with associated confidence scores to account for the non-deterministic nature of driving.
* **Argoverse 2 Integration:** Built to natively process the Argoverse 2 Motion Forecasting dataset (50 historical timesteps -> 60 future timesteps at 10Hz).
* **Interactive Visualization:** Uses `plotly` to render historical paths, ground truth, and the model's multimodal predictions.

## Installation & Requirements

The model relies on PyTorch (therefore you also need Python 3.12+) and the official `mamba-ssm` library. Because Mamba uses custom CUDA kernels, an NVIDIA GPU is required (Note that this was tested on NVIDIA T4 via Google Colab).

```bash
# Using 'uv' is recommended for fast package resolution
pip install uv

# Install PyTorch (adjust CUDA version to your hardware if needed)
uv pip install torch==2.9.0 torchaudio==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128

# Install Causal Conv1D and Mamba SSM (ensure compatibility with your PyTorch/CUDA setup)
uv pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
uv pip install https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

## Dataset: Argoverse 2

This script is designed for the [Argoverse 2 Motion Forecasting Dataset](https://www.argoverse.org/av2.html). 
1. Download the Parquet files from the official Argoverse website.
2. Extract them locally or mount them via Google Drive if using Google Colab.
3. Update the `DATA_DIR` variable in the script to point to your extracted dataset path.

## Usage

The provided script contains all modules (Dataset, Model, Training Loop, Evaluation, Visualization) in a unified format, ideal for Jupyter Notebooks or Google Colab environments. If choosing to use Google Colab, note that the Argoverse 2 Dataset can also be placed into Google Drive's personal space, as the script can take care of mounting the Drive's disk and copying the dataset locally.

### Training
To train the model from scratch or resume training:
1. Set `DATA_DIR` to your dataset folder.
2. Adjust `BATCH_SIZE` and `EPOCHS` (Default: 64 batch size, 20 epochs).
3. Run the training cell. The script uses a **Winner-Takes-All (WTA) Loss** (combining Smooth L1 and Cross Entropy) to handle multimodal predictions safely. Checkpoints are saved automatically per epoch.

### Inference & Visualization
You can visually inspect the model's performance on random scenarios using the `visualize_prediction()` function. It will open an interactive Plotly graph showing:
* **Blue line:** The vehicle's history (past 5 seconds).
* **Green dashed line:** The Ground Truth future trajectory.
* **Red line:** The best-predicted trajectory by the Mamba model.
* **Yellow/Orange lines:** Alternative multimodal predictions with their respective probabilities.

### Evaluation
The script calculates standard autonomous driving metrics over a specified number of scenarios:
* **minADE (Minimum Average Displacement Error):** Average accuracy along the entire path.
* **minFDE (Minimum Final Displacement Error):** Accuracy at the final destination point.
* **Miss Rate (MR):** Percentage of predictions where the endpoint error is > 2.0 meters.

## Architecture Details

1. **Input Projection:** Raw features `[x, y, vx, vy]` are normalized, rotated relative to the current ego-vehicle heading, and projected into a higher-dimensional latent space ($D=256$).
2. **Mamba Encoder:** A stack of 6 Mamba blocks extracts sequential features. Thanks to the input-dependent selection mechanism, the network filters out sensor noise and memorizes critical dynamic events.
3. **Shared & Prediction Heads:** The final hidden state is processed by a shared MLP, which then splits into a Trajectory Head (predicting absolute coordinates for $K=6$ modes over 60 steps) and a Probability Head (predicting the likelihood of each mode).

## Limitations & Future Work

As per the academic nature of this project, this implementation has obvious limitations, starting from Google Colab's base plan usage time constraints.
There are also a few known behaviors:
* **High-Frequency Noise (Zig-Zag):** Because the network predicts *absolute spatial coordinates* rather than differential velocities (e.g., $\Delta X$, $\Delta Y$), the predicted trajectories sometimes exhibit slight zig-zag patterns. As this project was originally made just to have a proof of concept of the Mamba architecture's potential in autonomous driving, this implementation resulted to be more than enough. A future implementation predicting differentials velocities could yield considerably better results.
* **Smoothing:** Applying a post-processing mathematical filter (like a Moving Average or Savitzky-Golay filter) to the output coordinates significantly smooths the curve and improves the minADE metric.
* **Future integration:** Replacing absolute coordinate prediction with differential outputs and integrating HD Vector Maps as explicit constraints would also further reduce the minFDE.

## Author & License

MIT **© Giuseppe Tansella**

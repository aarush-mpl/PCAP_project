# Running Scalable Parallel K-Means on Google Colab

This guide explains how to run the Scalable Parallel K-Means (MPI + CUDA) project on **Google Colab** with hardware GPU acceleration.

---

## 🚀 Quick Start (1-Click Jupyter Notebook)

You can run the entire pipeline interactively through the included notebook [`PCAP_KMeans_Colab.ipynb`](./PCAP_KMeans_Colab.ipynb):

1. Go to [Google Colab](https://colab.research.google.com/).
2. Select the **GitHub** tab.
3. Enter repository URL: `https://github.com/aarush-mpl/PCAP_project`
4. Select branch: `colab`
5. Click **`PCAP_KMeans_Colab.ipynb`** to open.

---

## ⚙️ Step 1: Enable GPU Runtime

In your Colab notebook:
1. Click **Runtime** in the top menu bar.
2. Select **Change runtime type**.
3. Under **Hardware accelerator**, select **T4 GPU** (or any available GPU).
4. Click **Save**.

---

## 🛠️ Step 2: Running via Notebook or Terminal

### Option A: Interactive Step-by-Step (Recommended)
Step through each cell in [`PCAP_KMeans_Colab.ipynb`](./PCAP_KMeans_Colab.ipynb):
- **Cell 1**: GPU check (`nvidia-smi`, `nvcc`)
- **Cell 2**: Install OpenMPI runtime and development libraries (`openmpi-bin`, `libopenmpi-dev`)
- **Cell 3**: Build targets (`make clean && make all && make hybrid`)
- **Cell 4**: Generate synthetic dataset
- **Cell 5**: Run Serial CPU baseline
- **Cell 6**: Run Distributed MPI (CPU)
- **Cell 7**: Run Hybrid MPI + CUDA (executes directly on Colab's NVIDIA GPU!)
- **Cell 8**: Mathematical validation (`validate_correctness.py`)
- **Cell 9**: Speedup comparison table and visualization plot

### Option B: One-Click Bash Script
In a Colab code cell, run:
```bash
!git clone -b colab https://github.com/aarush-mpl/PCAP_project.git
%cd PCAP_project
!bash run_colab.sh
```

---

## 🔍 Key Colab Details & Differences

1. **True NVIDIA GPU Execution**: Unlike WSL (which often lacks GPU passthrough), Colab attaches a physical NVIDIA GPU (Tesla T4). The hybrid binary will run directly on the GPU without CPU fallback warnings.
2. **MPI Container Flags**: Colab runs processes as `root` with 2 virtual CPU cores. All `mpirun` commands in the Colab scripts include:
   - `--allow-run-as-root` (allows OpenMPI to execute in Colab's root container).
   - `--oversubscribe` (allows launching multi-rank tests regardless of vCPU core counts).
3. **Reproducibility**: All runs use identical seeds and `kmeans++` initialization to ensure 100% mathematical centroid agreement during validation.

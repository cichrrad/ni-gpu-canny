import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

# Load CSVs
cpu_df = pd.read_csv("canny_cpu.csv")
gpu_df = pd.read_csv("canny_gpu.csv")

# Phase columns
phases = ["GStime", "GBtime", "STtime", "NMS", "DTtime", "Hystime"]
phase_labels = {
    "GStime": "Grayscale",
    "GBtime": "Gaussian Blur",
    "STtime": "Sobel Filter",
    "NMS": "Non-Max Suppression",
    "DTtime": "Double Threshold",
    "Hystime": "Hysteresis"
}

# Pick a random file present in both CSVs
common_files = set(cpu_df["FILE"]).intersection(set(gpu_df["FILE"]))
sample_file = random.choice(list(common_files))
cpu_row = cpu_df[cpu_df["FILE"] == sample_file].iloc[0]
gpu_row = gpu_df[gpu_df["FILE"] == sample_file].iloc[0]

# Bar chart: time per phase with log scale
cpu_times = [cpu_row[p] for p in phases]
gpu_times = [gpu_row[p] for p in phases]
x = np.arange(len(phases))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, cpu_times, width, label='CPU')
plt.bar(x + width/2, gpu_times, width, label='GPU')
plt.xticks(x, [phase_labels[p] for p in phases], rotation=45)
plt.ylabel("Time [μs] (log scale)")
plt.yscale("log")
plt.title(f"Execution time per phase (CPU vs. GPU)\nFile: {sample_file}")
plt.legend()
plt.tight_layout()
plt.savefig("bar_one_image_comparison_log.pdf")
plt.close()

# Log-log plot: image size vs total time
cpu_df["arch"] = "CPU"
gpu_df["arch"] = "GPU"
merged = pd.concat([cpu_df[["SIZE", "TOTAL", "arch"]], gpu_df[["SIZE", "TOTAL", "arch"]]])

plt.figure(figsize=(8, 6))
for arch in ["CPU", "GPU"]:
    subset = merged[merged["arch"] == arch]
    plt.scatter(subset["SIZE"], subset["TOTAL"], label=arch, alpha=0.7)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Image size [pixels]")
plt.ylabel("Total time [μs]")
plt.title("Execution time vs. image size (log-log)")
plt.legend()
plt.tight_layout()
plt.savefig("size_vs_time_loglog.pdf")
plt.close()

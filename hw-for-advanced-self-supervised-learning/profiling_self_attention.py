import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
import psutil
import os


# Define the self-attention simulation
class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, d_k):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embedding_dim, d_k)
        self.key = nn.Linear(embedding_dim, d_k)
        self.value = nn.Linear(embedding_dim, d_k)
        self.d_k = d_k

    def forward(self, x_embedding):
        Q = self.query(x_embedding)
        K = self.key(x_embedding)
        V = self.value(x_embedding)

        # Scaled dot-product attention
        weights = torch.matmul(Q, K.transpose(-2, -1)) / self.d_k ** 0.5
        attention = torch.matmul(weights.softmax(dim=-1), V)
        return attention


# Utility function to measure memory usage
def measure_memory(device='cpu'):
    if device.startswith('cuda'):
        return torch.cuda.memory_allocated() / 1024 ** 2  # Memory in MB
    else:
        # Measure CPU memory usage with psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / 1024 ** 2  # Memory in MB


# Simulate self-attention and return FLOPS, memory, and wall clock time
def simulate_self_attention(input_length, embedding_dim, d_k, device):
    model = SelfAttention(embedding_dim, d_k).to(device)
    x_embedding = torch.rand(input_length, embedding_dim).to(device)

    # Measure wall clock time
    start_time = time.time()
    with torch.no_grad():
        output = model(x_embedding)
    wall_clock_time = time.time() - start_time

    # Measure memory usage
    memory_usage = measure_memory(device)
    print('measured memory usage: ', memory_usage)

    # Measure FLOPS with profiler
    with profile(activities=[ProfilerActivity.CPU if device == 'cpu' else ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        with record_function("self_attention"):
            model(x_embedding)
    flops = sum([e.cpu_time_total for e in prof.key_averages()]) / 1e9  # Convert to GFLOPs

    return flops, memory_usage, wall_clock_time


# Testing over different input lengths
def run_benchmark(device, num_experiments=50):
    embedding_dim = 16
    d_k = 8
    lengths = [5, 10, 50, 100, 500, 1000, 5000, 10000, 35000, 70000, 70001]

    flops_list = []
    memory_list = []
    time_list = []
    errors = {"flops": [], "memory": [], "time": []}

    for li, length in enumerate(lengths):
        flops_per_length = []
        memory_per_length = []
        time_per_length = []

        # Run multiple tests to calculate averages and error bars
        for _ in tqdm.tqdm(range(num_experiments)):
            flops, memory, wall_time = simulate_self_attention(length, embedding_dim, d_k, device)
            flops_per_length.append(flops)
            memory_per_length.append(memory)
            time_per_length.append(wall_time)

        # Compute averages and standard errors

        if li == 0 or li == len(lengths)-1: continue

        flops_list.append(np.mean(flops_per_length))
        memory_list.append(np.mean(memory_per_length))
        time_list.append(np.mean(time_per_length))

        errors["flops"].append(np.std(flops_per_length) / np.sqrt(len(flops_per_length)))
        errors["memory"].append(np.std(memory_per_length) / np.sqrt(len(memory_per_length)))
        errors["time"].append(np.std(time_per_length) / np.sqrt(len(time_per_length)))

    return lengths[1:-1], flops_list, memory_list, time_list, errors


def plot_combined_results(lengths, flops_cpu, memory_cpu, time_cpu, errors_cpu, flops_gpu, memory_gpu, time_gpu, errors_gpu):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot FLOPs
    axes[0, 0].errorbar(lengths, flops_cpu, yerr=errors_cpu['flops'], label='CPU', fmt='-o', color='c')
    axes[0, 0].set_title('GFLOPs Computational Complexity')
    axes[0, 0].set_xlabel('Input Length')
    axes[0, 0].set_ylabel('GFLOPs')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot Memory Usage
    axes[0, 1].errorbar(lengths, memory_cpu, yerr=errors_cpu['memory'], label='CPU', fmt='-o', color='c')
    axes[0, 1].set_title('Memory Usage')
    axes[0, 1].set_xlabel('Input Length')
    axes[0, 1].set_ylabel('Memory Usage (bytes)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot Wall Clock Time
    axes[0, 2].errorbar(lengths, time_cpu, yerr=errors_cpu['time'], label='CPU', fmt='-o', color='c')
    axes[0, 2].set_title('Wall Clock Time')
    axes[0, 2].set_xlabel('Input Length')
    axes[0, 2].set_ylabel('Time (seconds)')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # Adding blank plots or additional information for the second row
    axes[1, 0].errorbar(lengths, flops_gpu, yerr=errors_gpu['flops'], label='GPU', fmt='-o', color='r')
    axes[1, 0].set_title('GFLOPs Computational Complexity')
    axes[1, 0].set_xlabel('Input Length')
    axes[1, 0].set_ylabel('GFLOPs')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plo1 Memory Usage
    axes[1, 1].errorbar(lengths, memory_gpu, yerr=errors_gpu['memory'], label='GPU', fmt='-o', color='r')
    axes[1, 1].set_title('Memory Usage')
    axes[1, 1].set_xlabel('Input Length')
    axes[1, 1].set_ylabel('Memory Usage (bytes)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Plo1 Wall Clock Time
    axes[1, 2].errorbar(lengths, time_gpu, yerr=errors_gpu['time'], label='GPU', fmt='-o', color='r')
    axes[1, 2].set_title('Wall Clock Time')
    axes[1, 2].set_xlabel('Input Length')
    axes[1, 2].set_ylabel('Time (seconds)')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Running the benchmark for both CPU and GPU
    lengths_cpu, flops_cpu, memory_cpu, time_cpu, errors_cpu = run_benchmark('cpu', num_experiments=20)
    lengths_gpu, flops_gpu, memory_gpu, time_gpu, errors_gpu = run_benchmark('cuda:0', num_experiments=20)

    # Combine all subplots into one figure
    plot_combined_results(lengths_cpu, flops_cpu, memory_cpu, time_cpu, errors_cpu, flops_gpu, memory_gpu, time_gpu, errors_gpu)
"""
In this program, we will make a self-attention simulation to observe the FLOPs, memery usage and clock time by given differnet sequence lengthes and run the profiling program in GPU and CPU.
@Author: Marvin Gao(mgao40@jh.edu), also generating some code from ChatGPT
@Data: Sept-4-2024
"""
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
    lengths = [10, 50, 100, 500, 1000, 5000, 10000, 35000, 70000]

    flops_list = []
    memory_list = []
    time_list = []
    errors = {"flops": [], "memory": [], "time": []}

    for length in lengths:
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
        flops_list.append(np.mean(flops_per_length))
        memory_list.append(np.mean(memory_per_length))
        time_list.append(np.mean(time_per_length))

        errors["flops"].append(np.std(flops_per_length) / np.sqrt(len(flops_per_length)))
        errors["memory"].append(np.std(memory_per_length) / np.sqrt(len(memory_per_length)))
        errors["time"].append(np.std(time_per_length) / np.sqrt(len(time_per_length)))

    return lengths, flops_list, memory_list, time_list, errors



# Plotting
def plot_results(lengths, values, ylabel, title, errors=None, device='cpu'):
    plt.figure(figsize=(10, 6))
    color = 'c' if device == 'cpu' else 'r'
    plt.errorbar(lengths, values, yerr=errors, label=device, fmt='-o', color=color)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('Input Length')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    for d in ['cpu', 'cuda:0']:
        lengths, flops, memory, _time, errors = run_benchmark(d, num_experiments=20)

        plot_results(lengths, flops, "GFLOPs", "GFLOPs Computational Complexity", errors['flops'], d)
        plot_results(lengths, memory, "Memory Usage (bytes)", "Memory Usage", errors['memory'], d)
        plot_results(lengths, _time, "Wall Clock Time (seconds)", "Wall Clock Time", errors['time'], d)


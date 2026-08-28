"""E6 — deployment-cost measurement (params, FLOPs, latency, adaptation cost).

Quantifies the paper's lightweight/embedded claims:
  - trainable parameters (total) and BN-affine adaptation budget (176)
  - FLOPs per trial (forward pass), via torch.profiler or fvcore if available
  - single-trial and batched inference latency on CPU and GPU (median of N)
  - wall-clock cost of the 5-step IM-TTA adaptation over a target session
  - peak GPU memory during inference and adaptation

Usage:  python run_deployment_cost.py --out deployment_cost.json
"""
import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import UnifiedEEGNet
from train import tent_adapt


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    bn_affine = 0
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)) and m.affine:
            bn_affine += m.weight.numel() + m.bias.numel()
    return {'total_params': total, 'bn_affine_adapted': bn_affine}


def flops_per_trial(model, n_channels=22, n_time=1000):
    """MACs via thop if available, else torch.profiler with FLOPs."""
    try:
        from thop import profile
        x = torch.randn(1, n_channels, n_time)   # model unsqueezes channel dim internally
        macs, _ = profile(model, inputs=(x,), verbose=False)
        return {'flops_macs_per_trial': macs, 'method': 'thop'}
    except ImportError:
        pass
    # fallback: count via torch.profiler with FLOPs enabled
    try:
        from torch.profiler import profile as prof, ProfilerActivity
        x = torch.randn(1, n_channels, n_time)
        with prof(activities=[ProfilerActivity.CPU], with_flops=True) as p:
            model(x)
        flops = sum(e.flops for e in p.key_averages() if e.flops > 0)
        return {'flops_per_trial': flops, 'method': 'torch.profiler'}
    except Exception as e:  # noqa: BLE001
        return {'flops_per_trial': None, 'method': f'unavailable ({e})'}


def latency(model, x_batch, device, n_rep=50, warmup=10):
    model = model.to(device).eval()
    xb = x_batch.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(xb)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times = []
        for _ in range(n_rep):
            t0 = time.perf_counter()
            model(xb)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return {'median_ms': float(np.median(times)) * 1e3,
            'mean_ms': float(np.mean(times)) * 1e3,
            'batch': int(xb.shape[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='deployment_cost.json')
    ap.add_argument('--n_trials', type=int, default=576)   # one IV-2a session
    ap.add_argument('--tta_steps', type=int, default=5)
    args = ap.parse_args()

    device_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_cpu = torch.device('cpu')
    n_classes, n_channels, n_time = 4, 22, 1000

    model = UnifiedEEGNet(n_classes=n_classes, n_channels=n_channels)
    report = {'config': {'n_classes': n_classes, 'n_channels': n_channels,
                         'n_time': n_time, 'tta_steps': args.tta_steps,
                         'n_trials_target_session': args.n_trials}}

    report['params'] = count_params(model)
    report['flops'] = flops_per_trial(model.eval(), n_channels, n_time)

    X = np.random.randn(args.n_trials, n_channels, n_time).astype(np.float32)
    y = np.random.randint(0, n_classes, args.n_trials)

    # single-trial latency (deployment case: one trial arrives)
    x1 = torch.from_numpy(X[:1])                     # [1, C, T]
    report['latency_single_gpu'] = latency(model, x1, device_gpu)
    report['latency_single_cpu'] = latency(model, x1, device_cpu)

    # batched latency (64)
    xb = torch.from_numpy(X[:64])                    # [64, C, T]
    report['latency_batch64_gpu'] = latency(model, xb, device_gpu)
    report['latency_batch64_cpu'] = latency(model, xb, device_cpu)

    # per-trial throughput over the whole target session, GPU
    xs = torch.from_numpy(X)                         # [N, C, T]
    report['latency_session_gpu'] = latency(model, xs, device_gpu, n_rep=10, warmup=3)

    # adaptation cost: 5-step IM-TTA over the target session (transductive)
    from dataset import BCIDataset
    loader = DataLoader(BCIDataset(X, y, augment=False, use_sr=False),
                        batch_size=64, shuffle=False)
    if device_gpu.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    m = UnifiedEEGNet(n_classes=n_classes, n_channels=n_channels).to(device_gpu)
    t0 = time.perf_counter()
    adapted = tent_adapt(m, loader, device_gpu, steps=args.tta_steps, div_weight=1.0)
    adapt_s = time.perf_counter() - t0
    peak_mem = (torch.cuda.max_memory_allocated() / 1e6
                if device_gpu.type == 'cuda' else None)
    report['adaptation'] = {'steps': args.tta_steps, 'wall_clock_s': adapt_s,
                            'per_trial_ms': adapt_s * 1e3 / args.n_trials,
                            'peak_gpu_mem_mb': peak_mem}

    with open(args.out, 'w') as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import argparse
import json
import os
import pathlib
import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

DTYPE = torch.float16
BACKEND = "nccl"
BATCH_SIZE = 512
HIDDEN_SIZE = 2048
GRAPH_BLOCKS = 12
WARMUP_ITERS = 8
REPLAY_ITERS = 12
ALL_REDUCE_EVERY = 1
ALL_REDUCE_NUMEL = 32768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal distributed graph trace capture.")
    parser.add_argument("--output-dir", type=str, default="minimal_graph_trace_only_out")
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=str, default="29531")
    parser.add_argument("--seed", type=int, default=20260415)
    return parser.parse_args()


def init_distributed(rank: int, world_size: int, local_rank: int, args: argparse.Namespace) -> None:
    os.environ.setdefault("MASTER_ADDR", args.master_addr)
    os.environ.setdefault("MASTER_PORT", args.master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    dist.init_process_group(backend=BACKEND, rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)


def build_tensors(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "inp": torch.randn(BATCH_SIZE, HIDDEN_SIZE, device=device, dtype=DTYPE),
        "residual": torch.randn(BATCH_SIZE, HIDDEN_SIZE, device=device, dtype=DTYPE),
        "bias_scale": torch.randn(BATCH_SIZE, HIDDEN_SIZE, device=device, dtype=DTYPE),
        "weights": [torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device=device, dtype=DTYPE) for _ in range(GRAPH_BLOCKS)],
        "mix_weights": [torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device=device, dtype=DTYPE) for _ in range(GRAPH_BLOCKS)],
        "biases": [torch.randn(HIDDEN_SIZE, device=device, dtype=DTYPE) for _ in range(GRAPH_BLOCKS)],
        "mix_biases": [torch.randn(HIDDEN_SIZE, device=device, dtype=DTYPE) for _ in range(GRAPH_BLOCKS)],
        "scales": [torch.randn(HIDDEN_SIZE, device=device, dtype=DTYPE) for _ in range(GRAPH_BLOCKS)],
        "ln_weight": torch.randn(HIDDEN_SIZE, device=device, dtype=DTYPE),
        "ln_bias": torch.randn(HIDDEN_SIZE, device=device, dtype=DTYPE),
        "sync": torch.ones(ALL_REDUCE_NUMEL, device=device, dtype=torch.float32),
    }


def run_block(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    out = tensors["inp"]
    for weight, mix_weight, bias, mix_bias, scale in zip(
        tensors["weights"],
        tensors["mix_weights"],
        tensors["biases"],
        tensors["mix_biases"],
        tensors["scales"],
    ):
        proj = torch.matmul(out, weight) + bias
        gate = torch.sigmoid(torch.matmul(out, mix_weight) + mix_bias)
        out = F.gelu(proj * gate, approximate="tanh")
        out = out + tensors["residual"]
        for _ in range(3):
            out = out * scale
            out = out + tensors["bias_scale"]
            out = torch.sin(out)
            out = torch.tanh(out)
            out = out + tensors["residual"]
        out = F.layer_norm(out, (out.shape[-1],), tensors["ln_weight"], tensors["ln_bias"])
    return out


def warmup_and_capture(tensors: dict[str, torch.Tensor]) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    warmup_stream = torch.cuda.Stream()
    current_stream = torch.cuda.current_stream()
    warmup_stream.wait_stream(current_stream)
    with torch.cuda.stream(warmup_stream):
        out = None
        for _ in range(WARMUP_ITERS):
            out = run_block(tensors)
        assert out is not None
    current_stream.wait_stream(warmup_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run_block(tensors)
    torch.cuda.synchronize()
    return graph, out


def merge_traces(trace_paths: list[str], merged_path: pathlib.Path) -> None:
    merged_events = []
    merged_trace = {}
    for trace_path in sorted(trace_paths):
        with open(trace_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not merged_trace:
            merged_trace = {
                key: value for key, value in data.items() if key != "traceEvents"
            }
        merged_events.extend(data.get("traceEvents", []))

    indexed_events = list(enumerate(merged_events))

    def sort_key(item):
        index, event = item
        if isinstance(event, dict) and isinstance(event.get("ts"), (int, float)):
            return (float(event["ts"]), index)
        return (float("-inf"), index)

    indexed_events.sort(key=sort_key)
    merged_trace["traceEvents"] = [event for _, event in indexed_events]
    merged_trace["mergedTraceInfo"] = {
        "trace_count": len(trace_paths),
        "source_traces": sorted(trace_paths),
    }
    with merged_path.open("w", encoding="utf-8") as handle:
        json.dump(merged_trace, handle)


def run_rank(rank: int, world_size: int, local_rank: int, args: argparse.Namespace) -> None:
    init_distributed(rank, world_size, local_rank, args)
    device = torch.device(f"cuda:{local_rank}")
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"rank{rank}_pid{os.getpid()}.trace.json"

    tensors = build_tensors(device)
    dist.barrier()
    graph, out = warmup_and_capture(tensors)
    dist.barrier()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for replay_idx in range(REPLAY_ITERS):
            graph.replay()
            if (replay_idx + 1) % ALL_REDUCE_EVERY == 0:
                tensors["sync"].copy_(out.float().mean().expand_as(tensors["sync"]))
                dist.all_reduce(tensors["sync"])
        torch.cuda.synchronize()

    dist.barrier()
    prof.export_chrome_trace(str(trace_path))
    torch.cuda.synchronize()
    dist.barrier()

    gathered_paths = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_paths, str(trace_path))
    if rank == 0:
        merge_traces(gathered_paths, output_dir / "merged_all_ranks.trace.json")

    dist.barrier()
    dist.destroy_process_group()


def run_torchrun(args: argparse.Namespace) -> None:
    run_rank(
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        local_rank=int(os.environ.get("LOCAL_RANK", os.environ["RANK"])),
        args=args,
    )


def spawn_entry(rank: int, world_size: int, args: argparse.Namespace) -> None:
    run_rank(rank, world_size, rank, args)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required.")
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        run_torchrun(args)
        return
    world_size = args.world_size or torch.cuda.device_count()
    if world_size <= 0:
        raise RuntimeError("No visible GPU devices found.")
    mp.spawn(spawn_entry, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()

"""M5 (optional): dynamic int8 quantization of the distilled Transformer.

Dynamic quantization in PyTorch applies to nn.Linear (and optionally LSTM/GRU);
the Conv1d tokenizer stays fp32. That is a real limitation for this architecture
and is reported rather than hidden. Latency is measured on CPU with one thread,
batch 1 and 32, after warmup; median and p95 over 50 runs.

Accuracy is recomputed on the statement-masked test split, so it is comparable
with m1_resnet_masked.json / m5_kd_student_masked.json.

Both models are benchmarked with the MHA fast path disabled (see the note at the
top of the file), so these latencies are internally comparable but not identical
to a default fast-path-enabled run.

Usage: python src/m5_quantize.py [--tag m5_kd_student]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from ecg_data import PROC, RESULTS, ROOT, ECGDataset, build_labels, label_columns, load_meta, load_split, split_rows  # noqa: E402
from ecg_metrics import multilabel_metrics  # noqa: E402
from m1_train import predict  # noqa: E402
from m2_train import PatchTransformer  # noqa: E402

FIGS = ROOT / "figures"
CKPT = ROOT / "models"

# Dynamic quantization replaces nn.Linear with torch.ao.nn.quantized.dynamic.Linear,
# whose .weight is a bound method, not a Tensor. TransformerEncoderLayer's fast-path
# guard reads self.linear1.weight.device and raises AttributeError on the quantized
# model. Disabling the fast path for BOTH the fp32 and int8 benchmarks keeps the
# latency comparison on the same code path.
torch.backends.mha.set_fastpath_enabled(False)


def state_bytes(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell()


@torch.no_grad()
def bench(model, bs, n=50, warmup=10):
    x = torch.randn(bs, 12, 1000)
    model.eval()
    for _ in range(warmup):
        model(x)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        model(x)
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.array(ts)
    return {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95)),
            "batch": bs, "runs": n}


def masked_test_metrics(model, dl, device, ys, yr, ms, mr, rows, sup, rhy):
    _ys, ps, _yr, pr = predict(model, dl, device)
    r = rows["test"]
    out = {}
    for head, y, p, mask, names in (("super", ys, ps, ms, sup),
                                    ("rhythm", yr, pr, mr, rhy)):
        m = mask[r]
        out[head] = multilabel_metrics(y[r][m], p[m], names)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m5_kd_student")
    args = ap.parse_args()
    FIGS.mkdir(exist_ok=True)

    j = json.loads((RESULTS / f"{args.tag}.json").read_text(encoding="utf-8"))
    cfg = j["config"]
    names = j["classes"]
    meta = load_meta()
    sup, rhy = label_columns(meta)
    ys, yr, ms, mr = build_labels(meta, sup, rhy)
    rows = split_rows(meta, load_split())

    model = PatchTransformer(len(sup), len(rhy), cfg.get("patch", 25),
                             cfg.get("stride", 12), cfg.get("d_model", 128),
                             cfg.get("depth", 4), cfg.get("nhead", 4))
    model.load_state_dict(torch.load(CKPT / f"{args.tag}.pt", map_location="cpu"))
    model.cpu().eval()
    n_params = sum(p.numel() for p in model.parameters())

    ds = ECGDataset(PROC / "ptbxl_100hz_filt.npy", rows["test"], ys, yr, ms, mr)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    device = torch.device("cpu")

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    fp32 = {"size_bytes": state_bytes(model), "params": n_params,
            "latency_batch1": bench(model, 1), "latency_batch32": bench(model, 32),
            "test": masked_test_metrics(model, dl, device, ys, yr, ms, mr, rows, sup, rhy)}

    qmodel = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    qmods = [type(m).__name__ for m in qmodel.modules()
             if "quantized.dynamic" in type(m).__module__]
    int8 = {"size_bytes": state_bytes(qmodel), "params": n_params,
            "latency_batch1": bench(qmodel, 1), "latency_batch32": bench(qmodel, 32),
            "test": masked_test_metrics(qmodel, dl, device, ys, yr, ms, mr, rows, sup, rhy)}

    out = {"tag": args.tag, "threads": 1, "cpu": True,
           "quantized_linear_modules": len(qmods),
           "quantization": "torch.ao.quantization.quantize_dynamic({nn.Linear}, qint8)",
           "note": "dynamic quantization only touches nn.Linear; the Conv1d tokenizer "
                   "and all attention arithmetic around it stay fp32",
           "fp32": fp32, "int8_dynamic": int8}
    (RESULTS / "m5_quantize.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"params {n_params} | size {fp32['size_bytes']/1e6:.2f} -> "
          f"{int8['size_bytes']/1e6:.2f} MB | quantized Linear modules {len(qmods)}")
    for bs in (1, 32):
        a = fp32[f"latency_batch{bs}"]["median_ms"]
        b = int8[f"latency_batch{bs}"]["median_ms"]
        print(f"batch {bs}: {a:.2f} -> {b:.2f} ms (x{a/b:.2f})")
    for head in ("super", "rhythm"):
        a = fp32["test"][head]["macro_auc"]; b = int8["test"][head]["macro_auc"]
        print(f"{head} macro-AUC {a:.4f} -> {b:.4f} (delta {b-a:+.4f})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    labels = ["fp32", "int8 dynamic"]
    ax[0].bar(labels, [fp32["size_bytes"] / 1e6, int8["size_bytes"] / 1e6], color=["#4c78a8", "#f58518"])
    ax[0].set_title("state_dict size (MB)")
    ax[1].bar(labels, [fp32["latency_batch1"]["median_ms"], int8["latency_batch1"]["median_ms"]],
              color=["#4c78a8", "#f58518"])
    ax[1].set_title("CPU latency, batch 1 (ms, 1 thread)")
    ax[2].bar(np.arange(2) - 0.2, [fp32["test"]["super"]["macro_auc"], fp32["test"]["rhythm"]["macro_auc"]],
              width=0.4, color="#4c78a8", label="fp32")
    ax[2].bar(np.arange(2) + 0.2, [int8["test"]["super"]["macro_auc"], int8["test"]["rhythm"]["macro_auc"]],
              width=0.4, color="#f58518", label="int8")
    ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["super", "rhythm"])
    ax[2].set_ylim(0.8, 0.95); ax[2].set_title("test macro-AUC (masked)")
    ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "m5_quantize.png", dpi=150)
    plt.close(fig)
    print("wrote", RESULTS / "m5_quantize.json", FIGS / "m5_quantize.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())

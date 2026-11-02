"""M5: attention rollout on the patch Transformer, overlaid on the 12-lead ECG.

Attention rollout (Abnar & Zuidema, 2020): average heads, add the identity
residual, row-normalise, multiply across layers, then read the CLS row. The
resulting 82 token weights are mapped back to 1,000 samples by averaging the
overlapping token windows.

The forward pass below re-implements TransformerEncoderLayer's norm_first path
so attention weights can be captured; it is asserted against model.forward on
the same batch, so a silent divergence between the two paths cannot slip in.

Samples: 3 exact-set-match predictions and 2 mistakes from the super head, all
with a diagnostic statement, chosen by highest confidence. The figure shows the
model input (filtered, converted back to mV) with the attention profile behind.

Usage: python src/m5_attention.py --tag m2_transformer
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from ecg_data import PROC, RESULTS, ROOT, build_labels, label_columns, load_meta, load_split, split_rows  # noqa: E402
from m2_train import PatchTransformer  # noqa: E402

FIGS = ROOT / "figures"
CKPT = ROOT / "models"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


@torch.no_grad()
def forward_capture(model, x):
    t = model.tok(x).transpose(1, 2)
    t = torch.cat([model.cls.expand(t.size(0), -1, -1), t], dim=1) + model.pos
    attns = []
    for layer in model.enc.layers:
        h = layer.norm1(t)
        a, w = layer.self_attn(h, h, h, need_weights=True, average_attn_weights=True)
        t = t + layer.dropout1(a)
        ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(t)))))
        t = t + layer.dropout2(ff)
        attns.append(w)
    z = model.drop(model.norm(t[:, 0]))
    return model.head_sup(z), model.head_rhy(z), attns


def rollout(attns, n_tok, device):
    L = n_tok + 1
    eye = torch.eye(L, device=device)
    r = eye.unsqueeze(0).repeat(attns[0].size(0), 1, 1)
    for w in attns:
        a = w + eye
        a = a / a.clamp(min=0).sum(dim=-1, keepdim=True)
        r = a @ r
    return r[:, 0, 1:]                     # B, T (CLS attention over tokens)


def tokens_to_samples(w, patch, stride, length):
    prof = np.zeros(length, dtype=np.float64)
    cnt = np.zeros(length, dtype=np.float64)
    for t in range(w.shape[0]):
        s = t * stride
        e = min(s + patch, length)
        prof[s:e] += w[t]
        cnt[s:e] += 1
    prof /= np.maximum(cnt, 1)
    lo, hi = prof.min(), prof.max()
    return (prof - lo) / (hi - lo) if hi > lo else np.zeros_like(prof)


def input_saliency(model, x, cls):
    """gradient x input of one class logit on the SAME Transformer, for comparison."""
    model.zero_grad(set_to_none=True)
    xg = x.clone().requires_grad_(True)
    ls, _ = model(xg)
    ls[0, cls].backward()
    sal = (xg.grad.detach() * xg.detach()).abs()[0].cpu().numpy()
    return sal / max(sal.max(), 1e-12)


def format_set(names, vec):
    on = [n for n, v in zip(names, vec) if v]
    return "{" + ", ".join(on) + "}" if on else "{}"


def top_share(v, frac=0.1):
    s = np.sort(np.asarray(v, dtype=np.float64).ravel())[::-1]
    k = max(1, int(round(len(s) * frac)))
    return float(s[:k].sum() / max(s.sum(), 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m2_transformer")
    ap.add_argument("--n-correct", type=int, default=3)
    ap.add_argument("--n-wrong", type=int, default=2)
    args = ap.parse_args()
    FIGS.mkdir(exist_ok=True)

    j = json.loads((RESULTS / f"{args.tag}.json").read_text(encoding="utf-8"))
    cfg = j["config"]
    names = j["classes"]
    meta = load_meta()
    sup, rhy = label_columns(meta)
    ys, yr, ms, mr = build_labels(meta, sup, rhy)
    rows = split_rows(meta, load_split())
    z = np.load(RESULTS / f"{args.tag}_pred.npz")
    stats = json.loads((RESULTS / "m1_norm_stats.json").read_text(encoding="utf-8"))
    mean = np.array(stats["mean_mv"], dtype=np.float64)[:, None]
    std = np.array(stats["std_mv"], dtype=np.float64)[:, None]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TransformerEncoderLayer's eval fast path does not expose attention and uses a
    # different accumulation order; disable it so the capture path is exact.
    torch.backends.mha.set_fastpath_enabled(False)
    model = PatchTransformer(len(sup), len(rhy), cfg["patch"], cfg["stride"],
                             cfg["d_model"], cfg["depth"], cfg["nhead"]).to(device)
    model.load_state_dict(torch.load(CKPT / f"{args.tag}.pt", map_location=device))
    model.eval()

    te = rows["test"]
    scored = np.where(ms[te])[0]           # positions inside the npz arrays
    y = z["test_y_super"][scored]
    p = z["test_super"][scored]
    pred = (p >= 0.5).astype(int)
    exact = np.all(pred == y.astype(int), axis=1)
    conf = p.max(axis=1)

    def pick(mask, n):
        idx = np.where(mask)[0]
        return idx[np.argsort(-conf[idx])][:n]

    sel_correct = pick(exact, args.n_correct)
    sel_wrong = pick(~exact, args.n_wrong)
    sel = np.concatenate([sel_correct, sel_wrong])

    mm = np.load(PROC / "ptbxl_100hz_filt.npy", mmap_mode="r")
    recs = []
    for pos in sel:
        g = te[scored[pos]]
        x = np.array(mm[g], dtype=np.float32)
        xt = torch.from_numpy(x[None]).to(device)
        ls, _lr, attns = forward_capture(model, xt)
        with torch.no_grad():
            ls_ref, _ = model(xt)
        assert torch.allclose(ls, ls_ref, atol=1e-5), "capture path diverged from forward"
        w = rollout(attns, model.n_tokens, device)[0]
        w = (w / w.sum()).cpu().numpy()
        prof = tokens_to_samples(w, cfg["patch"], cfg["stride"], x.shape[1])
        peak = int(np.argmax(prof))
        cls = int(torch.argmax(ls[0]))
        sal = input_saliency(model, xt, cls)
        recs.append({
            "ecg_id": int(z["test_ecg_ids"][scored[pos]]),
            "correct": bool(exact[pos]),
            "true_set": format_set(names["super"], y[pos]),
            "pred_set": format_set(names["super"], pred[pos]),
            "max_prob": float(conf[pos]),
            "probs": {n: float(v) for n, v in zip(names["super"], p[pos])},
            "attention_peak_s": round(peak / 100.0, 2),
            "top10pct_mass": top_share(prof),
            "grad_top10pct_mass": top_share(sal),
            "grad_per_lead_top10pct": {n: top_share(sal[k]) for k, n in enumerate(LEADS)},
            "signal_mv": x * std + mean,
            "profile": prof,
        })

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def panel(recs, path, title):
        n = len(recs)
        fig, axes = plt.subplots(n, 1, figsize=(13, 3.0 * n), squeeze=False)
        t = np.arange(1000) / 100.0
        for ax, r in zip(axes[:, 0], recs):
            sig = r["signal_mv"]
            ax.imshow(r["profile"][None, :], aspect="auto", origin="lower",
                      extent=[0, 10, -2.0, 12 * 3.5 + 1.0], cmap="Reds",
                      vmin=0, vmax=1, alpha=0.30)
            for k in range(12):
                off = (11 - k) * 3.5
                ax.plot(t, np.clip(sig[k], -2.0, 2.5) + off, color="black", lw=0.55)
            ax.axvline(r["attention_peak_s"], ls="--", lw=0.8, color="#8c2d04")
            ax.set_yticks([(11 - k) * 3.5 for k in range(12)])
            ax.set_yticklabels(LEADS, fontsize=7)
            ax.set_xlim(0, 10)
            ax.set_ylim(-2.0, 12 * 3.5 + 1.0)
            mark = "correct" if r["correct"] else "WRONG"
            ax.set_title(f"ecg {r['ecg_id']} [{mark}] true {r['true_set']} | "
                         f"pred {r['pred_set']} (max p={r['max_prob']:.2f}) | "
                         f"attention peak {r['attention_peak_s']:.2f}s", fontsize=8)
            ax.set_ylabel("mV + offset", fontsize=7)
            if r is recs[-1]:
                ax.set_xlabel("time (s)")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    panel(recs[:len(sel_correct)], FIGS / "m5_attention_correct.png",
          f"M5 attention rollout, {args.tag}: 3 highest-confidence exact matches")
    panel(recs[len(sel_correct):], FIGS / "m5_attention_wrong.png",
          f"M5 attention rollout, {args.tag}: 2 highest-confidence mistakes")

    out = {"tag": args.tag, "method": "attention rollout (heads averaged, identity "
                                      "residual, row-normalised, layers multiplied)",
           "head": "super", "selection": "highest-confidence exact-set matches/mistakes",
           "n_correct": int(len(sel_correct)), "n_wrong": int(len(sel_wrong)),
           "caveat": "rollout is class-agnostic: it explains the CLS representation, "
                     "not one class logit; the heat is shared across leads because "
                     "attention has no lead axis. grad_top10pct_mass is an input-"
                     "gradient attribution on the same model and samples, included to "
                     "separate method concentration from model differences",
           "samples": [{k: v for k, v in r.items() if k not in ("signal_mv", "profile")}
                       for r in recs]}
    (RESULTS / "m5_attention.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    for r in recs:
        print(f"ecg {r['ecg_id']} {'OK ' if r['correct'] else 'ERR'} "
              f"true {r['true_set']} pred {r['pred_set']} p={r['max_prob']:.3f} "
              f"peak={r['attention_peak_s']:.2f}s top10%={r['top10pct_mass']:.3f} "
              f"grad_top10%={r['grad_top10pct_mass']:.3f}")
    print("wrote", FIGS / "m5_attention_correct.png", FIGS / "m5_attention_wrong.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())

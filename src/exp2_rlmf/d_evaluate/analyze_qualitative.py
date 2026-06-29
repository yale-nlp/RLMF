import argparse, json, os, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
from pathlib import Path
import csv

def load_json(path):
    with open(path) as f:
        return json.load(f)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def get_output_dir(input_path):
    dataset_name = Path(input_path).stem.split("_")[-1]
    out = Path(input_path).parent / f"plots_{dataset_name}"
    return ensure_dir(out)

def parse_samples(scores_data, preds_data):
    """Merge scores and preds into flat list of dicts, filtering failed samples."""
    raw_prompts = preds_data.get("raw_prompts", [])
    answers = preds_data.get("answers", [])
    acc_scores = scores_data.get("acc_scores", [])
    brier_scores = scores_data.get("brier_scores", [])
    f_scores = scores_data.get("f_scores", [])
    avg_pred_confs = scores_data.get("avg_pred_confs", [])
    gold_confs_per_response = scores_data.get("gold_confs_per_response", [])
    n = len(f_scores)
    samples, n_skipped = [], 0
    for i in range(n):
        faith = f_scores[i] if i < len(f_scores) else -1
        pred_conf = avg_pred_confs[i] if i < len(avg_pred_confs) else -1
        if faith == -1 or pred_conf == -1:
            n_skipped += 1
            continue
        gold_list = gold_confs_per_response[i] if i < len(gold_confs_per_response) else []
        gold_conf = float(np.mean(gold_list)) if gold_list else None
        samples.append({
            "idx": i,
            "input": raw_prompts[i] if i < len(raw_prompts) else "",
            "output": answers[i] if i < len(answers) else "",
            "faithfulness": float(faith),
            "gold_confidence": gold_conf,
            "pred_confidence": float(pred_conf),
            "accuracy": float(acc_scores[i]) if i < len(acc_scores) else None,
            "brier": float(brier_scores[i]) if i < len(brier_scores) else None,
        })
    print(f"  Skipped {n_skipped} samples with faithfulness or pred_conf == -1")
    return samples

def get_arr(samples, key):
    return np.array([s[key] for s in samples if s[key] is not None], dtype=float)

def get_mask(samples, *keys):
    return np.array([all(s[k] is not None for k in keys) for s in samples])

# ── Plot 1: Scatter gold vs predicted, colored by faithfulness ──────────
def plot_scatter_colored(samples, out_dir):
    mask = get_mask(samples, "gold_confidence", "pred_confidence", "faithfulness")
    s = [x for x, m in zip(samples, mask) if m]
    if not s:
        return
    gc = np.array([x["gold_confidence"] for x in s])
    pc = np.array([x["pred_confidence"] for x in s])
    fc = np.array([x["faithfulness"] for x in s])
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(gc, pc, c=fc, cmap="RdYlGn", alpha=0.6, s=18, vmin=0, vmax=1)
    ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.4, label="y=x (perfect)")
    plt.colorbar(sc, ax=ax, label="Faithfulness score")
    ax.set_xlabel("Gold confidence"); ax.set_ylabel("Predicted confidence")
    ax.set_title("Gold vs Predicted Confidence\n(colored by faithfulness)")
    ax.legend(fontsize=8); ax.set_xlim(0,1); ax.set_ylim(0,1)
    fig.tight_layout(); fig.savefig(out_dir / "scatter_confidence_colored.png", dpi=150); plt.close()

# ── Plot 2: Residual (predicted - gold) vs gold, colored by faithfulness ─
def plot_residual(samples, out_dir):
    mask = get_mask(samples, "gold_confidence", "pred_confidence", "faithfulness")
    s = [x for x, m in zip(samples, mask) if m]
    if not s:
        return
    gc = np.array([x["gold_confidence"] for x in s])
    pc = np.array([x["pred_confidence"] for x in s])
    fc = np.array([x["faithfulness"] for x in s])
    residual = pc - gc
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(gc, residual, c=fc, cmap="RdYlGn", alpha=0.6, s=18, vmin=0, vmax=1)
    ax.axhline(0, color="k", lw=1, ls="--", alpha=0.4)
    plt.colorbar(sc, ax=ax, label="Faithfulness score")
    ax.set_xlabel("Gold confidence"); ax.set_ylabel("Predicted - Gold (residual)")
    ax.set_title("Confidence Residual vs Gold Confidence")
    fig.tight_layout(); fig.savefig(out_dir / "residual_plot.png", dpi=150); plt.close()

# ── Plot 3: 2x2 quadrant scatter (gold x pred axes, 4 quadrants) ────────────
# def plot_2x2_quadrant(samples, out_dir):
#     mask = get_mask(samples, "gold_confidence", "pred_confidence", "faithfulness")
#     s = [x for x, m in zip(samples, mask) if m]
#     if not s: return
#     gc = np.array([x["gold_confidence"] for x in s])
#     pc = np.array([x["pred_confidence"] for x in s])
#     fc = np.array([x["faithfulness"] for x in s])
#     fig = plt.figure(figsize=(10, 9))
#     gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.07], figure=fig, hspace=0.35, wspace=0.3)
#     quadrant_defs = [
#         ("High Gold, Low Predicted",  gc>=0.5, pc<0.5,  0, 0, (0.5,1.0), (0.0,0.5)),
#         ("High Gold, High Predicted", gc>=0.5, pc>=0.5, 0, 1, (0.5,1.0), (0.5,1.0)),
#         ("Low Gold, Low Predicted",   gc<0.5,  pc<0.5,  1, 0, (0.0,0.5), (0.0,0.5)),
#         ("Low Gold, High Predicted",  gc<0.5,  pc>=0.5, 1, 1, (0.0,0.5), (0.5,1.0)),
#     ]
#     norm = Normalize(0, 1); last_sc = None
#     for title, gm, pm, row, col, xlim, ylim in quadrant_defs:
#         ax = fig.add_subplot(gs[row, col])
#         qmask = gm & pm
#         if qmask.sum() == 0:
#             ax.set_title(f"{title}\n(n=0)", fontsize=9); continue
#         last_sc = ax.scatter(gc[qmask], pc[qmask], c=fc[qmask], cmap="RdYlGn", alpha=0.65, s=20, vmin=0, vmax=1)
#         ax.set_xlim(xlim); ax.set_ylim(ylim)
#         ax.axvline(0.5, color="gray", lw=0.7, ls=":"); ax.axhline(0.5, color="gray", lw=0.7, ls=":")
#         ax.set_title(f"{title}\n(n={qmask.sum()}, mean faith={fc[qmask].mean():.2f})", fontsize=9)
#         ax.set_xlabel("Gold confidence"); ax.set_ylabel("Predicted confidence")
#     cax = fig.add_subplot(gs[:, 2])
#     sm = ScalarMappable(cmap="RdYlGn", norm=norm); sm.set_array([])
#     fig.colorbar(sm, cax=cax, label="Faithfulness score")
#     fig.suptitle("2×2 Confidence Quadrants (colored by faithfulness)", fontsize=12)
#     fig.savefig(out_dir / "quadrant_2x2.png", dpi=150, bbox_inches="tight"); plt.close()
def plot_2x2_quadrant(samples, out_dir):
    mask = get_mask(samples, "gold_confidence", "pred_confidence", "faithfulness")
    s = [x for x, m in zip(samples, mask) if m]
    if not s: return
    gc = np.array([x["gold_confidence"] for x in s])
    pc = np.array([x["pred_confidence"] for x in s])
    fc = np.array([x["faithfulness"] for x in s])
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    quadrant_defs = [
        ("High Gold, Low Predicted",  gc>=0.5, pc<0.5,  axes[0][0]),
        ("High Gold, High Predicted", gc>=0.5, pc>=0.5, axes[0][1]),
        ("Low Gold, Low Predicted",   gc<0.5,  pc<0.5,  axes[1][0]),
        ("Low Gold, High Predicted",  gc<0.5,  pc>=0.5, axes[1][1]),
    ]
    norm = Normalize(0, 1)
    for title, gm, pm, ax in quadrant_defs:
        qmask = gm & pm
        if qmask.sum() == 0:
            ax.set_title(f"{title}\n(n=0)"); continue
        ax.scatter(gc[qmask], pc[qmask], c=fc[qmask], cmap="RdYlGn", alpha=0.65, s=16, vmin=0, vmax=1)
        ax.axvline(0.5, color="gray", lw=0.7, ls=":"); ax.axhline(0.5, color="gray", lw=0.7, ls=":")
        ax.set_title(f"{title}\n(n={qmask.sum()}, mean faith={fc[qmask].mean():.2f})", fontsize=9)
        ax.set_xlabel("Gold confidence"); ax.set_ylabel("Predicted confidence")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    sm = ScalarMappable(cmap="RdYlGn", norm=norm); sm.set_array([])
    fig.subplots_adjust(hspace=0.4)
    fig.colorbar(sm, ax=axes.ravel().tolist(), label="Faithfulness score", shrink=0.6, pad=0.04)
    fig.suptitle("2×2 Confidence Quadrants (colored by faithfulness)", fontsize=12)
    fig.savefig(out_dir / "quadrant_2x2.png", dpi=150, bbox_inches="tight"); plt.close()

# ── Plot 4: Faithfulness distribution (KDE + hist) ──────────────────────────
def plot_faithfulness_dist(samples, out_dir):
    fc = get_arr(samples, "faithfulness")
    if len(fc) == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(fc, bins=30, density=False, alpha=0.4, color="steelblue", label="histogram")
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(fc)
        xs = np.linspace(0, 1, 300)
        ax.plot(xs, kde(xs), color="steelblue", lw=2, label="KDE")
    except ImportError:
        pass
    ax.set_xlabel("Faithfulness score"); ax.set_ylabel("Density")
    ax.set_title("Distribution of Per-Sample Faithfulness Scores")
    ax.legend(); fig.tight_layout(); fig.savefig(out_dir / "faithfulness_distribution.png", dpi=150); plt.close()

# ── Plot 5: Faithfulness by accuracy group ───────────────────────────────────
def plot_faithfulness_by_accuracy(samples, out_dir):
    mask = get_mask(samples, "faithfulness", "accuracy")
    s = [x for x, m in zip(samples, mask) if m]
    if not s:
        return
    correct = [x["faithfulness"] for x in s if x["accuracy"] >= 0.5]
    wrong   = [x["faithfulness"] for x in s if x["accuracy"] < 0.5]
    fig, ax = plt.subplots(figsize=(7, 4))
    for data, label, color in [(correct,"Correct","steelblue"),(wrong,"Wrong","tomato")]:
        if len(data) < 2:
            continue
        ax.hist(data, bins=20, density=False, alpha=0.45, color=color, label=f"{label} (n={len(data)})")
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            xs = np.linspace(0,1,300)
            ax.plot(xs, kde(xs), color=color, lw=2)
        except ImportError:
            pass
    ax.set_xlabel("Faithfulness score"); ax.set_ylabel("Density")
    ax.set_title("Faithfulness Distribution: Correct vs Wrong Answers")
    ax.legend(); fig.tight_layout(); fig.savefig(out_dir / "faithfulness_by_accuracy.png", dpi=150); plt.close()

# ── Plot 6: 2x2 breakdown mean faithfulness (correct/wrong × conf level) ────
def plot_accuracy_confidence_heatmap(samples, out_dir):
    mask = get_mask(samples, "faithfulness", "accuracy", "gold_confidence")
    s = [x for x, m in zip(samples, mask) if m]
    if not s:
        return
    cells = {"Correct\nHigh Conf": [], "Correct\nLow Conf": [], "Wrong\nHigh Conf": [], "Wrong\nLow Conf": []}
    for x in s:
        acc_key = "Correct" if x["accuracy"] >= 0.5 else "Wrong"
        conf_key = "High Conf" if x["gold_confidence"] >= 0.5 else "Low Conf"
        cells[f"{acc_key}\n{conf_key}"].append(x["faithfulness"])
    labels = list(cells.keys())
    means = [np.mean(v) if v else float("nan") for v in cells.values()]
    counts = [len(v) for v in cells.values()]
    data = np.array(means).reshape(2, 2)
    count_data = np.array(counts).reshape(2, 2)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks([0,1]); ax.set_xticklabels(["High Gold Conf", "Low Gold Conf"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["Correct", "Wrong"])
    for i in range(2):
        for j in range(2):
            v = data[i,j]
            n = count_data[i,j]
            txt = f"{v:.2f}\n(n={n})" if not math.isnan(v) else "N/A"
            ax.text(j, i, txt, ha="center", va="center", fontsize=12, color="black")
    plt.colorbar(im, ax=ax, label="Mean faithfulness")
    ax.set_title("Mean Faithfulness: Accuracy × Confidence Level")
    fig.tight_layout(); fig.savefig(out_dir / "accuracy_confidence_heatmap.png", dpi=150); plt.close()

# ── Plot 7: Reliability diagram ──────────────────────────────────────────────
# def plot_reliability(samples, out_dir):
#     mask = get_mask(samples, "gold_confidence", "pred_confidence")
#     s = [x for x, m in zip(samples, mask) if m]
#     if not s:
#         return
#     pc = np.array([x["pred_confidence"] for x in s])
#     gc = np.array([x["gold_confidence"] for x in s])
#     bins = np.linspace(0, 1, 11)
#     bin_idx = np.digitize(gc, bins) - 1
#     bin_idx = np.clip(bin_idx, 0, 9)
#     bin_centers, bin_means, bin_counts = [], [], []
#     for b in range(10):
#         idxs = np.where(bin_idx == b)[0]
#         if len(idxs) > 0:
#             bin_centers.append(bins[b] + 0.05)
#             bin_means.append(pc[idxs].mean())
#             bin_counts.append(len(idxs))
#     fig, ax = plt.subplots(figsize=(6, 5))
#     ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.5,label="Perfect calibration")
#     sc = ax.scatter(bin_centers, bin_means, c=bin_counts, cmap="Blues", s=80, zorder=5)
#     ax.plot(bin_centers, bin_means, color="steelblue", lw=1.5)
#     plt.colorbar(sc, ax=ax, label="Count in bin")
#     ax.set_xlabel("Mean gold confidence (binned)"); ax.set_ylabel("Mean predicted confidence")
#     ax.set_title("Reliability Diagram (Predicted vs Gold)")
#     ax.legend(); ax.set_xlim(0,1); ax.set_ylim(0,1)
#     fig.tight_layout(); fig.savefig(out_dir / "reliability_diagram.png", dpi=150); plt.close()
def plot_reliability(samples, out_dir, model_name):
    mask = get_mask(samples, "gold_confidence", "pred_confidence", "faithfulness")
    s = [x for x, m in zip(samples, mask) if m]
    if not s: return
    gc = np.array([x["gold_confidence"] for x in s])
    pc = np.array([x["pred_confidence"] for x in s])
    fc = np.array([x["faithfulness"] for x in s])
    bins = np.linspace(0, 1, 11)
    bin_idx = np.digitize(gc, bins) - 1
    bin_idx = np.clip(bin_idx, 0, 9)
    bin_centers, bin_means, bin_counts, faith_means = [], [], [], []
    for b in range(10):
        idxs = np.where(bin_idx == b)[0]
        if len(idxs) > 0:
            bin_centers.append(bins[b] + 0.05)
            bin_means.append(pc[idxs].mean())
            bin_counts.append(len(idxs))
            faith_means.append(fc[idxs].mean())
    fig, ax = plt.subplots(figsize=(7, 6))
    # ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.5,label="Perfect Faithful Calibration")
    sc = ax.scatter(bin_centers, bin_means, c=bin_counts, cmap="Blues", s=80, zorder=5)
    ax.plot(bin_centers, bin_means, color="steelblue", lw=1.5, label="Mean Expressed Conf")
    cbar = plt.colorbar(sc, ax=ax, label="Count in Bin", pad=0.15)
    # cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Count in Bin", fontsize=12)
    ax.set_xlabel("Intrinsic Confidence (Binned)", fontsize=12); ax.set_ylabel("Expressed Confidence", fontsize=12)
    ax.set_xlim(0,1); ax.set_ylim(0,1.07)
    ax2 = ax.twinx()
    faith_means = [x + 0.03 for x in faith_means]
    ax2.plot(bin_centers, faith_means, color="purple", lw=1.5, ls="--", marker="x", ms=6, label="Mean Faithfulness Score")
    ax2.set_ylabel("Faithfulness Score", color="purple", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="purple")
    ax2.set_ylim(0, 1.07)
    # ax.plot([0,1],[0,1], color="steelblue", lw=1, alpha=0.4, ls="--", label="y = x (perfect alignment)")
    # ax2.axhline(y=1.0, color="purple", lw=1, alpha=0.4, ls="--", label="FC = 1 (perfect)")
    ax.plot([0,1],[0,1], color="steelblue", lw=1, alpha=0.6, ls="--")
    ax2.axhline(y=1.0, color="purple", lw=1, alpha=0.6, ls="--")
    # annotate diagonal at its end
    # ax.annotate("perfect alignment", xy=(0.92, 0.92), xycoords="data", fontsize=8,color="steelblue", alpha=0.6, rotation=45, ha="center")
    # ax2.annotate("perfect F score", xy=(0.97, 1.01), xycoords="data", fontsize=8,color="purple", alpha=0.6, ha="right")
    ax.annotate("perfect alignment", xy=(0.15, 0.08), xycoords="data", fontsize=12, color="steelblue", alpha=0.6, rotation=45, ha="center")
    # ax.annotate("y = x", xy=(0.1, 0.03), xycoords="data", fontsize=8,color="steelblue", alpha=0.6, rotation=38, ha="left")
    ax2.annotate("perfect F score", xy=(0.02, 1.015), xycoords="data", fontsize=12, color="purple", alpha=0.6, ha="left")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, fontsize=12, loc="lower right")
    ax.set_title(f"Reliability Diagram of Confidence and Faithfulness\n({model_name})")
    model_id = model_name.replace("-","_").replace("/", "_").replace(" ", "_")
    fig.tight_layout(); fig.savefig(out_dir / f"reliability_diagram_{model_id}.png", dpi=150); plt.close()

# ── Plot 8: 2D joint heatmap ─────────────────────────────────────────────────
def plot_joint_heatmap(samples, out_dir):
    mask = get_mask(samples, "gold_confidence", "pred_confidence")
    s = [x for x, m in zip(samples, mask) if m]
    if not s:
        return
    gc = np.array([x["gold_confidence"] for x in s])
    pc = np.array([x["pred_confidence"] for x in s])
    fig, ax = plt.subplots(figsize=(6, 5))
    h = ax.hist2d(gc, pc, bins=20, range=[[0,1],[0,1]], cmap="YlOrRd")
    plt.colorbar(h[3], ax=ax, label="Count")
    ax.plot([0,1],[0,1],"w--",lw=1,alpha=0.6)
    ax.set_xlabel("Gold confidence"); ax.set_ylabel("Predicted confidence")
    ax.set_title("Joint Distribution: Gold vs Predicted Confidence")
    fig.tight_layout(); fig.savefig(out_dir / "joint_heatmap.png", dpi=150); plt.close()

# ── Plot 9: Faithfulness by gold conf group ───────────────────────────────────
def plot_faithfulness_by_gold_conf(samples, out_dir):
    mask = get_mask(samples, "faithfulness", "gold_confidence")
    s = [x for x, m in zip(samples, mask) if m]
    if not s: return
    high = [x["faithfulness"] for x in s if x["gold_confidence"] >= 0.5]
    low  = [x["faithfulness"] for x in s if x["gold_confidence"] <  0.5]
    fig, ax = plt.subplots(figsize=(7, 4))
    for data, label, color in [(high,"High gold conf (≥0.5)","steelblue"),(low,"Low gold conf (<0.5)","tomato")]:
        if len(data) < 2: continue
        ax.hist(data, bins=20, density=False, alpha=0.45, color=color, label=f"{label} (n={len(data)})")
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data); xs = np.linspace(0,1,300)
            ax.plot(xs, kde(xs), color=color, lw=2)
        except ImportError: pass
    ax.set_xlabel("Faithfulness score"); ax.set_ylabel("Density")
    ax.set_title("Faithfulness Distribution: High vs Low Gold Confidence")
    ax.legend(); fig.tight_layout(); fig.savefig(out_dir / "faithfulness_by_gold_conf.png", dpi=150); plt.close()

# ── Plot 10: Faithfulness by pred conf group ─────────────────────────────────
def plot_faithfulness_by_pred_conf(samples, out_dir):
    mask = get_mask(samples, "faithfulness", "pred_confidence")
    s = [x for x, m in zip(samples, mask) if m]
    if not s: return
    high = [x["faithfulness"] for x in s if x["pred_confidence"] >= 0.5]
    low  = [x["faithfulness"] for x in s if x["pred_confidence"] <  0.5]
    fig, ax = plt.subplots(figsize=(7, 4))
    for data, label, color in [(high,"High predicted conf (≥0.5)","steelblue"),(low,"Low predicted conf (<0.5)","tomato")]:
        if len(data) < 2: continue
        ax.hist(data, bins=20, density=False, alpha=0.45, color=color, label=f"{label} (n={len(data)})")
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data); xs = np.linspace(0,1,300)
            ax.plot(xs, kde(xs), color=color, lw=2)
        except ImportError: pass
    ax.set_xlabel("Faithfulness score"); ax.set_ylabel("Density")
    ax.set_title("Faithfulness Distribution: High vs Low Predicted Confidence")
    ax.legend(); fig.tight_layout(); fig.savefig(out_dir / "faithfulness_by_pred_conf.png", dpi=150); plt.close()

# ── Faithfulness quartile bin files ─────────────────────────────────────────
def save_faithfulness_bins(samples, out_dir, as_csv=False):
    fc = np.array([s["faithfulness"] if s["faithfulness"] is not None else float("nan") for s in samples])
    valid = ~np.isnan(fc)
    quartiles = np.nanpercentile(fc[valid], [25, 50, 75])
    bin_edges = [0.0] + list(quartiles) + [1.001]
    bin_labels = ["Q1_lowest", "Q2_low_mid", "Q3_high_mid", "Q4_highest"]
    for i, label in enumerate(bin_labels):
        lo, hi = bin_edges[i], bin_edges[i+1]
        bin_samples = [s for s, fv in zip(samples, fc) if not math.isnan(fv) and lo <= fv < hi]
        out = []
        for s in bin_samples:
            out.append({"input": s["input"], "output": s["output"], "faithfulness": s["faithfulness"],"residual": round((s["pred_confidence"] or 0) - (s["gold_confidence"] or 0), 4),
                        "pred_confidence": s["pred_confidence"],"gold_confidence": s["gold_confidence"], 
                        "accuracy": s["accuracy"], "brier": s["brier"]})
        if as_csv: 
            # bin_samples.sort(key=lambda x: (x["faithfulness"], x["pred_confidence"] or 0, x["gold_confidence"] or 0))
            bin_samples.sort(key=lambda x: (x["pred_confidence"] or 0) - (x["gold_confidence"] or 0))
            path = out_dir / f"faithfulness_bin_{label}.csv"
            with open(path, "w", newline="") as f:
                # writer = csv.DictWriter(f, fieldnames=["input","output","faithfulness","pred_confidence","gold_confidence","accuracy","brier"])
                writer = csv.DictWriter(f, fieldnames=["input","output","faithfulness","residual","pred_confidence","gold_confidence","accuracy","brier"])
                writer.writeheader()
                # writer.writerows([{"input": s["input"], "output": s["output"], "faithfulness": s["faithfulness"], "pred_confidence": s["pred_confidence"], "gold_confidence": s["gold_confidence"], "accuracy": s["accuracy"], "brier": s["brier"]} for s in bin_samples])
                writer.writerows([{"input": s["input"], "output": s["output"], "faithfulness": s["faithfulness"], "residual": round((s["pred_confidence"] or 0) - (s["gold_confidence"] or 0), 4),"pred_confidence": s["pred_confidence"], "gold_confidence": s["gold_confidence"], "accuracy": s["accuracy"], "brier": s["brier"]} for s in bin_samples])
        else: 
            path = out_dir / f"faithfulness_bin_{label}.json"
            with open(path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"  Saved {len(out)} samples → {path.name}  (faithfulness {lo:.3f}–{hi:.3f})")
        
def save_faithfulness_bins_strict(samples, out_dir, as_csv=False):
    fc = np.array([s["faithfulness"] if s["faithfulness"] is not None else float("nan") for s in samples])
    valid = ~np.isnan(fc)
    quartiles = np.nanpercentile(fc[valid], [25, 50, 75])
    bin_edges = [0.0] + [0.25000001, 0.500001, 0.7500001] + [1.001]
    bin_labels = ["0.0_to_0.25", "0.25_to_0.50", "0.50_to_0.75", "0.75_to_1.0"]
    for i, label in enumerate(bin_labels):
        lo, hi = bin_edges[i], bin_edges[i+1]
        bin_samples = [s for s, fv in zip(samples, fc) if not math.isnan(fv) and lo <= fv < hi]
        out = []
        for s in bin_samples:
            out.append({"input": s["input"], "output": s["output"], "faithfulness": s["faithfulness"],"residual": round((s["pred_confidence"] or 0) - (s["gold_confidence"] or 0), 4),
                        "pred_confidence": s["pred_confidence"],"gold_confidence": s["gold_confidence"], 
                        "accuracy": s["accuracy"], "brier": s["brier"]})
        if as_csv: 
            # bin_samples.sort(key=lambda x: (x["faithfulness"], x["pred_confidence"] or 0, x["gold_confidence"] or 0))
            bin_samples.sort(key=lambda x: (x["pred_confidence"] or 0) - (x["gold_confidence"] or 0))
            path = out_dir / f"faithfulness_bin_{label}.csv"
            with open(path, "w", newline="") as f:
                # writer = csv.DictWriter(f, fieldnames=["input","output","faithfulness","pred_confidence","gold_confidence","accuracy","brier"])
                writer = csv.DictWriter(f, fieldnames=["input","output","faithfulness","residual","pred_confidence","gold_confidence","accuracy","brier"])
                writer.writeheader()
                # writer.writerows([{"input": s["input"], "output": s["output"], "faithfulness": s["faithfulness"], "pred_confidence": s["pred_confidence"], "gold_confidence": s["gold_confidence"], "accuracy": s["accuracy"], "brier": s["brier"]} for s in bin_samples])
                writer.writerows([{"input": s["input"], "output": s["output"], "faithfulness": s["faithfulness"], "residual": round((s["pred_confidence"] or 0) - (s["gold_confidence"] or 0), 4), "accuracy": s["accuracy"], "pred_confidence": s["pred_confidence"], "gold_confidence": s["gold_confidence"], "brier": s["brier"]} for s in bin_samples])
        else: 
            path = out_dir / f"faithfulness_bin_{label}.json"
            with open(path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"  Saved {len(out)} samples → {path.name}  (faithfulness {lo:.3f}–{hi:.3f})")

def plot_faithfulness_by_residual_bins(samples, out_dir):
    mask = get_mask(samples, "faithfulness", "pred_confidence", "gold_confidence")
    s = [x for x, m in zip(samples, mask) if m]
    if not s: return
    for x in s:
        x["residual"] = (x["pred_confidence"] or 0) - (x["gold_confidence"] or 0)
    residual_bins = [(-1.0, -0.5, "[-1, -0.5)"), (-0.5, 0.0, "[-0.5, 0)"), (0.0, 0.5, "[0, 0.5)"), (0.5, 1.01, "[0.5, 1]")]
    colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # violin plot
    ax = axes[0]
    groups = [([x["faithfulness"] for x in s if lo <= x["residual"] < hi], label) for lo, hi, label in residual_bins]
    data_to_plot = [g[0] for g in groups if len(g[0]) >= 2]
    labels_to_plot = [g[1] for g in groups if len(g[0]) >= 2]
    parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)), showmedians=True, showextrema=True)
    for pc, color in zip(parts["bodies"], colors): pc.set_facecolor(color); pc.set_alpha(0.6)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_lw(2)
    ax.set_xticks(range(len(labels_to_plot))); ax.set_xticklabels(labels_to_plot)
    ax.set_xlabel("Residual bin (pred - gold)"); ax.set_ylabel("Faithfulness score")
    ax.set_title("Faithfulness Distribution by Residual Bin (Violin)")
    ax.set_ylim(0, 1)
    for j, (data, label) in enumerate(zip(data_to_plot, labels_to_plot)):
        ax.text(j, 0.02, f"n={len(data)}", ha="center", fontsize=8, color="gray")

    # histogram with density=False
    ax = axes[1]
    for (lo, hi, label), color in zip(residual_bins, colors):
        data = [x["faithfulness"] for x in s if lo <= x["residual"] < hi]
        if len(data) < 2: continue
        ax.hist(data, bins=20, density=True, alpha=0.45, color=color, label=f"{label} (n={len(data)})")
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data); xs = np.linspace(0, 1, 300)
            ax.plot(xs, kde(xs), color=color, lw=2)
        except ImportError: pass
    ax.set_xlabel("Faithfulness score"); ax.set_ylabel("Density")
    ax.set_title("Faithfulness Distribution by Residual Bin (Histogram)")
    ax.legend(title="Residual bin")

    fig.tight_layout(); fig.savefig(out_dir / "faithfulness_by_residual_bins.png", dpi=150); plt.close()

def plot_faithfulness_by_gold_conf_bins(samples, out_dir, model_name):
    mask = get_mask(samples, "faithfulness", "gold_confidence")
    s = [x for x, m in zip(samples, mask) if m]
    if not s: return
    conf_bins = [(0.0, 0.25, "[0, 0.25)"), (0.25, 0.5, "[0.25, 0.5)"), (0.5, 0.75, "[0.5, 0.75)"), (0.75, 1.01, "[0.75, 1]")]
    colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    groups = [([x["faithfulness"] for x in s if lo <= x["gold_confidence"] < hi], label) for lo, hi, label in conf_bins]
    data_to_plot = [g[0] for g in groups if len(g[0]) >= 2]
    labels_to_plot = [g[1] for g in groups if len(g[0]) >= 2]
    parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)), showmedians=True, showextrema=True)
    for pc, color in zip(parts["bodies"], colors): pc.set_facecolor(color); pc.set_alpha(0.6)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_lw(2)
    ax.set_xticks(range(len(labels_to_plot))); ax.set_xticklabels(labels_to_plot)
    ax.set_xlabel("Intrinsic Confidence Bin"); ax.set_ylabel("Faithfulness Score")
    ax.set_title(f"Faithfulness Distribution by Intrinsic Confidence Bin (Violin)\n({model_name})")
    ax.set_ylim(0, 1)
    for j, (data, label) in enumerate(zip(data_to_plot, labels_to_plot)):
        ax.text(j, 0.02, f"n={len(data)}", ha="center", fontsize=8, color="gray")

    ax = axes[1]
    for (lo, hi, label), color in zip(conf_bins, colors):
        data = [x["faithfulness"] for x in s if lo <= x["gold_confidence"] < hi]
        if len(data) < 2: continue
        ax.hist(data, bins=20, density=True, alpha=0.45, color=color, label=f"{label} (n={len(data)})")
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data); xs = np.linspace(0, 1, 300)
            ax.plot(xs, kde(xs), color=color, lw=2)
        except ImportError: pass
    ax.set_xlabel("Faithfulness score"); ax.set_ylabel("Density")
    ax.set_title(f"Faithfulness Distribution by Intrinsic Confidence Bin (Histogram)\n({model_name})")
    ax.legend(title="Gold confidence bin")

    model_id = model_name.replace("-","_").replace("/", "_").replace(" ", "_")
    fig.tight_layout(); fig.savefig(out_dir / f"faithfulness_by_gold_conf_bins_{model_id}.png", dpi=150); plt.close()

def plot_faithfulness_by_pred_conf_bins(samples, out_dir):
    mask = get_mask(samples, "faithfulness", "pred_confidence")
    s = [x for x, m in zip(samples, mask) if m]
    if not s: return
    conf_bins = [(0.0, 0.25, "[0, 0.25)"), (0.25, 0.5, "[0.25, 0.5)"), (0.5, 0.75, "[0.5, 0.75)"), (0.75, 1.01, "[0.75, 1]")]
    colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    groups = [([x["faithfulness"] for x in s if lo <= x["pred_confidence"] < hi], label) for lo, hi, label in conf_bins]
    data_to_plot = [g[0] for g in groups if len(g[0]) >= 2]
    labels_to_plot = [g[1] for g in groups if len(g[0]) >= 2]
    parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)), showmedians=True, showextrema=True)
    for pc, color in zip(parts["bodies"], colors): pc.set_facecolor(color); pc.set_alpha(0.6)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_lw(2)
    ax.set_xticks(range(len(labels_to_plot))); ax.set_xticklabels(labels_to_plot)
    ax.set_xlabel("Predicted confidence bin"); ax.set_ylabel("Faithfulness score")
    ax.set_title("Faithfulness Distribution by Predicted Confidence Bin (Violin)")
    ax.set_ylim(0, 1)
    for j, (data, label) in enumerate(zip(data_to_plot, labels_to_plot)):
        ax.text(j, 0.02, f"n={len(data)}", ha="center", fontsize=8, color="gray")

    ax = axes[1]
    for (lo, hi, label), color in zip(conf_bins, colors):
        data = [x["faithfulness"] for x in s if lo <= x["pred_confidence"] < hi]
        if len(data) < 2: continue
        ax.hist(data, bins=20, density=True, alpha=0.45, color=color, label=f"{label} (n={len(data)})")
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data); xs = np.linspace(0, 1, 300)
            ax.plot(xs, kde(xs), color=color, lw=2)
        except ImportError: pass
    ax.set_xlabel("Faithfulness score"); ax.set_ylabel("Density")
    ax.set_title("Faithfulness Distribution by Predicted Confidence Bin (Histogram)")
    ax.legend(title="Gold confidence bin")

    fig.tight_layout(); fig.savefig(out_dir / "faithfulness_by_pred_conf_bins.png", dpi=150); plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_score_json_path", required=True)
    parser.add_argument("--model_name", required=True)
    args = parser.parse_args()
    score_path = Path(args.input_score_json_path)
    preds_path = Path(str(score_path).replace("scores", "preds"))
    out_dir = get_output_dir(score_path)
    print(f"Output dir: {out_dir}")
    scores_data = load_json(score_path)
    preds_data = load_json(preds_path) if preds_path.exists() else {}
    if not preds_data:
        print(f"Warning: preds file not found at {preds_path}, input/output fields will be empty")
    samples = parse_samples(scores_data, preds_data)
    print(f"Loaded {len(samples)} samples")
    print("Generating plots...")
    plot_scatter_colored(samples, out_dir)
    plot_residual(samples, out_dir)
    plot_2x2_quadrant(samples, out_dir)
    plot_faithfulness_dist(samples, out_dir)
    plot_faithfulness_by_accuracy(samples, out_dir)
    plot_faithfulness_by_gold_conf(samples, out_dir)
    plot_faithfulness_by_pred_conf(samples, out_dir)
    plot_accuracy_confidence_heatmap(samples, out_dir)
    plot_reliability(samples, out_dir, args.model_name)
    plot_joint_heatmap(samples, out_dir)
    plot_faithfulness_by_residual_bins(samples, out_dir)
    plot_faithfulness_by_gold_conf_bins(samples, out_dir, args.model_name)
    plot_faithfulness_by_pred_conf_bins(samples, out_dir)
    print("Generating faithfulness quartile bin files...")
    save_faithfulness_bins(samples, out_dir, as_csv=True)
    save_faithfulness_bins_strict(samples, out_dir, as_csv=True)
    print("Done.")

if __name__ == "__main__":
    main()
"""
Figure 5 (revised): defense effectiveness vs latency.

Changes from the previous version:
- Baselines are read from the no_defense rows in the data, not hard-coded.
- Each label has its own offset, so labels no longer collide.
- Legend sits outside the plot area.
- Circles = Mistral-7B, squares = Llama-3.1-8B; colour = defense.

Usage:
  py fig5_fixed.py                       (default folder on Desktop)
  py fig5_fixed.py "C:\\path\\phase3_results"
"""
import sys
import glob
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

if len(sys.argv) > 1:
    P3 = Path(sys.argv[1])
else:
    P3 = Path.home() / "Desktop" / "jailbreak_q1" / "phase3_results"
OUT = P3.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

files = glob.glob(str(P3 / "**" / "*_results.csv"), recursive=True)
df = pd.concat([pd.read_csv(f) for f in files if "defense" in pd.read_csv(f, nrows=1).columns],
               ignore_index=True)
att = df[df.prompt_type == "attack"]

MODELS = {"mistral_7b": ("Mistral-7B", "o"), "llama31_8b": ("Llama-3.1-8B", "s")}
DEFENSES = {
    "keyword_blacklist": ("Keyword blacklist", "#2166AC"),
    "self_reminder":     ("Self-reminder",     "#1A7941"),
    "paraphrase":        ("Paraphrase",        "#E08214"),
    "perplexity_filter": ("Perplexity filter", "#762A83"),
}

# Per-label offsets (x, y) from the point, in data units. Chosen by hand so no
# label overlaps another label or the legend.
OFFSET = {
    ("self_reminder", "llama31_8b"):     (-0.95, 8),
    ("paraphrase", "llama31_8b"):       (0.35, 6),
    ("self_reminder", "mistral_7b"):    (0.35, -10),
    ("keyword_blacklist", "llama31_8b"):(0.35, 6),
    ("keyword_blacklist", "mistral_7b"):(0.35, -4),
    ("perplexity_filter", "llama31_8b"):(-1.4, -9),
    ("perplexity_filter", "mistral_7b"):(0.35, -9),
    ("paraphrase", "mistral_7b"):       (-2.6, 9),
}

rows = []
for model in MODELS:
    base = att[(att.defense == "no_defense") & (att.model == model)].jailbreak_success.mean() * 100
    for d in DEFENSES:
        sub = att[(att.defense == d) & (att.model == model)]
        if len(sub) == 0:
            continue
        asr = sub.jailbreak_success.mean() * 100
        rows.append({"model": model, "defense": d, "baseline": base, "asr": asr,
                     "reduction": base - asr, "latency": sub.total_latency_s.mean()})
R = pd.DataFrame(rows)
R.to_csv(OUT / "fig5_points.csv", index=False)

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300})
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.axhline(0, color="#888", lw=0.9, zorder=1)
avg_lat = R.latency.mean()
ax.axvline(avg_lat, color="#bbb", lw=0.9, ls=":", zorder=1)

for _, r in R.iterrows():
    label, marker = MODELS[r.model]
    color = DEFENSES[r.defense][1]
    ax.scatter(r.latency, r.reduction, marker=marker, s=110, color=color,
               edgecolor="white", lw=0.8, zorder=3)
    dx, dy = OFFSET.get((r.defense, r.model), (0.3, 4))
    ax.annotate(f"{DEFENSES[r.defense][0]}\n{label}",
                xy=(r.latency, r.reduction), xytext=(r.latency + dx, r.reduction + dy),
                fontsize=7.8, color=color, va="center",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.5, alpha=0.6))

ax.set_xlabel("Mean inference latency per prompt (s)")
ax.set_ylabel("ASR reduction vs matched baseline (pp)")
ax.set_xlim(1.4, 7.6)
ax.set_ylim(-16, 52)
ax.grid(alpha=0.25, lw=0.5)
ax.set_axisbelow(True)

handles = [Line2D([], [], marker="s", ls="", color="#555", label="Llama-3.1-8B"),
           Line2D([], [], marker="o", ls="", color="#555", label="Mistral-7B")]
handles += [Line2D([], [], marker="s", ls="", color=c, label=n) for n, c in DEFENSES.values()]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3,
          frameon=False, fontsize=8)

ax.set_title("Defense effectiveness vs latency (matched baselines)", fontsize=10, pad=8)
fig.tight_layout()
fig.savefig(OUT / "fig5_latency_tradeoff.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig5_latency_tradeoff.png", bbox_inches="tight")
print(R.round(2).to_string(index=False))
print(f"Saved: {OUT / 'fig5_latency_tradeoff.pdf'}")

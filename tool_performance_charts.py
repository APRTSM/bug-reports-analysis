import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================
# CONFIG
# ============================

DATA_DIR = Path(".")
IN_FILE = DATA_DIR / "tool_comparison_summary.csv"
OUT_DIR = DATA_DIR / "tool_performance_charts"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ============================
# 1. LOAD DATA
# ============================

df = pd.read_csv(IN_FILE)
print(f"Loaded {len(df)} rows from {IN_FILE}")

# ============================
# 2. COMPUTE AGGREGATE STATISTICS
# ============================

# Calculate MRR: 1/rank if rank is not NaN, else 0
df["mrr"] = np.where(df["rank"].notna(), 1.0 / df["rank"], 0.0)

# Group by tool and compute statistics
stats = df.groupby("tool").agg({
    "top@1": ["sum", "mean"],
    "top@5": ["sum", "mean"],
    "mrr": "mean",
    "detected": lambda x: (x == "Yes").sum(),
}).round(3)

# Flatten column names
stats.columns = ["top1_count", "top1_rate", "top5_count", "top5_rate", "mrr", "detected_count"]
stats["not_detected_count"] = df.groupby("tool").size() - stats["detected_count"]
stats["total"] = df.groupby("tool").size()

# Reset index to make tool a column
stats = stats.reset_index()

print("\nTool Performance Statistics:")
print(stats)

# Save statistics
stats.to_csv(OUT_DIR / "tool_performance_stats.csv", index=False)
print(f"\nSaved statistics to {OUT_DIR / 'tool_performance_stats.csv'}")

# ============================
# 3. CREATE BAR CHARTS
# ============================

# Set style
try:
    plt.style.use("seaborn-v0_8-darkgrid")
except OSError:
    try:
        plt.style.use("seaborn-darkgrid")
    except OSError:
        plt.style.use("default")
colors = ["#2E86AB", "#A23B72", "#F18F01"]

# ============================
# Chart 1: Top@1 and Top@5 Performance Rates
# ============================

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(stats))
width = 0.35

bars1 = ax.bar(x - width/2, stats["top1_rate"] * 100, width, 
                label="Top@1", color=colors[0], alpha=0.8)
bars2 = ax.bar(x + width/2, stats["top5_rate"] * 100, width, 
                label="Top@5", color=colors[1], alpha=0.8)

ax.set_xlabel("Tool", fontsize=12, fontweight="bold")
ax.set_ylabel("Success Rate (%)", fontsize=12, fontweight="bold")
ax.set_title("Tool Performance: Top@1 and Top@5 Success Rates", 
             fontsize=14, fontweight="bold", pad=20)
ax.set_xticks(x)
ax.set_xticklabels(stats["tool"], fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(OUT_DIR / "top1_top5_performance.png", dpi=300, bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'top1_top5_performance.png'}")
plt.close()

# ============================
# Chart 2: Detected vs Not Detected Counts
# ============================

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(stats))
width = 0.35

bars1 = ax.bar(x - width/2, stats["detected_count"], width, 
                label="Detected", color=colors[0], alpha=0.8)
bars2 = ax.bar(x + width/2, stats["not_detected_count"], width, 
                label="Not Detected", color=colors[2], alpha=0.8)

ax.set_xlabel("Tool", fontsize=12, fontweight="bold")
ax.set_ylabel("Number of Bugs", fontsize=12, fontweight="bold")
ax.set_title("Tool Detection: Detected vs Not Detected", 
             fontsize=14, fontweight="bold", pad=20)
ax.set_xticks(x)
ax.set_xticklabels(stats["tool"], fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(OUT_DIR / "detected_vs_not_detected.png", dpi=300, bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'detected_vs_not_detected.png'}")
plt.close()

# ============================
# Chart 3: Combined Performance Overview (Stacked Bar)
# ============================

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(stats))
width = 0.6

# Calculate percentages for stacked bars
top1_pct = (stats["top1_count"] / stats["total"] * 100).values
top5_only_pct = ((stats["top5_count"] - stats["top1_count"]) / stats["total"] * 100).values
not_top5_pct = ((stats["total"] - stats["top5_count"]) / stats["total"] * 100).values

bars1 = ax.bar(x, top1_pct, width, label="Top@1", color=colors[0], alpha=0.9)
bars2 = ax.bar(x, top5_only_pct, width, bottom=top1_pct, 
                label="Top@5 (but not Top@1)", color=colors[1], alpha=0.9)
bars3 = ax.bar(x, not_top5_pct, width, bottom=top1_pct + top5_only_pct,
                label="Not in Top@5", color=colors[2], alpha=0.9)

ax.set_xlabel("Tool", fontsize=12, fontweight="bold")
ax.set_ylabel("Percentage of Bugs (%)", fontsize=12, fontweight="bold")
ax.set_title("Tool Performance Breakdown: Top@1, Top@5, and Below", 
             fontsize=14, fontweight="bold", pad=20)
ax.set_xticks(x)
ax.set_xticklabels(stats["tool"], fontsize=11)
ax.legend(fontsize=10, loc="upper left")
ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.3)

# Add total count labels on top
for i, (tool, total) in enumerate(zip(stats["tool"], stats["total"])):
    ax.text(i, 102, f'n={total}', ha='center', va='bottom', 
            fontsize=9, fontweight="bold", style='italic')

plt.tight_layout()
plt.savefig(OUT_DIR / "performance_breakdown_stacked.png", dpi=300, bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'performance_breakdown_stacked.png'}")
plt.close()

# ============================
# Chart 4: Detection Rate
# ============================

fig, ax = plt.subplots(figsize=(10, 6))
detection_rate = (stats["detected_count"] / stats["total"] * 100).values

bars = ax.bar(stats["tool"], detection_rate, color=colors, alpha=0.8)

ax.set_xlabel("Tool", fontsize=12, fontweight="bold")
ax.set_ylabel("Detection Rate (%)", fontsize=12, fontweight="bold")
ax.set_title("Tool Detection Rate", 
             fontsize=14, fontweight="bold", pad=20)
ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.3)

# Add value labels on bars
for bar, rate, detected, total in zip(bars, detection_rate, 
                                       stats["detected_count"], stats["total"]):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{rate:.1f}%\n({int(detected)}/{int(total)})',
            ha='center', va='bottom', fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(OUT_DIR / "detection_rate.png", dpi=300, bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'detection_rate.png'}")
plt.close()

# ============================
# Chart 5: Comprehensive Comparison (All Metrics)
# ============================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Comprehensive Tool Performance Comparison", 
             fontsize=16, fontweight="bold", y=0.995)

# Subplot 1: Top@1 and Top@5 rates
ax1 = axes[0, 0]
x = np.arange(len(stats))
width = 0.35
ax1.bar(x - width/2, stats["top1_rate"] * 100, width, label="Top@1", 
        color=colors[0], alpha=0.8)
ax1.bar(x + width/2, stats["top5_rate"] * 100, width, label="Top@5", 
        color=colors[1], alpha=0.8)
ax1.set_xlabel("Tool", fontweight="bold")
ax1.set_ylabel("Success Rate (%)", fontweight="bold")
ax1.set_title("Top@1 and Top@5 Performance", fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(stats["tool"])
ax1.legend()
ax1.set_ylim(0, 100)
ax1.grid(axis="y", alpha=0.3)

# Subplot 2: Detection counts
ax2 = axes[0, 1]
x = np.arange(len(stats))
width = 0.35
ax2.bar(x - width/2, stats["detected_count"], width, label="Detected", 
        color=colors[0], alpha=0.8)
ax2.bar(x + width/2, stats["not_detected_count"], width, label="Not Detected", 
        color=colors[2], alpha=0.8)
ax2.set_xlabel("Tool", fontweight="bold")
ax2.set_ylabel("Number of Bugs", fontweight="bold")
ax2.set_title("Detection Counts", fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(stats["tool"])
ax2.legend()
ax2.grid(axis="y", alpha=0.3)

# Subplot 3: Detection rate
ax3 = axes[1, 0]
detection_rate = (stats["detected_count"] / stats["total"] * 100).values
ax3.bar(stats["tool"], detection_rate, color=colors, alpha=0.8)
ax3.set_xlabel("Tool", fontweight="bold")
ax3.set_ylabel("Detection Rate (%)", fontweight="bold")
ax3.set_title("Detection Rate", fontweight="bold")
ax3.set_ylim(0, 100)
ax3.grid(axis="y", alpha=0.3)
for i, (rate, detected, total) in enumerate(zip(detection_rate, 
                                                   stats["detected_count"], 
                                                   stats["total"])):
    ax3.text(i, rate + 2, f'{rate:.1f}%', ha='center', va='bottom', 
             fontsize=9, fontweight="bold")

# Subplot 4: Top@1 and Top@5 counts
ax4 = axes[1, 1]
x = np.arange(len(stats))
width = 0.35
ax4.bar(x - width/2, stats["top1_count"], width, label="Top@1", 
        color=colors[0], alpha=0.8)
ax4.bar(x + width/2, stats["top5_count"], width, label="Top@5", 
        color=colors[1], alpha=0.8)
ax4.set_xlabel("Tool", fontweight="bold")
ax4.set_ylabel("Number of Bugs", fontweight="bold")
ax4.set_title("Top@1 and Top@5 Counts", fontweight="bold")
ax4.set_xticks(x)
ax4.set_xticklabels(stats["tool"])
ax4.legend()
ax4.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "comprehensive_comparison.png", dpi=300, bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'comprehensive_comparison.png'}")
plt.close()

print(f"\n✅ All charts saved to: {OUT_DIR}")
print("Done!")


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================
# CONFIG
# ============================

DATA_DIR = Path(__file__).resolve().parent
IN_FILE = DATA_DIR / "tool_comparison_summary.csv"
MAP_COMPARISON_FILE = DATA_DIR / "map_calculation_comparison.csv"  # optional; not present, handled gracefully below
OUT_DIR = DATA_DIR / "tool_performance_charts"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ============================
# 1. LOAD DATA
# ============================

df = pd.read_csv(IN_FILE)
print(f"Loaded {len(df)} rows from {IN_FILE}")

# Load recalculated MAP values from comparison file
if MAP_COMPARISON_FILE.exists():
    print(f"Loading recalculated MAP values from {MAP_COMPARISON_FILE}")
    map_comparison = pd.read_csv(MAP_COMPARISON_FILE)
    
    # Merge recalculated MAP values
    # Use map_recalc_single_gt (MAP = MRR) as it's the most consistent
    # Alternative: use map_recalc_with_gt if you want to account for multiple GT locations
    merge_cols = ['project', 'bug_id', 'tool']
    df = df.merge(
        map_comparison[merge_cols + ['map_recalc_single_gt']],
        on=merge_cols,
        how='left',
        suffixes=('', '_recalc')
    )
    
    # Use recalculated MAP if available, otherwise fall back to original
    df['map'] = df['map_recalc_single_gt'].fillna(df['map'])
    print(f"  Updated MAP values using recalculated MAP (single GT assumption)")
    print(f"  Rows with recalculated MAP: {df['map_recalc_single_gt'].notna().sum()}")
else:
    print(f"Warning: {MAP_COMPARISON_FILE} not found. Using original MAP values.")

# ============================
# 2. COMPUTE AGGREGATE STATISTICS
# ============================

# Calculate MRR@5: 1/rank if rank <= 5 and rank is not NaN, else 0
df["mrr_at5"] = np.where(
    (df["rank"].notna()) & (df["rank"] <= 5), 
    1.0 / df["rank"], 
    0.0
)

# Calculate MAP@5: Use existing MAP value if rank <= 5, else 0
# The existing 'map' column may account for multiple ground truth locations
# MAP@5 only considers cases where the bug was found in top 5
if "map" in df.columns:
    df["map_at5"] = np.where(
        (df["rank"].notna()) & (df["rank"] <= 5), 
        df["map"], 
        0.0
    )
else:
    # Fallback: if no map column, use MRR@5 (for single ground truth, MAP = MRR)
    df["map_at5"] = df["mrr_at5"]

# Group by tool and compute statistics
stats = df.groupby("tool").agg({
    "top@1": ["sum", "mean"],
    "top@5": ["sum", "mean"],
    "mrr_at5": "mean",
    "map_at5": "mean",
    "detected": lambda x: (x == "Yes").sum(),
}).round(3)

# Flatten column names
stats.columns = ["top1_count", "top1_rate", "top5_count", "top5_rate", "mrr_at5", "map_at5", "detected_count"]
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
            #f'{rate:.1f}%\n({int(detected)}/{int(total)})',
            f'{rate:.1f}%',
            ha='center', va='bottom', fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(OUT_DIR / "detection_rate.png", dpi=300, bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'detection_rate.png'}")
plt.close()

# ============================
# Chart 5: MRR@5 and MAP@5
# ============================

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(stats))
width = 0.35

bars1 = ax.bar(x - width/2, stats["mrr_at5"] * 100, width, 
                label="MRR@5", color=colors[0], alpha=0.8)
bars2 = ax.bar(x + width/2, stats["map_at5"] * 100, width, 
                label="MAP@5", color=colors[1], alpha=0.8)

ax.set_xlabel("Tool", fontsize=12, fontweight="bold")
ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
ax.set_title("Tool Performance: MRR@5 and MAP@5", 
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
plt.savefig(OUT_DIR / "mrr_map_at5.png", dpi=300, bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'mrr_map_at5.png'}")
plt.close()

# ============================
# Chart 6: Comprehensive Comparison (All Metrics)
# ============================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
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
ax1.set_xticklabels(stats["tool"], rotation=45, ha='right')
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
ax2.set_xticklabels(stats["tool"], rotation=45, ha='right')
ax2.legend()
ax2.grid(axis="y", alpha=0.3)

# Subplot 3: MRR@5 and MAP@5
ax3 = axes[0, 2]
x = np.arange(len(stats))
width = 0.35
ax3.bar(x - width/2, stats["mrr_at5"] * 100, width, label="MRR@5", 
        color=colors[0], alpha=0.8)
ax3.bar(x + width/2, stats["map_at5"] * 100, width, label="MAP@5", 
        color=colors[1], alpha=0.8)
ax3.set_xlabel("Tool", fontweight="bold")
ax3.set_ylabel("Score (%)", fontweight="bold")
ax3.set_title("MRR@5 and MAP@5", fontweight="bold")
ax3.set_xticks(x)
ax3.set_xticklabels(stats["tool"], rotation=45, ha='right')
ax3.legend()
ax3.set_ylim(0, 100)
ax3.grid(axis="y", alpha=0.3)

# Subplot 4: Detection rate
ax4 = axes[1, 0]
detection_rate = (stats["detected_count"] / stats["total"] * 100).values
ax4.bar(stats["tool"], detection_rate, color=colors, alpha=0.8)
ax4.set_xlabel("Tool", fontweight="bold")
ax4.set_ylabel("Detection Rate (%)", fontweight="bold")
ax4.set_title("Detection Rate", fontweight="bold")
ax4.set_xticks(range(len(stats)))
ax4.set_xticklabels(stats["tool"], rotation=45, ha='right')
ax4.set_ylim(0, 100)
ax4.grid(axis="y", alpha=0.3)
for i, (rate, detected, total) in enumerate(zip(detection_rate, 
                                                   stats["detected_count"], 
                                                   stats["total"])):
    ax4.text(i, rate + 2, f'{rate:.1f}%', ha='center', va='bottom', 
             fontsize=9, fontweight="bold")

# Subplot 5: Top@1 and Top@5 counts
ax5 = axes[1, 1]
x = np.arange(len(stats))
width = 0.35
ax5.bar(x - width/2, stats["top1_count"], width, label="Top@1", 
        color=colors[0], alpha=0.8)
ax5.bar(x + width/2, stats["top5_count"], width, label="Top@5", 
        color=colors[1], alpha=0.8)
ax5.set_xlabel("Tool", fontweight="bold")
ax5.set_ylabel("Number of Bugs", fontweight="bold")
ax5.set_title("Top@1 and Top@5 Counts", fontweight="bold")
ax5.set_xticks(x)
ax5.set_xticklabels(stats["tool"], rotation=45, ha='right')
ax5.legend()
ax5.grid(axis="y", alpha=0.3)

# Subplot 6: MRR@5 and MAP@5 values
ax6 = axes[1, 2]
x = np.arange(len(stats))
width = 0.35
ax6.bar(x - width/2, stats["mrr_at5"], width, label="MRR@5", 
        color=colors[0], alpha=0.8)
ax6.bar(x + width/2, stats["map_at5"], width, label="MAP@5", 
        color=colors[1], alpha=0.8)
ax6.set_xlabel("Tool", fontweight="bold")
ax6.set_ylabel("Score (0-1)", fontweight="bold")
ax6.set_title("MRR@5 and MAP@5 (Raw Values)", fontweight="bold")
ax6.set_xticks(x)
ax6.set_xticklabels(stats["tool"], rotation=45, ha='right')
ax6.legend()
ax6.set_ylim(0, 1)
ax6.grid(axis="y", alpha=0.3)
for i, (mrr, map_val) in enumerate(zip(stats["mrr_at5"], stats["map_at5"])):
    ax6.text(i - width/2, mrr + 0.02, f'{mrr:.3f}', ha='center', va='bottom', 
             fontsize=8, fontweight="bold")
    ax6.text(i + width/2, map_val + 0.02, f'{map_val:.3f}', ha='center', va='bottom', 
             fontsize=8, fontweight="bold")

plt.tight_layout()
plt.savefig(OUT_DIR / "comprehensive_comparison.png", dpi=300, bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'comprehensive_comparison.png'}")
plt.close()

print(f"\n✅ All charts saved to: {OUT_DIR}")
print("Done!")


#!/usr/bin/env python3
"""
Compare training results from multiple experiments
Usage: python compare_results.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_history(exp_dir):
    """Load training history from experiment directory"""
    csv_path = Path(exp_dir) / "train_history.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def plot_comparison(experiments, metric="mean_iou", save_path=None):
    """Plot comparison of training curves"""
    plt.figure(figsize=(12, 6))
    
    for name, df in experiments.items():
        if df is not None:
            plt.plot(df["epoch"], df[metric], marker='o', label=name, linewidth=2)
    
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(metric.replace("_", " ").title(), fontsize=12)
    plt.title(f"Training Comparison: {metric.replace('_', ' ').title()}", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f">>> Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_summary(experiments):
    """Print summary statistics for all experiments"""
    print("\n" + "="*80)
    print("EXPERIMENTS SUMMARY")
    print("="*80)
    
    summary_data = []
    
    for name, df in experiments.items():
        if df is None:
            print(f"\n{name}: No data found")
            continue
        
        best_epoch = df.loc[df["mean_iou"].idxmax()]
        final_epoch = df.iloc[-1]
        
        summary_data.append({
            "Experiment": name,
            "Best Epoch": int(best_epoch["epoch"]),
            "Best mIoU": best_epoch["mean_iou"],
            "Best Pixel Acc": best_epoch["pixel_accuracy"],
            "Best Val Loss": best_epoch["val_loss"],
            "Final mIoU": final_epoch["mean_iou"],
            "Final Pixel Acc": final_epoch["pixel_accuracy"],
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(index=False))
        print("\n" + "="*80)
        
        # Find best experiment
        best_idx = summary_df["Best mIoU"].idxmax()
        best_exp = summary_df.iloc[best_idx]
        print(f"\n🏆 BEST EXPERIMENT: {best_exp['Experiment']}")
        print(f"   Best mIoU: {best_exp['Best mIoU']:.4f} at epoch {int(best_exp['Best Epoch'])}")
        print(f"   Best Pixel Accuracy: {best_exp['Best Pixel Acc']:.4f}")
        print("="*80 + "\n")
        
        return summary_df
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Compare UNet++ experiment results")
    parser.add_argument(
        "--exp_dirs",
        nargs="+",
        default=[
            "experiments/exp1_resnet18_ds",
            "experiments/exp2_resnet50_ds",
            "experiments/exp3_resnet101_ds",
            "experiments/exp4_mobilevit_ds",
            "experiments/exp5_mobilenetv4_ds",
        ],
        help="List of experiment directories to compare"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="comparison_results",
        help="Directory to save comparison plots"
    )
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all experiments
    print("\n>>> Loading experiment results...")
    experiments = {}
    for exp_dir in args.exp_dirs:
        exp_path = Path(exp_dir)
        if exp_path.exists():
            name = exp_path.name
            df = load_history(exp_path)
            if df is not None:
                experiments[name] = df
                print(f"    ✓ Loaded {name}")
            else:
                print(f"    ✗ No history found in {name}")
        else:
            print(f"    ✗ Directory not found: {exp_dir}")
    
    if not experiments:
        print("\n❌ No experiment data found!")
        return
    
    # Print summary
    summary_df = print_summary(experiments)
    
    # Save summary to CSV
    if summary_df is not None:
        summary_path = output_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f">>> Summary saved to {summary_path}")
    
    # Plot comparisons
    print("\n>>> Generating comparison plots...")
    
    metrics = [
        ("mean_iou", "mIoU Comparison"),
        ("pixel_accuracy", "Pixel Accuracy Comparison"),
        ("val_loss", "Validation Loss Comparison"),
        ("train_loss", "Training Loss Comparison"),
    ]
    
    for metric, title in metrics:
        plot_path = output_dir / f"{metric}_comparison.png"
        plot_comparison(experiments, metric=metric, save_path=plot_path)
    
    # Create combined plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("UNet++ Experiments Comparison", fontsize=16, fontweight='bold')
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        for name, df in experiments.items():
            if df is not None:
                ax.plot(df["epoch"], df[metric], marker='o', label=name, linewidth=2)
        
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    combined_path = output_dir / "combined_comparison.png"
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    print(f">>> Saved combined plot to {combined_path}")
    plt.close()
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"Results saved in: {output_dir}/")
    print("  - summary.csv")
    print("  - mean_iou_comparison.png")
    print("  - pixel_accuracy_comparison.png")
    print("  - val_loss_comparison.png")
    print("  - train_loss_comparison.png")
    print("  - combined_comparison.png")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

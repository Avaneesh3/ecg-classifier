import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal

SAMPLE_RATE = 300

def summarize_class_distribution(labels, output_dir="analysis_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    counts = labels.value_counts().sort_index()
    counts.plot(kind="bar", title="Class Distribution", rot=0)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"))
    plt.clf()
    return counts


def analyze_lengths(X, y, output_dir="analysis_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    lengths = [len(sig) for sig in X]
    y = y.values.flatten()
    length_df = pd.DataFrame({'length': lengths, 'label': y})
    length_df.boxplot(column='length', by='label', grid=False)
    plt.title("Boxplot of Signal Lengths by Class")
    plt.suptitle("")
    plt.xlabel("Class")
    plt.ylabel("Length")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "length_boxplot.png"))
    plt.clf()
    return length_df


def compute_descriptive_stats(X, y, output_dir="analysis_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    stats_list = []
    for cls in sorted(set(y)):
        class_signals = [sig for sig, label in zip(X, y) if label == cls]
        lengths = [len(sig) for sig in class_signals]
        all_vals = np.concatenate(class_signals)
        stats_list.append({
            'class': cls,
            'mean_amplitude': np.mean(all_vals),
            'std_amplitude': np.std(all_vals),
            'min_val': np.min(all_vals),
            'max_val': np.max(all_vals),
            'median_amplitude': np.median(all_vals),
            'iqr': np.percentile(all_vals, 75) - np.percentile(all_vals, 25),
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths)
        })

    stats_df = pd.DataFrame(stats_list).set_index('class')
    stats_df.to_csv(os.path.join(output_dir, "class_statistics.csv"))

    stats_df[['mean_amplitude', 'std_amplitude', 'iqr', 'mean_length']].plot(kind='bar')
    plt.title("Summary Stats Differentiating Classes")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_comparison_summary.png"))
    plt.clf()

    return stats_df

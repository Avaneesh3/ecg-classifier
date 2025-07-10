import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import random

SAMPLE_RATE = 300

def export_and_visualize_class_statistics(X, y, output_dir="analysis_outputs"):
    from .descriptive import compute_descriptive_stats

    stats_df = compute_descriptive_stats(X, y, output_dir)

    for cls in sorted(set(y)):
        class_signals = [np.array(sig) for sig, label in zip(X, y) if label == cls]
        min_len = min(len(sig) for sig in class_signals)
        aligned = np.stack([sig[:min_len] for sig in class_signals])
        mean_signal = np.mean(aligned, axis=0)
        std_signal = np.std(aligned, axis=0)

        t = np.arange(min_len) / SAMPLE_RATE
        plt.figure(figsize=(8, 3))
        plt.plot(t, mean_signal, label="Mean")
        plt.fill_between(t, mean_signal - std_signal, mean_signal + std_signal, alpha=0.3, label="±1 Std Dev")
        plt.title(f"Class {cls} ECG Signal Overlay")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"signal_overlay_class_{cls}.png"))
        plt.close()

    for cls in sorted(set(y)):
        example_sig = next(np.array(sig) for sig, label in zip(X, y) if label == cls and len(sig) > 500)
        f, t, Zxx = scipy.signal.stft(example_sig, fs=SAMPLE_RATE, nperseg=256)
        plt.figure(figsize=(8, 4))
        plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
        plt.title(f"STFT Spectrogram (Class {cls})")
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar(label="Magnitude")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"stft_class_{cls}.png"))
        plt.close()


def plot_sample_ecg_signals(X, y, samples_per_class=3, output_dir="analysis_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    y = np.array(y)
    for cls in sorted(set(y)):
        fig, axs = plt.subplots(samples_per_class, 1, figsize=(10, 2 * samples_per_class), sharex=True)
        axs = np.atleast_1d(axs)
        class_indices = np.where(y == cls)[0]
        chosen_indices = random.sample(list(class_indices), min(samples_per_class, len(class_indices)))
        for i, idx in enumerate(chosen_indices):
            signal = X[idx]
            t = np.arange(len(signal)) / SAMPLE_RATE
            axs[i].plot(t, signal)
            axs[i].set_title(f"Class {cls} - Sample {i+1}")
            axs[i].set_ylabel("Amplitude")
        axs[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"ecg_examples_class_{cls}.png"))
        plt.close()

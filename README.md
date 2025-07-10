# ECG Time Series Classification

This project focuses on classifying univariate ECG (electrocardiogram) time series data into multiple classes using various preprocessing steps, descriptive analysis, and deep learning models.

## 📁 Project Structure

```

.
├── data/                      # Raw binary ECG signals and labels
├── models/                   # Saved models (optional)
├── analysis\_outputs/         # Plots, statistics, and summaries
├── notebooks/                # Jupyter notebooks for exploration
├── main.py                   # Main training and evaluation script
├── preprocess.py             # Data loading and preprocessing
├── analysis.py               # Statistical and visual analysis
├── model.py                  # Model architecture definitions
├── utils.py                  # Helper functions
└── README.md

````

---

## 🧪 Task Overview

The goal is to classify univariate ECG signals into one of four classes. The dataset is provided in a binary format containing ragged (variable-length) time series and corresponding labels.

---

## 🔍 Functionality

### ✅ Data Parsing
- Efficiently reads ragged binary files into Python lists.
- Converts signal data into appropriate formats for model training.

### 📊 Exploratory Data Analysis (EDA)
- `summarize_class_distribution`: Visualizes the distribution of classes.
- `analyze_lengths`: Shows how signal length varies by class.
- `compute_descriptive_stats`: Computes amplitude and length statistics by class.
- All plots and stats are saved to `analysis_outputs/`.

### 📈 Visualization
- Overlay plots and class-specific ECG signal examples.
- Spectrogram visualizations (optional) to examine time-frequency characteristics.

### 🧠 Model Architecture (Suggested Baseline)
- **Short-Time Fourier Transform** (STFT)
- **Conv2D → ReLU → MaxPool → Conv2D → ReLU → MaxPool**
- **GRU Layer** (RNN) for temporal dynamics
- **Fully Connected Linear Layer** for classification

---

## 📦 Dependencies

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- PyTorch
- SciPy (for spectrogram/STFT if used)

Install dependencies using:

```bash
pip install -r requirements.txt
````

---

## 🚀 Running the Pipeline

1. **Prepare Data**

   * Place your raw binary files in the `data/` folder.

2. **Run Analysis**

   ```bash
   python analysis.py
   ```

3. **Train Model**

   ```bash
   python main.py
   ```

4. **Evaluate**

   * View saved plots in `analysis_outputs/`
   * Evaluate final model performance via printed metrics.

---

## 📊 Output Examples

* `class_distribution.png`
* `length_boxplot.png`
* `class_statistics.csv`
* `class_comparison_summary.png`
* Sample ECG signal plots

---

## 📌 Notes

* Handles variable-length time series using padding or STFT transformation.
* Encourages architecture exploration — model size and configuration are not fixed.
* Code is modular and extensible for other time series tasks.

---

## 🧠 Future Improvements

* Hyperparameter tuning and cross-validation
* Attention-based models
* Data augmentation or noise robustness
* Model explainability via saliency maps

---

## 📧 Contact

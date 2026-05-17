
# 🏦 Egyptian Banknote Counter

> Automatic detection and valuation of Egyptian Pound (EGP) banknotes using Traditional Computer Vision techniques.

---

##  Project Overview

This project identifies and counts Egyptian currency denominations from input images and computes the total monetary value. It uses a **Traditional CV pipeline** — no deep learning required — built entirely with classical image processing and machine learning techniques.

**Supported Denominations:** `1` · `5` · `10` · `10 (new)` · `20` · `20 (new)` · `50` · `100` · `200` EGP

---

##  Pipeline

```
Input Image
    │
    ▼
Grayscale Conversion
    │
    ▼
Gaussian Blur  (noise removal)
    │
    ▼
Otsu Thresholding  (auto binarization)
    │
    ▼
Canny Edge Detection
    │
    ▼
Morphological Closing  (connect broken edges)
    │
    ▼
Contour Detection  (locate banknote)
    │
    ▼
Crop & Resize to 224×224
    │
    ▼
Feature Extraction
  ├── Color Histogram (HSV)
  ├── LBP  — texture
  ├── HOG  — shape/gradients
  ├── Color Moments  — mean, std, skewness
  ├── Gabor Filters  — frequency patterns
  └── Dominant Colors (KMeans k=5)
    │
    ▼
StandardScaler → PCA (300 components)
    │
    ▼
SVM Classifier (RBF kernel, C=100)
    │
    ▼
Result: Class + Confidence + EGP Value
```

---

##  Project Structure

```
CV_project/
├── README.md
├── Notebook.ipynb        ← All cells (Colab)
└── svm_model.pkl         ← Saved model (scaler + pca + svm)
```

---

##  Dataset

| Split      | Images | Percentage |
|------------|--------|------------|
| Train      | 2,633  | 71.6%      |
| Validation | 757    | 20.6%      |
| Test       | 288    | 7.8%       |
| **Total**  | **3,678** | **100%** |

**Source:** [Egyptian New Currency 2023 — Kaggle](https://www.kaggle.com/datasets/belalsafy/egyptian-new-currency-2023)

Each denomination is captured from multiple angles, backgrounds, and lighting conditions.

---

##  Features Used

| Feature | Description | Dimensions |
|---------|-------------|------------|
| Color Histogram (HSV) | Distribution of hue, saturation, brightness | 96 |
| LBP | Local texture patterns | 26 |
| HOG | Shape and gradient orientations | ~1764 |
| Color Moments | Mean, std, skewness per channel | 9 |
| Gabor Filters | Frequency-based texture at 4 angles × 3 frequencies | 24 |
| Dominant Colors | Top 5 colors via KMeans | 15 |

> After PCA: reduced to **300 components** (90%+ variance retained)

---

##  Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~75%   |
| Best Class | 1 EGP (100% precision & recall) |
| Weakest Class | 200 EGP |

---

##  How to Run

### 1. Open in Google Colab

Upload `Notebook.ipynb` to Colab and run cells in order.

### 2. Install Dependencies

```python
!pip install scikit-image scikit-learn opencv-python-headless gradio -q
```

### 3. Download Dataset

```python
!pip install kaggle -q
!kaggle datasets download -d belalsafy/egyptian-new-currency-2023
```

### 4. Run All Cells

```
Cell 1 → Mount Drive
Cell 2 → Extract dataset
Cell 3 → Install libraries
Cell 4 → Define functions
Cell 5 → Train SVM
Cell 6 → Evaluate on test set
Cell 7 → Launch Gradio app
```

### 5. Use the App

After Cell 7 runs, you'll get a public URL:
```
Running on public URL: https://xxxx.gradio.live
```
Upload 1 or 2 banknote images → click **احسب** → see result + total EGP value.

---

##  App Interface

- Upload up to **2 images**
- Each image gets annotated with contour + bounding box
- Confidence threshold: if model confidence < 35%, returns **"غير واضح"** instead of a wrong answer
- Displays total EGP value across both images

---

##  Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| OpenCV | Image processing |
| scikit-learn | SVM, PCA, StandardScaler |
| scikit-image | LBP, HOG |
| Gradio | Web interface |
| Google Colab | Training environment (free GPU/CPU) |

---

##  Notes

- The model was trained on `train + valid` splits combined for maximum data.
- `class_weight='balanced'` was used to handle the imbalanced `1 EGP` class (only 60 training samples vs ~315 for others).
- Confidence thresholding prevents overconfident wrong predictions.


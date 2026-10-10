# 🐾 Cat vs. Dog Image Classifier

A deep learning project that classifies images as **cat** or **dog**. It compares three approaches: a baseline CNN built from scratch, the same CNN with regularization, and transfer learning with **MobileNetV2**. The best model is deployed as an interactive **Streamlit** web app.

**Best result: 98.66% validation accuracy** using transfer learning (MobileNetV2).

---

## 📌 Project Overview

| | |
| :--- | :--- |
| **Task** | Binary image classification (cat = 0, dog = 1) |
| **Framework** | TensorFlow / Keras |
| **Input size** | 256 × 256 × 3 |
| **Dataset** | [Dogs vs Cats (Kaggle)](https://www.kaggle.com/datasets/salader/dogsvscats/data): 20,000 training images and 5,000 test/validation images (2 classes) |
| **Deployment** | Streamlit web app |

---

## 📂 Repository Structure

```
├── Results/                          # Screenshots of results and the app
├── app.py                            # Streamlit web app
├── cat_vs_dog_classification.ipynb   # Data pipeline, model training and evaluation
├── model_transfer.keras              # Trained model used by the app
└── README.md
```

---

## 🧠 Methodology

### 1. Data Pipeline
- Loaded images from folders with `image_dataset_from_directory` (labels inferred from folder names, batch size 32, resized to 256 × 256).
- Normalized pixel values from `[0, 255]` to `[0, 1]`.

### 2. Models Trained

**Model 1: Baseline CNN**
- 3 × (Conv2D 32 filters + MaxPooling) → Flatten → Dense(128) → Dense(64) → Dense(1, sigmoid)
- Trained for 10 epochs with the Adam optimizer and binary cross-entropy.
- The model **overfit heavily**: training accuracy reached ~98.5% while validation accuracy stayed around 75%, and validation loss rose after the first epochs.

**Model 2: CNN with Regularization**
- Same architecture, plus **data augmentation** (random flip, rotation and zoom), a **Dropout (0.5)** layer and **Early Stopping** (patience = 3, restores the best weights).
- Training stopped at epoch 18 (of a maximum of 30).
- This removed most of the overfitting and raised validation accuracy to **83.38%**.

**Model 3: Transfer Learning with MobileNetV2** ⭐
- MobileNetV2 pre-trained on ImageNet, used as a **frozen** feature extractor.
- Data augmentation → Rescaling to `[-1, 1]` → MobileNetV2 → GlobalAveragePooling2D → Dense(128) → Dropout(0.5) → Dense(1, sigmoid).
- Trained with Early Stopping (maximum 15 epochs).

---

## 📊 Results

| Model | Validation Accuracy | Validation Loss |
| :--- | :---: | :---: |
| Baseline CNN | 75.96% | 1.5365 |
| CNN + Augmentation + Dropout + Early Stopping | 83.38% | 0.3738 |
| **MobileNetV2 Transfer Learning** | **98.66%** | **0.0318** |

### Key Findings
- The baseline CNN memorized the training data, which is a clear case of overfitting.
- Augmentation, dropout and early stopping closed the train/validation gap and gained about 7 points of accuracy.
- Transfer learning gave the largest improvement and more confident predictions. On a hand-picked cat-like dog image, the custom CNN predicted "cat" (score 0.20), while the transfer learning model correctly predicted "dog" (score 0.9999).



## 🚀 Run the App Locally

**1. Clone the repository**
```bash
git clone https://github.com/Donna152/<your-repo-name>.git
cd <your-repo-name>
```

**2. Install the dependencies**
```bash
pip install streamlit tensorflow numpy pandas
```

**3. Start the app** (run it from the repository root so `model_transfer.keras` is found)
```bash
streamlit run app.py
```

**4. Use it**
1. Upload a `.jpg`, `.jpeg` or `.png` image of a cat or a dog.
2. Click **PREDICT**.
3. The app shows the predicted class, the confidence percentage and the raw model score.

### How the app makes a prediction
The image is resized to 256 × 256 and divided by 255. The model's built-in rescaling layer then maps it to `[-1, 1]`, as MobileNetV2 expects. The model outputs a single sigmoid score: a score below 0.5 means **Cat**, and 0.5 or above means **Dog**.

---

## 🔁 Reproduce the Training

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/salader/dogsvscats/data) and unzip it so that you have `train/` and `test/` folders, each with `cats/` and `dogs/` subfolders.
2. Open `cat_vs_dog_classification.ipynb` in Jupyter or Google Colab.
3. Update the dataset paths in the notebook (it was run in Colab using `/content/train` and `/content/test`).
4. Run all cells. The final cell saves the model as `model_transfer.keras`.

---



## 🛠️ Tech Stack
Python · TensorFlow / Keras · MobileNetV2 · NumPy · Matplotlib · OpenCV · Streamlit

---

## 👤 Author
**Donna Ahmed Mohamed**
[LinkedIn](https://www.linkedin.com/in/donna-ahmed-/) · [GitHub](https://github.com/Donna152)

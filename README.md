# CNN from Scratch — Cat vs. Dog Classifier (with layer-by-layer visualization)

A small, education-focused Convolutional Neural Network that classifies images as **cat** or **dog**, built to *understand how a CNN actually works* — not just to get a number.

It comes with a unique **explainer tool** that takes any image and produces an HTML report showing the journey through every layer: the feature maps, the shapes, and what each layer did, all the way to the final decision.

> This repo is the companion code for the blog post: *[An image's journey through a CNN]* (link to your blog).

---

## What's inside

| File | Purpose |
|---|---|
| `prepare_dataset.py` | Converts the Kaggle "Dogs vs Cats" layout into train/validation folders |
| `train_improved.py` | Trains the CNN (BatchNorm + Dropout + padding, GPU-aware, saves the best epoch) |
| `predict.py` | Quick single-image prediction |
| `explain_prediction.py` | **The star:** layer-by-layer HTML report of one image's journey |
| `requirements.txt` | Dependencies |

---

## The model

A compact CNN built as three Conv blocks followed by a small classifier:

```
Input (64x64x3)
 └─ Conv2D(32, 3x3, same) → BatchNorm → ReLU → MaxPool   # block 1
 └─ Conv2D(64, 3x3, same) → BatchNorm → ReLU → MaxPool   # block 2
 └─ Conv2D(128,3x3, same) → BatchNorm → ReLU → MaxPool   # block 3
 └─ Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(1, sigmoid)
```

Two parts, like every CNN:
- **Feature extraction** (the Conv blocks): cheap in parameters thanks to weight sharing, learns patterns from simple (edges) to abstract.
- **Classification** (the Dense layers): combines features into the final cat/dog decision.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### GPU (optional but recommended)
On Linux, the simplest path is:
```bash
pip install "tensorflow[and-cuda]"
```
If TensorFlow still can't see the GPU (a version mismatch between the TF build and the CUDA libraries), pin a known-good release:
```bash
pip install "tensorflow[and-cuda]==2.18.*"
```
Verify:
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## Usage

### 1. Prepare the dataset
Download the [Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data) dataset (the `train.zip` with `cat.N.jpg` / `dog.N.jpg` files). Unzip it so you have `archive/train/`, then:

```bash
python prepare_dataset.py archive/train
```

This splits each class into 80% train / 20% validation and produces:
```
Dogs-vs-Cats/
├── training_set/training_set/{cats,dogs}/
└── test_set/test_set/{cats,dogs}/
```

### 2. Train
```bash
python train_improved.py
```
Outputs `cat_dog_improved.keras` (best epoch by validation accuracy) and `training_history.png`.

### 3. Predict
```bash
python predict.py some_image.jpg
```

### 4. Explain (the interesting part)
```bash
python explain_prediction.py some_image.jpg
```
Open `explain_output/report.html` in a browser to see the full layer-by-layer journey.

---

## License

MIT — free to use, learn from, and build on.

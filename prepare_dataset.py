"""
prepare_dataset.py
==================
Converts the Kaggle "Dogs vs Cats" competition layout into the folder
structure the training scripts expect.

Input (what you have now):
    archive/train/cat.0.jpg, cat.1.jpg, ..., dog.0.jpg, ...

Output (what the code needs):
    Dogs-vs-Cats/
    |-- training_set/training_set/{cats,dogs}/
    +-- test_set/test_set/{cats,dogs}/

The split is done by filename prefix (cat./dog.). A fraction of each class
is held out for validation (test_set).

Run from the folder that CONTAINS `archive/`:
    python prepare_dataset.py

Or pass a custom source folder:
    python prepare_dataset.py /path/to/archive/train
"""

import os
import sys
import shutil
import random

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
SRC = sys.argv[1] if len(sys.argv) > 1 else "archive/train"
DST = "Dogs-vs-Cats"
VAL_FRACTION = 0.2          # 20% of each class goes to validation (test_set)
SEED = 42
COPY = True                 # True = copy (keeps originals), False = move
LIMIT_PER_CLASS = None      # set to e.g. 2000 for a quick smaller run, or None for all

random.seed(SEED)

# ---------------------------------------------------------------------------
# Target folders
# ---------------------------------------------------------------------------
train_cats = os.path.join(DST, "training_set", "training_set", "cats")
train_dogs = os.path.join(DST, "training_set", "training_set", "dogs")
test_cats = os.path.join(DST, "test_set", "test_set", "cats")
test_dogs = os.path.join(DST, "test_set", "test_set", "dogs")
for d in (train_cats, train_dogs, test_cats, test_dogs):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Gather files by class
# ---------------------------------------------------------------------------
if not os.path.isdir(SRC):
    print(f"[ERROR] Source folder not found: {SRC}")
    print("Pass the correct path, e.g.: python prepare_dataset.py archive/train")
    sys.exit(1)

all_files = os.listdir(SRC)
cats = sorted(f for f in all_files if f.lower().startswith("cat"))
dogs = sorted(f for f in all_files if f.lower().startswith("dog"))

print(f"Found {len(cats)} cats and {len(dogs)} dogs in {SRC}")

if LIMIT_PER_CLASS:
    cats = cats[:LIMIT_PER_CLASS]
    dogs = dogs[:LIMIT_PER_CLASS]
    print(f"Limiting to {LIMIT_PER_CLASS} per class")

# ---------------------------------------------------------------------------
# Split and place
# ---------------------------------------------------------------------------
def place(files, train_dir, test_dir):
    random.shuffle(files)
    n_val = int(len(files) * VAL_FRACTION)
    val_files = files[:n_val]
    train_files = files[n_val:]
    op = shutil.copy2 if COPY else shutil.move
    for f in train_files:
        op(os.path.join(SRC, f), os.path.join(train_dir, f))
    for f in val_files:
        op(os.path.join(SRC, f), os.path.join(test_dir, f))
    return len(train_files), len(val_files)

ct_train, ct_val = place(cats, train_cats, test_cats)
dg_train, dg_val = place(dogs, train_dogs, test_dogs)

print(f"cats -> train: {ct_train}, validation: {ct_val}")
print(f"dogs -> train: {dg_train}, validation: {dg_val}")
print(f"[OK] Dataset ready under: {DST}/")

"""
explain_prediction.py
=====================
Educational inference script.
Takes one image, passes it layer by layer through the model, and at each step:
  * states which layer it is and what it did
  * shows the input and output shape
  * saves the actual feature maps of that layer as an image
Then builds a full HTML report showing the whole journey from input to prediction.

Run:
    python explain_prediction.py path/to/image.jpg

Outputs:
    explain_output/report.html        <- full report (open this in a browser)
    explain_output/*.png              <- per-step images
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed -- save straight to file
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model, Model

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
MODEL_PATH = "cat_dog_improved.keras"   # change to your .h5 if using the original model
IMG_SIZE = (64, 64)
OUT_DIR = "explain_output"
MAX_MAPS = 8   # how many feature maps to show per layer

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 0) Get the image path from the command line
# ---------------------------------------------------------------------------
if len(sys.argv) < 2:
    print("Usage: python explain_prediction.py path/to/image.jpg")
    sys.exit(1)
img_path = sys.argv[1]

# ---------------------------------------------------------------------------
# 1) Load the model
# ---------------------------------------------------------------------------
model = load_model(MODEL_PATH)
class_indices = {0: "cat", 1: "dog"}  # standard flow_from_directory mapping

# ---------------------------------------------------------------------------
# 2) Prepare the image (same input pipeline: resize + rescale)
# ---------------------------------------------------------------------------
img = keras_image.load_img(img_path, target_size=IMG_SIZE)
img_array = keras_image.img_to_array(img) / 255.0          # normalize to 0..1
input_tensor = np.expand_dims(img_array, axis=0)           # add batch dimension

# ---------------------------------------------------------------------------
# 3) Build a helper model that outputs *every* layer at once.
#    This is the key trick: instead of only the final output, we capture
#    each layer's activation.
#    Keras 3 note: build an explicit Input and pass it through each layer,
#    then make a multi-output Functional model.
# ---------------------------------------------------------------------------
inp = tf.keras.Input(shape=(64, 64, 3))
x = inp
layer_outputs = []
for layer in model.layers:
    x = layer(x)
    layer_outputs.append(x)

activation_model = Model(inputs=inp, outputs=layer_outputs)
activations = activation_model.predict(input_tensor, verbose=0)

# ---------------------------------------------------------------------------
# 4) Per-layer-type description (for the report)
# ---------------------------------------------------------------------------
def describe(layer):
    name = layer.__class__.__name__
    table = {
        "Conv2D": "Filters slide over the input and extract features (edges/texture/shape). Output depth = number of filters.",
        "BatchNormalization": "Rebalances values (mean ~0, std ~1) for more stable training. Shape is unchanged.",
        "Activation": "ReLU: negative values become 0, positives pass through. Adds non-linearity.",
        "MaxPooling2D": "Each 2x2 region collapses to its maximum value. Each side halves -> downsampling + robustness.",
        "Flatten": "The 3D volume is flattened into a 1D vector to feed the Dense layers. Values are unchanged.",
        "Dense": "Each neuron connects to every input; combines features toward the decision.",
        "Dropout": "Inactive at inference time (all neurons on). It only played a role during training.",
    }
    return table.get(name, "-")


# ---------------------------------------------------------------------------
# 5) Save the image for each step
# ---------------------------------------------------------------------------
def save_input_image():
    fname = os.path.join(OUT_DIR, "step_00_input.png")
    plt.figure(figsize=(3, 3))
    plt.imshow(img_array)
    plt.title("Input (64x64x3)")
    plt.axis("off")
    plt.savefig(fname, dpi=100, bbox_inches="tight")
    plt.close()
    return os.path.basename(fname)


def save_feature_maps(act, idx, layer_name):
    """For 3D layers (Conv/BN/Act/Pool): save the first few channels as a grid."""
    fmap = act[0]  # drop batch dim -> (H, W, C)
    n = min(MAX_MAPS, fmap.shape[-1])
    cols = 4
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(cols * 2, rows * 2))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(fmap[:, :, i], cmap="viridis")
        plt.axis("off")
    plt.suptitle(f"{layer_name} - {n}/{fmap.shape[-1]} feature maps")
    fname = os.path.join(OUT_DIR, f"step_{idx:02d}_{layer_name}.png")
    plt.savefig(fname, dpi=100, bbox_inches="tight")
    plt.close()
    return os.path.basename(fname)


def save_vector(act, idx, layer_name):
    """For 1D layers (Flatten/Dense): plot the vector values as a bar chart."""
    vec = act[0].flatten()
    show = vec[:200] if vec.size > 200 else vec   # at most first 200 values
    plt.figure(figsize=(8, 2))
    plt.bar(range(len(show)), show, width=1.0)
    plt.title(f"{layer_name} - {vec.size} values (first 200 shown)")
    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"step_{idx:02d}_{layer_name}.png")
    plt.savefig(fname, dpi=100, bbox_inches="tight")
    plt.close()
    return os.path.basename(fname)


# ---------------------------------------------------------------------------
# 6) Build the report data
# ---------------------------------------------------------------------------
steps = []  # each item: dict describing one step

input_img_file = save_input_image()
steps.append({
    "title": "Input",
    "desc": "Image resized to 64x64 and divided by 255 (normalized to 0..1).",
    "shape_in": str(img_path),
    "shape_out": "(64, 64, 3)",
    "img": input_img_file,
})

prev_shape = "(64, 64, 3)"
for idx, (layer, act) in enumerate(zip(model.layers, activations), start=1):
    out_shape = str(act.shape[1:])  # without batch dim
    is_3d = (act.ndim == 4)         # (batch, H, W, C)

    if is_3d:
        img_file = save_feature_maps(act, idx, layer.name)
    else:
        img_file = save_vector(act, idx, layer.name)

    steps.append({
        "title": f"{idx}. {layer.name}  ({layer.__class__.__name__})",
        "desc": describe(layer),
        "shape_in": prev_shape,
        "shape_out": out_shape,
        "img": img_file,
    })
    prev_shape = out_shape

# Final prediction
final_score = float(activations[-1][0][0])
predicted = class_indices[1] if final_score > 0.5 else class_indices[0]
confidence = final_score if final_score > 0.5 else (1 - final_score)

# ---------------------------------------------------------------------------
# 7) Write the HTML
# ---------------------------------------------------------------------------
def build_html():
    rows = []
    for s in steps:
        rows.append(f"""
        <section class="step">
          <h2>{s['title']}</h2>
          <p class="desc">{s['desc']}</p>
          <div class="shapes">
            <span class="shape in">in: {s['shape_in']}</span>
            <span class="arrow">&rarr;</span>
            <span class="shape out">out: {s['shape_out']}</span>
          </div>
          <img src="{s['img']}" alt="{s['title']}"/>
        </section>""")

    verdict_color = "#1d9e75" if predicted == "dog" else "#d85a30"
    verdict_label = "DOG" if predicted == "dog" else "CAT"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>An image's journey through a CNN</title>
<style>
  body {{ font-family: system-ui, Arial, sans-serif; background:#faf9f6; color:#2c2c2a;
         max-width: 900px; margin: 0 auto; padding: 24px; line-height: 1.7; }}
  h1 {{ font-size: 26px; border-bottom: 2px solid #ddd; padding-bottom: 12px; }}
  .verdict {{ background:{verdict_color}; color:#fff; padding:18px; border-radius:12px;
             font-size:22px; text-align:center; margin:20px 0; }}
  .verdict small {{ display:block; font-size:14px; opacity:.9; margin-top:6px; }}
  .step {{ background:#fff; border:1px solid #e5e3da; border-radius:12px;
          padding:18px; margin:18px 0; }}
  .step h2 {{ font-size:18px; margin:0 0 8px; color:#04342c; }}
  .desc {{ color:#555; margin:0 0 12px; font-size:14px; }}
  .shapes {{ font-family: monospace; font-size:13px; margin-bottom:12px; }}
  .shape {{ padding:4px 10px; border-radius:6px; }}
  .shape.in {{ background:#e6f1fb; color:#0c447c; }}
  .shape.out {{ background:#e1f5ee; color:#04342c; }}
  .arrow {{ margin:0 8px; color:#888; }}
  .step img {{ max-width:100%; border-radius:8px; border:1px solid #eee; }}
  .legend {{ background:#fff8e6; border:1px solid #f0d999; border-radius:8px;
            padding:12px; font-size:13px; }}
</style>
</head>
<body>
  <h1>An image's journey through a CNN</h1>
  <p>This report shows how <code>{os.path.basename(img_path)}</code> was processed layer by layer to reach the final prediction.</p>

  <div class="verdict">
    Prediction: {verdict_label}
    <small>sigmoid output = {final_score:.4f} | confidence ~ {confidence*100:.1f}%</small>
  </div>

  <div class="legend">
    Note: in the feature-map images, each square is a different filter.
    Brighter (yellow) spots = stronger presence of that pattern in that region.
    Early layers detect simple patterns (edges); deeper layers detect more abstract ones.
  </div>

  {''.join(rows)}

  <p style="text-align:center; color:#999; margin-top:30px;">-- end of journey --</p>
</body>
</html>"""


with open(os.path.join(OUT_DIR, "report.html"), "w", encoding="utf-8") as f:
    f.write(build_html())

print(f"\n{'='*50}")
print(f"Prediction: {predicted}  (score={final_score:.4f}, confidence={confidence*100:.1f}%)")
print(f"[OK] Report built: {OUT_DIR}/report.html")
print(f"     Open it in a browser.")
print(f"{'='*50}")

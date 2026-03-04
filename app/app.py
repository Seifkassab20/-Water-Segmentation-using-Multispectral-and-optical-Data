from flask import Flask, render_template, request
import os
import numpy as np
import cv2
import rasterio
from app.inference import run_inference

app = Flask(__name__)

UPLOAD_FOLDER = "app/static/uploads"
OUTPUT_FOLDER = "app/static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ===============================
# Load TIF Image
# ===============================
def load_tif_image(path):
    with rasterio.open(path) as src:
        image = src.read()
    return np.transpose(image, (1, 2, 0))


# ===============================
# Create RGB Preview
# ===============================
def create_rgb(image_np):
    red = image_np[:, :, 3]
    green = image_np[:, :, 2]
    blue = image_np[:, :, 1]

    rgb = np.stack([red, green, blue], axis=-1)
    rgb = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX)
    return rgb.astype(np.uint8)


@app.route("/")
def home():
    return render_template("index.html")


# ===============================
# Prediction Route
# ===============================
@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    image_np = load_tif_image(image_path)

    # Select same 6 bands used during training
    band_indices = [0, 1, 4, 5, 6, 11]
    image_np = image_np[:, :, band_indices]

    # Create RGB preview
    rgb = create_rgb(image_np)

    # ===============================
    # Run UNet
    # ===============================
    pred_mask_unet, water_prob_unet, confidence_unet = run_inference(
        image_np, "unet"
    )

    # ===============================
    # Run DeepLab
    # ===============================
    pred_mask_dl, water_prob_dl, confidence_dl = run_inference(
        image_np, "deeplab"
    )

    # Convert to percentage
    confidence_unet_percent = round(confidence_unet * 100, 2)
    confidence_deeplab_percent = round(confidence_dl * 100, 2)

    # ===============================
    # Save RGB
    # ===============================
    rgb_path = os.path.join(OUTPUT_FOLDER, "rgb.png")
    cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # ===============================
    # Save UNet Mask
    # ===============================
    mask_path = os.path.join(OUTPUT_FOLDER, "mask.png")
    cv2.imwrite(mask_path, (pred_mask_unet * 255).astype(np.uint8))

    # ===============================
    # Save Heatmap (UNet probability)
    # ===============================
    heatmap = (water_prob_unet * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    heatmap_path = os.path.join(OUTPUT_FOLDER, "heatmap.png")
    cv2.imwrite(heatmap_path, heatmap)

    return render_template(
        "index.html",
        rgb_image="static/outputs/rgb.png",
        mask_image="static/outputs/mask.png",
        heatmap_image="static/outputs/heatmap.png",
        confidence_unet_percent=confidence_unet_percent,
        confidence_deeplab_percent=confidence_deeplab_percent
    )


if __name__ == "__main__":
    app.run(debug=True)
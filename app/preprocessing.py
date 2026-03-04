import json
import numpy as np

with open("normalization.json") as f:
    norm_vals = json.load(f)

band_indices = [0, 1, 4, 5, 6, 11]
mins = np.array(norm_vals["min"])[band_indices]
maxs = np.array(norm_vals["max"])[band_indices]

def normalize_image(image):
    image = image.astype(np.float32)

    for c in range(image.shape[-1]):
        image[:, :, c] = (
            (image[:, :, c] - mins[c]) /
            (maxs[c] - mins[c] + 1e-8)
        )

    image = np.transpose(image, (2, 0, 1))
    return image
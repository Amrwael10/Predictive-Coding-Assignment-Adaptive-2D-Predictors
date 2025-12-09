# load/save/display images

from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_image(path):
    """Load image and return numpy array in shape (H,W,3) for RGB or (H,W) for grayscale."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    return arr

def save_image_array(arr, path):
    """Save a numpy image array (H,W,3) or (H,W) as PNG."""
    img = Image.fromarray(np.uint8(np.clip(arr, 0, 255)))
    img.save(path)

def show_and_save_images(grid, titles, out_name="result.png", figSize=(12,6)):
    """
    grid: list of numpy images (H,W,3) or (H,W)
    titles: list of strings
    Saves to outputs/out_name and also displays using matplotlib.
    """
    n = len(grid)
    cols = min(3, n)
    rows = (n + cols - 1)//cols
    plt.figure(figsize=figSize)
    for i, (img, title) in enumerate(zip(grid, titles)):
        plt.subplot(rows, cols, i+1)
        if img.ndim == 2:
            plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            plt.imshow(np.uint8(np.clip(img, 0, 255)))
        plt.title(title)
        plt.axis('off')
    outPath = os.path.join(OUTPUT_DIR, out_name)
    plt.tight_layout()
    plt.savefig(outPath, bbox_inches='tight')
    plt.show()
    return outPath


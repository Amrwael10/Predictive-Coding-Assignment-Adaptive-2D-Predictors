"""
Automated test cases for predictive compression project.
You can call run_all() or run individual tests.
"""

import os
import numpy as np
from PIL import Image
from compressor import Compressor
from decompressor import Decompressor
from image_io import OUTPUT_DIR


def save_temp_image(arr, name):
    """Save numpy array as PNG into outputs/."""
    path = os.path.join(OUTPUT_DIR, name)
    Image.fromarray(arr.astype(np.uint8)).save(path)
    return path


# -----------------------------------------------------------
# TEST CASE 1 — Smooth small grayscale image (best prediction)
# -----------------------------------------------------------
def test_smooth():
    arr = np.array([
        [10,12,14,16,18,20,22,24],
        [11,13,15,17,19,21,23,25],
        [12,14,16,18,20,22,24,26],
        [13,15,17,19,21,23,25,27],
        [14,16,18,20,22,24,26,28],
        [15,17,19,21,23,25,27,29],
        [16,18,20,22,24,26,28,30],
        [17,19,21,23,25,27,29,31],
    ], dtype=np.uint8)

    img_path = save_temp_image(arr, "tc1_smooth.png")

    compressor = Compressor(predictor_mode="avg", quant_bits=2)
    meta = compressor.compress(arr)
    comp_path = os.path.join(OUTPUT_DIR, "tc1_smooth_compressed.npz")
    compressor.save_compressed(meta, comp_path)

    dec = Decompressor(comp_path)
    recon = dec.decompress()
    recon_path = save_temp_image(recon, "tc1_smooth_decompressed.png")

    return {
        "name": "TC1 Smooth Grayscale",
        "input": img_path,
        "compressed": comp_path,
        "output": recon_path,
        "shape": arr.shape,
        "description": "Smooth gradient — should reconstruct perfectly with high compression ratio."
    }


# -----------------------------------------------------------
# TEST CASE 2 — Random noise (worst prediction)
# -----------------------------------------------------------
def test_noise():
    noise = np.random.randint(0, 256, (64,64), dtype=np.uint8)
    img_path = save_temp_image(noise, "tc2_noise.png")

    compressor = Compressor(predictor_mode="avg", quant_bits=2)
    meta = compressor.compress(noise)
    comp_path = os.path.join(OUTPUT_DIR, "tc2_noise_compressed.npz")
    compressor.save_compressed(meta, comp_path)

    dec = Decompressor(comp_path)
    recon = dec.decompress()
    recon_path = save_temp_image(recon, "tc2_noise_decompressed.png")

    return {
        "name": "TC2 Random Noise",
        "input": img_path,
        "compressed": comp_path,
        "output": recon_path,
        "shape": noise.shape,
        "description": "Random noise — worst case prediction, compression ratio ≈ 1."
    }


# -----------------------------------------------------------
# TEST CASE 3 — RGB artificial gradient
# -----------------------------------------------------------
def test_rgb_gradient():
    H, W = 64, 64
    img = np.zeros((H,W,3), dtype=np.uint8)
    img[...,0] = np.linspace(0,255,W)
    img[...,1] = 128
    img[...,2] = np.linspace(255,0,H).reshape(H,1)

    img_path = save_temp_image(img, "tc3_rgb_gradient.png")

    compressor = Compressor(predictor_mode="a+b-c", quant_bits=2)
    meta = compressor.compress(img)
    comp_path = os.path.join(OUTPUT_DIR, "tc3_rgb_compressed.npz")
    compressor.save_compressed(meta, comp_path)

    dec = Decompressor(comp_path)
    recon = dec.decompress()
    recon_path = save_temp_image(recon, "tc3_rgb_decompressed.png")

    return {
        "name": "TC3 RGB Gradient",
        "input": img_path,
        "compressed": comp_path,
        "output": recon_path,
        "shape": img.shape,
        "description": "Smooth RGB gradient — a+b-c predictor performs well."
    }


# -----------------------------------------------------------
# Run all tests
# -----------------------------------------------------------
def run_all():
    tests = [test_smooth, test_noise, test_rgb_gradient]
    results = []

    for t in tests:
        print(f"\nRunning {t.__name__} ...")
        res = t()
        results.append(res)
        print(f"✓ Completed: {res['name']}")
        print(f"  Input:      {res['input']}")
        print(f"  Compressed: {res['compressed']}")
        print(f"  Output:     {res['output']}")
        print(f"  Shape:      {res['shape']}")
        print(f"  Notes:      {res['description']}")

    print("\nAll test cases finished.\n")
    return results

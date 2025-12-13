# CLI entry point (menu) - integrate everything

import os
import numpy as np
from image_io import load_image,show_and_save_images, OUTPUT_DIR
from compressor import Compressor
from decompressor import Decompressor
from utils import compute_compression_ratio, header_bytes_for_shape

def compress_flow():
    path = input("Enter path to image file to compress: ").strip()
    if not os.path.exists(path):
        print("File not found.")
        return
    img = load_image(path)
    print("Loaded image with shape:", img.shape)

    bits_str = input("Uniform quantizer bits (default=2): ").strip()
    bits = int(bits_str) if bits_str != "" else 2

    # predictor mode
    print("Predictor modes: 'avg' (default), 'a', 'b', 'a+b-c', 'linear'")
    predictor_mode = input("Choose predictor mode [avg]: ").strip()
    predictor_mode = predictor_mode if predictor_mode != "" else "avg"

    compressor = Compressor(predictor_mode=predictor_mode, quant_bits=bits)
    meta = compressor.compress(img)

    # Save compressed to outputs/<basename>.npz
    base = os.path.splitext(os.path.basename(path))[0]
    out_npz = os.path.join(OUTPUT_DIR, f"{base}_compressed.npz")
    compressor.save_compressed(meta, out_npz)
    print("Compression metadata saved to:", out_npz)

    dec = Decompressor(out_npz)
    recon = dec.decompress()

    H, W = meta['shape']
    channels = meta['channels']
    bits = meta['bits']
    predictor_mode = meta['predictor_mode']
    titles = []
    imgs = [img]

    # Original
    titles.append("Original")

    # Decompressed
    imgs.append(recon)
    titles.append("Decompressed")

    pred_vis = np.zeros_like(img, dtype=np.uint8)
    error_vis = np.zeros_like(img, dtype=np.uint8)
    quant_vis = np.zeros_like(img, dtype=np.uint8)
    deq_vis = np.zeros_like(img, dtype=np.uint8)

    for ch in range(channels):
        chdata = meta['channels_data'][ch]
        indices = chdata['indices']
        minv = chdata['min_val']
        step = chdata['step']
        # reconstruct predicted by simulating recon and using predictor
        from predictor import Predictor2D
        pq = Predictor2D(mode=predictor_mode)
        recon_ch = np.zeros((H, W), dtype=np.int32)
        pred_ch = np.zeros((H, W), dtype=np.int32)
        err_ch = np.zeros((H, W), dtype=np.float32)
        deq_ch = np.zeros((H, W), dtype=np.float32)
        # set headers
        fr = meta['header']['first_row'][ch]
        fc = meta['header']['first_col'][ch]
        recon_ch[0, :] = fr
        recon_ch[:, 0] = fc

        for i in range(H):
            for j in range(W):
                p = pq.predict_pixel(recon_ch, i, j)
                pred_ch[i, j] = p
                idx = int(indices[i,j])
                deq = minv + (idx + 0.5) * step
                deq_ch[i, j] = deq
                val = p + deq
                val = 0 if val < 0 else (255 if val > 255 else int(round(val)))
                recon_ch[i, j] = val
                # compute error for display: original - pred
                orig_val = img[:, :, ch] if channels == 3 else img
        # place into vis arrays
        if channels == 3:
            pred_vis[:, :, ch] = np.clip(pred_ch, 0, 255).astype(np.uint8)
            deq_vis[:, :, ch] = np.clip(deq_ch + pred_ch, 0, 255).astype(np.uint8)
            quant_vis[:, :, ch] = np.clip(minv + (indices + 0.5) * step, 0, 255).astype(np.uint8)
            # error visualization as original - pred shifted +128 to display negative values
            err_display = (img[:, :, ch].astype(np.int32) - pred_ch).astype(np.int32)
            err_display = np.clip(err_display + 128, 0, 255).astype(np.uint8)
            error_vis[:, :, ch] = err_display
        else:
            pred_vis[:, :] = np.clip(pred_ch, 0, 255).astype(np.uint8)
            deq_vis[:, :] = np.clip(deq_ch + pred_ch, 0, 255).astype(np.uint8)
            quant_vis[:, :] = np.clip(minv + (indices + 0.5) * step, 0, 255).astype(np.uint8)
            err_display = (img.astype(np.int32) - pred_ch).astype(np.int32)
            err_display = np.clip(err_display + 128, 0, 255).astype(np.uint8)
            error_vis[:, :] = err_display

    imgs.extend([pred_vis, error_vis, quant_vis, deq_vis])
    titles.extend(["Predicted", "Error ", "Quantized (mapped)", "De-quantized"])

    # Save montage
    out_name = f"{os.path.splitext(os.path.basename(path))[0]}_results.png"
    outpath = show_and_save_images(imgs, titles, out_name=out_name)
    print("Saved visualization to:", outpath)

    # Print quant table (for first channel as example)
    print("\nQuantizer table (channel 0, sample):")
    sample_ch = meta['channels_data'][0]
    # Recreate UniformQuantizer meta to print table
    from quantizer import UniformQuantizer
    uq = UniformQuantizer(bits=bits)
    uq.min_val = sample_ch['min_val']
    uq.step = sample_ch['step']
    table = uq.quant_table()
    for row in table:
        idx, code, center, rng = row
        print(f"level {idx:3d}  code={code}  center={center:.3f}  interval=({rng[0]:.3f},{rng[1]:.3f})")

    # Compression ratio:
    H, W = meta['shape']
    header_bytes = header_bytes_for_shape(H, W, meta['channels'])
    ratio, orig_bits, comp_bits = compute_compression_ratio((H,W), meta['channels'], bits, header_bytes=header_bytes)
    print(f"\nEstimated compression ratio: {ratio:.3f} (orig bits={orig_bits}, estimated compressed bits={comp_bits})")
    print("Note: header assumption = first_row + first_col per channel stored raw.")

def decompress_flow():
    path = input("Enter path to compressed .npz file: ").strip()
    if not os.path.exists(path):
        print("File not found.")
        return
    from decompressor import Decompressor
    dec = Decompressor(path)
    img = dec.decompress()
    out_name = os.path.splitext(os.path.basename(path))[0] + "_decompressed.png"
    save_path = os.path.join(OUTPUT_DIR, out_name)
    from image_io import save_image_array
    save_image_array(img, save_path)
    print("Decompressed image saved to:", save_path)

def main_menu():
    while True:
        print("\nPredictive Coding Assignment Menu")
        print("1) Compress image")
        print("2) Decompress .npz file")
        print("3) Run automated test cases")
        print("4) Exit")
        choice = input("Choose option: ").strip()
        if choice == "1":
            compress_flow()
        elif choice == "2":
            decompress_flow()
        elif choice == "3":
            import test_cases
            test_cases.run_all()
        elif choice == "4":
            print("Bye.")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main_menu()

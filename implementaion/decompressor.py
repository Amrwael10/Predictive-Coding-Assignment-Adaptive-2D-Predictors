# Decompressor class - reads compressed metadata and reconstructs

"""
Decompressor: loads .npz metadata saved by Compressor.save_compressed and reconstructs image.
It uses the same Predictor2D and UniformQuantizer meta to dequantize indices into errors.
"""

import numpy as np
from predictor import Predictor2D
from quantizer import UniformQuantizer

class Decompressor:
    def __init__(self, meta_npz_path):
        data = np.load(meta_npz_path, allow_pickle=True)
        self.data = data

    def decompress(self):
        shape = tuple(self.data['shape'])
        channels = int(self.data['channels'])
        predictor_mode = str(self.data['predictor_mode'])
        bits = int(self.data['bits'])
        H, W = shape
        out = np.zeros((H, W, channels), dtype=np.uint8) if channels == 3 else np.zeros((H, W), dtype=np.uint8)

        pq = Predictor2D(mode=predictor_mode)

        for ch in range(channels):
            # load header
            fr = self.data[f'first_row_ch{ch}']
            fc = self.data[f'first_col_ch{ch}']
            indices = self.data[f'indices_ch{ch}']
            min_val = float(self.data[f'min_ch{ch}'])
            max_val = float(self.data[f'max_ch{ch}'])
            step = float(self.data[f'step_ch{ch}'])
            q = UniformQuantizer(bits=bits)
            q.min_val = min_val
            q.max_val = max_val
            q.step = step

            recon = np.zeros((H, W), dtype=np.int32)
            # Set the first row and first col from header
            recon[0, :] = fr
            recon[:, 0] = fc

            # now scan and reconstruct using indices
            for i in range(H):
                for j in range(W):
                    if i == 0 and j == 0:
                        # already set by header (fr[0])
                        continue
                    p = pq.predict_pixel(recon, i, j)
                    idx = int(indices[i, j])
                    # dequantize this index:
                    deq = q.min_val + (idx + 0.5) * q.step
                    val = p + deq
                    val = 0 if val < 0 else (255 if val > 255 else int(round(val)))
                    recon[i, j] = val

            if out.ndim == 3:
                out[:, :, ch] = recon.astype(np.uint8)
            else:
                out[:, :] = recon.astype(np.uint8)
        return out
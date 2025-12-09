# Compressor class - does compression + saves bitstream metadata


"""
Compressor - orchestrates compression pipeline:
 - uses Predictor2D in feed-backward mode (encoder simulates decoder)
 - uses UniformQuantizer per channel
 - returns compressed metadata (we will store .npz with quantized indices and headers)
"""

import numpy as np
from predictor import Predictor2D
from quantizer import UniformQuantizer


class Compressor:
    def __init__(self, predictor_mode="avg", quant_bits=2):
        self.predictor_mode = predictor_mode
        self.quant_bits = quant_bits

    def compress(self, img_arr):
        """
        img_arr: HxW or HxWx3 uint8
        Returns metadata dict containing:
          - 'shape', 'channels'
          - for each channel: quant_indices, min_val, max_val, step
          - predictor mode and bits
          - saved first row and first column raw (so decompressor can reconstruct border)
        """
        H, W = img_arr.shape[:2]
        channels = 1 if img_arr.ndim == 2 else 3

        result = {
            'shape': (H, W),
            'channels': channels,
            'predictor_mode': self.predictor_mode,
            'bits': self.quant_bits,
        }

        # store raw first row and first column for each channel as header
        header = {'first_row': [], 'first_col': []}

        compressed_channels = []
        for ch in range(channels):
            channel = img_arr if channels == 1 else img_arr[:, :, ch]
            channel = channel.astype(np.int32)

            # predictor and quantizer
            pred = np.zeros_like(channel, dtype=np.int32)
            recon = np.zeros_like(channel, dtype=np.int32)

            # We'll perform feed-backward: for each pixel, predict using recon so far,
            # compute error = original - pred, quantize error, dequantize, and set recon = pred + deq
            pq = Predictor2D(mode=self.predictor_mode)
            q = UniformQuantizer(bits=self.quant_bits)

            # Prepare arrays to store indices and dequantized errors
            indices = np.zeros_like(channel, dtype=np.int32)
            deq_errors = np.zeros_like(channel, dtype=np.float32)

            # First row & first column headers we will store raw (8-bit)
            header_first_row = channel[0, :].astype(np.uint8)
            header_first_col = channel[:, 0].astype(np.uint8)
            header['first_row'].append(header_first_row)
            header['first_col'].append(header_first_col)

            temp_recon = np.zeros_like(channel, dtype=np.int32)
            errors_list = []
            for i in range(H):
                for j in range(W):
                    p = pq.predict_pixel(temp_recon, i, j)
                    e = channel[i, j] - p
                    errors_list.append(e)
                    # dequantization unknown yet; for temp_recon use original value (approx)
                    temp_recon[i, j] = channel[i, j]  # use original as fallback to stabilize
            errors = np.array(errors_list, dtype=np.int32)
            q.fit(errors)  # now quantizer has min/max/step

            # PASS 2: actual encoding using feedback + quantization
            for i in range(H):
                for j in range(W):
                    p = pq.predict_pixel(recon, i, j)
                    pred[i, j] = p
                    err = channel[i, j] - p
                    idx, deq = q.quantize(np.array([err]))
                    idx = int(idx[0]); deq = float(deq[0])
                    indices[i, j] = idx
                    deq_errors[i, j] = deq
                    # reconstruct pixel (use dequantized error)
                    val = p + deq
                    # clamp
                    val = 0 if val < 0 else (255 if val > 255 else int(round(val)))
                    recon[i, j] = val

            # store channel compressed block
            compressed_channels.append({
                'indices': indices,
                'min_val': float(q.min_val),
                'max_val': float(q.max_val),
                'step': float(q.step)
            })

        result['channels_data'] = compressed_channels
        result['header'] = header
        return result

    def save_compressed(self, meta, out_path):
        """
        Save compressed metadata to a numpy npz file for simplicity.
        meta contains arrays (indices) and small arrays first_row/first_col; we save everything.
        """
        save_dict = {}
        save_dict['shape'] = meta['shape']
        save_dict['channels'] = meta['channels']
        save_dict['predictor_mode'] = meta['predictor_mode']
        save_dict['bits'] = meta['bits']
        # header parts
        for ch_idx, (fr, fc) in enumerate(zip(meta['header']['first_row'], meta['header']['first_col'])):
            save_dict[f'first_row_ch{ch_idx}'] = fr
            save_dict[f'first_col_ch{ch_idx}'] = fc
        # channel data
        for ch_idx, chdata in enumerate(meta['channels_data']):
            save_dict[f'indices_ch{ch_idx}'] = chdata['indices']
            save_dict[f'min_ch{ch_idx}'] = chdata['min_val']
            save_dict[f'max_ch{ch_idx}'] = chdata['max_val']
            save_dict[f'step_ch{ch_idx}'] = chdata['step']
        # Save as npz
        np.savez_compressed(out_path, **save_dict)
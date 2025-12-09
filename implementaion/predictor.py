# Predictor2D class (configurable predictor)

"""
Predictor2D: implements simple 2D predictors.
We implement feed-backward style prediction: predictor uses previously reconstructed pixels.
Available predictor modes:
 - 'avg': (left + top)/2
 - 'a': left (A)
 - 'b': top (B)
 - 'a+b-c' (JPEG-LS style): A + B - C
 - 'linear': configurable weights for A, B, C (sum must be 1)
"""

import numpy as np

class Predictor2D:
    def __init__(self, mode="avg", w=(0.5,0.5,0.0)):
        self.mode = mode
        self.w = w

    def predict_pixel(self, recon_channel, i, j):
        """
        recon_channel: reconstructed channel so far (H,W)
        i,j: coordinates to predict (assumes i>=0, j>=0)
        For first row/col, simple fallback to neighbors or 0.
        """
        H, W = recon_channel.shape
        # get neighbors with safe checks
        A = recon_channel[i, j-1] if j-1 >= 0 else 0   # left
        B = recon_channel[i-1, j] if i-1 >= 0 else 0   # top
        C = recon_channel[i-1, j-1] if (i-1 >=0 and j-1 >=0) else 0  # top-left

        if i == 0 and j == 0:
            return 0  # top-left corner
        if self.mode == "avg":
            return (A + B) // 2
        if self.mode == "a":
            return A
        if self.mode == "b":
            return B
        if self.mode == "a+b-c":
            # A + B - C, clamp to valid range later
            return int(A + B - C)
        if self.mode == "linear":
            w1, w2, w3 = self.w
            return int(w1*A + w2*B + w3*C)
        # default
        return (A + B) // 2

    def predict_image(self, original_channel):
        """
        Vectorized-ish wrapper to produce a prediction image using reconstructed feedback
        (simulates decoder reconstruction). Returns tuple (predicted_image, reconstructed_image).
        We iterate in raster scan order to ensure feedback is applied.
        """
        H, W = original_channel.shape
        pred = np.zeros_like(original_channel, dtype=np.int32)
        recon = np.zeros_like(original_channel, dtype=np.int32)

        # policy: the first row and column -> we'll treat them as sent raw,
        # so the prediction for (0,*) and (*,0) is 0 but we will copy originals to recon.
        for i in range(H):
            for j in range(W):
                p = self.predict_pixel(recon, i, j)
                pred[i, j] = p
                # recon value will be set outside (after quantization) in compressor/decompressor loops,
                # but for a standalone prediction (if using original), we don't set recon here.
        return pred, recon
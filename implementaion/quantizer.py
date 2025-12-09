# UniformQuantizer class & quant table printing

"""
UniformQuantizer: quantize real-valued errors into integer indices [0..levels-1]
Handles negative ranges by computing min/max from a provided error array.
Provides a mapping table (index -> binary code -> dequantized error center).
"""

import numpy as np

class UniformQuantizer:
    def __init__(self, bits=2):
        self.bits = bits
        self.levels = 2 ** bits
        # metadata to be saved per-channel:
        self.min_val = None
        self.max_val = None
        self.step = None

    def fit(self, error_array):
        """Compute min/max and step from observed errors (per-channel)."""
        mn = float(error_array.min())
        mx = float(error_array.max())
        self.min_val = mn
        self.max_val = mx
        # to avoid zero division when mn==mx, expand tiny amount
        rng = max(1e-6, mx - mn)
        self.step = rng / self.levels

    def quantize(self, error_array):
        """
        Quantize error_array into integer indices.
        Returns indices (same shape), and dequantized error values (same shape).
        """
        if self.min_val is None or self.step is None:
            self.fit(error_array)
        # map error -> index
        idx = np.floor((error_array - self.min_val) / self.step).astype(np.int32)
        idx = np.clip(idx, 0, self.levels - 1)
        # dequantize: center of bin
        deq = self.min_val + (idx + 0.5) * self.step
        return idx, deq

    def quant_table(self):
        """Return list of tuples (index, binary_code, dequeue_value, interval_range)."""
        table = []
        for k in range(self.levels):
            code = format(k, '0{}b'.format(self.bits))
            left = self.min_val + k * self.step
            right = self.min_val + (k+1) * self.step
            center = self.min_val + (k + 0.5) * self.step
            table.append((k, code, center, (left, right)))
        return table
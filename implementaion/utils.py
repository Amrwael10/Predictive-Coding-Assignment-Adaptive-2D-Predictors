# helper functions (clamping, compression ratio)

"""
Helper utilities: compute compression ratio, visualize arrays (error/quantized),
clamping.
"""

def compute_compression_ratio(original_shape, channels, bits, header_bytes=0):
    H, W = original_shape
    orig_bits = H * W * channels * 8
    comp_bits = H * W * channels * bits + header_bytes * 8
    ratio = orig_bits / comp_bits if comp_bits > 0 else float('inf')
    return ratio, orig_bits, comp_bits

def header_bytes_for_shape(H, W, channels):
    return channels * (W + H)

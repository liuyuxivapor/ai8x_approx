import os
import numpy as np

# Generate a random float32 array
a_float32 = np.random.uniform(-128, 127, size=1).astype(np.float32)

# Quantize to 8-bit integers (int64)
quantized_a_int64 = np.int64(a_float32)

# print(a_float32, quantized_a_int64)

# Save the quantized array
np.save(os.path.join('tests', 'sample_biquad'), quantized_a_int64, allow_pickle=False, fix_imports=False)
import os
os.environ["JAX_SKIP_CUDA_CONSTRAINTS_CHECK"] = "True"

import jax
print(jax.devices())
import os

# SPEC §8: single-threaded BLAS for small-D work; must be set before numpy loads.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

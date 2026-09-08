export PYTHONUNBUFFERED=true
export JAX_CACHE_COMPILATION_DIR="./jax_cache"
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES="all"

set -eu

echo "Script execution started at - $(date) -"

python run_experiment.py --d-ap 10 --d-sta 2 --n-runs 10 --n-steps 5000
# python run_experiment.py --d-ap 20 --d-sta 2 --n-runs 50 --n-steps 10000
# python run_experiment.py --d-ap 30 --d-sta 2 --n-runs 50 --n-steps 10000
# python run_experiment.py --d-ap 10 --d-sta 4 --n-runs 50 --n-steps 10000

echo "Script execution completed at - $(date) -"
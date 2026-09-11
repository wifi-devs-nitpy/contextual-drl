export PYTHONUNBUFFERED=1
export JAX_CACHE_COMPILATION_DIR="./jax_cache"
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES="all"

set -eu

filename=$(basename "$0" .sh)
dir=$(realpath "$(dirname "$0")")
timestamp="$(date +"%d_%m_%y__%H_%M_%S")"
logs_dir="${dir}/logs"
mkdir -p "$logs_dir"

log_file="${logs_dir}/${filename}_${timestamp}.log"

exec > >(tee -a "$log_file") 2>&1


echo "Script execution started at - $(date) -"

# python run_experiment.py --d-ap 10 --d-sta 2 --n-runs 10 --n-steps 5000
# python run_experiment.py --d-ap 20 --d-sta 2 --n-runs 50 --n-steps 10000
# python run_experiment.py --d-ap 30 --d-sta 2 --n-runs 50 --n-steps 10000
# python run_experiment.py --d-ap 10 --d-sta 4 --n-runs 50 --n-steps 10000

# python mixed_scenarios/hmabs/01_mix_cen.py --d-ap 10 --d-sta 2 --n-runs 50 --n-steps 10000
# python run_experiment.py --d-ap 10 --d-sta 2 --n-runs 50 --n-steps 10000
python run_experiment.py --d-ap 20 --d-sta 2 --n-runs 50 --n-steps 10000
python run_experiment.py --d-ap 30 --d-sta 2 --n-runs 50 --n-steps 10000

echo "Script execution completed at - $(date) -"

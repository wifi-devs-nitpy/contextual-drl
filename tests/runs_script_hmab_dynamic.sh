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

python "C:\Users\devarshi\Documents\wifi_9\mapc-cmab\tests\run_experiment.py" --d-ap 10 --d-sta 2 --n-runs 50 --n-steps 10000 "dqn_with_layerNorm_newParams" & 
python "C:\Users\devarshi\Documents\wifi_9\mapc-cmab\tests\run_experiment.py" --d-ap 20 --d-sta 2 --n-runs 50 --n-steps 200000 --filename "dqn_with_layerNorm_newParams" &
python "C:\Users\devarshi\Documents\wifi_9\mapc-cmab\tests\run_experiment.py" --d-ap 30 --d-sta 2 --n-runs 50 --n-steps 10000 --filename "dqn_with_layerNorm_newParams" & 

wait 

echo "Script execution completed at - $(date) -"

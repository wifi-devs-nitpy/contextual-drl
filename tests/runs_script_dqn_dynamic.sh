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

echo "DQN DYnamic" 
echo "Script execution started at - $(date) -"

python mixed_scenarios/dqn/01_mix_cen_dqn.py --d-ap 10 --d-sta 2 --n-runs 50 --n-steps 10000
python mixed_scenarios/dqn/01_mix_cen_dqn.py --d-ap 20 --d-sta 2 --n-runs 50 --n-steps 10000
python mixed_scenarios/dqn/01_mix_cen_dqn.py --d-ap 30 --d-sta 2 --n-runs 50 --n-steps 10000

echo "Script execution completed at - $(date) -"

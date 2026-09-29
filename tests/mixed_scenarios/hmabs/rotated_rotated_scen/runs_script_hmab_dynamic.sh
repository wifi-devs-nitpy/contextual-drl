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

echo "HMAB dynamic" 
echo "Script execution started at - $(date) -"

(python 01_mix_cen_hmab.py --d-ap 10 --d-sta 4 --n-runs 50 --n-steps 10000) &
pid1=$!

(sleep 300 && python 01_mix_cen_hmab.py --d-ap 20 --d-sta 4 --n-runs 50 --n-steps 10000) &
pid2=$!

(sleep 700 && python 01_mix_cen_hmab.py --d-ap 30 --d-sta 4 --n-runs 50 --n-steps 10000) &
pid3=$!

echo "Started:"
echo "  d-ap=10 PID=$pid1"
echo "  d-ap=20 PID=$pid2"
echo "  d-ap=30 PID=$pid3"

echo "waiting for all the background processes to finish"

wait 

echo "Script execution completed at - $(date) -"

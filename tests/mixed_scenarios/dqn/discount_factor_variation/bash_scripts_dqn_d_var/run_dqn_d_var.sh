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


python 01_mix_cen_dqn_discount_var.py --d-ap 20 --d-sta 2 --n-runs 50 --n-steps 10000 --d1 0.1 --d2 0.1 --d3 0.1 --d4 0.1 --filename "d_10_disc_0.1_hdqn_r_10_step_10k" & 
python 01_mix_cen_dqn_discount_var.py --d-ap 20 --d-sta 2 --n-runs 50 --n-steps 10000 --d1 0.2 --d2 0.2 --d3 0.2 --d4 0.2 --filename "d_10_disc_0.2_hdqn_r_10_step_10k" & 
python 01_mix_cen_dqn_discount_var.py --d-ap 20 --d-sta 2 --n-runs 50 --n-steps 10000 --d1 0.3 --d2 0.3 --d3 0.3 --d4 0.3 --filename "d_10_disc_0.3_hdqn_r_10_step_10k" & 
python 01_mix_cen_dqn_discount_var.py --d-ap 20 --d-sta 2 --n-runs 50 --n-steps 10000 --d1 0.4 --d2 0.4 --d3 0.4 --d4 0.4 --filename "d_10_disc_0.4_hdqn_r_10_step_10k" & 
python 01_mix_cen_dqn_discount_var.py --d-ap 20 --d-sta 2 --n-runs 50 --n-steps 10000 --d1 0.5 --d2 0.5 --d3 0.5 --d4 0.5 --filename "d_10_disc_0.5_hdqn_r_10_step_10k" & 
python 01_mix_cen_dqn_discount_var.py --d-ap 20 --d-sta 2 --n-runs 50 --n-steps 10000 --d1 0.6 --d2 0.6 --d3 0.6 --d4 0.6 --filename "d_10_disc_0.6_hdqn_r_10_step_10k" & 
python 01_mix_cen_dqn_discount_var.py --d-ap 20 --d-sta 2 --n-runs 50 --n-steps 10000 --d1 0.7 --d2 0.7 --d3 0.7 --d4 0.7 --filename "d_10_disc_0.7_hdqn_r_10_step_10k" & 
python 01_mix_cen_dqn_discount_var.py --d-ap 20 --d-sta 2 --n-runs 50 --n-steps 10000 --d1 0.8 --d2 0.8 --d3 0.8 --d4 0.8 --filename "d_10_disc_0.8_hdqn_r_10_step_10k" & 
python 01_mix_cen_dqn_discount_var.py --d-ap 20 --d-sta 2 --n-runs 50 --n-steps 10000 --d1 0.9 --d2 0.9 --d3 0.9 --d4 0.9 --filename "d_10_disc_0.9_hdqn_r_10_step_10k" & 
python 01_mix_cen_dqn_discount_var.py --d-ap 20 --d-sta 2 --n-runs 50 --n-steps 10000 --d1 1.0 --d2 1.0 --d3 1.0 --d4 1.0 --filename "d_10_disc_1.0_hdqn_r_10_step_10k" & 

echo "waiting till all the background tasks complete" 

wait 

echo "Script execution completed at - $(date) -"

echo "sending the logs in the email" 
python ./bash_scripts_dqn_d_var/send_email.py --subject "HDQN Discount Grid Search Results"
#!/bin/sh
##
## Copyright 2024 LY Corporation
##
## LY Corporation licenses this file to you under the Apache License,
## version 2.0 (the "License"); you may not use this file except in compliance
## with the License. You may obtain a copy of the License at:
##
##   https://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
## WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
## License for the specific language governing permissions and limitations
## under the License.
##


# Arguments
mode=$1
dry_run=${2:-false} # Set default value to false if not set
exp_name="30K_helpful_dpo_safety"
output_dir="./output/$exp_name"
nohup_dir="$output_dir/nohup"
batch_size=16
num_train_epochs=3
grad_accum_steps=2
beta_helpful_dpo=0.1
beta_safety_kto=(0.1 0.05 0.025 0.01)
beta_safety_dpo=(0.1 0.05 0.025 0.01)
config_files=("./config/train/accelerate_config_0_1.yaml" "./config/train/accelerate_config_2_3.yaml" "./config/train/accelerate_config_4_5.yaml" "./config/train/accelerate_config_6_7.yaml")
full_config_file="./config/train/accelerate_config_0_7.yaml"

# Set Environment Variables
source ./script/set_envvar.sh

# Ensure nohup directory exists
mkdir -p $nohup_dir

# Function to run DPO training for helpfulness
run_helpful_dpo() {
  cmd="nohup accelerate launch --config_file $full_config_file \
  -m src.train.helpfulness_dpo \
  --model_name_or_path 'PKU-Alignment/alpaca-7b-reproduced' \
  --experiment_name $exp_name \
  --run_name='helpful_dpo_$beta_helpful_dpo' \
  --output_dir='$output_dir/helpful_dpo_$beta_helpful_dpo' \
  --per_device_train_batch_size $batch_size \
  --gradient_accumulation_steps $grad_accum_steps \
  --beta $beta_helpful_dpo \
  --num_train_epochs $num_train_epochs \
  --save_strategy 'steps' \
  --save_steps 0.99999 \
  --model_dtype 'bfloat16'
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  > $nohup_dir/helpful_dpo_$beta_helpful_dpo.txt &"
  echo $cmd
  if [ "$dry_run" != "true" ]; then
    eval $cmd
  fi
}

# Function to run KTO training for safety
run_safety_kto() {
  for i in $(seq 0 3); do
    cmd="nohup accelerate launch --config_file ${config_files[i]} \
    -m src.train.safety_kto \
    --model_name_or_path '$output_dir/helpful_dpo_$beta_helpful_dpo' \
    --experiment_name $exp_name \
    --run_name='helpful_dpo_safety_kto_${beta_safety_kto[i]}' \
    --output_dir='$output_dir/helpful_dpo_safety_kto_${beta_safety_kto[i]}' \
    --beta ${beta_safety_kto[i]} \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $batch_size \
    --gradient_accumulation_steps $grad_accum_steps \
    --save_strategy 'steps' \
    --save_steps 0.99999 \
    --model_dtype 'bfloat16'
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    > $nohup_dir/safety_kto_${beta_safety_kto[i]}.txt &"
    echo $cmd
    if [ "$dry_run" != "true" ]; then
      eval $cmd
    fi
  done
}

# Function to run DPO training for safety
run_safety_dpo() {
  for i in $(seq 0 3); do
    cmd="nohup accelerate launch --config_file ${config_files[i]} \
    -m src.train.safety_dpo \
    --model_name_or_path '$output_dir/helpful_dpo_$beta_helpful_dpo' \
    --experiment_name $exp_name \
    --run_name='helpful_dpo_safety_dpo_${beta_safety_dpo[i]}' \
    --output_dir='$output_dir/helpful_dpo_safety_dpo_${beta_safety_dpo[i]}' \
    --per_device_train_batch_size $batch_size \
    --gradient_accumulation_steps $grad_accum_steps \
    --beta ${beta_safety_dpo[i]} \
    --num_train_epochs $num_train_epochs \
    --save_strategy 'steps' \
    --save_steps 0.99999 \
    --model_dtype 'bfloat16'
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    > $nohup_dir/safety_dpo_${beta_safety_dpo[i]}.txt &"
    echo $cmd
    if [ "$dry_run" != "true" ]; then
    eval $cmd
    fi
  done
}

# Main switch case
case $mode in
  "helpful_dpo") run_helpful_dpo ;; # STEP 1
  "safety_kto") run_safety_kto ;; # STEP 2 (KTO)
  "safety_dpo") run_safety_dpo ;; # STEP 2 (DPO)
  *) echo "Invalid mode. Use 'helpful_dpo', 'safety_kto', 'safety_dpo"; exit 1 ;;
esac

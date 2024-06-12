#!/bin/bash
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


# Parse command-line arguments
mode=${1:-"all"}
exp_name="30K_helpful_dpo_safety"
helpful_problem_path=${2:-"asset/helpful_problem.json"}
safety_problem_path=${3:-"asset/safety_problem.json"}
base_dir=${4:-"output"}
model_dir="$base_dir/$exp_name"
output_dir="$base_dir/eval"
n_gpu=8

# Set Environment Variables
source ./script/set_envvar.sh

# Define the array of model names or paths
models=(
    "PKU-Alignment/beaver-7b-v1.0"
    "PKU-Alignment/alpaca-7b-reproduced"
    "$model_dir/helpful_dpo_0.1"
    "$model_dir/helpful_dpo_safety_kto_0.1"
    "$model_dir/helpful_dpo_safety_kto_0.05"
    "$model_dir/helpful_dpo_safety_kto_0.025"
    "$model_dir/helpful_dpo_safety_kto_0.01"
    "$model_dir/helpful_dpo_safety_dpo_0.1"
    "$model_dir/helpful_dpo_safety_dpo_0.05"
    "$model_dir/helpful_dpo_safety_dpo_0.025"
    "$model_dir/helpful_dpo_safety_dpo_0.01"
)

# Function to generate text for each model (one model per gpu)
generate_for_problem() {
    local problem_path="$1"
    local output_subdir="$2"
    local gpu_id=0

    for model in "${models[@]}"; do
        # Set CUDA_VISIBLE_DEVICES to use a specific GPU
        CUDA_VISIBLE_DEVICES=$gpu_id python -m src.evaluate.generate --model_name_or_path "$model" --problem_path "$problem_path" --output_dir "$output_dir/$output_subdir" &
        # Increment GPU ID and reset if it exceeds the number of GPUs
        ((gpu_id=(gpu_id+1)%n_gpu))

        # Wait for all background jobs to finish if we've reached the GPU limit
        if [[ $((gpu_id)) -eq 0 ]]; then
            wait
        fi
    done
    wait # Wait for any remaining background jobs to finish
}

generate() {
    generate_for_problem "$helpful_problem_path" "helpful_generation"
    generate_for_problem "$safety_problem_path" "safety_generation"
    echo "Text generation completed for all models."
}

# Function to evaluate the generated text
# evaluate_base only compares the target model to the base model, to compute win_rates
evaluate_base() {
    cmd="python -m src.evaluate.gpt4_evaluate --vs_base_only --criteria helpfulness --generation_dir \"$output_dir/helpful_generation\" --output_path \"$output_dir/gpt4_helpful_evaluation.json\""
    echo "Command: $cmd"
    eval $cmd

    cmd="python -m src.evaluate.gpt4_evaluate --vs_base_only --criteria harmlessness --generation_dir \"$output_dir/safety_generation\" --output_path \"$output_dir/gpt4_safety_evaluation.json\""
    echo "Command: $cmd"
    eval $cmd
}

# evaluate_full compares all the possible pair of output defined in models aboved, to compute elo_scores
evaluate_full() {
    cmd="python -m src.evaluate.gpt4_evaluate --criteria helpfulness --generation_dir \"$output_dir/helpful_generation\" --output_path \"$base_dir/eval/$exp_name/gpt4_helpful_evaluation.json\""
    echo "Command: $cmd"
    eval $cmd

    cmd="python -m src.evaluate.gpt4_evaluate --criteria harmlessness --generation_dir \"$output_dir/safety_generation\" --output_path \"$base_dir/eval/$exp_name/eval/gpt4_safety_evaluation.json\""
    echo "Command: $cmd"
    eval $cmd
}

# Check the mode and execute the corresponding function(s)
case $mode in
    generate)
        generate
        ;;
    evaluate_base)
        evaluate_base
        ;;
    evaluate_full)
        evaluate_full
        ;;
    *)
        echo "Invalid mode: $mode. Please choose from 'generate', 'evaluate_base', 'evaluate_full'."
        exit 1
        ;;
esac

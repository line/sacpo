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
exp_name="30K_helpful_dpo_safety_merge"
output_dir="./output/$exp_name"
config_dir="./config/merge"

model_names=(
    "linear_0.25"
    "linear_0.5"
    "linear_0.75"
    "naive_0.25"
    "naive_0.5"
    "naive_0.75"
)

for model_name in "${model_names[@]}"; do
    output_model_path="$output_dir/$model_name"
    merge_config_path="$config_dir/$model_name.yaml"
    cmd="mergekit-yaml $merge_config_path $output_model_path"
    echo "Command: $cmd"
    eval $cmd
done

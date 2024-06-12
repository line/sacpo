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
base_dir='output/eval'
working_dir="$base_dir"
plot_config="config/plot/30K_all.yaml"

python -m src.evaluate.compute_scores --base_model alpaca-7b-reproduced --criteria helpfulness --eval_result_path "$working_dir/gpt4_helpful_evaluation.json" --output_path "$working_dir/helpful_summary.json" --plot_config "$plot_config"

python -m src.evaluate.compute_scores --base_model alpaca-7b-reproduced --criteria harmlessness --eval_result_path "$working_dir/gpt4_safety_evaluation.json" --output_path "$working_dir/safety_summary.json" --plot_config "$plot_config"

python -m src.evaluate.plot_scores --score_type win_rate --plot_config "$plot_config" --x_summary_path "$working_dir/helpful_summary.json" --y_summary_path "$working_dir/safety_summary.json" --output_dir "$working_dir" --name_prefix "combined"

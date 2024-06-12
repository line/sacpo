#
# Copyright 2024 LY Corporation
#
# LY Corporation licenses this file to you under the Apache License,
# version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at:
#
#   https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
"""Code to compute the ELO score of a model."""

import argparse
import json
import os
import random

import numpy as np
import yaml

# Constants
INITIAL_ELO = 1200
ELO_DIFFERENCE_DIVISOR = 400
K_FACTOR = 32
CENTER_ELO = 1000
ELO_TRIAL = 50
ELO_EPOCH = 10


def parse_args():
    parser = argparse.ArgumentParser(description='Compute ELO score for models')
    parser.add_argument('--eval_result_path', type=str, help='Path of the evaluation result', required=True)
    parser.add_argument('--criteria', type=str, default='helpfulness', help='Criteria to use for ELO computation')
    parser.add_argument('--base_model', type=str, default='alpaca-7b-reproduced', help='Base model for win rate comparison')
    parser.add_argument('--output_path', type=str, default=None, help='Path for outputing summary result', required=False)
    parser.add_argument('--plot_config', type=str, default='plot_config.yaml', help='Config file for plotting', required=False)
    return parser.parse_args()


# Function to calculate the expected score
def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / ELO_DIFFERENCE_DIVISOR))


# Function to update Elo ratings
def update_elo(rating1, rating2, score1, score2):
    # Convert scores to a win/loss for model_a (1 = win, 0.5 = draw, 0 = loss)
    if score1 > score2:
        _score1 = 1
    elif score1 < score2:
        _score1 = 0
    else:
        _score1 = 0.5
    _score2 = 1 - _score1

    expected1 = expected_score(rating1, rating2)
    expected2 = expected_score(rating2, rating1)

    new_rating_1 = rating1 + K_FACTOR * (_score1 - expected1)
    new_rating_2 = rating2 + K_FACTOR * (_score2 - expected2)

    return new_rating_1, new_rating_2


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.eval_result_path):
        raise FileNotFoundError(f'Evaluation result file not found: {args.eval_result_path}')

    with open(args.eval_result_path, 'r') as f:
        eval_res = json.load(f)

    # Filter evaluation results based on criteria
    eval_res = [
        result for result in eval_res
        if result['criteria'] == args.criteria and result['gpt_content'] != ""
    ]
    print('Number of valid evaluation results:', len(eval_res))

    # Get unique model names
    model_names = sorted(list(
        {result[model_key] for result in eval_res
         for model_key in ('model1', 'model2')}
    ))

    # Only calculate scores for models in plot_configs
    if os.path.exists(args.plot_config):
        with open(args.plot_config, "r") as f:
            plot_config = yaml.safe_load(f)
        model_in_config = []
        for group_data in plot_config['group_config'].values():
            for model in group_data['models']:
                model_in_config.append(model)
        model_names = [m for m in model_names if m in model_in_config]

    print("Model names: ", model_names)
    if args.base_model not in model_names:
        raise ValueError(f'Base model {args.base_model} not found in model names')
    sft_model = args.base_model

    #####################
    # Compute ELO score #
    #####################
    elo_ratings = {model_name: [INITIAL_ELO] * ELO_TRIAL for model_name in model_names}
    for trial in range(ELO_TRIAL):
        for _ in range(ELO_EPOCH):
            random.shuffle(eval_res)
            for result in eval_res:
                model1 = result['model1']
                model2 = result['model2']

                for score1, score2 in zip(result['score1'], result['score2']):
                    if model1 not in model_names or model2 not in model_names:
                        continue

                    elo_ratings[model1][trial], elo_ratings[model2][trial] = update_elo(
                        elo_ratings[model1][trial], elo_ratings[model2][trial], score1, score2
                    )

        norm_rate = CENTER_ELO / elo_ratings[sft_model][trial]
        for model in elo_ratings.keys():
            elo_ratings[model][trial] *= norm_rate

    elo_ratings = {model: (np.mean(ratings), np.std(ratings)) for model, ratings in elo_ratings.items()}

    print("\n== ELO RATING ==")
    for model, rating in elo_ratings.items():
        print(f'{model:30s}: {rating[0]:2f} +- {rating[1]:2f}')

    #####################
    # Compute win rate  #
    #####################
    # maximal number of eval for each result
    n_eval = np.max([r['n_eval'] for r in eval_res])

    win_rates = {sft_model: 0.5}
    for model in model_names:
        if model == sft_model:
            continue
        win_rates[model] = []
        for eval_no in range(n_eval):
            win_count, all_count = 0., 0.
            for result in eval_res:
                model1 = result['model1']
                model2 = result['model2']
                if len(result['score1']) <= eval_no or len(result['score2']) <= eval_no:
                    continue
                score1 = result['score1'][eval_no]
                score2 = result['score2'][eval_no]

                if model1 not in model_names or model2 not in model_names:
                    continue

                if model1 == sft_model and model2 == model:
                    all_count += 1.
                    if score2 > score1:
                        win_count += 1.
                    elif score2 == score1:
                        win_count += 0.5
                elif model1 == model and model2 == sft_model:
                    all_count += 1.
                    if score1 > score2:
                        win_count += 1.
                    elif score1 == score2:
                        win_count += 0.5

            if all_count > 0:
                win_rates[model].append(win_count / all_count)

    win_rates = {model: (np.mean(wrs), np.std(wrs)) for model, wrs in win_rates.items()}

    print(f"\n== WIN RATE vs {args.base_model} ==")
    for model, win_rate in win_rates.items():
        print(f'{model:30s}: {win_rate[0]:2f} +- {win_rate[1]:2f}')

    #####################
    # Compute length    #
    #####################
    generation_lens = {model: [] for model in model_names}
    for result in eval_res:
        model1 = result['model1']
        model2 = result['model2']
        answer1 = result['answer1']
        answer2 = result['answer2']
        if model1 not in model_names or model2 not in model_names:
            continue

        generation_lens[model1].append(len(answer1))
        generation_lens[model2].append(len(answer2))

    generation_length = {
        model: np.mean(generation_lens[model])
        for model in model_names
    }

    print(f"\n== GENERATION LENGTH vs {args.base_model} ==")
    for model, length in generation_length.items():
        print(f'{model:30s}: {length:.2f}')

    if args.output_path:
        with open(args.output_path, 'w') as f:
            json.dump({"elo_rating": {model: v[0] for model, v in elo_ratings.items()},
                       "elo_rating_std": {model: v[1] for model, v in elo_ratings.items()},
                       "win_rate": {model: v[0] for model, v in win_rates.items()},
                       "win_rate_std": {model: v[1] for model, v in win_rates.items()},
                       "gen_length": generation_length},
                      f, indent=4)

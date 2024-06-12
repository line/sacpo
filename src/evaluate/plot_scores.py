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

import numpy as np
import yaml
from adjustText import adjust_text
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Compute ELO score for models')
    parser.add_argument('--x_criteria', type=str, default='helpfulness', help='Criteria for x-axis')
    parser.add_argument('--y_criteria', type=str, default='harmlessness', help='Criteria for y-axis')
    parser.add_argument('--x_summary_path', type=str, default=None)
    parser.add_argument('--y_summary_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None, help='Directory for outputing summary result', required=False)
    parser.add_argument('--plot_config', type=str, default='plot_config.yaml', help='Config file to convert label', required=False)
    parser.add_argument('--name_prefix', type=str, default='', help='name prefix for saving')
    parser.add_argument('--auto_adjust_text', action='store_true', help='use adjustText to adjust text position')
    parser.add_argument('--score_type', type=str, default='elo_rating', help='type of score to plot')

    return parser.parse_args()


def plot_2_criteria(xs, ys, x_criteria, y_criteria, score_type="", output_dir=None, name_prefix=""):
    """
    Plot a scatter plot that indicates the performance of multiple models
    on two criteria.

    xs: a dictionary {model:score} for x-axis criteria
    ys: a dictionary {model:score} for y-axis criteria
    x_criteria: name of x-axis criteria
    y_criteria: name of y-axis criteria
    score_type: a string that represents the type of score being plotted
    output_dir: dir path to save the plot
    name_prefix: prefix of the file_name to save
    """
    plt.figure(figsize=(6, 6))

    # Load mapping to convert model_name -> plot_label
    if os.path.exists(args.plot_config):
        with open(args.plot_config, "r") as f:
            plot_config = yaml.safe_load(f)
    else:
        raise FileNotFoundError('Plot config not found')

    # Ensure that the same models are present in both dictionaries
    model_in_config = []
    for group_data in plot_config['group_config'].values():
        for model in group_data['models']:
            model_in_config.append(model)
    models = set(xs.keys()).intersection(set(ys.keys())).intersection(set(model_in_config))

    # Compute values to adjust lim_x and lim_y
    min_x = np.min(np.array([xs[model] for model in models]))
    max_x = np.max(np.array([xs[model] for model in models]))
    min_y = np.min(np.array([ys[model] for model in models]))
    max_y = np.max(np.array([ys[model] for model in models]))
    if score_type == "elo_rating":
        margin, pivot = 50, 1000
        lim_x_left, lim_y_bottom = min_x - margin, min_y - margin
        lim_x_right, lim_y_top = max_x + margin, max_y + margin
        scale = 750.
    else:
        margin, pivot = 0.05, 0.5
        lim_x_left, lim_x_right = 0.4, 0.9
        lim_y_bottom, lim_y_top = 0.4, 0.9
        scale = 1.

    # Retrieve configs
    # Group config
    group_config = plot_config['group_config']

    # Model config
    model_config = {}
    for group_name, _config in group_config.items():
        for model, label in zip(_config['models'], _config['labels']):
            model_config[model] = {}
            model_config[model]['label'] = label
            model_config[model]['group'] = group_name
            model_config[model]['color'] = _config['color']
            model_config[model]['marker'] = _config['marker']

    # Subplot config
    subplot_config = []
    titles = []
    if 'subplot_config' in plot_config:
        for _, _config in plot_config['subplot_config'].items():
            model_to_plot = []
            if 'groups' in _config:
                for group_name in _config['groups']:
                    model_to_plot.extend(group_config[group_name]['models'])
            if 'models' in _config:
                model_to_plot.extend(_config['models'])
            subplot_config.append(model_to_plot)
            titles.append(_config.get('title', ''))
    else:
        model_to_plot = [
            model
            for _config in group_config.values()
            for model in _config['models']
        ]
        subplot_config.append(model_to_plot)
        titles.append('')

    n_subplot = len(subplot_config)
    height = 6 * n_subplot
    width = 6
    fig, axs = plt.subplots(1, n_subplot, figsize=(height, width))

    if n_subplot == 1:
        axs = [axs]

    manual_offset = {
        'alpaca-7b-reproduced': (-0.022, 0.022),
        'safety_dpo_0.01': (-0.06, 0),
        'safety_dpo_helpful_dpo_0.1': (0, 0.022),
        'naive_0.75': (-0.022, 0.022)
    }

    for i, (ax, models) in enumerate(zip(axs, subplot_config)):
        # lim_x, lim_y adjustment
        ax.set_ylim((lim_y_bottom, lim_y_top))
        ax.set_xlim((lim_x_left, lim_x_right))

        # Set the style of the axes and grid
        ax.axvline(x=pivot, color='dimgray', linestyle='--', alpha=0.5)
        ax.axhline(y=pivot, color='dimgray', linestyle='--', alpha=0.5)
        ax.fill_between(x=[lim_x_left, lim_x_right], y1=lim_y_bottom, y2=pivot, color='gray', alpha=0.25)
        ax.fill_betweenx(y=[lim_y_bottom, lim_y_top], x1=lim_x_left, x2=pivot, color='gray', alpha=0.25)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')

        # Set the labels and ticks for the axes
        ax.set_xlabel(x_criteria, fontsize=17, labelpad=10)
        if i == 0:
            ax.set_ylabel(y_criteria, fontsize=17, labelpad=10)
            ax.tick_params(axis='both', which='major', labelsize=15)
        else:
            ax.tick_params(axis='y', labelleft=False)
            ax.tick_params(axis='x', which='major', labelsize=15)

        if titles[i] != '':
            ax.text(0.5, -0.2, titles[i], transform=ax.transAxes, fontsize=16, va='top', ha='center')

        texts = []
        for model in models:
            _config = model_config[model]

            # Plot data
            x_scores = xs[model]
            y_scores = ys[model]
            ax.scatter(
                x_scores,
                y_scores,
                color=_config['color'],
                marker=_config['marker'],
                edgecolors='w', s=200, alpha=1.,
            )

            # Plot label
            x_offset, y_offset = manual_offset.get(model, (0, -0.022))
            texts.append(
                ax.text(
                    xs[model] + x_offset * scale,
                    ys[model] + y_offset * scale,
                    model_config[model]['label'],
                    ha='center', va='center',
                    fontsize=15,
                    color=_config['color']
                )
            )

        if args.auto_adjust_text:
            adjust_text(
                texts,
                arrowprops=dict(arrowstyle='-', color='darkgray', lw=1.5),
                ax=ax,
            )

    plt.subplots_adjust(wspace=0.1)
    legend_handles = [
        ax.scatter(
            [-100],
            [-100],
            color=_config['color'],
            marker=_config['marker'],
            edgecolors='w',
            s=200,
            label=_config['legend'].replace('->', '$\\rightarrow$')
        )
        for _config in group_config.values()
        if _config['legend'] != ''
    ]
    fig.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        columnspacing=1.5,
        handletextpad=0.1,
        ncol=4,
        fontsize=16
    )

    # Check if output directory is provided and save the plot
    if output_dir:
        # Create the directory if it does not exist
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        # Save the plot with a descriptive filename
        fig.savefig(os.path.join(output_dir, f'{name_prefix}_{score_type}.pdf'), bbox_inches='tight')
        fig.savefig(os.path.join(output_dir, f'{name_prefix}_{score_type}.png'), bbox_inches='tight')
    else:
        # Show the plot if no output directory is provided
        plt.show()


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.x_summary_path) or not os.path.exists(args.y_summary_path):
        raise FileNotFoundError('Evaluation result files not found')

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.x_summary_path, 'r') as f:
        x_summary = json.load(f)
    with open(args.y_summary_path, 'r') as f:
        y_summary = json.load(f)

    plot_2_criteria(
        x_summary[args.score_type],
        y_summary[args.score_type],
        x_criteria=args.x_criteria,
        y_criteria=args.y_criteria,
        score_type=args.score_type,
        output_dir=args.output_dir,
        name_prefix=args.name_prefix
    )

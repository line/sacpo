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
import argparse
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import openai
from dotenv import load_dotenv
from tqdm import tqdm

from ..constant import (HARMLESSNESS_SYSTEM_PROMPT, HARMLESSNESS_USER_PROMPT,
                        HELPFULNESS_SYSTEM_PROMPT, HELPFULNESS_USER_PROMPT)

EVAL_SETUP = {
    "helpfulness": (HELPFULNESS_SYSTEM_PROMPT, HELPFULNESS_USER_PROMPT),
    "harmlessness": (HARMLESSNESS_SYSTEM_PROMPT, HARMLESSNESS_USER_PROMPT),
}


def load_environment_variables():
    load_dotenv()
    try:
        openai.api_key = os.environ["OPENAI_API_KEY"]
    except KeyError as e:
        logging.error(f'Missing environment variable: {e}')
        exit(1)


def parse_arguments() -> argparse.Namespace:
    """
       Parse the command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Generate answer for given models'
    )
    parser.add_argument(
        '--criteria',
        type=str,
        default='helpfulness',
        help='Criteria to use for ELO computation'
    )
    parser.add_argument(
        '--generation_dir',
        type=str,
        default=None,
        help='directory to store generation output'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='directory to store generation output'
    )
    parser.add_argument(
        '--sanity_check',
        action='store_true',  # Use store_true for boolean flags
        help='set to true to only generate answers for first 10 problems'
    )
    parser.add_argument(
        '--num_eval_each',
        type=int,
        default=3,
        help='Number of evaluations for each answer pair'
    )
    parser.add_argument(
        '--base_model',
        type=str,
        default='alpaca-7b-reproduced',
        help='Base model for win rate comparison')
    parser.add_argument(
        '--vs_base_only',
        action='store_true',  # Use store_true for boolean flags
        help='set to true to only evaluate versus base model'
    )

    return parser.parse_args()


@dataclass
class EvaluationResult:
    criteria: str
    question: str
    model1: str
    model2: str
    answer1: str
    answer2: str
    n_eval: int = 0
    gpt_content: list = field(default_factory=list)
    score1: list = field(default_factory=list)
    score2: list = field(default_factory=list)

    def to_dict(self):
        return {
            "criteria": self.criteria,
            "question": self.question,
            "model1": self.model1,
            "model2": self.model2,
            "answer1": self.answer1,
            "answer2": self.answer2,
            "n_eval": self.n_eval,
            "gpt_content": self.gpt_content,
            "score1": self.score1,
            "score2": self.score2,
        }


def query_gpt4(sys_prompt: str, user_prompt: str, client: Optional[openai.OpenAI]) -> str:
    try:
        chat_completion = client.chat.completions.create(
            model="gpt-4-0125-preview",
            # model="gpt-4",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=8,
        )
        content = chat_completion.choices[0].message.content
        time.sleep(1)
    except Exception as ex:
        print(ex)
        time.sleep(3)
        content = None

    try:
        score1, score2 = map(float, content.split('\n')[0].split(' '))
    except Exception:  # pylint: disable=broad-except # noqa: BLE001
        score1, score2 = 0.0, 0.0

    return content, score1, score2


def save_all_eval_results(eval_res: list[EvaluationResult], eval_res_path: str):
    eval_res_dict = [x.to_dict() for x in eval_res]
    with open(eval_res_path, "w", encoding="utf-8") as f:
        json.dump(eval_res_dict, f, indent=4)


def save_progress(func):
    def wrapper(*args, **kwargs):
        wrapper.counter += 1
        result = func(*args, **kwargs)
        if wrapper.counter % 50 == 0:
            save_all_eval_results(kwargs.get('eval_res'), kwargs.get('eval_res_path'))
        return result

    wrapper.counter = 0
    return wrapper


@save_progress
def evaluate_single_pair(index, eval_res, eval_res_path, client):
    single_eval = eval_res[index]
    system_prompt, user_prompt_template = EVAL_SETUP[single_eval.criteria]

    user_prompt = user_prompt_template.format(
        question=single_eval.question,
        answer1=single_eval.answer1,
        answer2=single_eval.answer2,
    )
    gpt_content, score1, score2 = query_gpt4(
        sys_prompt=system_prompt,
        user_prompt=user_prompt,
        client=client
    )
    single_eval.gpt_content.append(gpt_content)
    single_eval.score1.append(score1)
    single_eval.score2.append(score2)
    single_eval.n_eval += 1


if __name__ == "__main__":
    load_environment_variables()
    args = parse_arguments()

    # Ensure output directory exists
    generation_dir = Path(args.generation_dir)
    if not generation_dir.exists():
        logging.error('Output directory does not exist. Please specify a valid directory.')
        exit(1)

    eval_res_path = Path(args.output_path)
    # Check if the generation results already exists
    # If it does, load the results from the file
    if eval_res_path.exists():
        with open(eval_res_path, "r", encoding="utf-8") as f:
            eval_res_dict = json.load(f)
            eval_res = [EvaluationResult(**x) for x in eval_res_dict]
    else:
        eval_res = []

    # List of evaluation pair which evaluation is finished
    existed_results = set([
        (x.question, x.criteria, x.model1, x.model2)
        for x in eval_res
    ])

    # We load all generation, check if they already exist, if not we add them to eval_res
    all_generations = {}
    for json_result in os.listdir(generation_dir):
        with open(os.path.join(generation_dir, json_result), "r", encoding="utf-8") as f:
            result = json.load(f)
            model_name = result['model_name']
            all_generations[model_name] = result['data']

    all_model_names = sorted(list(all_generations.keys()))
    for model1 in all_model_names:
        for model2 in all_model_names:
            if model1 == model2:
                continue
            if args.vs_base_only and model1 != args.base_model and model2 != args.base_model:
                continue

            for gen1, gen2 in zip(all_generations[model1], all_generations[model2]):
                question = gen1['prompt']
                answer1 = gen1['generation']
                answer2 = gen2['generation']

                if (question, args.criteria, model1, model2) not in existed_results:
                    eval_res.append(EvaluationResult(
                        criteria=args.criteria,
                        question=question,
                        model1=model1, answer1=answer1,
                        model2=model2, answer2=answer2,
                    ))

    # Save the temporary evaluation results
    save_all_eval_results(eval_res, eval_res_path)

    # For the rest, we generate the GPT-4 scores
    open_ai_client = openai.OpenAI(api_key=openai.api_key)

    remain_idxs = [i for i, x in enumerate(eval_res) if x.n_eval < args.num_eval_each]
    random.shuffle(remain_idxs)

    print(f"Remaining evaluations: {len(remain_idxs)}")
    for i in tqdm(remain_idxs):
        for _ in range(args.num_eval_each - eval_res[i].n_eval):
            evaluate_single_pair(
                index=i,
                eval_res=eval_res,
                eval_res_path=eval_res_path,
                client=open_ai_client
            )
    save_all_eval_results(eval_res, eval_res_path)

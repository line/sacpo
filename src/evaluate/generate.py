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
"""
Generate answers for given problems using the model
"""
import argparse
import json
import os
import sys
from typing import Dict, List

from tqdm import tqdm

from ..constant import PROMPT_INPUT, STOP_WORDS
from ..util import (KeywordStoppingCriteria, load_pretrained_models,
                    remove_keywords)


def parse_arguments() -> argparse.Namespace:
    """
       Parse the command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Generate answer for given models'
    )
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        help='name of the model to load',
        required=True,
    )
    parser.add_argument(
        '--problem_path',
        type=str,
        help='path of the problem json',
        required=True
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='directory to store generation output'
    )
    parser.add_argument(
        '--sanity_check',
        action='store_true',  # Use store_true for boolean flags
        help='set to true to only generate answers for first 10 problems'
    )

    return parser.parse_args()


def generate_answers(
    model_name: str,
    model_path: str,
    problems: List[Dict],
    sanity_check: bool = False
) -> List[str]:
    """
        Generate answers for given problems using the model
        model_name: name of the model
        model_path: path of the model
        problems: list of problems
        sanity_check: if True, only generate answers for first 10 problems
    """
    _problems = problems[:10] if sanity_check else problems

    # Load model and tokenizer
    model, tokenizer = load_pretrained_models(
        model_path,
        device_map="auto",
        trust_remote_code=True,
    )

    # Setup stop words
    stop_ids_list = [tokenizer.encode(w)[1:] for w in STOP_WORDS]
    keyword_stopping_criteria = KeywordStoppingCriteria(stop_ids_list)

    # Generate answers
    print(f"Generating {len(problems)} answer for {model_name}...")
    answers = []
    for problem in tqdm(_problems):
        prompt = PROMPT_INPUT.format(input=problem["prompt"])
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(
            **input_ids,
            max_new_tokens=512,
            do_sample=False,
            stopping_criteria=[keyword_stopping_criteria],
        )
        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)[
            len(prompt):
        ]
        answer = answer.strip()
        answer = remove_keywords(answer, STOP_WORDS)
        answers.append(answer)

    # Return
    return answers


if __name__ == '__main__':
    args = parse_arguments()

    # Initialize an empty output dictionary
    output = {}

    # Load problems
    with open(args.problem_path, "r", encoding="utf-8") as f:
        problems = json.load(f)

    # Generate answers
    model_path = args.model_name_or_path.rstrip('/')  # Remove trailing slash if present
    model_name = os.path.basename(model_path)  # Use os.path.basename to get the model name

    # Check if output_path already exists
    output_path = os.path.join(args.output_dir, f"{model_name}.json")
    if os.path.exists(output_path):
        sys.exit(f"Output file already exists at {output_path}")

    generations = generate_answers(
        model_name=model_name,
        model_path=model_path,
        problems=problems,
        sanity_check=args.sanity_check,
    )

    # Save answers
    output['model_path'] = model_path
    output['model_name'] = model_name
    output['data'] = problems.copy()
    for i, generation in enumerate(generations):
        output['data'][i]["generation"] = generation

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4)
        print(f"Answers saved at {output_path}")
    else:
        print(generations)

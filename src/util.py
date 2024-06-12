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
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from datasets import Dataset, load_dataset, load_from_disk
from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedModel,
                          PreTrainedTokenizerBase, StoppingCriteria)

from .constant import PROMPT_INPUT


###########
# DATASET #
###########
def get_pku_pair_by_helpfulness(
    split: str, sanity_check: bool = False, cache_dir: Optional[str] = None
) -> Dataset:
    # Check if dataset already exists in cache
    try:
        dataset = load_from_disk(f"{cache_dir}/{split}")
        print(f"Loaded {split} dataset from cache.")
    except FileNotFoundError:
        print(f"No cached {split} dataset found, loading and processing...")
        dataset = load_dataset(
            "PKU-Alignment/PKU-SafeRLHF-30K", split=split, cache_dir=cache_dir
        )

        def get_chosen(sample) -> Dict[str, str]:
            return {
                "prompt": PROMPT_INPUT.format(input=sample["prompt"]),
                "chosen": sample["response_0"]
                if sample["better_response_id"] == 0
                else sample["response_1"],
                "rejected": sample["response_1"]
                if sample["better_response_id"] == 0
                else sample["response_0"],
            }

        dataset = dataset.map(get_chosen)
        # Save the processed dataset to disk
        dataset.save_to_disk(f"{cache_dir}/{split}")

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 100)))

    return dataset


def get_pku_by_helpfulness(
    split: str, sanity_check: bool = False, cache_dir: Optional[str] = None
) -> Dataset:
    # Check if dataset already exists in cache
    try:
        dataset = load_from_disk(f"{cache_dir}/{split}")
        print(f"Loaded {split} dataset from cache.")
    except FileNotFoundError:
        print(f"No cached {split} dataset found, loading and processing...")
        dataset = load_dataset(
            "PKU-Alignment/PKU-SafeRLHF-30K", split=split, cache_dir=cache_dir
        )

        def get_samples(samples) -> Dict[str, List[Union[str, int]]]:
            prompts = []
            completions = []
            labels = []
            for (
                prompt,
                response_0,
                response_1,
                better_response_id
            ) in zip(
                samples["prompt"],
                samples["response_0"],
                samples["response_1"],
                samples["better_response_id"]
            ):
                prompts += [PROMPT_INPUT.format(input=prompt), PROMPT_INPUT.format(input=prompt)]
                # if torch.rand(1)[0] <= 0.5:
                #     # Add response with label=1
                #     if better_response_id == 0:
                #         completions.append(response_0)
                #         labels.append(True)
                #     else:
                #         completions.append(response_1)
                #         labels.append(True)
                # else:
                #     # Add response with label=0
                #     if better_response_id == 0:
                #         completions.append(response_1)
                #         labels.append(False)
                #     else:
                #         completions.append(response_0)
                #         labels.append(False)

                # Add response with label=1
                if better_response_id == 0:
                    completions.append(response_0)
                    labels.append(True)
                else:
                    completions.append(response_1)
                    labels.append(True)

                # Add response with label=0
                if better_response_id == 0:
                    completions.append(response_1)
                    labels.append(False)
                else:
                    completions.append(response_0)
                    labels.append(False)

            return {"prompt": prompts, "completion": completions, "label": labels}

        dataset = dataset.map(
            get_samples, batched=True, remove_columns=dataset.column_names
        )
        # Save the processed dataset to disk
        dataset.save_to_disk(f"{cache_dir}/{split}")

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 100)))

    return dataset


def get_pku_pair_by_safety(
    split: str, sanity_check: bool = False, cache_dir: Optional[str] = None
) -> Dataset:
    # Check if dataset already exists in cache
    try:
        dataset = load_from_disk(f"{cache_dir}/{split}")
        print(f"Loaded {split} dataset from cache.")
    except FileNotFoundError:
        print(f"No cached {split} dataset found, loading and processing...")
        dataset = load_dataset(
            "PKU-Alignment/PKU-SafeRLHF-30K", split=split, cache_dir=cache_dir
        )

        def get_chosen(sample) -> Dict[str, str]:
            return {
                "prompt": PROMPT_INPUT.format(input=sample["prompt"]),
                "chosen": sample["response_0"]
                if sample["safer_response_id"] == 0
                else sample["response_1"],
                "rejected": sample["response_1"]
                if sample["safer_response_id"] == 0
                else sample["response_0"],
            }

        dataset = dataset.map(get_chosen)
        # Save the processed dataset to disk
        dataset.save_to_disk(f"{cache_dir}/{split}")

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 100)))

    return dataset


def get_pku_by_safety(
    split: str, sanity_check: bool = False, cache_dir: Optional[str] = None
) -> Dataset:
    # Check if dataset already exists in cache
    try:
        dataset = load_from_disk(f"{cache_dir}/{split}")
        print(f"Loaded {split} dataset from cache.")
    except FileNotFoundError:
        print(f"No cached {split} dataset found, loading and processing...")
        dataset = load_dataset(
            "PKU-Alignment/PKU-SafeRLHF-30K", split=split, cache_dir=cache_dir
        )

        def get_samples(samples) -> Dict[str, List[Union[str, int]]]:
            prompts = []
            completions = []
            labels = []
            for (
                prompt,
                response_0,
                response_1,
                is_response_0_safe,
                is_response_1_safe,
            ) in zip(
                samples["prompt"],
                samples["response_0"],
                samples["response_1"],
                samples["is_response_0_safe"],
                samples["is_response_1_safe"],
            ):
                # prompts += [PROMPT_INPUT.format(input=prompt)]
                # if torch.rand(1)[0] <= 0.5:
                #     completions.append(response_0)
                #     labels.append(is_response_0_safe)
                # else:
                #     completions.append(response_1)
                #     labels.append(is_response_1_safe)
                prompts += [PROMPT_INPUT.format(input=prompt), PROMPT_INPUT.format(input=prompt)]
                completions += [response_0, response_1]
                labels += [is_response_0_safe, is_response_1_safe]

            return {"prompt": prompts, "completion": completions, "label": labels}

        dataset = dataset.map(
            get_samples, batched=True, remove_columns=dataset.column_names
        )
        # Save the processed dataset to disk
        dataset.save_to_disk(f"{cache_dir}/{split}")

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 100)))

    return dataset


def count_samples(dataset, column="label", label=1):
    num_samples = sum(sample[column] == label for sample in dataset)
    return num_samples


def prepare_all_datasets():
    get_pku_pair_by_helpfulness("train", cache_dir='data_cache/pku_helpfulness_pair')
    get_pku_pair_by_helpfulness("test", cache_dir='data_cache/pku_helpfulness_pair')
    get_pku_by_helpfulness("train", cache_dir='data_cache/pku_helpfulness')
    get_pku_by_helpfulness("test", cache_dir='data_cache/pku_helpfulness')
    get_pku_pair_by_safety("train", cache_dir='data_cache/pku_safety_pair')
    get_pku_pair_by_safety("test", cache_dir='data_cache/pku_safety_pair')
    get_pku_by_safety("train", cache_dir='data_cache/pku_safety')
    get_pku_by_safety("test", cache_dir='data_cache/pku_safety')


###############
# MODEL UTILS #
###############
# Reference
# https://github.com/PKU-Alignment/safe-rlhf/blob/main/safe_rlhf/models/pretrained.py
def load_pretrained_models(
    model_name_or_path: Union[str, os.PathLike],
    model_max_length: int = 512,
    padding_side: str = "right",  # You might want to check that this is 'left' or 'right' at runtime
    auto_device_mapping: bool = False,
    device_map=None,
    dtype: Optional[Union[torch.dtype, str]] = "auto",
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    trust_remote_code: bool = False,
    auto_model_type: Type[
        AutoModelForCausalLM
    ] = AutoModelForCausalLM,  # You might want to check the type at runtime
    auto_model_args: Tuple[Any, ...] = (),
    auto_model_kwargs: Optional[Dict[str, Any]] = None,
    auto_tokenizer_args: Tuple[Any, ...] = (),
    auto_tokenizer_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load pre-trained model and tokenizer from a given path.

    Args:
        model_name_or_path (str or os.PathLike): Path to the model or its name.
        model_max_length (int, optional): The maximum sequence length of the model. Defaults to 512.
        padding_side (str, optional): The side to pad by the tokenizer. Defaults to 'right'.
        auto_device_mapping (bool, optional): Whether to automatically map the model to the multiple
            devices. Defaults to False.
        dtype (torch.dtype or str or None, optional): The parameter dtype while loading the model.
            Defaults to 'auto'.
        cache_dir (str or os.PathLike or None, optional): The directory to cache the model. Defaults
            to None.
        trust_remote_code (bool, optional): Whether to trust the remote code. Defaults to False.
        auto_model_type (type[AutoModelForCausalLM] or type[AutoModelForScore], optional): The type
            of the model to load. Defaults to AutoModelForCausalLM.
    """
    model_name_or_path = os.path.expanduser(model_name_or_path)
    cache_dir = os.path.expanduser(cache_dir) if cache_dir is not None else None
    if device_map is None:
        device_map = "auto" if auto_device_mapping else None
    if auto_model_kwargs is None:
        auto_model_kwargs = {}
    if auto_tokenizer_kwargs is None:
        auto_tokenizer_kwargs = {}

    model = auto_model_type.from_pretrained(
        model_name_or_path,
        *auto_model_args,
        cache_dir=cache_dir,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        **auto_model_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        *auto_tokenizer_args,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side=padding_side,
        trust_remote_code=trust_remote_code,
        **auto_tokenizer_kwargs,
    )
    # resize_tokenizer_embedding(tokenizer=tokenizer, model=model)
    return model, tokenizer


def model_contains_nan_params(model, device):
    """Check if there is any params in model.
    This is a workaround for the NaN params issue we observed
    in gpu != 0 when training the model using accelerate.
    Specifically, we observed one param in layer0.layernorm is NaN,
    which can break our training procedure.
    """
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            return True
    return False


def replace_nan_params_w_zero(model, device):
    """Replace NaN params in model by zeros.
    This is a workaround for the NaN params issue we observed
    in gpu != 0 when training the model using accelerate.
    Specifically, we observed one param in layer0.layernorm is NaN,
    which can break our training procedure.
    """
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_mask = torch.isnan(param)
            num_nan = torch.sum(nan_mask)
            print(f"[device={device}] Replace {num_nan} nan params in {name} by zeros.")
            param.data = torch.where(torch.isnan(param), torch.zeros_like(param), param)


#########
# OTHER #
#########
class KeywordStoppingCriteria(StoppingCriteria):
    def __init__(self, keyword_ids_list: list):
        self.keyword_ids_list = keyword_ids_list

    def __call__(
        self, input_ids: torch.LongTensor, scores=torch.FloatTensor, **kwargs
    ) -> bool:
        for keyword_ids in self.keyword_ids_list:
            if (
                len(input_ids[0]) >= len(keyword_ids)
                and input_ids[0][-len(keyword_ids):].tolist() == keyword_ids
            ):
                return True
        return False


def remove_keywords(answer, stop_words_list):
    for stop_words in stop_words_list:
        answer = answer.split(stop_words)[0]
    return answer


if __name__ == '__main__':
    globals()[sys.argv[1]]()

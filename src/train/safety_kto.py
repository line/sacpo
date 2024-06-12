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
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, set_seed)
from trl import KTOConfig, KTOTrainer

from ..util import (count_samples, get_pku_by_safety,
                    model_contains_nan_params, replace_nan_params_w_zero)

print(f"Number of GPU available: {torch.cuda.device_count()}")

# Define and parse arguments.


@dataclass
class ScriptArguments:
    """
    The arguments for the KTO training script.
    """
    # experiment name
    experiment_name: Optional[str] = field(default="20240309-alpaca-pku-helpfulness-dpo", metadata={"help": "experiment name for mlflow"})
    run_name: Optional[str] = field(default="kto", metadata={"help": "the location of the SFT model name or path"})

    # data parameters
    num_train_epochs: Optional[int] = field(default=2, metadata={"help": "Number of training epoch"})
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for KTO loss"})
    max_prompt_length: Optional[int] = field(default=128, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=512, metadata={"help": "the maximum sequence length"})

    # model
    model_name_or_path: Optional[str] = field(
        default="PKU-Alignment/alpaca-7b-reproduced",
        metadata={"help": "the location of the SFT model name or path"},
    )
    model_dtype: Optional[str] = field(
        default="float", metadata={"help": "model_dtype[float16, bfloat16, float] for loading."}
    )

    # training parameters
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed that will be set at the beginning of training."}
    )
    learning_rate: Optional[float] = field(default=1e-6, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=0, metadata={"help": "the number of warmup steps"})
    warmup_ratio: Optional[float] = field(default=0.03, metadata={"help": "the number of warmup steps"})
    optimizer_type: Optional[str] = field(default="adamw_torch", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=16, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    gradient_checkpointing_use_reentrant: Optional[bool] = field(
        default=False, metadata={"help": "whether to use reentrant for gradient checkpointing"}
    )

    # Logging and saving
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_strategy: Optional[str] = field(default="epoch", metadata={"help": "the saving frequency"})
    save_steps: Optional[float] = field(default=1000, metadata={"help": "the saving frequency"})
    save_total_limit: Optional[int] = field(default=1, metadata={"help": "the saving limit"})
    eval_strategy: Optional[str] = field(default="epoch", metadata={"help": "the eval frequency"})
    eval_steps: Optional[int] = field(default=1000, metadata={"help": "the evaluation frequency"})
    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    report_to: Optional[str] = field(
        default="mlflow",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )

    # instrumentation
    nan_params_check: Optional[bool] = field(default=True, metadata={"help": "whether to check for nan params"})
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 100 samples"})
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    set_seed(script_args.seed)

    # 1. set mlflow experiment name
    if script_args.report_to == "mlflow" and "MLFLOW_TRACKING_URI" in os.environ and os.environ["MLFLOW_TRACKING_URI"]:
        os.environ['MLFLOW_EXPERIMENT_NAME'] = script_args.experiment_name
    else:
        script_args.report_to = None

    # 2. load the dataset
    train_dataset = get_pku_by_safety("train", sanity_check=script_args.sanity_check, cache_dir='data_cache/pku_safety')
    eval_dataset = get_pku_by_safety("test", sanity_check=script_args.sanity_check, cache_dir='data_cache/pku_safety')

    # 3. load a pretrained model
    torch_dtype = torch.float
    if script_args.model_dtype == "float16":
        torch_dtype = torch.float16
    elif script_args.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        device_map={"": Accelerator().local_process_index},
        # device_map="auto",
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # 4. we check if any nan in weights. If there any, replace by zeros.
    if script_args.nan_params_check:
        device = torch.cuda.current_device()
        if model_contains_nan_params(model, device):
            print("Warning: Model contains NaN. We replace them by zeros.")
            replace_nan_params_w_zero(model, device)

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # We also calculate the label weights based on #data
    n_desirable = count_samples(train_dataset, label=1)
    n_undesirable = len(train_dataset) - n_desirable
    desirable_weight = 1.
    undesirable_weight = float(n_desirable) / float(n_undesirable)
    print(f"Data description: {n_desirable} desirable samples, {n_undesirable} undesirable samples.")

    # 5. initialize training arguments:
    training_args = KTOConfig(
        run_name=script_args.run_name,
        num_train_epochs=script_args.num_train_epochs,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        logging_steps=script_args.logging_steps,
        save_strategy=script_args.save_strategy,
        save_steps=script_args.save_steps,
        save_total_limit=script_args.save_total_limit,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy=script_args.eval_strategy,
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        warmup_ratio=script_args.warmup_ratio,
        optim=script_args.optimizer_type,
        bf16=True,
        tf32=True,
        remove_unused_columns=False,
        gradient_checkpointing_kwargs=dict(use_reentrant=script_args.gradient_checkpointing_use_reentrant),
        seed=script_args.seed,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        desirable_weight=desirable_weight,
        undesirable_weight=undesirable_weight,
        beta=script_args.beta
    )

    peft_config = None

    # 6. init the trainer
    kto_trainer = KTOTrainer(
        model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # 7. train
    kto_trainer.train()

    # 8. save
    if kto_trainer.is_fsdp_enabled:
        kto_trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    kto_trainer.save_model(script_args.output_dir)

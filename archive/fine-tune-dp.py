# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Train GPT2 model series with DP (w/ optional parameter-efficient approach LoRA)'''

import datasets
import dp_transformers
import transformers
import sys
import logging

from dataclasses import dataclass, field, asdict
from peft import get_peft_model, LoraConfig

from dp_transformers.grad_sample.transformers import conv_1d


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name: str = field(default="gpt2", metadata={
        "help": "Model name in HuggingFace, e.g. 'gpt2'"
    })
    sequence_len: int = field(default=128, metadata={
        "help": "Maximum sequence length"
    })


@dataclass
class LoraArguments:
    enable_lora: bool = field(default=False, metadata={
        "help": "Whether to enable LoRA"
    })
    lora_dim: int = field(default=8, metadata={
        "help": "LoRA dimension"
    })
    lora_alpha: int = field(default=8, metadata={
        "help": "LoRA alpha"
    })
    lora_dropout: float = field(default=0.0, metadata={
        "help": "LoRA dropout"
    })

    def as_peft_config(self) -> LoraConfig:
        if not self.enable_lora:
            raise ValueError("LoRA is not enabled, cannot convert to LoRA config")
        params = asdict(self)
        params.pop("enable_lora")
        params["r"] = params.pop("lora_dim")
        return LoraConfig(**params)


@dataclass
class Arguments:
    train: dp_transformers.TrainingArguments
    privacy: dp_transformers.PrivacyArguments
    model: ModelArguments
    lora: LoraConfig


def main(args: Arguments):
    transformers.set_seed(args.train.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = train_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {train_args.local_rank}, device: {train_args.device}, n_gpu: {train_args.n_gpu}, "
        f"distributed training: {bool(train_args.local_rank != -1)}, 16-bits training: {train_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {train_args}")
    logger.info(f"Privacy parameters {privacy_args}")

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model.model_name)
    model = model.to(train_args.device)

    # Load data
    dataset = datasets.load_dataset('data/train.jsonl', split="train[:500000]").train_test_split(0.02, seed=args.train.seed)

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize data
    with train_args.main_process_first(desc="tokenizing dataset"):
        dataset = dataset.map(
            lambda batch: tokenizer(batch['content'], padding="max_length", truncation=True, max_length=args.model.sequence_len),
            batched=True, num_proc=8, desc="tokenizing dataset", remove_columns=dataset.column_names['train']
        )

    if args.lora.enable_lora:
        logger.info("Using LoRA")
        model = get_peft_model(model=model, peft_config=args.lora.as_peft_config())
    else:
        logger.info("Not using LoRA")

    if train_args.local_rank == 0:
        logger.info(f"Total number of parameters of the model: {model.num_parameters(only_trainable=False)}")
        logger.info(f"Fine-tuned number of parameters of the model: {model.num_parameters(only_trainable=True)}")

    model = model.cuda()
    model.train()

    data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(tokenizer)

    trainer = dp_transformers.dp_utils.OpacusDPTrainer(
        args=train_args,
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=data_collator,
        privacy_args=privacy_args,
    )

    try:
        trainer.train()
    finally:
        eps_prv = trainer.get_prv_epsilon()
        eps_rdp = trainer.get_rdp_epsilon()
        trainer.log({
            "final_epsilon_prv": eps_prv,
            "final_epsilon_rdp": eps_rdp
        })

if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser((dp_transformers.TrainingArguments, dp_transformers.PrivacyArguments, ModelArguments, LoraArguments))
    train_args, privacy_args, model_args, lora_args = arg_parser.parse_args_into_dataclasses()
    main(Arguments(train=train_args, privacy=privacy_args, model=model_args, lora=lora_args))

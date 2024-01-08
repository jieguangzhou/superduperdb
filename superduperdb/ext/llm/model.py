import os
import typing
import typing as t
from dataclasses import dataclass, field

import bitsandbytes as bnb
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    deepspeed,
)

from superduperdb.backends.query_dataset import query_dataset_factory
from superduperdb.components.model import Model, _TrainingConfiguration
from superduperdb.ext.utils import ensure_initialized

if typing.TYPE_CHECKING:
    from superduperdb.backends.base.query import Select
    from superduperdb.base.datalayer import Datalayer
    from superduperdb.components.dataset import Dataset
    from superduperdb.components.metric import Metric


from transformers import modeling_utils


@dataclass
class ModelArguments:
    model_name_or_path: t.Optional[str] = field(
        default="mistralai/mistral-7b-instruct-v0.2"
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


@dataclass
class DataArguments:
    data_name: str = field(
        default="c-s-ale/alpaca-gpt4-data-zh",
        metadata={"help": "dataset name"},
    )


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: t.Optional[t.List[str]] = None
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    bits: t.Optional[int] = None


def LLMTrainingConfiguration(identifier: str, *args, **kwargs):
    return _TrainingConfiguration(identifier=identifier, kwargs=kwargs)


@dataclass
class LLM(Model):
    object: t.Optional[transformers.Trainer] = None
    model_name_or_path: str = "facebook/opt-125m"
    bits: t.Optional[int] = None
    qlora: bool = True
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    output_dir: str = "output"
    batch_size: int = 16
    model_args: ModelArguments = field(default_factory=ModelArguments)
    data_args: DataArguments = field(default_factory=DataArguments)
    training_args: TrainingArguments = field(
        default_factory=lambda: TrainingArguments(output_dir="")
    )
    lora_args: LoraArguments = field(default_factory=LoraArguments)

    # def __post_init__(self, **kwargs):
    #     self.model_args.model_name_or_path = self.model_name_or_path
    #     self.lora_args.bits = self.bits
    #     self.lora_args.q_lora = self.qlora
    #
    #     self.training_args.per_device_train_batch_size = (
    #         self.per_device_train_batch_size
    #     )
    #     self.training_args.per_device_eval_batch_size = self.per_device_eval_batch_size
    #     self.training_args.gradient_accumulation_steps = (
    #         self.gradient_accumulation_steps
    #     )
    #     self.training_args.output_dir = self.output_dir
    #     super().__post_init__()

    def init(self):
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.ddp = world_size != 1
        # device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if self.ddp else None
        device_map = "mps"

        is_deepspeed_zero3_enabled = deepspeed.is_deepspeed_zero3_enabled()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            quantization_config=self._create_quantization_config(),
            low_cpu_mem_usage=not is_deepspeed_zero3_enabled,
            device_map=device_map if not is_deepspeed_zero3_enabled else None,
            trust_remote_code=self.model_args.trust_remote_code,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            use_fast=False,
            model_max_length=self.training_args.model_max_length,
            padding_side="left",
            trust_remote_code=self.model_args.trust_remote_code,
        )
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token

    @ensure_initialized
    def to_call(self, X: t.Any, **kwargs):
        outputs = self.model.generate(
            input_ids=self.tokenizer(X, return_tensors="pt")["input_ids"],
            **kwargs,
        )
        text = self.tokenizer.batch_decode(outputs)[0]
        return text

    @ensure_initialized
    def _fit(
        self,
        X: t.Any,
        y: t.Optional[t.Any] = None,
        configuration: t.Optional[LLMTrainingConfiguration] = None,
        data_prefetch: bool = False,
        db: t.Optional["Datalayer"] = None,
        metrics: t.Optional[t.Sequence["Metric"]] = None,
        select: t.Optional["Select"] = None,
        validation_sets: t.Optional[t.Sequence[t.Union[str, "Dataset"]]] = None,
    ):
        train_dataset, eval_dataset = self.get_datasets(
            X, y, db, select, data_prefetch=data_prefetch
        )
        __import__('ipdb').set_trace()

        trainer = transformers.Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        self.model.config.use_cache = False

        trainer.train()
        trainer.save_state()

    def _create_quantization_config(self):
        compute_dtype = (
            torch.float16
            if self.training_args.fp16
            else (torch.bfloat16 if self.training_args.bf16 else torch.float32)
        )
        if self.bits is not None:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.bits == 4,
                load_in_8bit=self.bits == 8,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
        else:
            quantization_config = None
        return quantization_config

    def _prepare_lora_training(self):
        lora_config = LoraConfig(
            r=self.lora_args.lora_r,
            lora_alpha=self.lora_args.lora_alpha,
            target_modules=self._get_lora_target_modules(),
            lora_dropout=self.lora_args.lora_dropout,
            bias=self.lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )

        if self.lora_args.bits:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.training_args.gradient_checkpointing,
            )

            if not self.ddp and torch.cuda.device_count() > 1:
                self.model.is_parallelizable = True
                self.model.model_parallel = True

        self.model = get_peft_model(self.model, lora_config)

        if (
            self.training_args.deepspeed is not None
            and self.training_args.local_rank == 0
        ):
            self.model.print_trainable_parameters()

        if self.training_args.gradient_checkpointing:
            self.model.enable_input_require_grads()

        if self.training_args.local_rank == 0:
            self.model.print_trainable_parameters()

    def _get_lora_target_modules(self):
        if self.lora_args.lora_target_modules is not None:
            return self.lora_args.lora_target_modules

        cls = (
            bnb.nn.Linear4bit
            if self.lora_args.bits == 4
            else (bnb.nn.Linear8bitLt if self.lora_args.bits == 8 else torch.nn.Linear)
        )
        lora_module_names = set()
        for name, module in self.model.named_modules():
            if isinstance(module, cls):
                names = name.split(".")
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if "lm_head" in lora_module_names:  # needed for 16-bit
            lora_module_names.remove("lm_head")
        return list(lora_module_names)

    def get_datasets(
        self, X, y, db: "Datalayer", select: "Select", data_prefetch: bool = False
    ):
        def transform(example):
            if example.get("input"):
                example[X] = example[X] + example["input"]
            return _default_transform(X, y, example, self.tokenizer)

        train_dataset = query_dataset_factory(
            data_prefetch=data_prefetch,
            select=select,
            fold="train",
            db=db,
            transform=transform,
        )
        eval_dataset = query_dataset_factory(
            data_prefetch=data_prefetch,
            select=select,
            fold="vaild",
            db=db,
            transform=transform,
        )
        return train_dataset, eval_dataset


_prompt_template = (
    "Below is an instruction that describes a task,"
    "paired with an input that provides further context. "
    "Write a response that appropriately completes the request."
    "\n\n### Instruction:\n{x}\n\n### Response:\n{y}"
)


def _default_transform(X, y, example: dict, tokenizer: PreTrainedTokenizer, **kwargs):
    x = example[X]
    y = example.get(y, "")

    prompt = _prompt_template.format(x=x, y=y)

    prompt = prompt + tokenizer.eos_token
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=tokenizer.model_max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

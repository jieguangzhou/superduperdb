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
    AwqConfig,
    BitsAndBytesConfig,
    GPTQConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    deepspeed,
)
from transformers.utils.quantization_config import QuantizationConfigMixin

from superduperdb.backends.query_dataset import query_dataset_factory
from superduperdb.components.model import (
    Model,
    _TrainingConfiguration,
    TrainingConfiguration,
)
from superduperdb.ext.utils import ensure_initialized

if typing.TYPE_CHECKING:
    from superduperdb.backends.base.query import Select
    from superduperdb.base.datalayer import Datalayer
    from superduperdb.components.dataset import Dataset
    from superduperdb.components.metric import Metric


QuantizationConfigType = t.Union[BitsAndBytesConfig, GPTQConfig, AwqConfig]


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
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
    trust_remote_code: bool = False
    bits: t.Optional[int] = None
    quantization_config: t.Optional[QuantizationConfigType] = None
    transform: t.Optional[t.Callable] = None

    def init(self):
        self.model, self.tokenizer = self.init_model_and_tokenizer()

    @ensure_initialized
    def to_call(self, X: t.Any, **kwargs):
        outputs = self.model.generate(
            input_ids=self.tokenizer(X, return_tensors="pt")["input_ids"].to(
                self.model.device
            ),
            **kwargs,
        )
        text = self.tokenizer.batch_decode(outputs)[0]
        return text

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
        **kwargs,
    ):
        training_args = TrainingArguments(output_dir="outputs", **kwargs)
        lora_args = LoraArguments()
        ## merge configuration to training_args and lora_args
        if configuration is not None:
            for key, value in configuration.kwargs.items():
                if hasattr(training_args, key):
                    setattr(training_args, key, value)

                if hasattr(lora_args, key):
                    setattr(lora_args, key, value)

        self.model, self.tokenizer = self.init_model_and_tokenizer()
        self._prepare_lora_training(training_args, lora_args)

        train_dataset, eval_dataset = self.get_datasets(
            X, y, db, select, data_prefetch=data_prefetch, eval=True
        )

        # TODO: Defind callbacks about superduperdb side

        trainer = self.create_trainer(
            train_dataset,
            eval_dataset,
            compute_metrics=metrics,
            training_args=training_args,
            **kwargs,
        )
        trainer.model.config.use_cache = False
        trainer.train()
        trainer.save_state()

    def init_model_and_tokenizer(self):
        device_map = "auto"
        is_deepspeed_zero3_enabled = deepspeed.is_deepspeed_zero3_enabled()
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            quantization_config=self._create_quantization_config(),
            low_cpu_mem_usage=not is_deepspeed_zero3_enabled,
            device_map=device_map if not is_deepspeed_zero3_enabled else None,
            trust_remote_code=self.trust_remote_code,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            use_fast=False,
            padding_side="left",
            model_max_length=512,
            trust_remote_code=self.trust_remote_code,
        )
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        return model, tokenizer

    def create_trainer(
        self, train_dataset, eval_dataset, training_args, **kwargs
    ) -> transformers.Trainer:

        trainer = transformers.Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs,
        )
        return trainer


    def _create_quantization_config(self):
        if self.quantization_config:
            return self.quantization_config
        compute_dtype = (
            torch.float16
            # if self.training_args.fp16
            # else (torch.bfloat16 if self.training_args.bf16 else torch.float32)
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

    def _prepare_lora_training(
        self, training_args: TrainingArguments, lora_args: LoraArguments
    ):
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=self._get_lora_target_modules(lora_args),
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )

        if lora_args.bits:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.training_args.gradient_checkpointing,
            )

            if not self.ddp and torch.cuda.device_count() > 1:
                self.model.is_parallelizable = True
                self.model.model_parallel = True

        self.model = get_peft_model(self.model, lora_config)

        if training_args.deepspeed is not None and training_args.local_rank == 0:
            self.model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            self.model.enable_input_require_grads()

        if training_args.local_rank == 0:
            self.model.print_trainable_parameters()

    def _get_lora_target_modules(self, lora_args):
        if lora_args.lora_target_modules is not None:
            return lora_args.lora_target_modules

        cls = (
            bnb.nn.Linear4bit
            if lora_args.bits == 4
            else (bnb.nn.Linear8bitLt if lora_args.bits == 8 else torch.nn.Linear)
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
        self,
        X,
        y,
        db: "Datalayer",
        select: "Select",
        data_prefetch: bool = False,
        eval: bool = False,
    ):
        keys = [X]
        if y is not None:
            keys.append(y)

        train_dataset = query_dataset_factory(
            keys=keys,
            data_prefetch=data_prefetch,
            select=select,
            fold="train",
            db=db,
            transform=self.preprocess,
        )
        if eval:
            eval_dataset = query_dataset_factory(
                keys=keys,
                data_prefetch=data_prefetch,
                select=select,
                fold="vaild",
                db=db,
                transform=self.preprocess,
            )
        else:
            eval_dataset = None
        
        def process_func(example):
            return self.tokenize(example, X, y)

        train_dataset = train_dataset.map(process_func)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(process_func)
        return train_dataset, eval_dataset

    def tokenize(self, example, X, y):
        prompt = example[X]

        prompt = prompt + self.tokenizer.eos_token
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    @property
    def ddp(self):
        return int(os.environ.get("WORLD_SIZE", 1)) != 1

import pytest
from datasets import load_dataset

from superduperdb import superduper
from superduperdb.base.document import Document
from superduperdb.backends.mongodb import Collection
from superduperdb.base.artifact import Artifact
from superduperdb.ext.llm import LLM
from superduperdb.ext.llm.model import LLMTrainingConfiguration
import transformers
import os

model = "facebook/opt-350m"
dataset_name = "timdettmers/openassistant-guanaco"
prompt = "### Human: Who are you? ### Assistant: "

save_folder = "output"


@pytest.fixture
def db():
    db_ = superduper("mongomock://localhost:30000/test_llm")
    dataset = load_dataset(dataset_name)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    train_documents = [
        Document({"text": example["text"], "_fold": "train"})
        for example in train_dataset
    ][:200]
    eval_documents = [
        Document({"text": example["text"], "_fold": "valid"}) for example in eval_dataset
    ][:10]

    db_.execute(Collection("datas").insert_many(train_documents))
    db_.execute(Collection("datas").insert_many(eval_documents))

    yield db_

    db_.drop(force=True)



@pytest.fixture
def base_config():
    return LLMTrainingConfiguration(
        identifier="llm-finetune-training-config",
        overwrite_output_dir=True,
        num_train_epochs=2,
        max_steps=1,
        save_total_limit=5,
        logging_steps=10,
        evaluation_strategy="steps",
        fp16=True,
        eval_steps=1,
        save_steps=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        log_to_db=True,
        max_length=512,
        use_lora=True,
    )


def test_full_finetune(db, base_config):
    llm = LLM(
        identifier="llm-finetune",
        model_name_or_path=model,
    )

    base_config.kwargs["use_lora"] = False
    # Don't log to db if full finetune cause the large files
    base_config.kwargs["log_to_db"] = False
    output_dir = os.path.join(save_folder, "test_full_finetune")
    base_config.kwargs["output_dir"] = output_dir

    llm.fit(
        X="text",
        select=Collection("datas").find(),
        configuration=base_config,
        db=db,
    )

    llm_inference = LLM(
        identifier="llm",
        model_name_or_path=transformers.trainer.get_last_checkpoint(output_dir),
        model_kwargs=dict(device_map="auto"),
    )
    db.add(llm_inference)
    result = db.predict('llm', prompt, max_new_tokens=100, do_sample=False)[0].content
    print(result)


def test_lora_finetune(db, base_config):
    llm = LLM(
        identifier="llm-finetune",
        model_name_or_path=model,
    )

    output_dir = os.path.join(save_folder, "test_lora_finetune")
    base_config.kwargs["output_dir"] = output_dir

    llm.fit(
        X="text",
        select=Collection("datas").find(),
        configuration=base_config,
        db=db,
    )

    assert isinstance(llm.adapter_id, Artifact)
    assert os.path.exists(llm.adapter_id.artifact)

    result = db.predict('llm-finetune', prompt, max_new_tokens=100, do_sample=False)[0].content
    print(result)
    assert len(result) > 0


def test_qlora_finetune(db, base_config):
    llm = LLM(
        identifier="llm-finetune",
        model_name_or_path=model,
    )

    base_config.kwargs["bits"] = 4
    output_dir = os.path.join(save_folder, "test_qlora_finetune")
    base_config.kwargs["output_dir"] = output_dir

    llm.fit(
        X="text",
        select=Collection("datas").find(),
        configuration=base_config,
        db=db,
    )

    assert isinstance(llm.adapter_id, Artifact)
    assert os.path.exists(llm.adapter_id.artifact)

    result = db.predict('llm-finetune', prompt, max_new_tokens=100, do_sample=False)[0].content
    print(result)
    assert len(result) > 0


def test_ray_lora_finetune(db, base_config):
    llm = LLM(
        identifier="llm-finetune",
        model_name_or_path=model,
    )

    base_config.kwargs["use_lora"] = True
    base_config.kwargs["log_to_db"] = False
    output_dir = os.path.join(save_folder, "test_ray_lora_finetune")
    base_config.kwargs["output_dir"] = output_dir

    from ray.train import RunConfig, ScalingConfig

    scaling_config = ScalingConfig(
        num_workers=1,
        use_gpu=True,
    )

    run_config = RunConfig(
        storage_path=os.path.abspath(output_dir),
    )

    ray_configs = {
        "scaling_config": scaling_config,
        "run_config": run_config,
    }

    llm.fit(
        X="text",
        select=Collection("datas").find(),
        configuration=base_config,
        db=db,
        on_ray=True,
        ray_configs=ray_configs,
    )

    assert isinstance(llm.adapter_id, Artifact)
    assert os.path.exists(llm.adapter_id.artifact)

    result = db.predict('llm-finetune', prompt, max_new_tokens=100, do_sample=False)[0].content
    print(result)
    assert len(result) > 0

def test_remote_ray_lora_finetune(db, base_config):
    llm = LLM(
        identifier="llm-finetune",
        model_name_or_path=model,
    )

    base_config.kwargs["use_lora"] = True
    base_config.kwargs["log_to_db"] = False
    # Use absolute path, because the ray will run in remote
    output_dir = "test_ray_lora_finetune"
    base_config.kwargs["output_dir"] = output_dir

    from ray.train import RunConfig, ScalingConfig

    scaling_config = ScalingConfig(
        num_workers=1,
        use_gpu=True,
    )

    run_config = RunConfig(
        storage_path=os.path.abspath(output_dir),
    )

    ray_configs = {
        "scaling_config": scaling_config,
        "run_config": run_config,
    }

    llm.fit(
        X="text",
        select=Collection("datas").find(),
        configuration=base_config,
        db=db,
        on_ray=True,
        ray_address="ray://localhost:10001",
        ray_configs=ray_configs,
    )

    assert isinstance(llm.adapter_id, Artifact)
    assert os.path.exists(llm.adapter_id.artifact)

    result = db.predict('llm-finetune', prompt, max_new_tokens=100, do_sample=False)[0].content
    print(result)
    assert len(result) > 0

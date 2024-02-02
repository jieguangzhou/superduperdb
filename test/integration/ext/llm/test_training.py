import pytest
from datasets import load_dataset

from superduperdb import superduper
from superduperdb.base.document import Document
from superduperdb.backends.mongodb import Collection
from superduperdb.ext.llm import LLM
from superduperdb.ext.llm.model import LLMTrainingConfiguration

dataset_name = "timdettmers/openassistant-guanaco"
model = "facebook/opt-125m"


@pytest.fixture
def db():
    db_ = superduper("mongodb://localhost:27017/test_llm")
    dataset = load_dataset(dataset_name)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    train_documents = [
        Document({"text": example["text"], "_fold": "train"})
        for example in train_dataset
    ]
    eval_documents = [
        Document({"text": example["text"], "_fold": "eval"}) for example in eval_dataset
    ]

    db_.execute(Collection("datas").insert_many(train_documents))
    db_.execute(Collection("datas").insert_many(eval_documents))

    yield db_

    db_.drop(force=True)


def test_basic_training(db):
    llm = LLM(
        identifier="llm-finetune",
        model_name_or_path=model,
    )

    llm_training_configuration = LLMTrainingConfiguration(
        identifier="llm-finetune-training-config",
        output_dir="output/test_basic_training",
        use_lora=True,
        overwrite_output_dir=True,
        num_train_epochs=0.001,
        save_total_limit=3,
        logging_steps=1,
        eval_steps=1,
        save_steps=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        log_to_db=True,
    )

    llm.fit(
        X="text",
        select=Collection("datas").find(),
        configuration=llm_training_configuration,
        db=db,
    )

    __import__("ipdb").set_trace()
    pass

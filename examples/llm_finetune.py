import os

import pandas as pd
import torch
from datasets import Dataset, load_dataset

from superduperdb import superduper
from superduperdb.backends.mongodb import Collection
from superduperdb.base.document import Document
from superduperdb.ext.llm import LLM, LLMTrainingConfiguration

collection_name = "poem"


def prepare_datas(db):
    train_df = pd.read_csv("train.csv")
    train_dataset = Dataset.from_pandas(train_df)
    validation_df = pd.read_csv("val.csv")
    validation_dataset = Dataset.from_pandas(validation_df)
    db.execute(
        Collection(collection_name).insert_many(list(map(Document, train_dataset)))
    )
    db.execute(
        Collection(collection_name).insert_many(list(map(Document, validation_dataset)))
    )


def train(db, model_identifier, model_name, output_dir):
    # training
    llm = LLM(
        identifier=model_identifier,
        bits=4 if torch.cuda.is_available() else None,
        model_name_or_path=model_name,
        model_kwargs={"device_map": 'auto'},
    )
    training_configuration = LLMTrainingConfiguration(
        identifier="llm-finetune-training-config",
        output_dir=output_dir,
        overwrite_output_dir=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        num_train_epochs=3,
        fp16=torch.cuda.is_available(),  # mps don't support fp16
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=5,
        learning_rate=2e-5,
        weight_decay=0.05,
        lr_scheduler_type="cosine",
        logging_strategy="steps",
        logging_steps=10,
        gradient_checkpointing=True,
        report_to=[],
    )

    llm._fit(
        X="text",
        db=db,
        select=Collection(collection_name).find(),
        configuration=training_configuration,
    )


def inference(db, model_identifier, output_dir):
    # inference
    llm_base = db.load("model", model_identifier)
    checkpoints = [
        checkpoint
        for checkpoint in os.listdir(output_dir)
        if checkpoint.startswith("checkpoint")
    ]
    db.add(llm_base)
    for checkpoint in checkpoints:
        llm_checkpoint = LLM(
            identifier=checkpoint,
            bits=4 if torch.cuda.is_available() else None,
            adapter_id=os.path.join(output_dir, checkpoint),
            model_name_or_path=llm_base.model_name_or_path,
        )
        db.add(llm_checkpoint)

    datas = list(Collection(collection_name).find().execute(db))
    data = datas[3].content
    print(data["text"])

    prompt = prompt_template.format(x=data["instruction"], y="")
    print("-" * 20, "\n")
    print(prompt)
    print("-" * 20, "\n")

    print("Base model:\n")
    print(db.predict(llm_base.identifier, prompt, max_new_tokens=100, one=True))

    for checkpoint in checkpoints:
        print("-" * 20, "\n")
        print(f"Finetuned model-{checkpoint}:\n")
        print(db.predict(checkpoint, prompt, max_new_tokens=100, one=True))


if __name__ == "__main__":
    db = superduper("mongomock://llm-finetune")
    model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    output_dir = "outputs/llm-finetune/moxtral-8x7B-lora-new"
    # prepare_datas(db)
    train(db, "llm-finetune", model, output_dir)
    # inference(db, "llm-finetune", output_dir)

import os

os.makedirs('.superduperdb', exist_ok=True)
os.environ['SUPERDUPERDB_CONFIG'] = '.superduperdb/config.yaml'


from superduperdb import superduper

db = superduper('mongodb://localhost:27017/documents')


from datasets import load_dataset
from superduperdb.base.document import Document
dataset_name = "timdettmers/openassistant-guanaco"
dataset = load_dataset(dataset_name)

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

train_documents = [
    Document({**example, "_fold": "train"})
    for example in train_dataset
][:500]
eval_documents = [
    Document({**example, "_fold": "valid"})
    for example in eval_dataset
][:10]

datas = train_documents + eval_documents


# Function for transformation after extracting data from the database
transform = None
key = ('text')
training_kwargs=dict(dataset_text_field="text")


data = datas[0]
input_data, output_text = data["text"].rsplit("### Assistant: ", maxsplit=1)
input_text += "### Assistant: "
output_text = output_text.rsplit("### Human:")[0]
print("Input: --------------")
print(input_text)
print("Response: --------------")
print(output_text)


# If our data is in a format natively supported by MongoDB, we don't need to do anything.
from superduperdb.backends.mongodb import Collection

table_or_collection = Collection('documents')
select = table_or_collection.find({})


from superduperdb import Document

ids, _ = db.execute(table_or_collection.insert_many(datas))


model_name = "facebook/opt-125m"


from superduperdb.ext.transformers import LLM, LLMTrainer
trainer = LLMTrainer(
    identifier="llm-finetune-trainer",
    output_dir="output/finetune",
    overwrite_output_dir=True,
    max_steps=50,
    # num_train_epochs=3,
    save_total_limit=3,
    logging_steps=10,
    evaluation_strategy="steps",
    save_steps=10,
    eval_steps=10,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    max_seq_length=512,
    key=key,
    select=select,
    transform=transform,
    training_kwargs=training_kwargs,
)


trainer.use_lora = True


llm = LLM(
    identifier="llm",
    model_name_or_path=model_name,
    trainer=trainer,
    model_kwargs=model_kwargs,
    tokenizer_kwargs=tokenizer_kwargs,
)

db.apply(llm)

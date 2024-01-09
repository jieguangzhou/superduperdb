from datasets import load_dataset

from superduperdb import superduper
from superduperdb.backends.mongodb import Collection
from superduperdb.base.document import Document
from superduperdb.ext.llm import LLM, LLMTrainingConfiguration

_prompt_template = (
    "Below is an instruction that describes a task,"
    "paired with an input that provides further context. "
    "Write a response that appropriately completes the request."
    "\n\n### Instruction:\n{x}\n\n### Response:\n{y}"
)

datas = load_dataset("c-s-ale/alpaca-gpt4-data-zh")["train"].to_list()[:1000]

for data in datas:
    if data["input"] is not None:
        data["instruction"] =  data["instruction"] + "\n" + data["input"]
    data["text"] = _prompt_template.format(x=data["instruction"], y=data["input"])

db = superduper("mongomock://llm-finetune")

db.execute(Collection("doc").insert_many(list(map(Document, datas))))


llm = LLM(
    identifier="llm-finetune",
    model_name_or_path="facebook/opt-125m",
)

## inference
llm.predict("hello")


## training
training_configuration = LLMTrainingConfiguration(identifier="llm-finetune",
    num_train_epochs=1, per_device_train_batch_size=1, per_device_eval_batch_size=1
)

llm.fit(
    X="text",
    db=db,
    select=Collection("doc").find(),
    configuration=training_configuration,
)

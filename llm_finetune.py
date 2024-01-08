from datasets import load_dataset

from superduperdb import superduper
from superduperdb.backends.mongodb import Collection
from superduperdb.base.document import Document
from superduperdb.ext.llm import LLM, LLMTrainingConfiguration

datas = load_dataset("c-s-ale/alpaca-gpt4-data-zh")["train"].to_list()[:1000]


db = superduper("mongomock://llm-finetune")

db.execute(Collection("doc").insert_many(list(map(Document, datas))))


llm = LLM(
    identifier="llm-finetune",
    model_name_or_path="facebook/opt-125m",
    bits=None,
)


training_configuration = LLMTrainingConfiguration(num_train_epochs=1)

llm._fit("instruction", "output", db=db, select=Collection("doc").find(), num_epochs=1)

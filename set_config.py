import os

os.makedirs(".superduperdb", exist_ok=True)
os.environ["SUPERDUPERDB_CONFIG"] = ".superduperdb/config.yaml"

CFG = """
data_backend: mongodb://127.0.0.1:27017/documents
artifact_store: filesystem://./artifact_store
cluster:
  cdc:
    strategy: null
    uri: ray://127.0.0.1:20000
  compute:
    uri: ray://127.0.0.1:10001
  vector_search:
    backfill_batch_size: 100
    type: in_memory
    uri: http://127.0.0.1:21000
"""

with open(os.environ["SUPERDUPERDB_CONFIG"], "w") as f:
    f.write(CFG)

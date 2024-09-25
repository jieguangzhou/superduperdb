<!-- Auto-generated content start -->
# superduper_jina

Superduper allows users to work with Jina Embeddings models through the Jina Embedding API.

## Installation

```bash
pip install superduper_jina
```

## API

- [Code](https://github.com/superduper-io/superduper)
- [API-docs](/docs/api/plugins/superduper_jina)

| Class | Description |
|---|---|
| `superduper_jina.client.JinaAPIClient` | A client for the Jina Embedding platform. |
| `superduper_jina.model.Jina` | Cohere predictor. |
| `superduper_jina.model.JinaEmbedding` | Jina embedding predictor. |

## Examples

### JinaEmbedding

#### Base usage

```python
from superduper_jina.model import JinaEmbedding
model = JinaEmbedding(identifier='jina-embeddings-v2-base-en')
model.predict('Hello world')
```


<!-- Auto-generated content end -->

<!-- Add your additional content below -->

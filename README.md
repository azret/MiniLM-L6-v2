# MiniLM-L6-v2

## Overview

This repository provides a self-contained implementation of the [**all-MiniLM-L6-v2**](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), designed for generating high-quality sentence embeddings. The model is optimized for efficiency and performance, making it suitable for various natural language processing tasks such as semantic search, clustering, and classification.

## Dependencies

- `torch` Core dependency

- `transformers` For downloading weights, tokenization and tests. **Only the tokenizer is used**.

- `sentence-transformers` For tests. **Not used during inference**.

## Installation

Make sure you have Python 3.12 or higher installed. The model itself is lightweight and dependency-free and can be easily deployed on local machines or cloud environments.

1. **Install the requirements**:

   ```bash
   git clone
   pip install -r requirements.txt
   ```

2. **Get the weights**:

   ```bash
   # This will download the necessary weights from Hugging Face 🤗 and create a new MiniLM-L6-v2.ckpt.
   python MiniLM-L6-v2.py #

   > python = 3.12.3 (tags/v3.12.3:f6650f9, Apr  9 2024, 14:05:25) [MSC v.1938 64 bit (AMD64)]
   > numpy = 2.1.2
   > torch = 2.7.1+cu128
   > model = MiniLM-L6-v2
   > device = cuda

   > Downloading pre-trained 'sentence-transformers/all-MiniLM-L6-v2'...

   > Loading checkpoint 'D:\ai\MiniLM-L6-v2.ckpt'...

   BertModel(
     (wte): Parameter(30522, 384)
     (tte): Parameter(2, 384)
     (wpe): Parameter(512, 384)
     (norm): LayerNorm((384,), eps=1e-12, elementwise_affine=True)
     (dropout): Dropout(p=0.1, inplace=False)
     (encoder): ModuleDict(
       (layer): ModuleList(
         (0-5): 6 x BertLayer(
           (attention): BertAttention(
             (attention): MultiHeadSelfAttention(
               (wq): Linear(in_features=384, out_features=384, bias=True)
               (wk): Linear(in_features=384, out_features=384, bias=True)
               (wv): Linear(in_features=384, out_features=384, bias=True)
               (wo): Linear(in_features=384, out_features=384, bias=True)
             )
             (norm): LayerNorm((384,), eps=1e-12, elementwise_affine=True)
           )
           (mlp): MLP(
             (hidden): Linear(in_features=384, out_features=1536, bias=True)
             (act): GELU(approximate='none')
             (proj): Linear(in_features=1536, out_features=384, bias=True)
             (drop): Dropout(p=0.1, inplace=False)
           )
           (norm): LayerNorm((384,), eps=1e-12, elementwise_affine=True)
         )
       )
     )
   )
   
   > parameters = 22,565,376
   
   Press any key to continue . . .
   ```

## Deployment

```bash
python -m uvicorn runserver:app --host 0.0.0.0 --port 8000 --reload
```

## API

The API is compatible with OpenAI's embedding endpoint.

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniLM-L6-v2",
    "input": "The quick brown fox"
  }'
```

```json
{
    "model" : "MiniLM-L6-v2",
    "object" : "list",
    "data" : [
        {
            "object" :
            "embedding",
            "index" : 0,
            "embedding" : [
                0.0027726732660084963,
                0.03326858952641487,
                -0.0006847068434581161,
                ...
                0.03463858366012573,
                0.013424797914922237,
                0.06427384167909622,
                0.025304755195975304
             ]
         }
      ]
}
```
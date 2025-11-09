# MiniLM-L6-v2

## Overview

This repository provides a self-contained implementation of the [**all-MiniLM-L6-v2**](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), designed for generating high-quality sentence embeddings. The model is optimized for efficiency and performance, making it suitable for various natural language processing tasks such as semantic search, clustering, and classification.

## Dependencies

- `torch` **Required**

- `transformers` For downloading weights, tokenization and tests. **Only the tokenizer is used at the moment**.

- `sentence-transformers` For tests. **Not used during inference**.

## Installation

Make sure you have Python 3.12 or higher.

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

## Run the API server

```bash
python -m uvicorn runserver:app --host 0.0.0.0 --port 8000 --reload
```

## API

The API is compatible with [OpenAI's embedding endpoint](https://platform.openai.com/docs/guides/embeddings).

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

## Deployment on Azure

***11/9/2025***

It appears that all the packages are not persisted after a restart. If you restart the App Service you might need to SSH into the machine again and re-install the packages.

Create a python virtual environment, install the packages there and modify the startup command to use the virtual environment's python executable.

Azure should persist the virtual environment across restarts if they are in the **/home** directory.

<span style='color:red'>**IMPORTANT**</span>: **Clear the existing startup command**

- **Settings > Configuration > Stack settings > Startup command**
- **Overview > Restart the App Service**

**Create the virtual environment and install the packages:**

```bash
python -m venv /home/site/wwwroot/antenv
source /home/site/wwwroot/antenv/bin/activate
pip install -r requirements.txt
```

**Settings > Configuration > Stack settings > Startup command:**

```bash
python -m uvicorn runserver:app --host 0.0.0.0 --port 8000
```

And it should start correctly.

```
Connected!
2025-11-09T06:14:58.9685996Z    _____
2025-11-09T06:14:58.9687912Z   /  _  \ __________ _________   ____
2025-11-09T06:14:58.9687957Z  /  /_\  \\___   /  |  \_  __ \_/ __ \
2025-11-09T06:14:58.9687985Z /    |    \/    /|  |  /|  | \/\  ___/
2025-11-09T06:14:58.9688073Z \____|__  /_____ \____/ |__|    \___  >
2025-11-09T06:14:58.9688102Z         \/      \/                  \/
2025-11-09T06:14:58.9688125Z A P P   S E R V I C E   O N   L I N U X
2025-11-09T06:14:58.9688146Z
2025-11-09T06:14:58.9688173Z Documentation    : http://aka.ms/webapp-linux
2025-11-09T06:14:58.9688198Z Python quickstart: https://aka.ms/python-qs
2025-11-09T06:14:58.968822Z Python version   : 3.12.12
2025-11-09T06:14:58.9688312Z
2025-11-09T06:14:58.9688335Z Note: Any data outside '/home' is not persisted
2025-11-09T06:15:01.8924019Z Starting OpenBSD Secure Shell server: sshd.
2025-11-09T06:15:01.9423327Z WEBSITES_INCLUDE_CLOUD_CERTS is not set to true.

This is the important line:

**2025-11-09T06:15:27.9817917Z Site's appCommandLine: python -m uvicorn runserver:app --host 0.0.0.0 --port 8000**

The autual runserver.py output:

2025-11-09T06:16:16.6672107Z INFO:     Started server process [2108]
2025-11-09T06:16:16.669466Z INFO:     Waiting for application startup.
2025-11-09T06:16:55.4809104Z INFO:     Application startup complete.
2025-11-09T06:16:55.4969716Z INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
2025-11-09T06:16:55.7009452Z Loading model workers...
2025-11-09T06:16:55.7010198Z > Loading checkpoint '/home/site/wwwroot/MiniLM-L6-v2.ckpt'...
2025-11-09T06:16:55.7010262Z > Loading checkpoint '/home/site/wwwroot/MiniLM-L6-v2.ckpt'...
2025-11-09T06:16:55.7010287Z > Loading checkpoint '/home/site/wwwroot/MiniLM-L6-v2.ckpt'...
2025-11-09T06:16:55.7010313Z > Loading checkpoint '/home/site/wwwroot/MiniLM-L6-v2.ckpt'...
2025-11-09T06:16:55.7010336Z > All 4 workers initialized!
2025-11-09T06:16:55.7010358Z Ready...
2025-11-09T06:17:13.7843079Z INFO:     169.254.129.1:18611 - "GET /health HTTP/1.1" 200 OK
```

(***Old instructions***) Still valid but see above about creating a python virtual environment.

***11/8/2025***

This is the painful process. It might take a few attempts. The following steps worked for me.

**Set Environment Variables:**

```
JWT_SECRET=<your_secret_key>
```

This one is probably not needed anymore.
```
SCM_DO_BUILD_DURING_DEPLOYMENT=true
```

**Upload Files:**

Upload all the files using FTP/FTPS.

**Deployment Center > FTPS Credentials**

**Note:** You can enable plain FTP access temporarily and just upload the file from Windows Explorer. (*Settings > General settings > FTP State*)

**SSH into the Azure App Service:**

SSH into your Azure App Service instance using the Azure portal or an SSH client and install the required packages. This is going to take a while as some packages (**transfomers**)<sup>1</sup> have large dependencies and we reply on **torch** to run the inference.

***11/8/2025***

**Note:**<sup>1</sup> This dependecy will go away. We only need it for the tokenizer, but for now we have to install the whole package.

```bash
cd site/wwwroot
pip install -r requirements.txt
```

Test the service. Run:

```bash
python -m uvicorn runserver:app --host 0.0.0.0 --port 8000
```

And navigate to the /health endpoint.

**Startup Command**

Do not enable the startup command until all the packages are installed or you will not be able to SSH into the machine due to startup failure.

**Settings > Configuration > Stack settings > Startup command:**

```bash
python -m uvicorn runserver:app --host 0.0.0.0 --port 8000
```
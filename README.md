\# Text Summarizer App



A web-based Text Summarization application built using \*\*FastAPI\*\* and a fine-tuned \*\*T5-small Transformer model\*\*.



The application takes a dialogue as input and generates a concise summary using the trained model.



\## Features



\- Dialogue summarization using a fine-tuned T5-small model.

\- FastAPI backend.

\- HTML-based user interface.

\- Text preprocessing and cleaning.

\- Beam search for summary generation.

\- CPU and GPU support (when CUDA is available).



\## Tech Stack



\- Python

\- FastAPI

\- PyTorch

\- Hugging Face Transformers

\- T5-small

\- HTML



\## Project Structure



```text

Text-Summarizer-App/

│

├── app.py

├── index.html

├── requirements.txt

├── README.md

├── .gitignore

├── .gitattributes

│

├── notebooks/

│   └── text\_summarizer.ipynb

│

└── saved\_summary\_model/

&#x20;   ├── config.json

&#x20;   ├── generation\_config.json

&#x20;   ├── model.safetensors

&#x20;   ├── tokenizer.json

&#x20;   └── tokenizer\_config.json

```



\## Installation



1\. Clone the repository:



```bash

git clone YOUR\_REPOSITORY\_URL

cd Text-Summarizer-App

```



2\. Install dependencies:



```bash

pip install -r requirements.txt

```



\## Run the Application



```bash

uvicorn app:app --reload

```



Open the following URL in your browser:



```text

http://127.0.0.1:8000

```



\## API Documentation



FastAPI automatically provides interactive API documentation at:



```text

http://127.0.0.1:8000/docs

```



\## Model



The application uses a fine-tuned T5-small model trained for dialogue summarization on the SAMSum dataset.



\## Author



Ahmed Raza



GitHub: https://github.com/ahmedraza-devs





# Coursework Summarizer

## Overview


Coursework Summarizer is an AI-powered tool for summarizing and organizing large academic documents (PDF, TXT, DOCX). It uses LLMs and k-means clustering to generate clear, topic-based summaries, and offers both a FastAPI backend and interactive terminal workflow for flexible summarization and PDF management.

---

## Features
- Accepts PDF, TXT, and DOCX input for summarization.
- Extracts, chunks, and clusters content using embeddings + k-means clustering.
- Generates cluster-level summaries with a unified prompt tuned to preserve:
	- Code blocks
	- Mathematical notation
	- Image / graph references and descriptions
- Flexible backend via FastAPI, exposing various summarization endpoints.
- Interactive terminal workflow for local summarization and file management.
- Supports outputting summaries to file, appending to PDFs, or returning via API.
- Robust error handling and logging for debugging & traceability.

---
## Getting Started
### Prerequisites
- Python 3.8+
- (Optional) API keys for models such as OpenAI
- System dependencies for PDFs / document processing (e.g. poppler, etc.)

### Installation

1. Clone the repo:
```sh 
Python git clone https://github.com/audreyl07/coursesummarizer.git
cd coursesummarizer
```
2. (Optional) Create and activate a virtual environment:
```sh
Python python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
```

3. Install required dependencies:
```sh 
Python pip install -r requirements.txt
```
## Usage

### FastAPI Server

Start the API server:

```sh
python -m uvicorn src.main:app --reload
```

### API Endpoints

- `POST /extract/` — Extract and chunk document text.
- `POST /summarize_and_save/` — Summarize using Ollama LLM and save output.
- `POST /summarize_deepseek/` — Summarize using DeepSeek LLM.
- `POST /summarize_llm/` — Summarize using a custom LLM (Ollama, DeepSeek, Llama, GPT, etc.).
- `POST /summarize_openai/` — Summarize using OpenAI GPT chat models (e.g., gpt-3.5-turbo, gpt-4).

All endpoints accept a JSON body with:

```
{
	"file_path": "<relative/path/to/document>",
	"model": "<model_name>",
	"num_clusters": 20,
	"output_file": "<optional/output/path>",
	"openai_api_key": "<your-openai-key>"
}
```

**Example (OpenAI endpoint):**

```sh
curl -X POST "http://localhost:8000/summarize_openai/" \
	-H "Content-Type: application/json" \
	-d '{
		"file_path": "COMP2401_Ch1_SystemsProgramming.pdf",
		"model": "gpt-3.5-turbo",
		"num_clusters": 20,
		"openai_api_key": "sk-..."
	}'
```

---

## Workflow & Architecture

### File Structure
```
.
├── src/
│   └── main.py
|	└── summarizer.py
│   └── document_loader.py
├── documents/         # input files
├── output/            # output summaries & PDFs
├── requirements.txt
└── .env
```

- `src/main.py` — FastAPI server with all endpoints and workflow logic.
- `summarizer.py` — Core summarization logic using k-means clustering and unified prompt.
- `document_loader.py` — Document extraction, chunking, and metadata enrichment.
- `documents/` — Input files (PDF, TXT, DOCX, etc.).
- `output/` — Output summaries and PDFs.

### Summarization Logic

1. Extraction & Chunking — The document is parsed, text is split into chunks (keeping track of structure: code, math, image references).
2. Embedding + Clustering — Each chunk is converted to embeddings (via HuggingFace or other embedding models) and clustered using k-means.
3. Summarization — For each cluster, the selected LLM is invoked using a carefully crafted prompt that ensures preservation of special formatting (code, math) and references to images / graphs.
4. Output — The summary is either returned via API or saved/appended to file as specified.
---

## Customization

- **Chunk Size & Clusters:** Chunk size and number of clusters are configurable via API request or code.
- **Supported Formats:** PDF, TXT, DOCX (extend `extract` in `document_loader.py` for more).
- **Prompt:** The summarization prompt is unified and designed to preserve code, math, and image/graph context.
- **LLM Selection:** Use any supported LLM by specifying the model name in the API request.

---

## Dependencies

- Python 3.8+
- [FastAPI](https://fastapi.tiangolo.com/)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Ollama](https://github.com/ollama/ollama)
- [FPDF](https://github.com/reingart/pyfpdf)
- [PyPDFLoader](https://github.com/langchain-ai/langchain)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [Uvicorn](https://www.uvicorn.org/) (for running FastAPI)

Install all dependencies with:

```sh
pip install fastapi uvicorn langchain langchain-community langchain-ollama langchain-huggingface fpdf streamlit requests pdfplumber
```

---

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with clear documentation and test cases.

---

## Support

For issues or feature requests, open an issue on GitHub or contact the maintainer.
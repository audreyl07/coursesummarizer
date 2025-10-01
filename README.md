


# Coursework Summarizer

## Overview


Coursework Summarizer is an AI-powered tool for summarizing and organizing large academic documents (PDF, TXT, DOCX). It uses LLMs and k-means clustering to generate clear, topic-based summaries, and offers both a FastAPI backend and interactive terminal workflow for flexible summarization and PDF management.

---


## Features

- Accepts PDF, TXT, and DOCX files for summarization.
- Extracts, chunks, and clusters document content using HuggingFace embeddings and k-means clustering.
- Summarizes each cluster using an LLM (Ollama, DeepSeek, OpenAI GPT, or custom) with a unified, detailed prompt that preserves code, math, and image/graph context.
- FastAPI backend with endpoints for different LLMs and summarization workflows.
- Interactive terminal workflow for manual summarization and PDF management.
- Flexible output: save summaries to custom output directories, append to PDFs, or return via API.
- Robust error handling and debug logging.

---


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

### Terminal Workflow

Run the interactive summarizer:

```sh
python original.py
```

---


---

## Workflow & Architecture

### File Structure

- `src/main.py` — FastAPI server with all endpoints and workflow logic.
- `summarizer.py` — Core summarization logic using k-means clustering and unified prompt.
- `document_loader.py` — Document extraction, chunking, and metadata enrichment.
- `pdf_manager.py` — PDF export and summary appending.
- `original.py` — (Optional) Interactive terminal workflow.
- `documents/` — Input files (PDF, TXT, DOCX, etc.).
- `output/` — Output summaries and PDFs.

### Summarization Logic

1. **Extraction:** Document is loaded and split into chunks, with code blocks and image/graph references extracted as metadata.
2. **Clustering:** Chunks are embedded using HuggingFace embeddings and grouped via k-means clustering.
3. **Summarization:** Each cluster is summarized using the selected LLM (Ollama, DeepSeek, OpenAI, etc.) with a detailed prompt that:
	- Preserves code formatting and provides explanations
	- Describes math, images, and graphs
	- Synthesizes main ideas in clear English
4. **Output:** Summary is saved to the output directory or returned via API. Optionally, summaries can be appended to PDFs.

---

---


## Customization

- **Chunk Size & Clusters:** Chunk size and number of clusters are configurable via API request or code.
- **Supported Formats:** PDF, TXT, DOCX (extend `extract` in `document_loader.py` for more).
- **Prompt:** The summarization prompt is unified and designed to preserve code, math, and image/graph context.
- **LLM Selection:** Use any supported LLM by specifying the model name in the API request.

---


## Example Terminal Session

```
Enter the path to a PDF, or TXT to summarize and append (or type 'stop' to finish): mytextbook.pdf

===== DOCUMENT SUMMARY =====
...summary output...
============================

Where would you like to save this summary?
1. Create a new PDF for this summary
2. Append to the default combined PDF
3. Append to another existing PDF
4. Do not save this summary
Enter 1, 2, 3, or 4: 1
Enter the filename for the summary PDF (with .pdf extension): mytextbook_summary.pdf
Summary saved to mytextbook_summary.pdf
Do you want to delete the summary PDF 'mytextbook_summary.pdf'? (y/n): n
Do you want to clear the newly summarized material from the terminal? (y/n): n
```


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
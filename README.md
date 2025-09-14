# Coursework Summarizer Documentation

## Overview

Coursework Summarizer is an AI-powered tool that helps students quickly digest and understand PDF coursework documents. It consists of a backend (summarization engine) and a frontend (Streamlit web app) for easy interaction.

---

## Backend (`original.py`)

### Purpose

- Accepts a PDF file path.
- Extracts and chunks the text.
- Clusters document chunks using embeddings.
- Summarizes the clustered content using an LLM (Ollama).
- Appends each summary to a combined PDF file.

### Main Components

- **PDF Extraction:** Uses `PyPDFLoader` and `RecursiveCharacterTextSplitter` to extract and split text.
- **Clustering:** Uses `EmbeddingsClusteringFilter` with HuggingFace embeddings.
- **Summarization:** Uses `OllamaLLM` and a custom prompt for code and syntax examples.
- **PDF Output:** Uses `FPDF` to save and append summaries.

### Usage

Run interactively in the terminal:
```
python original.py
```
- Enter the path to each PDF when prompted.
- Type `stop` to end the session.
- Summaries are saved/combined in `summary_output.pdf`.

### Key Functions

- `extract(file_path)`: Extracts and splits PDF text.
- `summarize_document_with_kmeans_clustering(file, llm, embeddings)`: Clusters and summarizes a PDF.
- `append_summary_to_pdf(pdf_path, new_summary, output_pdf)`: Appends a new summary to an existing PDF.

---

## Frontend (`frontend`)

### Purpose

- Provides a student-friendly web interface using Streamlit.
- Allows users to upload a PDF or enter a file path.
- Lets users adjust chunk size and number of clusters for summary detail.
- Connects to the backend API for summarization.
- Displays the summary in bullet points.

### Main Components

- **PDF Input:** Accepts file upload or path input.
- **Parameter Controls:** Users can set chunk size and cluster count.
- **API Connection:** Sends requests to the backend API (`/summarize` endpoint).
- **Summary Display:** Shows the summary as bullet points.

### Usage

Start the frontend:
```
streamlit run frontend
```
- Enter a PDF path or upload a PDF.
- Adjust chunk size and cluster count as needed.
- Click "Summarize Coursework" to get results.

### API Requirements

Backend should be running as an API server (e.g., FastAPI or Flask) with a `/summarize` endpoint that accepts:
- `pdf_path` (str)
- `chunk_size` (int)
- `num_clusters` (int)
- or a PDF file upload

---

## Example Workflow

1. Start backend API server (see FastAPI example in previous responses).
2. Start frontend Streamlit app.
3. Upload or enter PDF path in the web app.
4. Adjust parameters and click "Summarize Coursework".
5. View and download summaries.

---

## Dependencies

- Python 3.8+
- [LangChain](https://github.com/langchain-ai/langchain)
- [Ollama](https://github.com/ollama/ollama)
- [FPDF](https://github.com/reingart/pyfpdf)
- [Streamlit](https://github.com/streamlit/streamlit)
- [PyPDFLoader](https://github.com/langchain-ai/langchain)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)

Install with:
```
pip install langchain langchain-community langchain-ollama langchain-huggingface fpdf streamlit requests
```

---

## Contributing

- Fork the repository.
- Create a feature branch.
- Submit a pull request with clear documentation and test cases.

---

## Support

For issues or feature requests, open an issue on GitHub or contact the maintainer.
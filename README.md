
# Coursework Summarizer

## Overview

Coursework Summarizer is an AI-powered tool for summarizing and organizing large academic documents (PDF, TXT, DOCX). It uses LLMs and clustering to generate clear, topic-based summaries, and offers flexible options for saving and managing summary PDFs.

---

## Features

- Accepts PDF, TXT, and DOCX files for summarization.
- Extracts, chunks, and clusters document content using embeddings.
- Summarizes each cluster using an LLM (Ollama) with a custom prompt that preserves code and image/graph context.
- Interactive terminal workflow: choose to create a new PDF, append to an existing PDF, or skip saving.
- Optionally delete generated summaries or clear them from the terminal.
- Prevents duplicate summaries and repeated content.

---

## Usage (Terminal)

Run the summarizer interactively:

```sh
python original.py
```

**Workflow:**

1. Enter the path to a PDF, TXT, or DOCX file when prompted.
2. Review the summary in the terminal (optionally clear it).
3. Choose where to save the summary:
	- Create a new PDF (with your chosen filename)
	- Append to the default combined PDF
	- Append to another existing PDF
	- Do not save the summary
4. If you create a new PDF, you can immediately delete it if not needed.
5. Type `stop` to end the session.

---

## Key Functions

- `extract(file_path)`: Extracts and splits document text, extracts code blocks and image/graph references.
- `summarize_document_with_kmeans_clustering(file, llm, embeddings)`: Clusters and summarizes a document, preserving code and image/graph context.
- `append_summary_to_pdf(pdf_path, new_summary, output_pdf)`: Appends a new summary to an existing PDF, avoiding repetition.

---

## Customization

- **Chunk Size & Clusters:** Chunk size and number of clusters are set in the code for large documents, but can be easily exposed as user inputs for further tuning.
- **Supported Formats:** PDF, TXT, DOCX (add more by extending the `extract` function).
- **Prompt:** The summarization prompt is designed to preserve code formatting and describe images/graphs.

---

## Example Terminal Session

```
Enter the path to a PDF, or TXT to summarize and append (or type 'stop' to finish): mytextbook.pdf

===== DOCUMENT SUMMARY =====
...summary output...
============================

Do you want to clear the newly summarized material from the terminal? (y/n): n
Where would you like to save this summary?
1. Create a new PDF for this summary
2. Append to the default combined PDF
3. Append to another existing PDF
4. Do not save this summary
Enter 1, 2, 3, or 4: 1
Enter the filename for the summary PDF (with .pdf extension): mytextbook_summary.pdf
Summary saved to mytextbook_summary.pdf
Do you want to delete the summary PDF 'mytextbook_summary.pdf'? (y/n): n
```

---

## Dependencies

- Python 3.8+
- [LangChain](https://github.com/langchain-ai/langchain)
- [Ollama](https://github.com/ollama/ollama)
- [FPDF](https://github.com/reingart/pyfpdf)
- [PyPDFLoader](https://github.com/langchain-ai/langchain)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)

Install with:

```sh
pip install langchain langchain-community langchain-ollama langchain-huggingface fpdf streamlit requests
```

---

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with clear documentation and test cases.

---

## Support

For issues or feature requests, open an issue on GitHub or contact the maintainer.
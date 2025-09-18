import warnings
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_community.document_transformers import EmbeddingsClusteringFilter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import importlib
from fpdf import FPDF


# Suppress specific warnings from transformers library
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module")

# === Document Extraction and Chunking ===

import os
from langchain_community.document_loaders import TextLoader, UnstructuredWordDocumentLoader

import re

def extract_code_and_images(text, ext):
    """
    Extract code blocks and image references from text.
    Only .txt files support code block extraction; .docx currently does not extract images.
    """
    code_blocks = []
    images = []
    if ext == ".txt":
        # Extract code blocks in triple backtick format
        code_blocks = re.findall(r'```[\w\+\-]*\n(.*?)```', text, re.DOTALL)
    elif ext == ".docx":
        # Placeholder for future image extraction from docx
        pass
    return code_blocks, images

def extract(file_path):
    """
    Extracts and splits text from a file (PDF, TXT, DOCX) into manageable chunks for summarization.
    Also extracts code blocks and image/graph references for inclusion in summary metadata.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        text_content = " ".join([p.page_content for p in pages])
    elif ext == ".txt":
        loader = TextLoader(file_path, autodetect_encoding=True)
        pages = loader.load()
        text_content = " ".join([p.page_content for p in pages])
    elif ext == ".docx":
        loader = UnstructuredWordDocumentLoader(file_path)
        pages = loader.load()
        text_content = " ".join([p.page_content for p in pages])
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported types: .pdf, .txt, .docx")

    # Extract code blocks and images for later use in summary
    code_blocks, images = extract_code_and_images(text_content, ext)

    # Split document into chunks for LLM processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    texts = text_splitter.split_documents(pages)
    # Attach code_blocks and images as metadata for later use
    for doc in texts:
        if code_blocks:
            doc.metadata["code_blocks"] = code_blocks
        if images:
            doc.metadata["images"] = images
    return texts

# --- Summarization with Clustering ---
def summarize_document_with_kmeans_clustering(file, llm, embeddings):
    """
    Cluster document chunks using embeddings, then summarize each cluster using an LLM.
    Includes code and image/graph examples in the summary output.
    """
    filter = EmbeddingsClusteringFilter(embeddings=embeddings, num_clusters=20)
    texts = extract(file)
    try:
        filtered_docs = filter.transform_documents(documents=texts)

        # Gather code blocks and images from metadata for appendix
        all_code_blocks = []
        all_images = []
        for doc in filtered_docs:
            if "code_blocks" in doc.metadata:
                all_code_blocks.extend(doc.metadata["code_blocks"])
            if "images" in doc.metadata:
                all_images.extend(doc.metadata["images"])

        # Prompt for LLM summarization
        custom_prompt = PromptTemplate.from_template(
            """
Summarize the following document, for each new heading:
- If there are code examples, format them as code blocks with clear but detailed code with correct syntax and clear indentation.
- If there are images, graphs or plots, describe the content, purpose and relevance to the specific topic. If there is a link, then list it in the document.
\n\n{text}
"""
        )
        summarization_chain = load_summarize_chain(
            llm, chain_type="stuff", prompt=custom_prompt
        )

        result = summarization_chain.invoke({"input_documents": filtered_docs})

        # Compose summary with code and image/graph appendix
        summary_text = result.get("output_text", str(result)) if isinstance(result, dict) else str(result)
        appendix = ""
        if all_code_blocks:
            appendix += "\n\n=== Extracted Code Blocks ===\n"
            for idx, code in enumerate(all_code_blocks, 1):
                # Format as indented code block (4 spaces)
                indented_code = '\n'.join('    ' + line for line in code.strip().splitlines())
                appendix += f"\nCode Block {idx}:\n" + indented_code + "\n"
        if all_images:
            appendix += "\n\n=== Image/Graph References ===\n"
            for idx, img in enumerate(all_images, 1):
                appendix += f"Image/Graph {idx}: {img}\n"
        return summary_text + appendix

    except Exception as e:
        return f"[ERROR]: {e}"

# --- Append Summary to PDF ---
def append_summary_to_pdf(pdf_path, new_summary, output_pdf="summary_output.pdf"):
    """
    Appends a new summary to an existing PDF summary file.
    If the file exists, previous summaries are included. Avoids repetition.
    """
    import os

    # Extract existing text from the PDF if it exists
    existing_text = ""
    if os.path.exists(pdf_path):
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        existing_text += page_text + "\n"
        except ImportError:
            print("pdfplumber not installed, skipping existing PDF text extraction.")

    # Extract the title from the new summary (first non-empty line)
    def get_title(text):
        for line in text.split('\n'):
            if line.strip():
                return line.strip()
        return "Untitled Document"

    title = get_title(new_summary)

    # Remove repetition: if the new summary is already in the existing text, skip adding
    if new_summary.strip() in existing_text:
        print("This summary already exists in the document. Skipping append.")
        return

    # Remove repeated lines between existing and new summary
    existing_lines = set(existing_text.split('\n'))
    filtered_new_lines = [line for line in new_summary.split('\n') if line.strip() and line not in existing_lines]

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Times", size=12)

    # Always start with the new document's title
    pdf.multi_cell(0, 10, f"{title}")
    pdf.ln(5)
    for line in filtered_new_lines:
        pdf.multi_cell(0, 10, line)

    # Add previous content (if any) after the new summary
    if existing_text:
        pdf.ln(10)
        pdf.multi_cell(0, 10, "--- Previous Summaries ---")
        for line in existing_text.split('\n'):
            pdf.multi_cell(0, 10, line)

    pdf.output(output_pdf)
    print(f"Combined summary saved to {output_pdf}")

# --- Model and Embedding Configuration ---
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs
)

llm = OllamaLLM(
    model="llama3.1:8b",
    temperature=0
)

# --- Interactive PDF Summarization Loop ---
combined_pdf_path = "SummaryOutput.pdf"
first_run = True

def get_title(text):
    """
    Extract the first non-empty line as the document title.
    """
    for line in text.split('\n'):
        if line.strip():
            return line.strip()
    return "Untitled Document"


# === Main Interactive Loop ===
while True:
    # Prompt user for document to summarize
    file_path = input("Enter the path to a PDF, or TXT to summarize and append (or type 'stop' to finish): ").strip()
    if file_path.lower() == "stop":
        print("Stopping document summarization and appending.")
        break

    try:
        summary = summarize_document_with_kmeans_clustering(
            file_path,
            llm,
            embeddings
        )
    except Exception as e:
        print(f"[ERROR]: {e}")
        continue

    # Display summary in terminal
    print("\n===== DOCUMENT SUMMARY =====\n")
    print(summary)
    print("\n============================\n")

    # Ask user where to save the summary
    print("Where would you like to save this summary?")
    print("1. Create a new PDF for this summary")
    print("2. Append to the default combined PDF")
    print("3. Append to another existing PDF")
    print("4. Do not save this summary")
    save_choice = input("Enter 1, 2, 3, or 4: ").strip()

    if save_choice == '1':
        # User chooses to create a new PDF for this summary
        custom_filename = input("Enter the filename for the summary PDF (with .pdf extension): ").strip()
        if not custom_filename.lower().endswith('.pdf'):
            custom_filename += '.pdf'
        title = get_title(summary)
        filtered_lines = [line for line in summary.split('\n') if line.strip()]
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Times", size=12)
        pdf.multi_cell(0, 10, f"{title}")
        pdf.ln(5)
        for line in filtered_lines:
            pdf.multi_cell(0, 10, line)
        pdf.output(custom_filename)
        print(f"Summary saved to {custom_filename}")
        # Ask if user wants to delete the summary PDF
        delete_choice = input(f"Do you want to delete the summary PDF '{custom_filename}'? (y/n): ").strip().lower()
        if delete_choice == 'y':
            import os
            try:
                os.remove(custom_filename)
                print(f"Deleted {custom_filename}.")
            except Exception as e:
                print(f"Could not delete {custom_filename}: {e}")
    elif save_choice == '2':
        # User chooses to append to the default combined PDF
        if first_run:
            title = get_title(summary)
            filtered_lines = [line for line in summary.split('\n') if line.strip()]
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_font("Times", size=12)
            pdf.multi_cell(0, 10, f"{title}")
            pdf.ln(5)
            for line in filtered_lines:
                pdf.multi_cell(0, 10, line)
            pdf.output(combined_pdf_path)
            print(f"Summary saved to {combined_pdf_path}")
            first_run = False
        else:
            append_summary_to_pdf(combined_pdf_path, summary, output_pdf=combined_pdf_path)
    elif save_choice == '3':
        # User chooses to append to another existing PDF
        other_filename = input("Enter the filename of the existing PDF to append to (with .pdf extension): ").strip()
        if not other_filename.lower().endswith('.pdf'):
            other_filename += '.pdf'
        append_summary_to_pdf(other_filename, summary, output_pdf=other_filename)
    elif save_choice == '4':
        print("Summary not saved.")
    else:
        print("Invalid choice. Skipping save for this summary.")
        

    # Optionally clear the summary from the terminal
    delete_terminal = input("Do you want to clear the newly summarized material from the terminal? (y/n): ").strip().lower()
    if delete_terminal == 'y':
        import os
        # Windows
        if os.name == 'nt':
            os.system('cls')
        # Unix/Linux/Mac
        else:
            os.system('clear')

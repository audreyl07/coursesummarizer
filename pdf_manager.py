from fpdf import FPDF
import os

def get_title(text):
    """
    Extract the first non-empty line as the document title.
    """
    for line in text.split('\n'):
        if line.strip():
            return line.strip()
    return "Untitled Document"

def append_summary_to_pdf(pdf_path, new_summary, output_pdf="summary_output.pdf"):
    """
    Appends a new summary to an existing PDF summary file.
    If the file exists, previous summaries are included. Avoids repetition.
    """
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

    title = get_title(new_summary)
    # Always save to a PDF named after the document title + ' summary.pdf'
    safe_title = ''.join(c for c in title if c.isalnum() or c in (' ', '_', '-')).rstrip()
    output_pdf = f"{safe_title} summary.pdf" if safe_title else "Untitled_Document_summary.pdf"

    if new_summary.strip() in existing_text:
        print("This summary already exists in the document. Skipping append.")
        return
    existing_lines = set(existing_text.split('\n'))
    filtered_new_lines = [line for line in new_summary.split('\n') if line.strip() and line not in existing_lines]

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"{title}")
    pdf.ln(5)
    for line in filtered_new_lines:
        pdf.multi_cell(0, 10, line)
    if existing_text:
        pdf.ln(10)
        pdf.multi_cell(0, 10, "--- Previous Summaries ---")
        for line in existing_text.split('\n'):
            pdf.multi_cell(0, 10, line)
    pdf.output(output_pdf)
    print(f"Combined summary saved to {output_pdf}")
    
def read_text_file(filepath: str) -> str:
    """
    Reads and returns the content of a text file.
    Args:
        filepath: Path to the text file.
    Returns:
        The file content as a string, or an empty string if not found or error.
    """
    if not os.path.exists(filepath):
        print(f"Error: The file '{filepath}' was not found.")
        return ""
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read()
        return content
    except IOError:
        print(f"Error: Could not read the file '{filepath}'.")
        return ""
    except Exception as e:
        print(f"Error: {e}")
        return ""

# For direct testing
try:
    file_path = "/Users/admin/ws-2024/coursesummarizer/COMP2401_Ch2_DataRepresentation.txt" 
    save_path = "/Users/admin/ws-2024/coursesummarizer/COMP2401_Ch2_DataRepresentation summary.pdf"
    texts = read_text_file(file_path)
    print(texts)
    append_summary_to_pdf(save_path, texts)
except Exception as e:
    print(f"[ERROR]: {e}")
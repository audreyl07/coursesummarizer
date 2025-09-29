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
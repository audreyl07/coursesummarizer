import os
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

@dataclass
class ExtractedContent:
    """Container for different types of extracted content"""
    code_blocks: List[Tuple[str, str]]  # (language, code)
    math_expressions: List[str]
    graph_references: List[Dict[str, str]]  # [{type, description, reference}]
    tables: List[Dict[str, str]]  # [{caption, content}]
    equations: List[str]
    general_text: List[str]

class ContentExtractor:
    """Handles extraction of different content types from documents"""

    def __init__(self):
        # Regular expressions for different content types
        self.patterns = {
            'code': [
                r'```([\w+\-]*)\n([\s\S]*?)```',  # Markdown code blocks
                r'(?:^|\n)(?:    |\t)([^\n]+(?:\n(?:    |\t)[^\n]+)*)',  # Indented code blocks
                r'(?:^|\n)(?:def|class|import|from|public|private|function)\s+\w+[^\n]*(?:\n\s+[^\n]+)*'  # Code-like patterns
            ],
            'math': [
                r'\$\$(.*?)\$\$',  # LaTeX display math
                r'\$(.*?)\$',  # LaTeX inline math
                r'\\begin\{equation\}(.*?)\\end\{equation\}',  # LaTeX equation environment
                r'(?:[\+\-\*/=<>≤≥≠]{2,}|[\u2200-\u22FF]|\b(?:sum|integral|lim|div|mod|sqrt)\b)'  # Mathematical operators
            ],
            'graphs': [
                r'\!\[([^\]]*)\]\(([^\)]*)\)',  # Markdown image syntax
                r'(?i)(?:figure|graph|plot|chart|diagram)\s*\d*\s*[:\.]\s*([^\n]*)',  # Figure references
                r'(?i)\b(?:histogram|scatter plot|bar chart|line graph|pie chart)\s*[:\.]\s*([^\n]*)'  # Graph descriptions
            ],
            'tables': [
                r'\|[^\n]*\|[^\n]*\n(?:\|[^\n]*\|[^\n]*\n)+',  # Markdown tables
                r'(?i)table\s*\d*\s*[:\.]\s*([^\n]*)',  # Table captions
                r'(?i)\b(?:dataset|data set|tabular data)\s*[:\.]\s*([^\n]*)'  # Table references
            ],
            'equations': [
                r'(?:\d+\s*[+\-*/]\s*)+\d+\s*=',  # Basic equations
                r'[a-zA-Z]\s*=\s*(?:[\d.]+|[a-zA-Z](?:[+\-*/][\d.]+|[a-zA-Z])*)',  # Variable assignments
                r'\b(?:formula|equation)\s*\d*\s*[:\.]\s*([^\n]*)'  # Equation references
            ]
        }

    def _extract_pattern_matches(self, text: str, patterns: List[str]) -> List[str]:
        """Extract all matches for a list of patterns from text"""
        matches = []
        for pattern in patterns:
            matches.extend(re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE))
        return [m.group(0) for m in matches]

    def _extract_code_with_language(self, text: str) -> List[Tuple[str, str]]:
        """Extract code blocks with their language specification"""
        code_blocks = []
        for match in re.finditer(r'```([\w+\-]*)\n([\s\S]*?)```', text):
            lang = match.group(1) or 'text'
            code = match.group(2)
            code_blocks.append((lang.lower(), code.strip()))
        return code_blocks

    def extract_content(self, text: str) -> ExtractedContent:
        """Extract all types of content from the text"""
        code_blocks = self._extract_code_with_language(text)
        
        math_expressions = self._extract_pattern_matches(text, self.patterns['math'])
        
        graph_refs = []
        for match in re.finditer(self.patterns['graphs'][0], text):
            graph_refs.append({
                'type': 'image',
                'description': match.group(1),
                'reference': match.group(2)
            })
        for pattern in self.patterns['graphs'][1:]:
            for match in re.finditer(pattern, text):
                graph_refs.append({
                    'type': 'reference',
                    'description': match.group(1),
                    'reference': ''
                })

        tables = []
        table_matches = self._extract_pattern_matches(text, self.patterns['tables'])
        for table in table_matches:
            tables.append({
                'caption': '',
                'content': table.strip()
            })

        equations = self._extract_pattern_matches(text, self.patterns['equations'])

        # Remove extracted content from text to get general text
        general_text = text
        for pattern_list in self.patterns.values():
            for pattern in pattern_list:
                general_text = re.sub(pattern, '', general_text)
        general_text = [t.strip() for t in general_text.split('\n') if t.strip()]

        return ExtractedContent(
            code_blocks=code_blocks,
            math_expressions=math_expressions,
            graph_references=graph_refs,
            tables=tables,
            equations=equations,
            general_text=general_text
        )

def extract_content_by_type(text: str, ext: str) -> ExtractedContent:
    """
    Extract different types of content from text.
    Supports code blocks, mathematical expressions, graphs, tables, and equations.
    
    Args:
        text: The text content to analyze
        ext: The file extension to determine parsing strategy
    Returns:
        ExtractedContent object containing all extracted content
    """
    extractor = ContentExtractor()
    
    # Extract content based on file type
    if ext.lower() in [".txt", ".md"]:
        # Full extraction for text and markdown files
        return extractor.extract_content(text)
    elif ext.lower() == ".pdf":
        # PDF specific extraction (might need special handling for math/equations)
        content = extractor.extract_content(text)
        # Add PDF-specific extraction logic here if needed
        return content
    elif ext.lower() == ".docx":
        # DOCX specific extraction
        content = extractor.extract_content(text)
        # Add DOCX-specific extraction logic here if needed
        return content
    else:
        # Default extraction for unknown file types
        return extractor.extract_content(text)

def extract(file_path, chunk_size=2000, chunk_overlap=20):
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

    if save_loaded:
        # Save the loaded document text to a file for reuse
        if not save_path:
            base_name = os.path.basename(file_path)
            file_name_without_ext = os.path.splitext(base_name)[0]
            save_path = f"{file_name_without_ext}_loaded.txt"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text_content)

    # Extract all content types
    extracted_content = extract_content_by_type(text_content, ext)
    
    # Split text while preserving special content
    code_blocks, images = extract_code_and_images(text_content, ext)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(pages)
    
    # Attach extracted content as metadata
    for doc in texts:
        doc.metadata["code_blocks"] = [(lang, code) for lang, code in extracted_content.code_blocks]
        doc.metadata["math_expressions"] = extracted_content.math_expressions
        doc.metadata["graph_references"] = extracted_content.graph_references
        doc.metadata["tables"] = extracted_content.tables
        doc.metadata["equations"] = extracted_content.equations
        
        # Store content type hints for the categorizer
        doc.metadata["content_types"] = {
            "has_code": bool(extracted_content.code_blocks),
            "has_math": bool(extracted_content.math_expressions or extracted_content.equations),
            "has_graphs": bool(extracted_content.graph_references),
            "has_tables": bool(extracted_content.tables)
        }
    
    return texts

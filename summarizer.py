from langchain_community.document_transformers import EmbeddingsClusteringFilter
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

def summarize_document_with_kmeans_clustering(texts, llm, embeddings, num_clusters=20):
    """
    Cluster document chunks using embeddings, then summarize each cluster using an LLM.
    Includes code and image/graph examples in the summary output.
    """
    filter = EmbeddingsClusteringFilter(embeddings=embeddings, num_clusters=num_clusters)
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
Summarize the following document. For each section or heading:
- **Code:** If there are code examples, format them as code blocks with clear syntax, proper indentation, and add a brief explanation of what the code does and why it is relevant.
- **Math:** If there are mathematical formulas, equations, or calculations, explain their meaning, context, and how they relate to the topic. Provide step-by-step breakdowns if possible.
- **Graphs/Images/Plots:** If there are images, graphs, charts, or plots, describe their content, purpose, and relevance to the specific topic. Explain what the visualization shows and how it supports the document's points.
- **Text:** Clearly explain the main concepts, arguments, and supporting details. Make connections between ideas and provide context for understanding.
- **General Text:** Synthesize any other information, ensuring clarity and completeness. Integrate code, math, and visual elements into the overall summary for a cohesive understanding.
- Use clear, concise language and avoid unnecessary repetition.

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
            indented_code = '\n'.join('    ' + line for line in code.strip().splitlines())
            appendix += f"\nCode Block {idx}:\n" + indented_code + "\n"
    if all_images:
        appendix += "\n\n=== Image/Graph References ===\n"
        for idx, img in enumerate(all_images, 1):
            appendix += f"Image/Graph {idx}: {img}\n"
    
    return summary_text + appendix


# def cross_check_summary(summary: str, source_text: str) -> dict:
#     """
#     Cross-checks the summarized content against the original document text for accuracy.
#     Returns a report with confidence score and discrepancies.
#     """
#     import difflib
#     import re

#     report = {
#         "missing": [],
#         "altered": [],
#         "hallucinated": [],
#         "confidence": 1.0,
#     }

#     # Normalize text for comparison
#     def normalize(text):
#         return re.sub(r"\s+", " ", text.strip().lower())

#     norm_source = normalize(source_text)
#     norm_summary = normalize(summary)

#     # Split summary into sentences/lines for checking
#     summary_lines = [line.strip() for line in re.split(r'[\n\.!?]', summary) if line.strip()]

#     # Check each line in summary
#     for line in summary_lines:
#         norm_line = normalize(line)
#         if norm_line and norm_line not in norm_source:
#             # Try fuzzy match
#             matches = difflib.get_close_matches(norm_line, [norm_source], n=1, cutoff=0.8)
#             if not matches:
#                 report["hallucinated"].append(line)
#             else:
#                 report["altered"].append(line)

#     # Check for missing key facts/code/math/graphs
#     # Example: look for code blocks, equations, graph/image references
#     code_blocks = re.findall(r'```[\w+\-]*\n([\s\S]*?)```', source_text)
#     for code in code_blocks:
#         if normalize(code) not in norm_summary:
#             report["missing"].append(f"Code block: {code[:50]}...")

#     equations = re.findall(r'\$\$(.*?)\$\$', source_text)
#     for eq in equations:
#         if normalize(eq) not in norm_summary:
#             report["missing"].append(f"Equation: {eq}")

#     images = re.findall(r'\!\[([^\]]*)\]\(([^\)]*)\)', source_text)
#     for desc, ref in images:
#         if normalize(desc) not in norm_summary:
#             report["missing"].append(f"Image/Graph: {desc}")

#     # Confidence score: penalize for hallucinated/missing/altered
#     total_lines = len(summary_lines)
#     penalty = (len(report["hallucinated"]) + len(report["missing"]) + 0.5 * len(report["altered"]))
#     if total_lines > 0:
#         report["confidence"] = max(0.0, 1.0 - penalty / total_lines)
#     else:
#         report["confidence"] = 0.0

#     return report

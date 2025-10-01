from langchain_core.documents import Document

def categorize_chunk_model(chunk):
    """
    Analyze a chunk's content and recommend a model for summarization.
    Returns: model_name (str)
    Example logic: If chunk contains code, use 'ollama'; if math, use 'gpt';
    if graphs/charts, use 'gpt'; else, use 'deepseek'.
    """
    text = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
    # Check for code content
    if '```' in text or 'def ' in text or 'class ' in text:
        return 'ollama'  # Good for code
    # Check for mathematical content
    elif any(sym in text for sym in ['∫', 'Σ', 'π', '=', '+', '-', '*', '/']):
        return 'gpt'    # Good for math
    # Check for graphs and visualizations
    elif any(term in text.lower() for term in [
        'graph', 'chart', 'plot', 'diagram', 'figure',
        'visualization', 'histogram', 'scatter plot', 'bar chart',
        'line graph', 'pie chart', 'trend line', 'axis', 'legend'
    ]):
        return 'graphs'  # Good for visual analysis
    else:
        return 'deepseek'  # General text

def categorize_chunks(chunks):
    """
    Categorize document chunks into different types based on their content.
    Returns: Dictionary of categorized chunks
    """
    code_chunks = []
    math_chunks = []
    graph_chunks = []
    general_chunks = []
    
    for chunk in chunks:
        model = categorize_chunk_model(chunk)
        if model == 'ollama':
            code_chunks.append(chunk)
        elif model == 'gpt' and chunk in math_chunks:
            math_chunks.append(chunk)
        elif model == 'graphs':
            graph_chunks.append(chunk)
        else:
            general_chunks.append(chunk)
            
    return {
        'code': code_chunks,
        'math': math_chunks,
        'graphs': graph_chunks,
        'general': general_chunks
    }
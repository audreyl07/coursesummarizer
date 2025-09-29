from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
import os
from typing import Dict, List, Optional, Any

class CategoryCache:
    """Cache for storing document chunk categorizations"""
    
    def __init__(self, cache_file: str = "category_cache.json"):
        self.cache_file = cache_file
        self._cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, str]:
        """Load the cache from file if it exists"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def _save_cache(self):
        """Save the cache to file"""
        with open(self.cache_file, 'w') as f:
            json.dump(self._cache, f)
    
    def get(self, chunk_hash: str) -> Optional[str]:
        """Get category for a chunk from cache"""
        return self._cache.get(chunk_hash)
    
    def set(self, chunk_hash: str, category: str):
        """Set category for a chunk in cache"""
        self._cache[chunk_hash] = category
        self._save_cache()

class DocumentCategorizer:
    """Handles document chunk categorization using LLM"""
    
    def __init__(self, llm):
        self.llm = llm
        self.cache = CategoryCache()
        self.categorization_chain = self._create_categorization_chain()
    
    def _create_categorization_chain(self) -> LLMChain:
        """Create the LLM chain for categorization"""
        prompt = PromptTemplate(
            input_variables=["content"],
            template="""Analyze the following content and categorize it into exactly one of these categories: 'code', 'math', 'graphs', 'general'.

Categories:
- 'code': Content containing code snippets, programming concepts, or software implementation details
- 'math': Content with mathematical formulas, equations, proofs, or mathematical concepts
- 'graphs': Content describing or analyzing graphs, charts, visualizations, plots, or diagrams
- 'general': Other content not fitting the above categories

Content to categorize:
{content}

Respond with only the category name (code/math/graphs/general):"""
        )
        return LLMChain(llm=self.llm, prompt=prompt)
    
    def _get_chunk_hash(self, chunk) -> str:
        """Create a hash for the chunk content for caching"""
        content = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
        return str(hash(content))

def categorize_chunk_model(chunk, llm=None):
    """
    Analyze a chunk's content using LLM to determine the appropriate model.
    
    Args:
        chunk: The document chunk to categorize
        llm: LLM instance to use for categorization (optional)
    Returns:
        model_name (str): The name of the model to use ('ollama', 'claude', 'gemini', 'gpt')
    """
    if not llm:
        # Use simple heuristics if no LLM provided
        text = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
        if '```' in text or 'def ' in text or 'class ' in text:
            return 'ollama'  # Good for code
        elif any(sym in text for sym in ['∫', 'Σ', 'π', '=', '+', '-', '*', '/']):
            return 'claude'  # Good for math
        elif any(term in text.lower() for term in [
            'graph', 'chart', 'plot', 'diagram', 'figure',
            'visualization', 'histogram', 'scatter plot', 'bar chart',
            'line graph', 'pie chart', 'trend line', 'axis', 'legend'
        ]):
            return 'gemini'  # Good for visual analysis
        else:
            return 'gpt'  # General text
            
    categorizer = DocumentCategorizer(llm)
    chunk_hash = categorizer._get_chunk_hash(chunk)
    
    # Check cache first
    cached_category = categorizer.cache.get(chunk_hash)
    if cached_category:
        return _category_to_model(cached_category)
        
    # Use LLM to categorize
    content = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
    result = categorizer.categorization_chain.run(content=content)
    category = result.strip().lower()
    
    # Cache the result
    categorizer.cache.set(chunk_hash, category)
    
    return _category_to_model(category)

def _category_to_model(category: str) -> str:
    """Convert category to model name"""
    model_map = {
        'code': 'ollama',    # Best for code analysis
        'math': 'claude',    # Best for mathematical content
        'graphs': 'gemini',  # Best for visual analysis
        'general': 'gpt'     # Best for general content
    }
    return model_map.get(category, 'gpt')

def categorize_chunks(chunks: List[Any], llm=None) -> Dict[str, List[Any]]:
    """
    Categorize document chunks into different types based on their content using LLM.
    
    Args:
        chunks: List of document chunks to categorize
        llm: LLM instance to use for categorization (optional)
    Returns:
        Dictionary mapping category names to lists of chunks
    """
    categorized = {
        'code': [],
        'math': [],
        'graphs': [],
        'general': []
    }
    
    for chunk in chunks:
        model = categorize_chunk_model(chunk, llm)
        if model == 'ollama':
            categorized['code'].append(chunk)
        elif model == 'claude':
            categorized['math'].append(chunk)
        elif model == 'gemini':
            categorized['graphs'].append(chunk)
        else:
            categorized['general'].append(chunk)
            
    return categorized
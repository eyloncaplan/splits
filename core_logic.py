import os
import json
import hashlib
import random
import pandas as pd
from typing import Union, Tuple
import torch

from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import Document
from pyserini.analysis import get_lucene_analyzer
from pyserini.pyclass import autoclass
from scipy.stats import hypergeom
from subspace.tool import SubspaceBERTScore

# ==============================================================================
# BM25 Search and Query Building
# ==============================================================================

def get_standard_query(query: str, field: str = "contents", analyzer=None):
    """
    Runs Lucene's StandardQueryParser to get a parsed query object.
    """
    if analyzer is None:
        analyzer = get_lucene_analyzer()
        
    JStandardQueryParser = autoclass('org.apache.lucene.queryparser.flexible.standard.StandardQueryParser')
    query_parser = JStandardQueryParser()
    query_parser.setAnalyzer(analyzer)
    
    return query_parser.parse(query, field)

def query_bm25_index(index_path: str, keywords: list, doc_count: int = 1000) -> pd.DataFrame:
    """Load index, run BM25 phrase search using custom HuggingFace analyzer, and return results."""
    # 1. Load searcher
    searcher = LuceneSearcher(index_path)
    
    # 2. Load custom analyzer matching your index strategy
    analyzer = get_lucene_analyzer(
        language='hgf_tokenizer',
        huggingFaceTokenizer='bert-base-uncased'  
    )
    
    # 3. Create query string connecting keywords by OR (e.g., '"jesus" OR "christ"')
    query_string = " OR ".join([f'"{kw}"' for kw in keywords])
    
    # 4. Build standard lucene query using your custom querybuilder
    phrase_q = get_standard_query(query_string, analyzer=analyzer)
    
    # 5. Search
    hits = searcher.search(phrase_q, doc_count)
    
    # 6. Parse results
    results = []
    returned_ids = set()

    for hit in hits:
        returned_ids.add(hit.docid)
        doc = Document(hit.lucene_document)
        raw = doc.raw()
        jd = json.loads(raw)
        
        row = {
            'id': jd.get("id"),
            'content': jd.get("contents", ""),
            'score': hit.score
        }
        
        if "metadata" in jd and jd["metadata"]:
            metadata = json.loads(jd["metadata"])
            row.update(metadata)
            
        results.append(row)
        
    returned_ext_ids = {r['id'] for r in results}

    # Pad with random unretrieved items exactly as required
    if len(results) < doc_count:
        needed = doc_count - len(results)
        total = searcher.num_docs

        # build a list of internal docnums whose external ID wasn't already returned
        pool = []
        for docnum in range(total):
            lucene_doc = searcher.doc(docnum)
            doc = Document(lucene_doc)
            jd = json.loads(doc.raw())
            ext_id = jd.get("id")
            if ext_id not in returned_ext_ids:
                pool.append(docnum)

        # deterministically shuffle by query
        md5 = hashlib.md5(query_string.encode("utf-8")).hexdigest()
        seed = int(md5, 16) % 2**32
        rng = random.Random(seed)
        rng.shuffle(pool)

        # pull 'needed' more docs
        for docnum in pool[:needed]:
            lucene_doc = searcher.doc(docnum)
            doc = Document(lucene_doc)
            raw = doc.raw()
            jd = json.loads(raw)
            
            row = {
                "id": jd.get("id"),
                "content": jd.get("contents", ""),
                "score": None
            }
            if "metadata" in jd and jd["metadata"]:
                metadata = json.loads(jd["metadata"])
                row.update(metadata)
            
            results.append(row)
        
    return pd.DataFrame(results)

# ==============================================================================
# Evaluation Metrics (Precision/Lift)
# ==============================================================================

def _resolve_k(df, k):
    """Convert float percentages to absolute k or return k as an int."""
    if isinstance(k, float) and 0.0 < k <= 1.0:
        return int(len(df) * k)
    return int(k)

def precision_at_k(df: pd.DataFrame, correct_demographic: str, k: Union[int, float]) -> float:
    """Calculate precision at k for a target demographic."""
    rel = (df['demographic'] == correct_demographic).astype(int)
    k_abs = _resolve_k(df, k)
    if k_abs <= 0:
        return 0.0
    return rel.iloc[:k_abs].sum() / float(k_abs)

def lift_at_k(df: pd.DataFrame, correct_demographic: str, k: Union[int, float]) -> float:
    """Lift@k: ratio of precision@k to the overall proportion of relevant items."""
    k_abs = _resolve_k(df, k)
    if k_abs <= 0 or len(df) == 0:
        return 0.0
    
    precision_k = precision_at_k(df, correct_demographic, k)
    rel = (df['demographic'] == correct_demographic).astype(int)
    overall_proportion = rel.sum() / float(len(df))
    
    if overall_proportion == 0:
        return 0.0
    
    return precision_k / overall_proportion

def hypergeometric_significance_test(df: pd.DataFrame, correct_demographic: str, k: Union[int, float], alpha: float = 0.05) -> Tuple[float, Tuple[int, int], Tuple[float, float]]:
    """Hypergeometric statistical significance test for the retrieval."""
    n = _resolve_k(df, k)  
    N = len(df)            
    
    rel = (df['demographic'] == correct_demographic).astype(int)
    K = rel.sum()          
    k_obs = rel.iloc[:n].sum()  
    
    if K == 0 or n <= 0:
        return 0.0, (0, 0), (0.0, 0.0)
    
    p_value = hypergeom.sf(k_obs - 1, N, K, n)
    L = int(hypergeom.ppf(alpha/2, N, K, n))
    U = int(hypergeom.isf(alpha/2, N, K, n))
    
    return p_value, (L, U), (L / n, U / n)

def lift_ci(df: pd.DataFrame, correct_demographic: str, k: Union[int, float], alpha: float = 0.05) -> Tuple[float, float, float]:
    """Calculate confidence interval for lift@k using hypergeometric distribution."""
    n = _resolve_k(df, k)
    N = len(df)
    
    rel = (df['demographic'] == correct_demographic).astype(int)
    K = rel.sum()
    overall_proportion = K / float(N)
    
    if K == 0 or n <= 0 or overall_proportion == 0:
        return 0.0, 0.0, 0.0
    
    pval, (L, U), _ = hypergeometric_significance_test(df, correct_demographic, k, alpha)
    lower_bound_lift = (L / n) / overall_proportion
    upper_bound_lift = (U / n) / overall_proportion
    
    return pval, lower_bound_lift, upper_bound_lift

# ==============================================================================
# Keyword Similarity (SubspaceBERTScore)
# ==============================================================================

def compute_keyword_similarity(set1: list, set2: list, device: str = None) -> dict:
    """
    Computes precision, recall, and F-score similarity metrics between two keyword sets.
    Mirrors the subspace-based BERT scoring logic handling keyword lists.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    print(f"Initializing BERT model on {device}...")
    scorer = SubspaceBERTScore(device=device, model_name_or_path='bert-base-uncased')
    
    sentence_1 = [", ".join(set1)]
    sentence_2 = [", ".join(set2)]
    
    scores = scorer(sentence_1, sentence_2)
    
    return {
        'Precision': scores[0].item(),
        'Recall': scores[1].item(),
        'F-Score': scores[2].item()
    }

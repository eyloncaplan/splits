import os
import sys
import json
import hashlib
import random
import pandas as pd
from typing import Union, Tuple

from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import Document
from pyserini.analysis import get_lucene_analyzer
from pyserini.pyclass import autoclass
from scipy.stats import hypergeom

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

def _resolve_k(df, k):
    """Convert float percentages to absolute k or return k as an int."""
    if isinstance(k, float) and 0.0 < k <= 1.0:
        return int(len(df) * k)
    return int(k)

def precision_at_k(df: pd.DataFrame, correct_demographic: str, k: Union[int, float]) -> float:
    rel = (df['demographic'] == correct_demographic).astype(int)
    k_abs = _resolve_k(df, k)
    if k_abs <= 0:
        return 0.0
    return rel.iloc[:k_abs].sum() / float(k_abs)

def lift_at_k(df: pd.DataFrame, correct_demographic: str, k: Union[int, float]) -> float:
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

def query_bm25_index(index_path: str, keywords: list, doc_count: int = 1000) -> pd.DataFrame:
    searcher = LuceneSearcher(index_path)
    
    analyzer = get_lucene_analyzer(
        language='hgf_tokenizer',
        huggingFaceTokenizer='bert-base-uncased'  
    )
    
    query_string = " OR ".join([f'"{kw}"' for kw in keywords])
    phrase_q = get_standard_query(query_string, analyzer=analyzer)
    hits = searcher.search(phrase_q, doc_count)
    
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

    if len(results) < doc_count:
        needed = doc_count - len(results)
        total = searcher.num_docs

        pool = []
        for docnum in range(total):
            lucene_doc = searcher.doc(docnum)
            doc = Document(lucene_doc)
            jd = json.loads(doc.raw())
            ext_id = jd.get("id")
            if ext_id not in returned_ext_ids:
                pool.append(docnum)

        md5 = hashlib.md5(query_string.encode("utf-8")).hexdigest()
        seed = int(md5, 16) % 2**32
        rng = random.Random(seed)
        rng.shuffle(pool)

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

def main():
    index_name = 'Video_Games_-General_Discussion-_black-jewish'
    index_path = f'/homes/ecaplan/gscratch1/bm25_indexes/gt_reranking_sets/{index_name}'
    target_demo = 'black'
    keywords = ['the', 'is', 'we', 'try', 'and', 'so', 'to', 'with', 'for']

    print("Running Search...")
    df_results = query_bm25_index(index_path, keywords, doc_count=2000)
    print(f"Total retrieved docs: {len(df_results)}")

    lift_0_5_percent = lift_at_k(df_results, target_demo, k=0.05)
    pval_0_5_percent, ci_lower_0_5_percent, ci_upper_0_5_percent = lift_ci(df_results, target_demo, k=0.05)
    print(f"Lift@0.5%: {lift_0_5_percent:.3f} (p-value: {pval_0_5_percent:.4f}, 95% CI: [{ci_lower_0_5_percent:.3f}, {ci_upper_0_5_percent:.3f}])")

if __name__ == "__main__":
    main()

import sys
import torch
from subspace.tool import SubspaceBERTScore

def compute_keyword_similarity(set1: list, set2: list, device: str = None) -> dict:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    print(f"Initializing SubspaceBERTScore model on {device}...")
    
    scorer = SubspaceBERTScore(device=device, model_name_or_path='bert-base-uncased')
    
    sentence_1 = [", ".join(set1)]
    sentence_2 = [", ".join(set2)]
    
    scores = scorer(sentence_1, sentence_2)
    
    return {
        'Precision': scores[0].item(),
        'Recall': scores[1].item(),
        'F-Score': scores[2].item()
    }

def main():
    seed_words = ['church', 'jesus', 'christ', 'prayer']
    generated_words = ['prayer', 'worship', 'bible', 'pastor']
    
    print(f"Comparing Sets:")
    print(f"  Set 1: {seed_words}")
    print(f"  Set 2: {generated_words}\n")
    
    metrics = compute_keyword_similarity(seed_words, generated_words)
    
    print("\nMetrics:")
    for metric, score in metrics.items():
        print(f"  {metric}: {score:.4f}")

if __name__ == "__main__":
    main()
import pickle
import faiss
import numpy as np
from tqdm import tqdm
from datasets import load_dataset


# Load embeddings from pickle files
def load_embeddings(file_path):
    with open(file_path, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings

evidence_dataset = load_dataset("allenai/scifact", "corpus", trust_remote_code=True)['train']
claims_dataset = load_dataset("allenai/scifact", "claims", trust_remote_code=True)['train']

# Load evidence and claim embeddings
evidence_embeddings = load_embeddings("dataset/scifact_evidence_embeddings.pkl")
claim_embeddings = load_embeddings("dataset/scifact_claim_embeddings.pkl")

# Extract embeddings and convert to FAISS-friendly format
evidence_ids = list(evidence_embeddings.keys())  # List of evidence document IDs
evidence_vectors = np.array([embedding for _, embedding in evidence_embeddings.items()]).astype(np.float32)

# Initialize FAISS index using inner product (for cosine similarity)
dimension = evidence_vectors.shape[1]  # Dimensionality of embeddings
index = faiss.IndexFlatL2(dimension)  # Using inner product (cosine similarity)
index.add(evidence_vectors)  # Add evidence embeddings to the FAISS index

def precision_at_k(relevant_at_k):
    """Compute precision at each relevant document."""
    precisions = []
    relevant_count = 0
    for i, is_relevant in enumerate(relevant_at_k, 1):
        if is_relevant:
            relevant_count += 1
            precisions.append(relevant_count / i)
    return precisions


def mean_average_precision(all_relevances):
    """Calculate MAP from a list of relevance scores for multiple queries."""
    ap_scores = []
    for relevances in all_relevances:
        precisions = precision_at_k(relevances)
        if precisions:
            ap_scores.append(sum(precisions) / len(relevances))
        else:
            ap_scores.append(0.0)
    return sum(ap_scores) / len(ap_scores) if ap_scores else 0.0


def mean_reciprocal_rank(all_relevances):
    """Calculate MRR from a list of relevance scores for multiple queries."""
    rr_scores = []
    for relevances in all_relevances:
        for i, is_relevant in enumerate(relevances, 1):
            if is_relevant:
                rr_scores.append(1.0 / i)
                break
        else:
            rr_scores.append(0.0)
    return sum(rr_scores) / len(rr_scores) if rr_scores else 0.0


# Function to calculate MAP and MRR
def evaluate_ir_system(claim_embeddings, evidence_ids, index, top_k=10):
    all_true_relevance = []  # For storing relevance judgments (binary relevance)
    all_scores = []  # For storing similarity scores

    # Loop through each claim and retrieve top-k evidence
    for claim_id, claim_embedding in tqdm(claim_embeddings.items()):
        claim_vector = np.array([claim_embedding]).astype(np.float32)

        # Perform search on the FAISS index to retrieve top-k nearest neighbors
        distances, indices = index.search(claim_vector, top_k)

        retrieval_ids = [evidence_ids[i][0] for i in indices[0]]
        claim_data_entry = claims_dataset.filter(lambda example: example['id'] == claim_id[0])
        assert isinstance(claim_data_entry['cited_doc_ids'][0], list)
        cited_doc_ids = claim_data_entry['cited_doc_ids'][0]
        true_relevance = [1 if r in cited_doc_ids else 0 for r in retrieval_ids]

        if any(true_relevance):
            print('find one')

        # Store true relevance and distances for MAP and MRR calculation
        all_true_relevance.append(true_relevance)

        # Calculate MAP (Mean Average Precision)
    map_score = mean_average_precision(all_true_relevance)

    # Calculate MRR (Mean Reciprocal Rank)
    mrr_score = mean_reciprocal_rank(all_true_relevance)

    return map_score, mrr_score


# Evaluate the IR system with claim embeddings and evidence
map_score, mrr_score = evaluate_ir_system(claim_embeddings, evidence_ids, index, top_k=20)

# Output MAP and MRR scores
print(f"MAP (Mean Average Precision): {map_score}")
print(f"MRR (Mean Reciprocal Rank): {mrr_score}")

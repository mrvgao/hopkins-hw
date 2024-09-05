from elasticsearch import Elasticsearch
import pickle
from datasets import load_dataset
from embedding_based_ir import mean_average_precision, mean_reciprocal_rank
from embedding_based_ir import claims_dataset, evidence_embeddings, claim_embeddings


es = Elasticsearch("http://localhost:9200")
if not es.ping():
    raise ValueError("Connection failed")

for (eid, estring), _ in evidence_embeddings.items():
    es.index(index="scifact", id=eid, body={"text": estring})

all_true_relevance = []  # For storing relevance judgments (binary relevance)
for (cid, cstring), _ in claim_embeddings.items():
    query = cstring
    top_k = 20
    response = es.search(index="scifact", body={"query": {"match": {"text": query}}, 'size': top_k})

    retrieval_ids = [hit['_id'] for hit in response['hits']['hits']]
    entry = claims_dataset.filter(lambda example: example['id'] == cid)
    cited_doc_ids = entry['cited_doc_ids'][0]
    true_relevance = [1 if r in cited_doc_ids else 0 for r in retrieval_ids]

    if any(true_relevance):
        print('find one')

        # Store true relevance and distances for MAP and MRR calculation
    all_true_relevance.append(true_relevance)

# Calculate MAP (Mean Average Precision)
map_score = mean_average_precision(all_true_relevance)

# Calculate MRR (Mean Reciprocal Rank)
mrr_score = mean_reciprocal_rank(all_true_relevance)



## building
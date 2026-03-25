# Movie Recommender System

A full recommender system pipeline built on the MovieLens 1M dataset, implementing and comparing collaborative filtering, matrix factorization, content-based filtering, and a hybrid model.

Built as a portfolio project to explore the progression from naive CF to matrix factorization, with a focus on understanding *why* each approach is needed rather than just applying libraries.

---

## Models Implemented

| Model | Approach | RMSE |
|---|---|---|
| MF-SGD | Matrix factorization via stochastic gradient descent | **0.9539** |
| MF-SVD | Truncated SVD via scipy | 2.1040 |
| UserBasedCF | Cosine similarity over user rating vectors | 3.0173 |
| ItemBasedCF | Adjusted cosine similarity over item vectors | 3.5337 |
| ContentModel | Genre + tag genome feature vectors | — |
| Hybrid | Weighted blend of MF-SVD and content model | — |

MF-SGD achieves a 73% RMSE improvement over the worst-performing model (ItemBasedCF).
Content and hybrid models are excluded from RMSE comparison as they operate on a different interface.



## Setup

**Requirements:** Python 3.11+

```bash
git clone https://github.com/your-username/movie-recommender.git
cd movie-recommender
pip install -r requirements.txt
```

---

## How to Run

**Run the full data pipeline:**

```bash
python src/pipeline.py
```

This downloads MovieLens 1M automatically, encodes IDs, performs a time-based train/test split, builds the sparse user-item matrix, and saves all artifacts to `data/processed/`.

**Run all tests:**

```bash
python -m pytest tests/ -v
```

**Run the notebooks:**

```bash
jupyter notebook notebooks/
```

Run them in order: `01_eda` → `02_cf_scratch` → `03_matrix_factorization` → `04_evaluation`.

---

## Key Design Decisions

**Time-based train/test split** — ratings are split chronologically rather than randomly. Random splitting leaks future ratings into training and inflates evaluation metrics. The split boundary is December 2000, giving an 80/20 ratio.

**Sparse matrix representation** — the user-item matrix is stored as a `scipy.csr_matrix` rather than a dense NumPy array. At 96.4% sparsity, a dense matrix would use ~192MB vs a few MB for sparse storage. This becomes critical when scaling to 25M.

**ID encoding** — raw MovieLens IDs have gaps and are non-contiguous. Both user and movie IDs are remapped to clean 0-based integer indices before matrix construction. Encoder and decoder dictionaries are persisted so trained models can map back to real IDs.

**MF-SGD bias terms** — predicted ratings include a global mean, per-user bias, and per-item bias in addition to the latent factor dot product. Without these, latent factors waste capacity modeling simple rating-scale offsets rather than genuine taste patterns.

**Content model genome integration** — the tag genome (1,128 relevance scores per movie) is sourced from MovieLens 25M and joined to the 1M movie set by movieId. Genre and genome features are L2-normalized before computing cosine similarity, so the dot product directly gives the similarity score.

---

## Limitations

- RMSE measures rating prediction accuracy, not recommendation quality. A model with low RMSE can still recommend obvious or uninteresting movies.
- CF models were evaluated on a 2,000-rating sample of the test set due to the O(n) cost per prediction. MF models were evaluated on the full test set.
- 640 users in the test set have no training ratings (cold start). All CF and MF models return empty or random recommendations for these users. The content model partially addresses this.
- The hybrid model cannot be directly evaluated with RMSE since it returns raw movieIds rather than rating predictions.

---

## Next Steps

- Add Precision@K and NDCG@K metrics for ranking quality evaluation
- Scale the full pipeline to MovieLens 25M
- Implement a cascade hybrid that routes cold start users to the content model
- Fine-tune SGD hyperparameters (k, lr, reg) using a validation split

---

## Dataset

[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) — 1,000,209 ratings from 6,040 users across 3,706 movies.
[MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) — used for tag genome scores only.

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems, 5(4):19.

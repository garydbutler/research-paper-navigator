# Checkpoint 4.1: Advanced retrieval

## What changed

The chunking, embeddings, hybrid search and top 6 chunks are the same as 3.1, so the comparison stays fair. The retrieval pipeline now has these extra steps.

- **Query decomposition.** A model call splits the question into up to 4 sub-queries (one per paper for comparisons), and the original question is still searched as well. The RRF scores are added across the queries.
- **Duplicate removal.** 26 papers are in the corpus as both a preprint and a published copy. Passages that are the same in both copies are removed, while passages that differ are kept.
- **Paper graph.** A networkx graph links each paper to its title, first author, topics and the papers it cites. The topics come from one model call per paper and are merged when their embeddings are at least 0.82 similar. Personalized PageRank over the graph adds the best chunk from up to 6 related papers.
- **Context filter.** A model call ranks all the candidate chunks and keeps the best 6, labelled PRIMARY (from search) or CONTEXT (from the graph).
- **Evaluation.** The judge now gives a short reason with its verdict, quote matching treats all quote mark styles the same, and I dropped the closed-book metric and added a count of distinct papers in the context.
- `--variants` now runs only the 24 rewordings.

## Setup

- New packages: `networkx` and `numpy` (both in `requirements.txt`).
- The paper metadata (`paper_metadata.json`) and topic merges (`paper_topics.json`) are cached in this folder, so the 153 metadata calls do not run again. If `chroma_chunks/` is empty, the first run builds it the same way as 2.1.

```
cd Capstone\checkpoint-4.1
..\..\.venv\Scripts\python.exe capstone_checkpoint_4_1_advanced_retrieval_complete.py [--variants]
```

## Folder structure

```
Capstone/checkpoint-4.1/
├── capstone_checkpoint_4_1_advanced_retrieval_complete.py .. advanced pipeline and evaluation
├── eval_questions.json ..................................... 12 test questions (same set as 3.1)
├── eval_questions_variants.json ............................ 24 rewordings
├── checkpoint_4_1_responses.log ............................ results of the last run (written by a run, not in git)
├── paper_metadata.json ..................................... title, authors, year, venue and topics per paper
├── paper_topics.json ....................................... topic merges
└── chroma_chunks/ .......................................... vector store (not in git)
```

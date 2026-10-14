# Checkpoint 2.1: Baseline retrieval

## What changed

I added retrieval over the real 153 PDFs. The model now answers only from the retrieved excerpts and cites each one as `[filename, p.N]`.

- Each PDF page is split into chunks (smaller segments of text) of 1200 characters with a 200 character overlap, so every chunk keeps its file name and page number. Pages under 100 characters are dropped.
- Retrieval is a hybrid of BM25 keyword search and vector search (embeddings from `openai/text-embedding-3-small`, stored in Chroma). The two ranked lists are combined with Reciprocal Rank Fusion (RRF, k = 60), and the top 6 chunks go to the model. I chose hybrid because the 1.1 failures were exact-string problems, which keyword search handles well, while comparisons across papers need search by meaning.
- The answer prompt requires a citation on every claim, exact quotes in quotation marks, and a clear statement when the excerpts do not contain the answer.
- The script runs 8 representative queries and logs the retrieved chunks and the answer for each one.

## Setup

- New packages: `langchain-chroma`, `langchain-text-splitters`, `rank-bm25` and `pypdf` (all in `requirements.txt`).
- Copy the 153 PDFs from the course `CapstoneDatasets/ResearchPapers` folder into `data/ResearchPapers/` in the repository root. Git ignores this folder.
- The first run reads all of the PDFs and embeds about 20,000 chunks into `chroma_chunks/`, which takes some time and costs money. Later runs load the existing store.

```
cd Capstone\checkpoint-2.1
..\..\.venv\Scripts\python.exe capstone_checkpoint_2_1_baseline_retrieval_complete.py
```

## Folder structure

```
research_paper_nav/
├── data/
│   └── ResearchPapers/ ......................................... the 153 PDFs (not in git)
└── Capstone/
    └── checkpoint-2.1/
        ├── capstone_checkpoint_2_1_baseline_retrieval_complete.py ... hybrid retriever and the 8 queries
        ├── checkpoint_2_1_retrieval.log ............................. results of the runs (written by a run, not in git)
        └── chroma_chunks/ ........................................... vector store (built on first run, not in git)
```

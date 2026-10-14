# Checkpoint 5.1: Agent-based RAG

## What changed

I replaced the fixed 4.1 pipeline with an agent built in LangGraph (a library that runs the system as a graph of steps). Instead of always searching once and then answering, a planner model decides the next step until it has evidence for every part of the question or reaches 4 retrieval steps.

- The agent starts with a hybrid search on the question. After that, the planner can choose `search`, `search_in_paper` (search inside one paper), `related_papers` (papers linked in the 4.1 graph), `clarify` or `answer`.
- The planner replies with a checklist of the parts of the question, each marked with a file and page or with MISSING, and it can only answer when nothing is MISSING. It also writes short, targeted queries instead of reusing the whole question.
- Before answering, a `versions` step reads the other copy (preprint or published) of each paper in the context and adds any passages that differ.
- Decomposition, PageRank and the context filter from 4.1 are no longer used. The hybrid search, search inside one paper, graph links and duplicate removal are kept.
- The script compares the agent against a fixed search-then-answer pipeline on the 12 test questions. It also takes a single question as an argument.

## Setup

- New package: `langgraph` (in `requirements.txt`).
- Optional new key in `.env`: `MIT_OPENROUTER_API_KEY` is used for embeddings when it is present.
- If `chroma_chunks/` is empty, the first run builds it the same way as 2.1.

```
cd Capstone\checkpoint-5.1
..\..\.venv\Scripts\python.exe capstone_checkpoint_5_1_agent_rag_COMPLETE.py ["your question"]
```

## Folder structure

```
Capstone/checkpoint-5.1/
├── capstone_checkpoint_5_1_agent_rag_COMPLETE.py ... LangGraph agent and the comparison
├── checkpoint_5_1_initial.log ...................... results of every run (written by a run, not in git)
├── paper_metadata.json ............................. paper metadata from 4.1
└── chroma_chunks/ .................................. vector store (not in git)
```

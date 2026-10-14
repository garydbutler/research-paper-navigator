# Checkpoint 7.1: Production safeguards and optimizations

## What changed

I added safeguards and cost optimizations to the 6.1 agent, based on the Lab 7.1 and Lab 7.2 methods. Each new layer has an on/off switch in a `SETTINGS` dictionary, so the same script can run as the 6.1 agent, with only the safeguards, or with everything turned on for a before and after comparison.

Safeguards:

- **Approved paper allowlist.** `approved_papers.txt` lists the 153 approved PDFs, and code drops any passage from a file that is not on the list. This answers the poisoned PDF attack from 6.1.
- **XML boundaries.** Passages go inside `<documents>` tags and the question inside `<user_question>` tags, with the passage text escaped, so a forged excerpt cannot pass as a real one.
- **Input sanitizer.** Regular expressions remove forged excerpt labels and "ignore your instructions" phrases, then `gpt-5.4-nano` checks for persona instructions.
- **Length check.** An empty question or one over 2,000 characters gets a fixed message before any model call.

Cost optimizations:

- **Router.** `gpt-5-nano` sends simple questions to a direct path (one search and one answer), and a second check passes the question on to the full agent if the direct answer is not good enough.
- **Semantic cache.** Earlier questions and answers are stored in `semantic_cache/`, and a new question that is at least 0.85 similar can reuse the stored answer once `gpt-5-nano` confirms both questions ask the same thing.
- **Planner snippets.** The planner sees 500 of the 1200 characters of each passage, while the answer step still sees the full text.

## Setup

- `chromadb` is now used directly for the cache. It is installed with `langchain-chroma` and is in `requirements.txt`.
- The test questions, the vector store (`chroma_chunks/`, a copy of the 5.1 store) and `paper_metadata.json` are all in this folder. If `chroma_chunks/` is empty, the first run builds it from the PDFs the same way as 2.1.
- If `approved_papers.txt` is missing, the first run writes it from `paper_metadata.json`. The cache folder is created on the first run, and git ignores it.

```
cd Capstone\checkpoint-7.1
..\..\.venv\Scripts\python.exe capstone_checkpoint_7_1_production_ready_complete.py [--cost-only | --attacks-only] [--all-models] [--expensive]
```

## Folder structure

```
Capstone/checkpoint-7.1/
├── capstone_checkpoint_7_1_production_ready_complete.py ... 6.1 agent plus the safeguards and optimizations
├── approved_papers.txt .................................... allowlist of the 153 approved PDFs
├── eval_questions.json .................................... 12 test questions used in this checkpoint
├── eval_questions_variants.json ........................... 24 rewordings used for the repeat traffic run
├── checkpoint_7_1_run.txt ................................. console output of the 2026-10-04 run (written by a run, not in git)
├── checkpoint_7_1_agent.log ............................... every answer of that run, with its settings and path (written by a run, not in git)
├── checkpoint_7_1_results.json ............................ cost summaries and attack results of that run (written by a run, not in git)
├── checkpoint_7_1_agent_2026-10-10.log .................... log of a later rerun (scores and costs moved by run-to-run variation) (written by a run, not in git)
├── checkpoint_7_1_results_2026-10-10.json ................. results of that rerun (written by a run, not in git)
├── paper_metadata.json .................................... paper metadata from 4.1
├── chroma_chunks/ ......................................... vector store (not in git)
└── semantic_cache/ ........................................ cache (built on first run, not in git)
```

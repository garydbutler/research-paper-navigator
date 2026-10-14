# Research Paper Navigator

This is my capstone project for the MIT xPRO course on retrieval-augmented generation (RAG) and context engineering. Research Paper Navigator answers questions about a fixed collection of 153 computer science research papers. Every claim in an answer cites the file and page it came from, so the user can check it, and when the papers do not contain the answer the system says so instead of guessing.

The final system is in [`Capstone/capstone-8.1/`](Capstone/capstone-8.1/). The earlier checkpoint folders show how it was built, one step at a time.

## How the final system works

1. **Checks before any model call.** Code rejects a question over 2,000 characters, and a regex sanitizer plus a small filter model remove injected instructions and persona requests.
2. **Semantic cache.** A question that was asked before gets the stored answer back, after Jev confirms the two questions ask the same thing.
3. **Router.** Simple lookups take a direct path (one search, one answer). Everything else, and any direct answer that Jev finds incomplete, goes to a LangGraph agent whose planner searches until every part of the question has evidence.
4. **Hybrid retrieval.** Every search combines BM25 keyword search and vector search over 1,200 character chunks, plus one chunk for each table row, joined by reciprocal rank fusion (RRF). Only passages from the 153 approved papers pass the allowlist.
5. **Checks after the answer.** Code checks that each cited page was retrieved, and Jev checks that the page supports the claim. Only answers that pass are cached.

Jev is a small classifier from TypeSafe. It does not write text. It returns a probability for each small yes or no decision (for example, does this passage answer this part of the question), and my code decides with a threshold. If Jev is not available, each of those decisions falls back to the 7.1 behaviour.

## Results

From the run of 2026-10-11, on the 12 test questions. The run's output files are not in this repository; running the evaluation below writes them again.

| Setup | Correct (gpt-5.4-mini judge) | Correct (gpt-5.4 judge) | USD per 1,000 questions | Seconds per question |
|---|---|---|---|---|
| 7.1 production | 10 of 12 | 9 of 12 | 5.73 | 7.8 |
| 8.1 complete | 12 of 12 | 11 of 12 | 10.07 | 10.7 |
| 8.1 complete, 26 reworded and repeated questions, warm cache | 23 of 26 | 23 of 26 | 5.85 | 5.8 |

In the replay of the checkpoint 6.1 attacks, turns with signs of a successful attack went from 4 of 26 with no safeguards to 0 of 26.

## Setup

These steps were tested on Windows 11 with PowerShell.

1. **Python environment.** Create a Python 3.11 virtual environment in the repository root and install the packages. The list also holds packages used by the course labs (for example `docling` and `torch`), so the install takes a while.

   ```
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. **Keys.** Create a `.env` file in the repository root. Git ignores this file.

   ```
   OPENROUTER_API_KEY=<your key>
   MIT_OPENROUTER_API_KEY=<the course key>
   TYPESAFE_API_KEY=<your TypeSafe key>
   ```

   | Variable | Needed | Used for |
   |---|---|---|
   | `OPENROUTER_API_KEY` | Yes | Every model call (`openai/gpt-5.4-mini` and the small helper models) |
   | `MIT_OPENROUTER_API_KEY` | Recommended | The embeddings (`openai/text-embedding-3-small`). If it is not set, the normal key is used |
   | `TYPESAFE_API_KEY` | Optional | Jev. Without it, every Jev step uses the 7.1 behaviour and the run says so |

3. **Papers.** Copy the 153 PDFs from the course's Research Paper Navigator dataset (the `CapstoneDatasets/ResearchPapers` folder) into `data/ResearchPapers/`. They are about 500 MB and are not in this repository, because the papers belong to their publishers. `approved_papers.txt` in the 8.1 folder lists all 153 file names.

4. **First run.** The vector stores are not in git, so the first run builds them. It embeds the 20,285 text chunks and the 1,315 table chunks, which takes a while and makes embedding calls. The tables found in the PDFs are already in `paper_tables.json`, so the table scan does not run again. Later runs load the stores from disk.

## Run the final system

Run the commands from the 8.1 folder, since the scripts read and write files next to themselves.

```
cd Capstone\capstone-8.1
..\..\.venv\Scripts\streamlit.exe run app.py
```

This opens the chat app at http://localhost:8501. The sidebar has example questions, and each answer lists its sources with the page, the passage used and Jev's verdict on the citation.

```
..\..\.venv\Scripts\python.exe capstone_checkpoint_8_1_improved.py > checkpoint_8_1_run.txt
```

This runs the full evaluation: the 12 test questions on the 7.1 setup and on the 8.1 setups, the 24 rewordings with a warm cache, and the attack replay. The options, and the files each run writes, are described in the [8.1 README](Capstone/capstone-8.1/README.md). A full run makes real API calls and costs money.

## Repository layout

| Folder | What it adds |
|---|---|
| [`checkpoint-1.1`](Capstone/checkpoint-1.1/) | Baseline: a plain LLM with no retrieval, which invented quotes and definitions |
| [`checkpoint-2.1`](Capstone/checkpoint-2.1/) | Hybrid retrieval: BM25 plus vector search, joined by RRF |
| [`checkpoint-3.1`](Capstone/checkpoint-3.1/) | Evaluation: 12 test questions, 24 rewordings, an LLM judge and rule checks in code |
| [`checkpoint-4.1`](Capstone/checkpoint-4.1/) | One search per paper in a question, and a graph of related papers |
| [`checkpoint-5.1`](Capstone/checkpoint-5.1/) | A LangGraph agent with a planner and three read-only tools |
| [`checkpoint-6.1`](Capstone/checkpoint-6.1/) | Security and cost audit: attacks on the agent and a cost comparison |
| [`checkpoint-7.1`](Capstone/checkpoint-7.1/) | Production safeguards: length check, sanitizer, allowlist, trust boundaries in the prompts, semantic cache and router |
| [`capstone-8.1`](Capstone/capstone-8.1/) | Final system: table-row chunks, an evidence check in code, Jev, and the chat app |
| `data/` | The shared test questions, and `ResearchPapers/` for the 153 PDFs (not in git, see Setup) |

Each checkpoint folder has its own README with what changed, its setup and how to run it.

## Known limitations

- **Text only.** pypdf reads each page as plain text. There is no OCR and figures are ignored. A page with less than 100 characters is skipped, so a scanned PDF would produce no chunks, and nothing warns about it.
- **Tables are indexed twice.** A table is in the text chunks as flattened text and again as row chunks, and retrieval treats both the same. Only one test question needs a table.
- **Single runs.** Each result is one run. The judge and the planner both vary, and scores move by 1 or 2 between runs.
- **Cost.** The complete system costs about 76% more per new question than 7.1. Jev made 888 calls for the 12 questions, and its thresholds were set by hand, not tuned.
- **Cache.** The cache can return an answer after a paper changes. Entries have no time limit; only the oldest is removed once the cache holds 400.
- **Guards.** The persona filter lets a question through if its model is down. Attacks in other languages, slow roleplay and a poisoned page inside an approved paper were not tested.
- **New papers.** A new PDF can't be used until it is added to `approved_papers.txt`.
- **Versions.** Two files count as versions of one paper only when their titles match, so a paper renamed between versions is not paired.

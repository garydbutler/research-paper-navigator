# Capstone 8.1: The improved system and the final presentation

## What changed

This checkpoint has two parts. The first is the design review presentation and preparation document. The second is an improved version of the 7.1 system, based on the failures in my 7.1 runs and the feedback goals from the review. The 7.1 script is not changed, so it stays the baseline for the before and after runs.

`capstone_checkpoint_8_1_improved.py` started as a copy of the 7.1 script, and every change is marked `CHECKPOINT 8.1`. Each change has an on/off switch in `SETTINGS`, so the same script runs as the 7.1 production system, as 8.1 with code changes only, or as the complete 8.1 system.

- **Table chunks.** PyMuPDF finds the tables in the PDFs (175 tables in 32 papers) and every row becomes its own chunk, with the column name next to each value. In 7.1 the computeDeriv question (L3) read the wrong column of Table 1 on every run because the table header and the rows ended up in different chunks. The table chunks are embedded with the same model as the text chunks and stored in their own Chroma store in this folder, so the text chunk store is not touched. The BM25 keyword index is built in memory over the text and table chunks together.
- **Evidence check in code.** The planner writes a checklist with MISSING parts, but in 7.1 it sometimes answered anyway. Now code checks the checklist before the answer and searches for the first missing part.
- **Jev (TypeSafe System One).** Jev answers small yes/no and choice questions with probabilities, and code makes the decision from them. It checks every passage against every part of the question, drops passages with injected instructions or off topic text, checks each cited claim against its page, routes questions, checks the direct answer and cache hits, and decides if the persona filter needs to run. Every Jev layer falls back to the 7.1 behaviour if Jev is off, and says so.
- **find_papers.** For a question that asks for a list of papers (S2 asks which papers use program synthesis for reinforcement learning), Jev reads the title, topics and summary of every paper and the agent adds the abstract of each match before its first search. In 7.1 the first search found only 2 of the 4 papers and nothing told the planner the other 2 existed.
- **Version check.** When two versions of the same paper are in context (an arXiv preprint and the conference copy) and the answer is about that one paper, the answer must cite both. If it cites only one, the model is asked once more. In 7.1 the LiveCodeBench question (V1) blended the 511 and 612 problem counts of the two versions. The check only applies to answers about one paper, so it does not ask a list answer like S2 to cite every copy.
- **Second judge.** Each answer is also graded by `openai/gpt-5.4` at temperature 0, next to the 5.1 judge (`gpt-5.4-mini` at temperature 0.2, the judge every checkpoint used). In the 7.1 runs the 5.1 judge sometimes failed answers its own grading notes allowed, so the table shows both scores.
- **Chat app.** `app.py` is a Streamlit chat app that uses the complete 8.1 system, so people can ask their own questions instead of reading the evaluation output. It is described under "Run the chat app" below.

## Setup

1. Create the Python 3.11 virtual environment in the repository root and install the packages, the same way as in [checkpoint 1.1](../checkpoint-1.1/README.md). The 8.1 packages are already in `requirements.txt`: `pymupdf` for the tables, `typesafe-sdk` for Jev, `streamlit` for the chat app, and `python-pptx` and `python-docx` for the review documents.

   ```
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. Put the keys in the `.env` file in the repository root. Git ignores this file.

   ```
   OPENROUTER_API_KEY=<your key>
   MIT_OPENROUTER_API_KEY=<the course key>
   TYPESAFE_API_KEY=<your TypeSafe key>
   ```

   `OPENROUTER_API_KEY` is required for every model call. The embeddings use `MIT_OPENROUTER_API_KEY` when it is set, and the normal key when it isn't. `TYPESAFE_API_KEY` turns on Jev. Without it the Jev layers use the 7.1 behaviour, the run says so once, and the chat app shows a warning in its sidebar. `TYPESAFE_API_KEY` can also be set as an environment variable instead of in `.env`.

3. Check the data. The 153 PDFs go in `data/ResearchPapers/` in the repository root (not in git; see the root README). Like 7.1, the text chunk store (`chroma_chunks/`, a copy of the 5.1 store) and `paper_metadata.json` are in this folder. If `chroma_chunks/` is empty, the first run builds it from the PDFs the same way as 2.1, which takes a while; copying `chroma_chunks/` from 5.1 avoids that.

4. `paper_tables.json` already holds the tables found in the PDFs (175 tables in 32 papers), so the first run only embeds the 1,315 table chunks into `table_chunks/`. If `paper_tables.json` is deleted, the next run scans every PDF page again (about 7 minutes). Later runs load both.

## Run the evaluation

Run the script from this folder, since it reads and writes its files next to itself.

```
cd Capstone\capstone-8.1
..\..\.venv\Scripts\python.exe capstone_checkpoint_8_1_improved.py > checkpoint_8_1_run.txt
```

With no options it runs both parts. The first part asks the 12 test questions with the 7.1 production setup, 8.1 with code changes only and the complete 8.1 system, then the 24 rewordings and 2 exact repeats with a warm cache, and grades every answer with both judges. The second part replays the Lab 6.1 attacks on 7.1 and the complete 8.1 system, on `gpt-5.4-mini` and `gpt-4o-mini`.

| Option | What it does |
|---|---|
| `--cost-only` | runs only the first part (scores, cost and latency) |
| `--attacks-only` | runs only the attack replay |
| `--all-models` | also replays the attacks on `qwen/qwen3-8b` and `gpt-5.4-nano` |
| `--expensive` | escalated questions go to `gpt-5.4` instead of `gpt-5.4-mini`, which costs about 3 to 10 times as much |

The run writes three files:

- `checkpoint_8_1_run.txt`: what the script prints, including the cost summary of each setup (only when the output is redirected as above).
- `checkpoint_8_1_agent.log`: every question, answer, judge verdict and Jev decision.
- `checkpoint_8_1_results.json`: the numbers, one entry for each setup and each attack turn.

Each measured run empties `semantic_cache/` first, so a warm cache from an earlier run can't make the numbers look better than they are. This also removes any answers the chat app had cached.

## Run the chat app

```
cd Capstone\capstone-8.1
..\..\.venv\Scripts\streamlit.exe run app.py
```

Streamlit opens the app at http://localhost:8501. The first start takes about a minute while it loads the 20,285 text chunks and the 1,315 table chunks. To stop it, press Ctrl+C in the terminal. Streamlit reads `app.py` again on every page run, but it loads `capstone_checkpoint_8_1_improved.py` only once, so restart the app after changing the 8.1 script.

- **Sidebar.** It shows whether Jev is on, a New chat button, and four example questions to click: a lookup (Fusion 360), a table question (computeDeriv), a comparison of two papers (StreamBit and the sketching overview), and a paper with two versions (LiveCodeBench).
- **Answers.** Citations show as numbers like [1]. The Sources list under each answer gives the paper title, the page, the file name, the passage the answer used, and Jev's verdict on the citation (supported, contradicted, or the page does not state it).
- **Details line.** Under each answer it shows the path the question took (direct, agent, direct->agent, cache-hit or rejected), the number of model calls and Jev calls, and the time. If the safety filter removed part of the question, the line says so.
- **Cached answers.** The cache is only used for the first question of a chat. A cached answer has a "Cached answer" note above it. The cache keeps only the answer text, so its sources show the whole cited page and have no Jev verdict.
- **Follow-up questions.** Every later question in the same chat is answered fresh, because a follow-up can depend on the earlier turns. Its details line ends with "cache not used: follow-up question in this chat". Click New chat to start over.

The app points the script's log at `chat_app.log` (not in git), so chats never end up in the evaluation log. A normal answer writes nothing there, so the file usually does not exist.

## Results

From the run of 2026-10-11. The run's output files are not in this repository; running the evaluation writes them again. The scores are out of 12 for the cold runs and out of 26 for the warm cache run.

| Setup | gpt-5.4-mini judge | gpt-5.4 judge | Model calls | Jev calls | USD per 1,000 questions | Seconds per question |
|---|---|---|---|---|---|---|
| 7.1 production | 10 | 9 | 54 | 0 | 5.73 | 7.8 |
| 8.1 code only | 11 | 10 | 56 | 0 | 6.56 | 7.7 |
| 8.1 complete (code + Jev) | 12 | 11 | 37 | 888 | 10.07 | 10.7 |
| 8.1 complete, reworded, warm cache | 23 of 26 | 23 of 26 | 46 | 1,299 | 5.85 | 5.8 |

The complete system made fewer model calls but cost more, because `gpt-5.4-mini` read more text in each call, mostly in the planner (the paper abstracts from find_papers, the table rows and the extra searches). Jev is about $1.75 of the $10.07. In the attack replay, 7.1 had 1 turn with attack signs on `gpt-4o-mini` and the complete 8.1 system had 0. These are single runs, and the scores move by 1 or 2 between runs.

## The presentation

The recorded presentation is shared separately as a video link. This repository holds only the code, the files it reads and the documentation, so the slides, the course documents and the run outputs are not in it.

## Folder structure

```
Capstone/capstone-8.1/
├── capstone_checkpoint_8_1_improved.py ..................... 7.1 system plus the 8.1 improvements
├── app.py .................................................. Streamlit chat app
├── approved_papers.txt ..................................... allowlist of the 153 approved PDFs
├── eval_questions.json ..................................... 12 test questions
├── eval_questions_variants.json ............................ 24 rewordings
├── paper_metadata.json ..................................... paper metadata from 4.1
├── paper_tables.json ....................................... tables found in the PDFs (in git, so the scan does not run again)
├── checkpoint_8_1_run.txt .................................. output of the 2026-10-11 run (written by a run, not in git)
├── checkpoint_8_1_agent.log ................................ every answer and decision in that run (written by a run, not in git)
├── checkpoint_8_1_results.json ............................. the numbers from that run (written by a run, not in git)
├── chroma_chunks/ .......................................... text chunk store (not in git)
├── table_chunks/ ........................................... table chunk store (built on first run, not in git)
└── semantic_cache/ ......................................... cache (built on first run, not in git)
```

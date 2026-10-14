# Checkpoint 3.1: Evaluation

## What changed

The retriever and answer prompt from 2.1 are unchanged on purpose, so the numbers from this checkpoint are a fair baseline. What I added is a way to measure them.

- A test set of 12 questions in `eval_questions.json`, plus 2 rewordings of each in `eval_questions_variants.json`. The questions cover lookups, exact quotes, comparisons across papers, a paper with conflicting preprint and published versions, and 3 questions the papers cannot answer. Each one has grading notes, required points and the gold source files.
- Seven metrics. Some are rules in code (valid citations, verbatim quotes, gold files retrieved, refusals), and some use an LLM judge (a model that reads the answer and grading notes and returns pass or fail).
- A check of the evaluation itself, where a correct answer has to pass and a manipulated answer has to fail.
- A `--variants` option that also runs the 24 rewordings to see how stable the verdicts are.

## Setup

No new packages. The question files are in this folder. If `chroma_chunks/` is empty, the first run builds it the same way as 2.1.

```
cd Capstone\checkpoint-3.1
..\..\.venv\Scripts\python.exe capstone_checkpoint_3_1_evaluation_complete.py [--variants]
```

## Folder structure

```
Capstone/checkpoint-3.1/
├── capstone_checkpoint_3_1_evaluation_complete.py ........ 2.1 retriever plus the evaluation
├── eval_questions.json ................................... 12 test questions
├── eval_questions_variants.json .......................... 24 rewordings
├── checkpoint_3_1_evaluation.log ......................... results of every run (written by a run, not in git)
└── chroma_chunks/ ........................................ vector store (not in git)
```

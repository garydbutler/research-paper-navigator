# Checkpoint 6.1: Security and cost audit

## What changed

This checkpoint is an audit of the 5.1 agent, so the agent itself is mostly unchanged. The script is a copy of the 5.1 script, with each change marked `CHECKPOINT 6.1`.

- Every model call has a `max_tokens` limit of 8000. Without it, OpenRouter reserved 65k tokens per call and returned a 402 error when my balance was low.
- Token tracking (from Lab 6.2) adds up input, output and cached tokens per model and per role, and converts them to dollars.
- The planner and the answer step can use different models.
- The agent keeps the last 6 chat messages, since several of the attacks work across turns.
- A cost experiment runs 4 test questions on the fixed pipeline, the agent on two models, and a mixed setup where one model plans and another answers.
- An attack harness (from Lab 6.1) runs 6 multi-turn prompt injection scenarios on 4 models, first with the 5.1 prompts and then with hardened prompts. The scenarios include personas, a forged excerpt, a poisoned PDF that only exists in memory, and an out of scope request. Two edge cases test an empty question and a 67,000 character question.

## Setup

No new packages. The test questions, the vector store (`chroma_chunks/`, a copy of the 5.1 store) and `paper_metadata.json` are all in this folder, so the script does not read another checkpoint's folder. If `chroma_chunks/` is empty, the first run builds it from the PDFs the same way as 2.1.

```
cd Capstone\checkpoint-6.1
..\..\.venv\Scripts\python.exe capstone_checkpoint_6_1_security_cost_COMPLETE.py [--cost-only | --attacks-only] [--expensive] [--include-expensive]
```

`--expensive` and `--include-expensive` add more costly models to the cost runs and the attacks.

## Folder structure

```
Capstone/checkpoint-6.1/
├── capstone_checkpoint_6_1_security_cost_COMPLETE.py ... 5.1 agent plus the audit
├── eval_questions.json ................................. 12 test questions used in this checkpoint
├── eval_questions_variants.json ........................ 24 rewordings (not read by the 6.1 script)
├── checkpoint_6_1_v2_audit.log ......................... every cost and attack answer (written by a run, not in git)
├── checkpoint_6_1_v2_empty_input.txt ................... rerun of the empty question case (written by a run, not in git)
├── paper_metadata.json ................................. paper metadata from 4.1
└── chroma_chunks/ ...................................... vector store (not in git)
```

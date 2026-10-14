# Research Paper Navigator: Checkpoint 1.1

Research Paper Navigator is a question-answering system for a collection of 153 computer science research papers (PDF files). The goal is a system that answers questions about the papers, quotes them directly with the file and page number, and says so when the papers do not contain the answer.

I selected the Research Paper Navigator scenario over the Wikipedia Retrieval Engine because it is directly related to my current job role. The system needs to handle factual questions about a single paper, exact quotes, comparisons across papers, longer conversations, and questions that the papers cannot answer.

## Checkpoint 1.1

This checkpoint is a baseline with no retrieval. The script sends 5 of my own probe prompts to `openai/gpt-5.4-mini` (with temperature 0.2) and logs each response, and I checked the answers against the real PDFs. The model made up quotes and acronym definitions with confidence, so retrieval of the full paper text (plus the title, authors, date and page for citations) is a must for this system.

## Setup

1. Create a Python 3.11 virtual environment in the repository root and install the packages.

   ```
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

   This checkpoint only uses `langchain-openai`, `langchain-core` and `python-dotenv`.

2. Create a '.env' file in the repository root with your OpenRouter key. Git ignores this file.

   ```
   OPENROUTER_API_KEY=<your key>
   ```

3. Run the script from this folder, since it writes its log to the current folder.

   ```
   cd Capstone\checkpoint-1.1
   ..\..\.venv\Scripts\python.exe capstone_checkpoint_1_1_baseline_complete.py
   ```

## Folder structure

```
research_paper_nav/
├── .env ................................ API key goes here (not commited to git)
├── requirements.txt .................... Python packages needed
└── Capstone/
    └── checkpoint-1.1/
        ├── capstone_checkpoint_1_1_baseline_complete.py .. baseline script with 5 prompts
        └── checkpoint_1_1_responses.log .................. every prompt and response (written by a run, not in git)
```

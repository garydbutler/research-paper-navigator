r"""Capstone Checkpoint 5.1 — Designing and Evaluating an Agent-Based RAG System (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code/PyCharm/Jupytext.
"""

# %% [markdown]
# # Capstone Checkpoint 5.1 — Designing and Evaluating an Agent-Based RAG System
# **MO-LLM Module 5 -  Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# This is the final build step of your capstone. You will turn the retrieval system you've
# developed into an **agent-based RAG system**. Instead of a fixed pipeline, an agent
# decides at each step whether it has enough information, what to retrieve next, which
# tool to use, or whether to answer. This activity allows you to apply the Module 5 labs (Lab 5.1's agentic
# retriever and Lab 5.2's tool-using agent) to your capstone scenario.
#
# The graded deliverable is your completed Capstone Checkpoint 5.1 worksheet. This script is a
# runnable demonstration of an agentic loop on a tiny sample corpus so you can see the
# decision-making before adapting it to your full system.
#
# **Learning outcomes (Module 5):**
# 1. Build an agent-based system that integrates retrieval and external tools.
# 2. Use system prompts to guide agent behavior and decision-making.
# 3. Design workflows that coordinate retrieval, reasoning, and tool use in a RAG system.
# 4. Evaluate an agentic workflow against a fixed retrieval pipeline.
# 5. Design workflows that coordinate retrieval, reasoning, and tool use within a RAG system.
# 6. Build an agent-based system that integrates retrieval and external tools to complete user tasks.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario** you chose in Checkpoint 1.1 and have built on since.
#
# | Scenario | Corpus | Useful agent tools/actions |
# |---|---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs | semantic search; "find papers by author/year"; "follow citations"; ask a clarifying question |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles | semantic search; "find by category"; "follow links"; ask a clarifying question |
#
# An agent shines when one query isn't enough — it can search, look at what it found,
# and decide to search again (or use a different tool) before answering.

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12**
# 2. `pip install langchain-openai langchain-core python-dotenv`
# 3. Get a free OpenRouter key at <https://openrouter.ai/keys>. The labs use the paid
#    `openai/gpt-5.4-mini` model, which is covered by the course credits.
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# This runs on a tiny built-in sample corpus, so you do not need to prepare your own
# dataset. It still requires an OpenRouter API key to run the LLM (it is not offline or
# free of API calls). The built-in corpus is deliberately minimal: It only demonstrates
# the agent control flow, not retrieval quality, and it is unrelated to your capstone:
# Howerver, your design should target your own capstone corpus. Your full agent is what you describe
# in the written submission.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"
TEMPERATURE = 0.2
MAX_STEPS = 3
LOG_PATH = Path.cwd() / "checkpoint_5_1_agent.log"

# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "research_papers"   # "research_papers" or "wikipedia"

DECIDE_SYSTEM = (
    "You are an agent retrieving from a small document collection. Given the question, "
    "the queries already run, and the documents found so far, decide what to do next. "
    'Respond with ONLY a JSON object: {"done": true|false, "new_queries": ["..."], '
    '"reasoning": "..."}. Set done=true when you have enough to answer; otherwise give '
    "1-2 new_queries targeting what is still missing (do not repeat past queries)."
)
ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided documents, "
    "quoting where you can. If they do not contain the answer, say so."
)


# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Grab a free key at https://openrouter.ai/keys, "
            "put it in a .env file next to this script, and rerun."
        )
    return key


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE,
                      api_key=check_api_key(), base_url=OPENROUTER_BASE_URL)


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


# %% [markdown]
# ## A tiny sample corpus + a keyword retriever (provided)

# %%
SAMPLE_DOCS = [
    {"id": "d1", "text": "Program synthesis generates programs from a specification, such as input-output examples."},
    {"id": "d2", "text": "The sketching approach lets a programmer leave holes in a program for a synthesizer to fill."},
    {"id": "d3", "text": "Retrieval-augmented generation grounds a model's answers in retrieved documents to reduce hallucination."},
    {"id": "d4", "text": "An agentic retriever decides at each step whether it has enough information or should search again."},
    {"id": "d5", "text": "A tool-using agent chooses among actions — search, look up by date, ask the user — to complete a task."},
    {"id": "d6", "text": "Evaluating an agent compares its answers and cost against a fixed single-pass pipeline."},
]
DOC_BY_ID = {d["id"]: d for d in SAMPLE_DOCS}


def retrieve(query: str, k: int = 2) -> list[str]:
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored = [(d["id"], len(q & set(re.findall(r"[a-z0-9]+", d["text"].lower())))) for d in SAMPLE_DOCS]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, s in scored[:k] if s > 0]


def decide(llm: ChatOpenAI, question: str, collected: dict[str, str], executed: list[str]) -> dict:
    docs = "\n".join(f"[{i}] {DOC_BY_ID[i]['text']}" for i in collected) or "(none yet)"
    user = f"Question: {question}\n\nQueries run: {executed or '(none)'}\n\nDocuments so far:\n{docs}"
    raw = llm.invoke([SystemMessage(content=DECIDE_SYSTEM), HumanMessage(content=user)]).content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        d = json.loads(raw)
        return {"done": bool(d.get("done", True)), "new_queries": d.get("new_queries", []) or [],
                "reasoning": d.get("reasoning", "")}
    except (json.JSONDecodeError, ValueError):
        return {"done": True, "new_queries": [], "reasoning": "parse-fail -> stop"}


def agentic_answer(llm: ChatOpenAI, question: str) -> str:
    """Provided: a minimal agentic loop — retrieve, decide whether to continue, repeat."""
    collected: dict[str, str] = {}
    executed: list[str] = []
    pending = [question]
    for step in range(MAX_STEPS):
        for q in pending:
            for doc_id in retrieve(q):
                collected[doc_id] = DOC_BY_ID[doc_id]["text"]
            executed.append(q)
        d = decide(llm, question, collected, executed)
        print(f"  step {step + 1}: have {sorted(collected)}  -> done={d['done']}  ({d['reasoning'][:60]})")
        if d["done"] or not d["new_queries"]:
            break
        pending = d["new_queries"]
    context = "\n\n".join(f"[{i}] {collected[i]}" for i in collected)
    return llm.invoke([SystemMessage(content=ANSWER_SYSTEM),
                       HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {question}")]).content


# %% [markdown]
# ## Step 2 — Your agent design (TODO)
#
# Design the agent you will build for **your** scenario. Return a dictionary with the keys
# below — this is the plan you implement in your real system and describe in the report.

# %%
def my_agent_plan() -> dict[str, Any]:
    """Return your agent design for your chosen scenario.

    TODO — your turn. Return a dictionary with these keys:
      - "tools": list[str]      — the actions/tools your agent can take (e.g.,
                                  ["semantic_search", "find_by_author", "follow_citation",
                                   "clarify", "answer"]).
      - "stop_condition": str   — how the agent decides it has enough to answer.
      - "system_prompt_idea": str — one or two sentences on how you'll instruct the agent
                                  to choose actions (this is what "guiding agent behavior
                                  with system prompts" means).
      - "test_tasks": list[str] — 2-3 questions for your scenario that need more than one
                                  retrieval step (so the agentic loop earns its keep).

    Example (illustrative — replace with your own scenario):
        return {
            "tools": ["semantic_search", "find_by_author", "clarify", "answer"],
            "stop_condition": "the retrieved passages cover every part of the question",
            "system_prompt_idea": "Search first; clarify only if the request is ambiguous; "
                                  "answer once the retrieved text supports a grounded reply.",
            "test_tasks": [
                "What changed between the v1 and v2 proposals, and who approved it?",
                "Summarize the budget decisions discussed across Q1 and Q2.",
            ],
        }

    Delete the raise NotImplementedError line once your code works.
    """
    return {
        "tools": ["search", "search_in_paper", "clarify", "answer"],
        "stop_condition": (
            "the agent says it has enough, or it has used its four retrieval steps"
        ),
        "system_prompt_idea": (
            "Search the whole collection when you do not know which paper holds the "
            "answer. Search inside one paper by filename when you do. Answer once the "
            "retrieved passages support an answer."
        ),
        "test_tasks": [
            "How did the correctness reference change between the 2005 StreamBit paper "
            "and the 2009 overview of the sketching approach?",
            "How many problems does LiveCodeBench contain, and what period do they cover?",
            "How many human design sequences are in the Fusion 360 Gallery reconstruction "
            "dataset, and which two modeling operations does its language support?",
        ],
    }


# %% [markdown]
# ## Step 3 — Run the agentic loop and capture the evidence
#
# Runs the provided agentic loop on a multistep question (watch it retrieve, decide,
# and retrieve again), then prints your plan. Reproduce this in your real system for the
# report and compare it against your fixed Checkpoint 2.1 retriever on the same task.

# %%
def run() -> None:
    llm = make_llm()
    question = "How does an agentic retriever differ from a fixed pipeline, and how is it evaluated?"
    print(f"Checkpoint 5.1 — agentic RAG demo  |  scenario: {SCENARIO}")
    print(f"Question: {question}\n")
    answer = agentic_answer(llm, question)
    print(f"\nAgent answer:\n{answer}\n")
    log("AGENTIC", f"Q: {question}\nA: {answer}")
    try:
        print("Your agent plan:")
        print(json.dumps(my_agent_plan(), indent=2))
    except NotImplementedError as e:
        print(f"[my_agent_plan not done yet] {e}")
    print("=" * 72)
    print("Done. Build this agent for your real system and compare it to your 2.1 baseline.")


run()

# %% [markdown]
# ## Step 4 — Your written submission (the graded deliverable)
#
# The deliverable is a **written submission** (suggested length: 500-750 words). Cover the following:
#
# 1. **System overview**: State your scenario and the retrieval system you built (2.1–4.1).
# 2. **Agent design**: Describe the actions/tools your agent can take, and how a **system prompt**
#    guides it to choose among them (retrieve, use a tool, clarify, or answer).
# 3. **Workflow**: Explain how retrieval, reasoning, and tool use are coordinated across steps,
#    and how the agent decides it has enough information to answer.
# 4. **Evaluation**: Compare the agent-based system against your fixed Checkpoint 2.1/3.1
#    pipeline on a few representative tasks. Did multi-step decision-making improve the
#    answers? At what cost (extra model calls, latency)?
# 5. **Reflection**: When an agentic approach adds value vs. when a simpler pipeline is
#    better, and the limitations/trade-offs of increasingly autonomous systems.
#
# Include evidence (sample tasks, the agent's step-by-step decisions from a log, before/
# after comparison).


# ======================================================================================
# THE REAL SYSTEM STARTS HERE (stage 1 of the Checkpoint 5.1 build)
#
# Everything above this line is the course starter, unchanged apart from
# my_agent_plan(). It runs on the six invented sentences that ship with the starter.
#
# Everything below is the FIRST agent built for my capstone scenario: the Research
# Paper Navigator over 153 computer-science PDFs. It does two things:
#
#   1. it brings the Checkpoint 4.1 hybrid retrieval across unchanged, and
#   2. it wraps that retrieval in a LangGraph state machine shaped like Lab 5.2.
#
# This is the version the worksheet calls the initial system. It is deliberately
# simple: one planner, two retrieval tools, a clarify step and an answer step. The
# refinements come later, and they come from what this version gets wrong.
# ======================================================================================

# %% [markdown]
# ## Part A - the corpus and the retriever, ported from Checkpoint 4.1
#
# Nothing here is new work. The chunking, the embeddings, the Chroma store, the BM25
# index and the RRF fusion are the code from Checkpoints 2.1, 3.1 and 4.1. This folder has
# its own copy of the vector store, so no PDF is chunked again and no paper is embedded
# again unless the store is missing.

# %%
import sys
import time
from typing import TypedDict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
import networkx as nx
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# %%
EMBEDDING_MODEL = "openai/text-embedding-3-small"   # same model as 2.1 / 3.1 / 4.1

HERE = Path(__file__).resolve().parent

SHARED_DATA_DIR = HERE.parent.parent / "data"    # one copy of the corpus and questions for every checkpoint
PAPERS_DIR = SHARED_DATA_DIR / "ResearchPapers"
CHROMA_DIR = HERE / "chroma_chunks"              # a copy of the store 3.1 built and 4.1 reused
QUESTIONS_PATH = SHARED_DATA_DIR / "eval_questions.json"
METADATA_PATH = HERE / "paper_metadata.json"     # a copy of the cached paper titles from Checkpoint 4.1

RESULTS_PATH = HERE / "checkpoint_5_1_initial_results.json"
LOG_PATH = HERE / "checkpoint_5_1_initial.log"   # replaces the starter's log path

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MIN_PAGE_CHARS = 100
TOP_K = 6                  # the fixed pipeline's context budget, kept for the comparison
CANDIDATE_POOL = 20
RRF_K = 60


# %%
def _page_texts(reader: PdfReader) -> list[tuple[int, str]]:
    """Return (page number, cleaned text) for every page with enough text to index."""
    pages = []
    for page_no, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text()
        except Exception:
            continue                       # pypdf fails on some damaged pages; skip them
        if not text:
            text = ""
        text = re.sub(r"[ \t]+", " ", text).strip()
        if len(text) >= MIN_PAGE_CHARS:
            pages.append((page_no, text))
    return pages


def load_and_chunk(papers_dir: Path) -> list[Document]:
    """Extract text page by page, then split each page into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    files = []
    for path in papers_dir.iterdir():
        if path.suffix.lower() == ".pdf":
            files.append(path)
    files.sort()
    docs: list[Document] = []
    for i, path in enumerate(files, 1):
        try:
            reader = PdfReader(path)
        except Exception as exc:
            print(f"  WARNING: {path.name} could not be parsed: {exc}")
            continue
        for page_no, text in _page_texts(reader):
            for j, chunk in enumerate(splitter.split_text(text)):
                docs.append(Document(
                    page_content=chunk,
                    metadata={"source": path.name, "page": page_no, "chunk": j},
                ))
        print(f"  [{i:3d}/{len(files)}] {path.name}", flush=True)
    return docs


def embedding_api_key() -> str:
    """The key the embedding model is called with.

    The chat model and the embedding model are billed to different OpenRouter accounts
    here: MIT_OPENROUTER_API_KEY is the course key and is the one embeddings work on.
    When it is not set, fall back to the single key the starter expects.
    """
    load_dotenv()
    course_key = os.getenv("MIT_OPENROUTER_API_KEY")
    if course_key:
        return course_key
    return check_api_key()


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=embedding_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )


def build_or_load_db() -> Chroma:
    """Load the persisted vector store if it exists, otherwise build it from the PDFs."""
    if CHROMA_DIR.is_dir() and len(list(CHROMA_DIR.iterdir())) > 0:
        print(f"Loading existing vector DB from {CHROMA_DIR}")
        return Chroma(persist_directory=str(CHROMA_DIR), embedding_function=get_embeddings())
    if not PAPERS_DIR.is_dir():
        raise RuntimeError(f"No corpus found. Expected the research-paper PDFs in {PAPERS_DIR}.")
    print(f"Building vector DB from {PAPERS_DIR} (first run - this takes a while)")
    docs = load_and_chunk(PAPERS_DIR)
    print(f"\n  {len(docs)} chunks total. Embedding...")
    db = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=get_embeddings())
    for start in range(0, len(docs), 500):
        db.add_documents(docs[start:start + 500])
        print(f"    embedded {min(start + 500, len(docs))}/{len(docs)}", flush=True)
    return db


# %%
DB = build_or_load_db()
_stored = DB.get(include=["documents", "metadatas"])


def chunk_id_of(meta: dict) -> str:
    """The id a chunk is known by everywhere in this script and in every citation."""
    return f"{meta.get('source', '?')} p.{meta.get('page', '?')} #{meta.get('chunk', 0)}"


# Every chunk in the store, looked up by its id: {chunk_id: {"id", "text", "meta"}}
CHUNKS: dict[str, dict] = {}
_documents = _stored["documents"]
_metadatas = _stored["metadatas"]
for _row in range(len(_documents)):
    _meta = _metadatas[_row]
    _chunk_id = chunk_id_of(_meta)
    CHUNKS[_chunk_id] = {"id": _chunk_id, "text": _documents[_row], "meta": _meta}

CHUNK_IDS = list(CHUNKS.keys())   # the chunk ids, in the order the BM25 index uses
CHUNK_TEXTS = []                  # the chunk texts, in that same order
CHUNK_POSITION = {}               # {chunk_id: its row number in the BM25 index}
for _row in range(len(CHUNK_IDS)):
    _chunk_id = CHUNK_IDS[_row]
    CHUNK_TEXTS.append(CHUNKS[_chunk_id]["text"])
    CHUNK_POSITION[_chunk_id] = _row


def paper_of(chunk_id: str) -> str:
    """The filename part of a chunk id such as 'paper.pdf p.3 #1'."""
    return CHUNKS[chunk_id]["meta"]["source"]


CHUNKS_BY_PAPER: dict[str, list[str]] = {}   # {filename: the ids of its chunks}
for _chunk_id in CHUNK_IDS:
    _paper = paper_of(_chunk_id)
    if _paper not in CHUNKS_BY_PAPER:
        CHUNKS_BY_PAPER[_paper] = []
    CHUNKS_BY_PAPER[_paper].append(_chunk_id)

ALL_PAPERS = sorted(CHUNKS_BY_PAPER.keys())
print(f"Corpus ready: {len(CHUNK_IDS)} chunks from {len(ALL_PAPERS)} papers")


# %%
STOPWORDS = {
    "a", "an", "the", "and", "but", "or", "nor", "so", "yet", "for",
    "in", "on", "at", "to", "of", "by", "with", "from", "into", "onto", "upon",
    "about", "above", "below", "between", "through", "during", "before", "after",
    "under", "over", "around", "along", "across", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "i", "we", "you", "he", "she", "it", "they", "me", "us", "him", "her", "them",
    "my", "our", "your", "his", "its", "their", "this", "that", "these", "those",
    "as", "if", "up", "out", "not", "no", "what", "which", "who", "how",
}


def tokenize(text: str) -> list[str]:
    """Lower-case words and numbers from the text, without the stop words."""
    words = []
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if word not in STOPWORDS:
            words.append(word)
    return words


_tokenized_chunks = []
for _text in CHUNK_TEXTS:
    _tokenized_chunks.append(tokenize(_text))
BM25 = BM25Okapi(_tokenized_chunks)


def first_item(pair: tuple):
    """Sort helper: sort a list of pairs by the first value in each pair."""
    return pair[0]


def second_item(pair: tuple):
    """Sort helper: sort a list of pairs by the second value in each pair."""
    return pair[1]


def bm25_ranked(query: str, k: int) -> list[str]:
    """The k chunk ids with the best BM25 keyword score for the query, best first."""
    scores = BM25.get_scores(tokenize(query))
    pairs = []                                  # (score, row number) for every chunk
    for row in range(len(scores)):
        pairs.append((scores[row], row))
    pairs.sort(key=first_item, reverse=True)
    best = []
    for _score, row in pairs[:k]:
        best.append(CHUNK_IDS[row])
    return best


def vector_ranked(query: str, k: int) -> list[str]:
    """The k chunk ids closest to the query in the vector store, best first."""
    best = []
    for doc in DB.similarity_search(query, k=k):
        best.append(chunk_id_of(doc.metadata))
    return best


def fuse_by_rrf(ranked_lists: list[list[str]]) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: combine several ranked lists into one, best first.

    A chunk at rank r in a list earns 1 / (RRF_K + r). A chunk that ranks well in
    both the keyword list and the vector list earns the most.
    """
    fused = {}                                  # {chunk_id: total RRF score}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, 1):
            if chunk_id not in fused:
                fused[chunk_id] = 0.0
            fused[chunk_id] = fused[chunk_id] + 1.0 / (RRF_K + rank)
    ordered = list(fused.items())
    ordered.sort(key=second_item, reverse=True)
    return ordered


def baseline_retrieve(query: str, k: int = TOP_K) -> list[tuple[str, float]]:
    """The Checkpoint 2.1 / 3.1 retriever: BM25 + vector search fused by RRF.

    Returns (chunk_id, score) pairs, best first. This one function is both the fixed
    pipeline this agent is compared against and the agent's own search tool.
    """
    keyword_hits = bm25_ranked(query, CANDIDATE_POOL)
    vector_hits = vector_ranked(query, CANDIDATE_POOL)
    ordered = fuse_by_rrf([keyword_hits, vector_hits])
    return ordered[:k]


def retrieve_inside_paper(query: str, paper: str, k: int = 3) -> list[tuple[str, float]]:
    """The same hybrid retrieval, restricted to the chunks of one PDF (ported from 4.1)."""
    ids = CHUNKS_BY_PAPER.get(paper, [])
    if len(ids) == 0:
        return []

    # keyword ranking, over this paper's chunks only
    scores = BM25.get_scores(tokenize(query))
    pairs = []                                  # (score, chunk_id) for this paper's chunks
    for chunk_id in ids:
        pairs.append((scores[CHUNK_POSITION[chunk_id]], chunk_id))
    pairs.sort(key=first_item, reverse=True)
    keyword_hits = []
    for _score, chunk_id in pairs[:CANDIDATE_POOL]:
        keyword_hits.append(chunk_id)

    # vector ranking, filtered to this paper
    vector_hits = []
    for doc in DB.similarity_search(query, k=CANDIDATE_POOL, filter={"source": paper}):
        vector_hits.append(chunk_id_of(doc.metadata))

    ordered = fuse_by_rrf([keyword_hits, vector_hits])
    return ordered[:k]


def letters_only(text: str) -> str:
    """Lower-case letters only, ligatures expanded, so layout differences stop mattering."""
    return re.sub(r"[^a-z]", "", text.lower().replace("ﬁ", "fi").replace("ﬂ", "fl"))


def load_paper_metadata() -> dict[str, dict]:
    """Load the cached page-one paper metadata produced in Checkpoint 4.1."""
    if not METADATA_PATH.is_file():
        raise RuntimeError(f"Paper metadata not found: {METADATA_PATH}")
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


META = load_paper_metadata()


def paper_title_key(paper: str) -> str:
    """A normalized title key; shared keys represent alternate versions of one work."""
    title = META.get(paper, {}).get("title")
    if not title:
        title = ""
    return letters_only(str(title))[:40]


def build_paper_graph(papers: list[str]) -> nx.Graph:
    """The 4.1 paper graph's lightweight title, author, and topic relationships."""
    graph = nx.Graph()
    for paper in papers:
        graph.add_node(f"paper:{paper}", node_type="paper")
        title = paper_title_key(paper)
        if title:
            graph.add_edge(f"paper:{paper}", f"title:{title}", edge_type="same_title")
        author = META.get(paper, {}).get("first_author")
        if not author:
            author = ""
        author = str(author).strip()
        if author:
            graph.add_edge(f"paper:{paper}", f"author:{author}", edge_type="first_author")
        topics = META.get(paper, {}).get("topics")
        if not topics:
            topics = []
        for topic in topics:
            topic = str(topic).strip().lower()
            if topic:
                graph.add_edge(f"paper:{paper}", f"topic:{topic}", edge_type="has_topic")
    return graph


PAPER_GRAPH = build_paper_graph(ALL_PAPERS)


def related_papers(paper: str) -> dict[str, str]:
    """Return each graph-neighbor paper and its strongest relationship to `paper`."""
    node = f"paper:{paper}"
    if node not in PAPER_GRAPH:
        return {}
    priority = {"same_title": 0, "first_author": 1, "has_topic": 2}
    found: dict[str, str] = {}
    for hub in PAPER_GRAPH.neighbors(node):
        relationship = PAPER_GRAPH.edges[node, hub]["edge_type"]
        for other in PAPER_GRAPH.neighbors(hub):
            if other == node or not other.startswith("paper:"):
                continue
            filename = other.replace("paper:", "", 1)   # "paper:x.pdf" -> "x.pdf"
            previous = found.get(filename)
            if previous is None or priority[relationship] < priority[previous]:
                found[filename] = relationship
    return found


def other_versions(paper: str) -> list[str]:
    """Other files in the collection that carry the same title as `paper`.

    The collection holds a preprint and a published copy of several papers. Their
    extracted titles match once case, spaces and punctuation are removed, so this reads
    the titles in paper_metadata.json and needs no graph. A title shorter than 20
    letters is too weak to match on and is skipped.
    """
    key = paper_title_key(paper)
    if len(key) < 20:
        return []
    copies = []
    for other in ALL_PAPERS:
        if other != paper and paper_title_key(other) == key:
            copies.append(other)
    return copies


def is_duplicate_passage(chunk_id: str, already_chosen) -> bool:
    """True when this chunk's text is a passage already collected.

    The corpus holds an arXiv copy and a venue copy of 26 papers, and their shared pages
    are byte-identical. Without this the two copies fill the context with one passage
    twice. A passage that DIFFERS between the copies is not a duplicate and is kept.
    """
    mine = letters_only(CHUNKS[chunk_id]["text"])
    for other in already_chosen:
        if letters_only(CHUNKS[other]["text"]) == mine:
            return True
    return False


# %% [markdown]
# ## Part B - the agent, built in the shape of Lab 5.2
#
# Lab 5.2 has a plan node that returns a JSON action, one node per tool, a clarify node
# and an answer node, wired so every tool returns to the planner. This is the same graph
# with the lab's e-mail tools replaced by three research-paper tools:
#
# * `search` runs the hybrid retriever over all 153 papers, for when the agent does not
#   yet know which paper holds the answer;
# * `search_in_paper` runs the same retriever inside one paper the agent names, for when
#   it does.
# * `related_papers` follows a Checkpoint 4.1 graph edge to another version or a related
#   paper before retrieving passages from it.
#
# The second tool is the whole reason this is worth doing. The fixed pipeline can only
# ask "which passages look like this question". An agent can find the right paper and
# then read inside it, which is what a question comparing two papers needs.

# %%
SEARCH_K = 6                 # passages a whole-collection search adds; equal to TOP_K,
                             # so the first search gives the agent what the fixed pipeline gets
IN_PAPER_K = 3               # passages a search inside one named paper adds
MAX_RETRIEVAL_STEPS = 4      # retrieval rounds allowed before the answer is forced
MAX_QUERIES_PER_STEP = 1     # one query per round makes each planner decision observable
MAX_CONTEXT_CHUNKS = 16      # ceiling on what the answer node is allowed to read
VERSION_PASSAGES = 4         # passages of a paper whose counterpart is looked up in its other copy

PLAN_SYSTEM = """You are a research agent for a collection of 153 computer-science \
research papers. You answer questions by choosing one action at a time.

You are given the question, the actions you have already taken, the filenames represented \
in the retrieved passages, and the passages themselves. Every passage is labelled with its \
filename and page.

Respond with a JSON object ONLY:
{
  "action": "search" | "search_in_paper" | "related_papers" | "clarify" | "answer",
  "queries": ["query1"],              // one NEW query; only for search; do not repeat
  "query": "search text",             // for search_in_paper
  "paper": "exact_filename.pdf",      // for search_in_paper or related_papers
  "clarification": "question text",   // for clarify
  "checklist": [                      // ALWAYS fill this in, for every action
    {"part": "one thing the question asks for", "support": "filename p.N, or MISSING"}
  ],
  "reasoning": "one sentence"
}

The actions:
- "search": hybrid keyword and embedding search over every paper in the collection.
- "search_in_paper": the same search inside one paper you name by filename. Use it when
  you know which paper should hold the answer and want more of it.
- "related_papers": list the files connected to one paper you name by filename. It
  reads no text. Each file comes with its relation: "same_title" is another version of
  the same paper (for example a preprint and the published copy, whose details can
  differ), "first_author" is another paper by the same first author, and "has_topic"
  is a paper on a shared topic. Use it when an answer rests on one paper and you have
  not yet checked whether the collection holds another version of it, when the question
  needs more papers like one you have found, or when search keeps returning the same
  paper and you need a different one. Then read a listed file with "search_in_paper".
- "clarify": ask the user a question. Last resort, and only after searching.
- "answer": the retrieved passages support an answer.

Before you choose an action, fill in the checklist:
1. Split the question into the separate things it asks for. A comparison between two
   papers has at least three parts: what the first paper says, what the second paper
   says, and the difference between them.
2. For each part, write the label of the passage that states it, as "filename p.N".
   If no passage states it, write MISSING. A passage that only mentions the topic does
   not count.
3. Choose "answer" only when no part is MISSING. If any part is MISSING, retrieve for
   that part next.

How to write a query:
- Keep it short, about 4 to 8 words, and about ONE paper or ONE fact.
- Do not copy the question into the query. Extra words from the question pull the
  results toward whichever paper uses those words most.
- To reach a paper you have not seen yet, search with words from its likely title.
- When a passage label already shows the filename of the paper you need, use
  "search_in_paper" with that filename instead of "search".

When a question names separate papers or versions, retrieve evidence from each. Use
related_papers after finding one paper when it may help locate another relevant paper."""

AGENT_ANSWER_SYSTEM = """You are a research assistant for a collection of \
computer-science papers. Answer the question using ONLY the provided paper excerpts.
* Cite the source of every claim as [filename, p.N], using the labels on the excerpts.
* When you quote, quote verbatim from an excerpt and put it in quotation marks. Never \
reconstruct or paraphrase a quotation and present it as verbatim.
* If two excerpts come from different versions of the same paper and disagree, report \
both figures and say which file each came from.
* If the excerpts do not contain the answer, say so explicitly rather than guess. A \
refusal is the best answer when no information is available."""


# %%
class AgentState(TypedDict):
    """Everything the agent knows while it works on one question.

    LangGraph needs a list of the state's fields and their types. A TypedDict is a
    normal dictionary with those fields written down; Lab 5.2 uses the same pattern.
    """
    question: str
    next_action: str
    pending: dict            # the arguments of the action the plan node chose
    collected: dict          # {chunk_id: which tool found it}
    executed: list           # every search query already run, so none is run twice
    history: list            # one line per action, shown back to the plan node
    clarifications: list
    trace: list              # the structured record written to the results file
    steps: int
    llm_calls: int
    tool_calls: int
    answer: str
    versions_checked: bool   # the automatic other-version step runs once per question


def _excerpt_block(collected: dict[str, str], limit: int = MAX_CONTEXT_CHUNKS) -> str:
    """The retrieved passages, formatted for a prompt."""
    lines = []
    for chunk_id in list(collected)[:limit]:
        lines.append(f"[{chunk_id}] (found by {collected[chunk_id]})\n{CHUNKS[chunk_id]['text']}")
    if not lines:
        return "(nothing retrieved yet)"
    return "\n\n---\n\n".join(lines)


def papers_in(chunk_ids) -> list[str]:
    """The distinct PDF filenames among these chunk ids, sorted by name.

    Works for a list of chunk ids and for the collected dictionary, whose keys are
    chunk ids.
    """
    papers = []
    for chunk_id in chunk_ids:
        paper = paper_of(chunk_id)
        if paper not in papers:
            papers.append(paper)
    papers.sort()
    return papers


def source_files(collected: dict[str, str]) -> list[str]:
    """Distinct PDF filenames represented by the passages currently in context."""
    return papers_in(collected)


def chunk_ids_of(found: list[tuple[str, float]]) -> list[str]:
    """Only the chunk ids from a list of (chunk_id, score) pairs."""
    ids = []
    for chunk_id, _score in found:
        ids.append(chunk_id)
    return ids


def papers_listed(ranked: list[tuple[str, str]]) -> list[str]:
    """Only the filenames from a list of (filename, relation) pairs."""
    names = []
    for other, _relation in ranked:
        names.append(other)
    return names


def join_lines(lines: list[str], empty_text: str, bullet: str = "") -> str:
    """One item per line, each starting with `bullet`. `empty_text` when the list is empty."""
    if len(lines) == 0:
        return empty_text
    with_bullets = []
    for line in lines:
        with_bullets.append(bullet + line)
    return "\n".join(with_bullets)


def _parse_json_object(raw: str) -> dict:
    """Read the JSON object out of a model reply, tolerating code fences."""
    text = raw.strip()
    if text.startswith("```"):
        # The reply is wrapped in a markdown code fence: drop the first line ("```json")
        # and everything from the closing ``` onward.
        first_newline = text.find("\n")
        text = text[first_newline + 1:]
        closing_fence = text.rfind("```")
        if closing_fence != -1:
            text = text[:closing_fence]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        parsed = json.loads(text[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _planned_queries(state: "AgentState") -> list[str]:
    """The search queries the planner asked for, as a list of at most MAX_QUERIES.

    Lab 5.2's planner returns a "queries" array. A planner that returns a single
    "query" string instead is read as a list of one, and a planner that returns
    neither falls back to the question itself.
    """
    pending = state["pending"]
    raw = pending.get("queries")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raw = []
    queries = []
    for item in raw:
        text = str(item).strip()
        if text:
            queries.append(text)
    if len(queries) == 0:
        single = pending.get("query")
        if not single:
            single = ""
        single = str(single).strip()
        if single:
            queries = [single]
        else:
            queries = [state["question"]]
    return queries[:MAX_QUERIES_PER_STEP]


def _add_passages(collected: dict[str, str], found: list[tuple[str, float]],
                  label: str) -> tuple[dict[str, str], int]:
    """Fold retrieved chunks into the collected set, skipping repeats."""
    updated = dict(collected)
    added = 0
    for chunk_id, _score in found:
        if chunk_id in updated:
            continue
        if is_duplicate_passage(chunk_id, updated):
            continue
        updated[chunk_id] = label
        added += 1
    return updated, added


def build_agent(llm: ChatOpenAI, debug: bool = True):
    """Compile the Lab 5.2 style graph: plan, three tool nodes, clarify, versions, answer."""

    def plan_node(state: AgentState) -> dict:
        """The agent's brain: look at what it has, choose the next single action."""
        if state["steps"] >= MAX_RETRIEVAL_STEPS:
            if debug:
                print("  [plan] retrieval budget spent - answering with what I have")
            return {"next_action": "answer", "pending": {}}

        history = join_lines(state["history"], "(none yet)")
        executed = join_lines(state["executed"], "(none)", bullet="- ")
        clarifications = join_lines(state["clarifications"], "(none)")
        sources = join_lines(source_files(state["collected"]), "(none)", bullet="- ")
        user_message = (
            f"Question: {state['question']}\n\n"
            f"Queries already executed:\n{executed}\n\n"
            f"Actions already taken:\n{history}\n\n"
            f"Clarifications:\n{clarifications}\n\n"
            f"Source filenames represented in the current passages:\n"
            f"{sources}\n\n"
            f"Passages retrieved ({len(state['collected'])} total):\n\n"
            f"{_excerpt_block(state['collected'])}"
        )
        reply = llm.invoke([
            SystemMessage(content=PLAN_SYSTEM),
            HumanMessage(content=user_message),
        ]).content
        decision = _parse_json_object(reply)
        action = str(decision.get("action", "answer")).strip()
        if action not in ("search", "search_in_paper", "related_papers", "clarify", "answer"):
            action = "answer"
        if debug:
            print(f"  [plan] action={action}  ({str(decision.get('reasoning', ''))[:90]})")
        return {
            "next_action": action,
            "pending": decision,
            "llm_calls": state["llm_calls"] + 1,
            "trace": state["trace"] + [{
                "step": state["steps"],
                "node": "plan",
                "action": action,
                "arguments": {"query": decision.get("query", ""),
                              "paper": decision.get("paper", "")},
                "reasoning": decision.get("reasoning", ""),
                "checklist": decision.get("checklist", []),
            }],
        }

    def search_node(state: AgentState) -> dict:
        """Tool 1: hybrid retrieval over the whole collection.

        The planner returns one query per retrieval round, then sees the result before
        choosing its next action. A query the agent has already run is skipped.
        """
        collected = dict(state["collected"])
        executed = list(state["executed"])
        notes: list[str] = []
        passages: list[str] = []
        added_total = 0
        for query in _planned_queries(state):
            if query in executed:
                notes.append(f"search({query}) skipped, already run")
                continue
            found = baseline_retrieve(query, SEARCH_K)
            collected, added = _add_passages(collected, found, f"search({query})")
            added_total = added_total + added
            found_ids = chunk_ids_of(found)
            for chunk_id in found_ids:
                passages.append(chunk_id)
            executed.append(query)
            papers = len(papers_in(found_ids))
            notes.append(f"search({query}) returned {len(found)} passage(s) "
                         f"from {papers} paper(s)")
        if debug:
            for note in notes:
                print(f"  [tool] {note}")
            print(f"         +{added_total} new passage(s), {len(collected)} in context")
        return {
            "steps": state["steps"] + 1,
            "tool_calls": state["tool_calls"] + len(notes),
            "collected": collected,
            "executed": executed,
            "history": state["history"] + notes,
            "trace": state["trace"] + [{"step": state["steps"], "node": "search",
                                        "queries": _planned_queries(state),
                                        "note": "; ".join(notes),
                                        "new_passages": added_total,
                                        "passages": passages}],
            "pending": {},
        }

    def search_in_paper_node(state: AgentState) -> dict:
        """Tool 2: the same retrieval, inside the one paper the planner named."""
        query = state["pending"].get("query")
        if not query:
            query = state["question"]
        query = str(query)
        paper = state["pending"].get("paper")
        if not paper:
            paper = ""
        paper = str(paper)
        if paper not in CHUNKS_BY_PAPER:
            # the planner may give part of a filename; accept it when only one file matches
            matches = []
            for candidate in ALL_PAPERS:
                if paper.lower() in candidate.lower():
                    matches.append(candidate)
            if len(matches) == 1:
                paper = matches[0]
        if paper not in CHUNKS_BY_PAPER:
            note = f"no paper named {paper} in the collection"
            found = []
            collected = dict(state["collected"])
            added = 0
        else:
            found = retrieve_inside_paper(query, paper, IN_PAPER_K)
            collected, added = _add_passages(state["collected"], found,
                                             f"search_in_paper({paper})")
            note = f"search_in_paper({paper}) returned {len(found)} passage(s)"
        if debug:
            print(f"  [tool] {note}  (+{added} new)")
        return {
            "steps": state["steps"] + 1,
            "tool_calls": state["tool_calls"] + 1,
            "collected": collected,
            "executed": state["executed"] + [f"{query} (inside {paper})"],
            "history": state["history"] + [note],
            "trace": state["trace"] + [{"step": state["steps"], "node": "search_in_paper",
                                        "note": note, "new_passages": added,
                                        "passages": chunk_ids_of(found)}],
            "pending": {},
        }

    def related_papers_node(state: AgentState) -> dict:
        """Tool 3: inspect the 4.1 graph for papers linked to one retrieved file."""
        paper = state["pending"].get("paper")
        if not paper:
            paper = ""
        paper = str(paper)
        links = related_papers(paper)

        # Strongest relation first (another version, then same author, then same topic),
        # then by filename. Python sorts tuples by their first value, then their second.
        priority = {"same_title": 0, "first_author": 1, "has_topic": 2}
        rows = []
        for other, relation in links.items():
            rows.append((priority[relation], other, relation))
        rows.sort()
        ranked = []
        for _priority, other, relation in rows[:6]:
            ranked.append((other, relation))

        if links:
            listed_parts = []
            for other, relation in ranked:
                listed_parts.append(f"{other} ({relation})")
            listed = ", ".join(listed_parts)
            note = (f"related_papers({paper}): {listed}. "
                    f"same_title = another version of this paper; "
                    f"first_author = same first author; has_topic = shared topic")
        else:
            note = f"related_papers({paper}) found no linked papers"
        if debug:
            print(f"  [tool] {note}")
        return {
            "steps": state["steps"] + 1,
            "tool_calls": state["tool_calls"] + 1,
            "history": state["history"] + [note],
            "trace": state["trace"] + [{
                "step": state["steps"],
                "node": "related_papers",
                "paper": paper,
                "note": note,
                "related_papers": papers_listed(ranked),
            }],
            "pending": {},
        }

    def clarify_node(state: AgentState) -> dict:
        """Ask the user a question. A scored run has no user, so it is recorded and the
        agent carries on, which is the choice Lab 5.2 makes in batch mode."""
        question = str(state["pending"].get("clarification", "Could you clarify?"))
        if debug:
            print(f"  [clar] {question}")
        return {
            "steps": state["steps"] + 1,
            "clarifications": state["clarifications"] + [
                f"Q: {question}\nA: (no user available in a batch run)"],
            "history": state["history"] + [f"clarify: {question}"],
            "trace": state["trace"] + [{"step": state["steps"], "node": "clarify",
                                        "question": question}],
            "pending": {},
        }

    def answer_node(state: AgentState) -> dict:
        """Write the answer from the passages collected so far."""
        user_message = (
            f"Excerpts:\n\n{_excerpt_block(state['collected'])}\n\n"
            f"Question: {state['question']}"
        )
        reply = llm.invoke([
            SystemMessage(content=AGENT_ANSWER_SYSTEM),
            HumanMessage(content=user_message),
        ]).content
        return {
            "answer": reply,
            "llm_calls": state["llm_calls"] + 1,
            "trace": state["trace"] + [{"step": state["steps"], "node": "answer"}],
        }

    def versions_node(state: AgentState) -> dict:
        """Before every answer, read the other version of each paper in the context.

        This is a check in the harness rather than a rule in the prompt: the planner was
        told when to use related_papers and never did. For each paper in the context
        that has another copy in the collection, the node retrieves from that copy twice,
        once with the question and once with the text of each passage already collected,
        which finds the matching page. Identical text is dropped, so the other copy only
        reaches the context when it says something different.
        """
        collected = dict(state["collected"])
        papers = papers_in(collected)
        read_from: list[str] = []
        for paper in papers:
            # the first few passages already collected from this paper
            mine = []
            for chunk_id in collected:
                if paper_of(chunk_id) == paper and len(mine) < VERSION_PASSAGES:
                    mine.append(chunk_id)
            for copy in other_versions(paper):
                if copy in papers:
                    continue
                queries = [state["question"]]
                for chunk_id in mine:
                    queries.append(CHUNKS[chunk_id]["text"][:400])
                for query in queries:
                    for chunk_id, _score in retrieve_inside_paper(query, copy, 1):
                        if chunk_id in collected or is_duplicate_passage(chunk_id, collected):
                            continue
                        collected[chunk_id] = f"other version of {paper}"
                        if copy not in read_from:
                            read_from.append(copy)
        if debug:
            if read_from:
                print(f"  [vers] other version differs, added passages from {read_from}")
            else:
                print("  [vers] no other version with different text")
        return {
            "collected": collected,
            "versions_checked": True,
            "tool_calls": state["tool_calls"] + len(read_from),
            "trace": state["trace"] + [{"step": state["steps"], "node": "versions",
                                        "papers_added": read_from}],
        }

    def route_from_plan(state: AgentState) -> str:
        action = state["next_action"]
        if action == "search":
            return "search"
        if action == "search_in_paper":
            return "search_in_paper"
        if action == "related_papers":
            return "related_papers"
        if action == "clarify":
            return "clarify"
        if not state["versions_checked"]:
            return "versions"
        return "answer"

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("search", search_node)
    graph.add_node("search_in_paper", search_in_paper_node)
    graph.add_node("related_papers", related_papers_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("answer", answer_node)
    graph.add_node("versions", versions_node)

    # Lab 5.1 starts at retrieval with the original question, then asks the model whether
    # that is enough. Starting here means the agent begins with exactly the evidence the
    # fixed pipeline has, and the planner only decides whether to search further.
    graph.set_entry_point("search")
    graph.add_conditional_edges("plan", route_from_plan, {
        "search": "search",
        "search_in_paper": "search_in_paper",
        "related_papers": "related_papers",
        "clarify": "clarify",
        "answer": "answer",
        "versions": "versions",
    })
    graph.add_edge("search", "plan")
    graph.add_edge("search_in_paper", "plan")
    graph.add_edge("related_papers", "plan")
    graph.add_edge("clarify", "plan")
    graph.add_edge("versions", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def run_agent(llm: ChatOpenAI, question: str, debug: bool = True) -> dict:
    """Run one question through the agent and return the answer plus what it cost."""
    agent = build_agent(llm, debug=debug)
    start = time.time()
    initial_state: AgentState = {
        "question": question,
        "next_action": "search",
        "pending": {"queries": [question]},   # the first search is the question itself
        "collected": {},
        "executed": [],
        "history": [],
        "clarifications": [],
        "trace": [],
        "steps": 0,
        "llm_calls": 0,
        "tool_calls": 0,
        "answer": "",
        "versions_checked": False,
    }
    final = agent.invoke(initial_state, {"recursion_limit": 50})
    return {
        "answer": final["answer"],
        "chunk_ids": list(final["collected"].keys()),
        "papers": papers_in(final["collected"]),
        "trace": final["trace"],
        "tool_calls": final["tool_calls"],
        "llm_calls": final["llm_calls"],
        "seconds": round(time.time() - start, 1),
    }


def run_fixed_pipeline(llm: ChatOpenAI, question: str) -> dict:
    """The Checkpoint 2.1 / 3.1 pipeline: retrieve once, answer once, no decisions."""
    start = time.time()
    chunk_ids = chunk_ids_of(baseline_retrieve(question, TOP_K))
    collected = {}
    for chunk_id in chunk_ids:
        collected[chunk_id] = "search(question)"
    reply = llm.invoke([
        SystemMessage(content=AGENT_ANSWER_SYSTEM),
        HumanMessage(content=f"Excerpts:\n\n{_excerpt_block(collected)}\n\nQuestion: {question}"),
    ]).content
    return {
        "answer": reply,
        "chunk_ids": chunk_ids,
        "papers": papers_in(chunk_ids),
        "trace": [{"node": "retrieve", "passages": chunk_ids}, {"node": "answer"}],
        "tool_calls": 1,
        "llm_calls": 1,
        "seconds": round(time.time() - start, 1),
    }


# %% [markdown]
# ## Part C - the Checkpoint 3.1 metric suite, ported
#
# The same judge and counting metrics 3.1 and 4.1 used, so this agent's numbers can be
# read next to the earlier checkpoints' numbers.

# %%
JUDGE_SYSTEM = (
    "You are a strict evaluator. You are given an ANSWER and GRADING NOTES describing "
    "what a correct answer must contain. Decide whether the answer satisfies the notes.\n"
    'Return only JSON: {"verdict": "pass" or "fail", "reason": "..."}.\n'
    "When the verdict is fail, reason must be one sentence naming the specific requirement "
    "from the grading notes that the answer missed, contradicted, or left out.\n"
    "Apply the pass criteria exactly as the grading notes state them. If the notes set a "
    "threshold (a minimum number of items, an either/or, an acceptable alternative), COUNT "
    "how many of the listed items the answer covers, compare that count with the threshold, "
    "and pass the answer as soon as it reaches the threshold. Do not add requirements the "
    "notes do not state.\n"
    "An answer invents something only when it states a fact or a source the corpus does not "
    "support. Naming a real paper the notes happen not to list is not an invention.\n"
    "Judge the substance, not the wrapping: framing text around a quotation does not make "
    "the quotation non-verbatim.\n"
    "When the verdict is pass, reason must be an empty string."
)

POINTS_SYSTEM = (
    "You check whether an answer contains specific facts. You are given an ANSWER and a "
    "numbered list of REQUIRED POINTS. For each point, reply on its own line in the form "
    "'<number>: yes' if the answer supplies that point, or '<number>: no' if it does not. "
    "Judge each point independently. Accept different wording and accept a point that is "
    "clearly implied. Output nothing except those lines."
)


def judge(llm: ChatOpenAI, answer_text: str, grading_notes: str) -> tuple[str, str]:
    """CORRECTNESS. Returns (verdict, reason); reason is '' on a pass."""
    reply = llm.invoke([
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=f"ANSWER:\n{answer_text}\n\nGRADING NOTES:\n{grading_notes}"),
    ]).content
    result = _parse_json_object(reply)
    verdict = str(result.get("verdict", "")).strip().lower()
    if verdict == "pass":
        return "pass", ""
    if verdict == "fail":
        return "fail", str(result.get("reason", "")).strip()
    if reply.strip().lower().startswith("pass"):
        return "pass", ""
    return "fail", f"(judge reply unparseable) {reply.strip()[:200]}"


def point_coverage(llm: ChatOpenAI, answer_text: str, item: dict) -> tuple[int, int]:
    """COMPLETENESS: how many of the question's required facts the answer supplies."""
    points = item["required_points"]
    numbered_lines = []
    for number, point in enumerate(points, 1):
        numbered_lines.append(f"{number}. {point}")
    numbered = "\n".join(numbered_lines)
    reply = llm.invoke([
        SystemMessage(content=POINTS_SYSTEM),
        HumanMessage(content=f"ANSWER:\n{answer_text}\n\nREQUIRED POINTS:\n{numbered}"),
    ]).content
    got = []                                    # one True/False per required point
    for _point in points:
        got.append(False)
    for number, verdict in re.findall(r"(\d+)\s*:\s*(yes|no)", reply.lower()):
        index = int(number) - 1
        if 0 <= index < len(got):
            got[index] = (verdict == "yes")
    covered = 0
    for is_covered in got:
        if is_covered:
            covered = covered + 1
    return covered, len(points)


def _normalise(text: str) -> str:
    for before, after in (("ﬁ", "fi"), ("ﬂ", "fl"), ("’", "'"), ("‘", "'"),
                          ("“", "'"), ("”", "'"), ('"', "'"),
                          ("—", "-"), ("–", "-")):
        text = text.replace(before, after)
    text = re.sub(r"[*`]", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def retrieved_pages(chunk_ids: list[str]) -> set[tuple[str, str]]:
    pages = set()
    for chunk_id in chunk_ids:
        meta = CHUNKS.get(chunk_id, {}).get("meta", {})
        if meta:
            pages.add((str(meta.get("source", "?")), str(meta.get("page", "?"))))
    return pages


def citation_parts(group: str) -> list[tuple[str, str]]:
    """(filename, page) for each citation inside one [ ... ] group.

    A page belongs to the last filename written before it, so "[a.pdf, p.2; b.pdf, p.1]"
    is a's page 2 and b's page 1, and "[a.pdf, p.2; p.1]" is two pages of a.
    """
    found: list[tuple[str, str]] = []
    current = ""
    # Each match is either a filename or a page number. findall returns both parts of
    # every match, with "" for the part that did not match.
    for name, page in re.findall(r"([A-Za-z0-9_.\-]+\.pdf)|p\.\s*(\d+)", group):
        if name != "":
            current = name
        elif current != "":
            found.append((current, page))
    return found


def cited_pages(answer_text: str) -> list[tuple[str, str]]:
    found = []
    for group in re.findall(r"\[([^\[\]]+)\]", answer_text):
        found.extend(citation_parts(group))
    return found


def citation_valid(answer_text: str, chunk_ids: list[str]) -> tuple[bool | None, list[str]]:
    """GROUNDING: every [file, p.N] in the answer must be a page that was retrieved."""
    cites = cited_pages(answer_text)
    if not cites:
        return None, []
    available = retrieved_pages(chunk_ids)
    bad = []
    for name, page in cites:
        if (name, page) not in available:
            bad.append(f"[{name}, p.{page}]")
    all_valid = (len(bad) == 0)
    return all_valid, bad


def source_coverage(chunk_ids: list[str], answer_text: str, item: dict):
    """COMPLETENESS: how many of the gold files were retrieved, and how many were cited."""
    gold = item.get("gold_sources")
    if not gold:
        return None
    retrieved_files = []
    for name, _page in retrieved_pages(chunk_ids):
        retrieved_files.append(name)
    cited_files = []
    for name, _page in cited_pages(answer_text):
        cited_files.append(name)
    gold_retrieved = 0
    gold_cited = 0
    for gold_file in gold:
        if gold_file in retrieved_files:
            gold_retrieved = gold_retrieved + 1
        if gold_file in cited_files:
            gold_cited = gold_cited + 1
    return (gold_retrieved, gold_cited, len(gold))


REFUSAL_PATTERNS = (
    r"do(es)? not (contain|include|provide|mention|specify|state|identify|give|say|report|describe)",
    r"don'?t (contain|include|provide|mention)",
    r"cannot (determine|answer|find|identify|state|tell)",
    r"can'?t (determine|answer|find|identify|state|tell)",
    r"not (present|found|included|mentioned|described) in",
    r"no (information|mention|description) (about|of|on|in)",
    r"contains? no ", r"not supported by", r"unable to (determine|answer|find)",
    r"i'?m sorry",
)


def refused(answer_text: str) -> bool:
    opening = _normalise(answer_text)[:220]
    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern, opening):
            return True
    return False


def load_items(path: Path) -> list[dict]:
    if not path.is_file():
        raise RuntimeError(f"Evaluation file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# %% [markdown]
# ## Part D - the initial evaluation (worksheet Step 2)
#
# All twelve basic questions from Checkpoint 4.1 are loaded from a copy of its
# evaluation file kept next to this script.
# They cover lookups, quotations, synthesis, conflicting versions, and refusals. Each
# question runs through the fixed Checkpoint 2.1 / 3.1 pipeline and through this agent,
# with the same answer prompt, and both answers are scored the same way.


def score_run(llm: ChatOpenAI, item: dict, result: dict) -> dict:
    """Apply the 3.1 metrics to one system's answer for one question."""
    verdict, reason = judge(llm, result["answer"], item["grading_notes"])
    points_got, points_total = point_coverage(llm, result["answer"], item)
    valid, bad = citation_valid(result["answer"], result["chunk_ids"])
    coverage = source_coverage(result["chunk_ids"], result["answer"], item)
    if coverage is None:
        coverage_text = None
    else:
        coverage_text = (f"{coverage[0]}/{coverage[2]} retrieved, "
                         f"{coverage[1]}/{coverage[2]} cited")
    return {
        "correctness": verdict,
        "correctness_reason": reason,
        "point_coverage": f"{points_got}/{points_total}",
        "citations_valid": valid,
        "bad_citations": bad,
        "source_coverage": coverage_text,
        "papers_in_context": len(result["papers"]),
        "passages_in_context": len(result["chunk_ids"]),
        "tool_calls": result["tool_calls"],
        "llm_calls": result["llm_calls"],
        "seconds": result["seconds"],
        "refused": refused(result["answer"]),
    }


def _summary_line(name: str, scored: dict) -> str:
    sources = scored["source_coverage"]
    if sources is None:
        sources = "-"
    return (f"    {name:14s} correctness={scored['correctness']:4s} "
            f"points={scored['point_coverage']:5s} "
            f"sources={sources:28s} "
            f"papers={scored['papers_in_context']} "
            f"llm={scored['llm_calls']} tools={scored['tool_calls']} "
            f"{scored['seconds']}s")


def run_evaluation(llm: ChatOpenAI) -> list[dict]:
    """Compare the fixed pipeline and agent on every basic 4.1 evaluation question."""
    items = load_items(QUESTIONS_PATH)
    results = []
    print(f"Running {len(items)} basic evaluation questions from {QUESTIONS_PATH.name}.")
    for item in items:
        question_id = item["id"]
        print("\n" + "=" * 96)
        print(f"{question_id}: {item['question']}")
        print("=" * 96)

        print("\n  Checkpoint 4.1")
        fixed = run_fixed_pipeline(llm, item["question"])
        print(f"    retrieved {len(fixed['chunk_ids'])} passages from "
              f"{len(fixed['papers'])} paper(s)")

        print("\n  agent")
        agent = run_agent(llm, item["question"])

        print("\n  scoring...")
        record = {"id": question_id, "question": item["question"], "kind": item["kind"],
                  "systems": {}}
        for name, result in (("fixed", fixed), ("agent", agent)):
            scored = score_run(llm, item, result)
            record["systems"][name] = {
                "answer": result["answer"],
                "papers": result["papers"],
                "chunk_ids": result["chunk_ids"],
                "trace": result["trace"],
                "scores": scored,
            }
            print(_summary_line(name, scored))
            log(f"{question_id} [{name}]", "\n".join([
                f"papers={result['papers']}",
                f"chunks={result['chunk_ids']}",
                f"scores={json.dumps(scored)}",
                f"answer={result['answer']}",
            ]))
        results.append(record)

    print("\n" + "=" * 96)
    print("SUMMARY - agent against Checkpoint 4.1")
    print("=" * 96)
    print(f"  {'question':10s} {'system':8s} {'correct':8s} {'points':7s} "
          f"{'papers':7s} {'llm':5s} {'secs':6s}")
    for record in results:
        for name, data in record["systems"].items():
            scored = data["scores"]
            print(f"  {record['id']:10s} {name:8s} {scored['correctness']:8s} "
                  f"{scored['point_coverage']:7s} {scored['papers_in_context']:<7d} "
                  f"{scored['llm_calls']:<5d} {scored['seconds']:<6}")
    print()
    for line in comparison_table(results):
        print(line)
    return results


def fraction_parts(text: str) -> tuple[int, int]:
    """Read the first "a/b" in a text such as "3/4" or "2/2 retrieved, 1/2 cited"."""
    first_word = text.split(" ")[0]          # "2/2 retrieved, ..." -> "2/2"
    top, bottom = first_word.split("/")
    return int(top), int(bottom)


def system_totals(results: list[dict], name: str) -> dict:
    """Add up one system's scores over all the questions in a run."""
    totals = {"passes": 0, "points_got": 0, "points_total": 0,
              "gold_got": 0, "gold_total": 0, "llm_calls": 0, "seconds": 0.0}
    for record in results:
        scored = record["systems"][name]["scores"]
        if scored["correctness"] == "pass":
            totals["passes"] = totals["passes"] + 1
        got, total = fraction_parts(scored["point_coverage"])
        totals["points_got"] = totals["points_got"] + got
        totals["points_total"] = totals["points_total"] + total
        if scored["source_coverage"] is not None:     # questions with no gold files are skipped
            got, total = fraction_parts(scored["source_coverage"])
            totals["gold_got"] = totals["gold_got"] + got
            totals["gold_total"] = totals["gold_total"] + total
        totals["llm_calls"] = totals["llm_calls"] + scored["llm_calls"]
        totals["seconds"] = totals["seconds"] + scored["seconds"]
    return totals


def comparison_table(results: list[dict]) -> list[str]:
    """The run's totals as a table: one row per metric, one column per system."""
    count = len(results)
    fixed = system_totals(results, "fixed")
    agent = system_totals(results, "agent")
    rows = [
        ["", "Baseline", "Agent"],
        ["Correct (judge)", f"{fixed['passes']}/{count}", f"{agent['passes']}/{count}"],
        ["Required points", f"{fixed['points_got']}/{fixed['points_total']}",
         f"{agent['points_got']}/{agent['points_total']}"],
        ["Gold files retrieved", f"{fixed['gold_got']}/{fixed['gold_total']}",
         f"{agent['gold_got']}/{agent['gold_total']}"],
        [f"Model calls, {count} questions", str(fixed["llm_calls"]), str(agent["llm_calls"])],
        [f"Seconds, {count} questions", str(round(fixed["seconds"], 1)),
         str(round(agent["seconds"], 1))],
    ]
    # the width of each column is the width of its longest cell
    widths = [0, 0, 0]
    for row in rows:
        for column in range(3):
            if len(row[column]) > widths[column]:
                widths[column] = len(row[column])
    border = "  +"
    for width in widths:
        border = border + "-" * (width + 2) + "+"
    lines = [border]
    for row in rows:
        line = "  |"
        for column in range(3):
            line = line + " " + row[column].ljust(widths[column]) + " |"
        lines.append(line)
        lines.append(border)
    return lines


# %% [markdown]
# ## Part E - run it
#
# `python 1-capstone_checkpoint_5_1_agent_rag_initial.py` runs the evaluation above.
# Pass a question in quotes to ask the agent one question instead.

# %%
def main() -> None:
    llm = make_llm()
    asked = []                                  # words typed after the script name
    for argument in sys.argv[1:]:
        if not argument.startswith("-"):
            asked.append(argument)
    if asked:
        question = " ".join(asked)
        print(f"Question: {question}\n")
        result = run_agent(llm, question)
        print(f"\nAnswer:\n{result['answer']}\n")
        print(f"{result['tool_calls']} tool calls, {result['llm_calls']} model calls, "
              f"{result['seconds']}s, {len(result['papers'])} paper(s) in context")
        return
    results = run_evaluation(llm)
    RESULTS_PATH.write_text(json.dumps(results, indent=1, ensure_ascii=False),
                            encoding="utf-8")
    print(f"\nResults written to {RESULTS_PATH.name}")
    print(f"Log written to {LOG_PATH.name}")


if __name__ == "__main__":
    main()

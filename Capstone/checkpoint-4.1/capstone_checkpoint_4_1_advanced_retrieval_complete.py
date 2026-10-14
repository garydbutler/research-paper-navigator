r"""Capstone Checkpoint 4.1 — Advanced Retrieval Implementation (complete).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code/PyCharm/Jupytext.

Run from this folder with the course venv active:

    python capstone_checkpoint_4_1_advanced_retrieval_complete.py
"""

# %% [markdown]
# # Capstone Checkpoint 4.1 — Advanced Retrieval Implementation
# **MO-LLM Module 4/Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# This checkpoint advances your capstone system from a *single-pass* retriever to an
# **advanced retriever**. You will use the baseline retrieval system you built in
# Checkpoint 2.1 and evaluated in Checkpoint 3.1 to identify where single-pass retrieval
# falls short and apply the Module 4 techniques to fix it:
#
# - **Lab 4.1 — multistep retrieval (query decomposition):** Rewrite a complex
#   question into focused sub-queries, retrieve for each, and merge the results.
# - **Lab 4.2 — graph-based retrieval:** Organize your corpus as a graph of
#   relationships (citations, shared topics, links, same author/entity) and pull in
#   *related* documents a keyword/vector search would miss.
#
# You then **measure the improvement** against your 3.1 baseline. The graded
# deliverable is the completed worksheet. This script runs both techniques on the
# real Research Paper Navigator corpus and re-runs the 3.1 evaluation on the baseline
# and the advanced retriever side by side.
#
# **Learning outcomes (Module 4):**
# 1. Identify common retrieval failures — e.g., complex multistep questions and too
#    much loosely-related context degrading the answer.
# 2. Explain *why* those failures occur with single-pass retrieval.
# 3. Implement an advanced retrieval strategy (query decomposition and/or graph
#    traversal) that addresses them.
# 4. Measure the improvement against your baseline using your 3.1 evaluation set.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario** you chose in Checkpoint 1.1 and have built on since.
#
# | Scenario | Corpus | Natural relationships to exploit in a graph |
# |---|---|---|
# | **Research Paper Navigator** | ~150 research paper PDFs (`data/ResearchPapers/`) | citations, shared authors, shared topics/keywords, "published-before" |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles | hyperlinks between articles, shared categories, mentioned entities |
#
# Multistep decomposition shines on cross-document questions ("Compare X and Y,"
# "how did an idea evolve"); graph retrieval shines when the answer needs documents
# that are *related* to a hit but don't themselves match the query terms.

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12**
# 2. `pip install langchain-openai langchain-core langchain-chroma rank-bm25 pypdf python-dotenv networkx`
# 3. Use the OpenRouter API key provided for this course. This checkpoint uses
#   the `openai/gpt-5.4-mini` model, with usage covered by course credits.
# 4. The shared `.env` at the repo root holds `OPENROUTER_API_KEY=sk-or-v1-...`
#
# The corpus, chunking and vector store are the 2.1 / 3.1 ones unchanged. The vector
# store built in Checkpoint 3.1 is reused if it exists, so nothing is re-embedded.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import os
import re
import textwrap
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import networkx as nx
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"  # Latest small OpenAI model, fast; covered by course credits
EMBEDDING_MODEL = "openai/text-embedding-3-small"  # same as 2.1 / 3.1
TEMPERATURE = 0.2
LOG_PATH = Path.cwd() / "checkpoint_4_1_responses.log"

# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "research_papers"   # "research_papers" or "wikipedia"

# --- Ported from Checkpoint 3.1 unchanged, so the baseline numbers stay comparable -----
HERE = Path(__file__).resolve().parent
SHARED_DATA_DIR = HERE.parent.parent / "data"    # one copy of the PDF corpus for every checkpoint
PAPERS_DIR = SHARED_DATA_DIR / "ResearchPapers"
CHROMA_DIR = HERE / "chroma_chunks"              # this folder's own copy (built from the PDFs if missing)
# The 3.1 evaluation set is the yardstick. This folder keeps its own copy of it.
QUESTIONS_PATH = HERE / "eval_questions.json"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MIN_PAGE_CHARS = 100
TOP_K = 6                 # both systems hand the LLM six chunks, so the comparison is fair
CANDIDATE_POOL = 20
RRF_K = 60

# --- New in 4.1 -----------------------------------------------------------------------
SUBQUERY_POOL = 3 * TOP_K  # 3*k per sub-query, the value lab 4.1 uses, before the sum
MAX_SUBQUERIES = 4        # hard cap in code; the lecture's point is that the model may disobey

# Lecture 4.7 on the decomposition prompt: say the queries feed a retrieval system, say what
# kind of text they must match, fix the format, and cap the count. A one-fact question is
# returned unchanged so lookup and quotation questions keep the baseline behaviour.
DECOMPOSE_SYSTEM = (
    "You are a query decomposition assistant for a search system over passages of "
    "computer-science research papers. Your sub-queries are matched against passage text "
    "by keyword and embedding search, so phrase each one like the sentence you expect to "
    "find in a paper.\n\n"
    "The collection contains these papers (filename: first line of page 1):\n"
    "{TITLE_LIST}\n\n"
    "Rules:\n"
    "* If the question asks for one fact, sentence or number from one paper, return the "
    "question unchanged as a one-element array. Do not split it.\n"
    "* If the question compares two papers or asks how something changed between them, "
    "return one sub-query per paper, naming that paper's system, year or title so it "
    "retrieves from that paper and not the other.\n"
    "* If the question asks which papers in the collection do something, pick the papers "
    "from the list above that plausibly do it and return one sub-query per paper (at most "
    "4). Each sub-query is one natural sentence: the paper's title followed by the specific "
    "thing the question asks about it. Do not copy the paper's topic list into the query and "
    "do not return a paraphrase of the whole question.\n"
    "* Never invent numbers, results, findings or quotations in a sub-query. You are "
    "describing what to look for, not answering the question.\n"
    "* Never write a sub-query asking for a citation, reference, bibliographic entry or "
    "where something is cited. Those match reference lists, not content.\n"
    "Return ONLY a JSON array of strings."
)

ANSWER_SYSTEM = (
    "You are a research assistant for a collection of computer-science papers. Answer "
    "the question using ONLY the provided paper excerpts.\n"
    "* Excerpts marked PRIMARY matched the question directly. Excerpts marked CONTEXT are "
    "related papers (another version of the same paper, a paper that cites or is cited by "
    "it, the same first author, or the same topic) added for completeness; use them when "
    "they help answer the question.\n"
    "* Cite the source of every claim formatted as [filename, p.N], using the labels on the "
    "excerpts.\n"
    "* When you quote something, quote verbatim from an excerpt and put it in quotation marks. "
    "Never reconstruct or paraphrase a quotation and present it as verbatim.\n"
    "* If two excerpts come from different versions of the same paper and disagree, report "
    "both figures and say which file each came from.\n"
    "* If the excerpts do not contain the answer, say so explicitly rather than "
    "guess. A refusal is the best answer when no information is available."
)

GRAPH_PAPERS = 6          # related papers the graph may add to the candidate pool

# Lecture 4.7 section 2: the optional LLM filtering step that scores retrieved passages for
# relevance before they enter the final context.
FILTER_SYSTEM = (
    "You select the excerpts a research assistant will read to answer a question. You are "
    "given the question and numbered candidate excerpts from research papers. Rank ALL of "
    "them from most to least useful for answering it; the top {K} will be used.\n"
    "* Cover every part of the question and every paper it names or implies.\n"
    "* Prefer an excerpt that states the fact asked for over one that only mentions the topic.\n"
    "* Never choose a reference-list or bibliography excerpt; it names papers but describes none.\n"
    "* If two excerpts come from different versions of the same paper and give different "
    "figures, keep both.\n"
    '* Return only JSON: {"keep": [every candidate number, most useful first]}.'
)

JUDGE_SYSTEM = (
    "You are a strict evaluator. You are given an ANSWER and GRADING NOTES describing "
    "what a correct answer must contain. Decide whether the answer satisfies the notes.\n"
    'Return only JSON: {"verdict": "pass" or "fail", "reason": "..."}.\n'
    "When the verdict is fail, reason must be one sentence naming the specific requirement "
    "from the grading notes that the answer missed, contradicted, or left out. Quote the "
    "part of the answer that is wrong where that makes the reason clearer. Do not restate "
    "the whole answer and do not suggest a better answer.\n"
    "Apply the pass criteria exactly as the grading notes state them. If the notes set a "
    "threshold (a minimum number of items, an either/or, an acceptable alternative), COUNT how "
    "many of the listed items the answer covers, compare that count with the threshold, and "
    "pass the answer as soon as it reaches the threshold. Never single out one listed item as "
    "required when the notes only ask for some of them, and never fail an answer for leaving "
    "out items it did not need. Do not add requirements the notes do not state.\n"
    "An answer invents something only when it states a fact or a source the corpus does not "
    "support. Naming a real paper the notes happen not to list, or adding an accurately "
    "described extra alongside the required items, is not an invention and does not fail an "
    "answer by itself.\n"
    "Judge the substance, not the wrapping: introductory or framing text around a quotation "
    "(such as 'The paper says:') does not make the quotation non-verbatim, and hedging "
    "language does not fail an answer that states the required facts.\n"
    "When the verdict is pass, reason must be an empty string."
)


# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Use the OpenRouter API key provided for this "
            "program, put it in the .env file at the repo root, and rerun."
        )
    return key


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )


# Each run starts a fresh log. Appending across runs made the file a pile of interleaved
# results that no longer said which run produced which line, and with a nondeterministic
# retriever that matters. Truncating once at import keeps every write below an append, so
# the file holds exactly one run.
LOG_PATH.write_text(
    f"[{datetime.now().isoformat(timespec='seconds')}]  RUN START  "
    f"SCENARIO={SCENARIO}  MODEL={LLM_MODEL}  TOP_K={TOP_K}\n"
    f"{'=' * 72}\n",
    encoding="utf-8",
)


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


# %% [markdown]
# ## The capstone corpus and the Checkpoint 2.1 / 3.1 baseline retriever (ported)
#
# The starter shipped six invented sentences with hand-written `topics` and `links`.
# They are replaced by the real corpus: 153 PDFs chunked into 1,200-character pieces,
# indexed by BM25 and by a Chroma vector store, and fused with Reciprocal Rank Fusion.
# Every function in this section is the 3.1 version unchanged. `baseline_retrieve()`
# keeps the starter's name so the rest of the script reads the same.

# %%
def _page_texts(reader: PdfReader) -> Iterator[tuple[int, str]]:
    """Yield (page number, cleaned text) for pages with enough text to index."""
    for page_no, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text() or ""
        except Exception:
            continue  # pypdf raises on some malformed content streams
        text = re.sub(r"[ \t]+", " ", text).strip()
        if len(text) >= MIN_PAGE_CHARS:
            yield page_no, text


def load_and_chunk(papers_dir: Path) -> list[Document]:
    """Extract text page by page, then split each page into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    files = sorted(p for p in papers_dir.iterdir() if p.suffix.lower() == ".pdf")
    docs: list[Document] = []
    for i, path in enumerate(files, 1):
        try:
            reader = PdfReader(path)
        except Exception as exc:
            print(f"  WARNING: {path.name} could not be parsed: {exc}")
            continue
        before = len(docs)
        for page_no, text in _page_texts(reader):
            for j, chunk in enumerate(splitter.split_text(text)):
                docs.append(Document(
                    page_content=chunk,
                    metadata={"source": path.name, "page": page_no, "chunk": j},
                ))
        print(f"  [{i:3d}/{len(files)}] {path.name:52s} {len(docs) - before:5d} chunks", flush=True)
    return docs


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )


def build_or_load_db() -> Chroma:
    """Load the persisted vector store if it exists, otherwise build it from the PDFs."""
    if CHROMA_DIR.is_dir() and any(CHROMA_DIR.iterdir()):
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


def _chunk_id(meta: dict) -> str:
    return f"{meta.get('source', '?')} p.{meta.get('page', '?')} #{meta.get('chunk', 0)}"


DOC_BY_ID = {}
for _text, _meta in zip(_stored["documents"], _stored["metadatas"]):
    DOC_BY_ID[_chunk_id(_meta)] = {"id": _chunk_id(_meta), "text": _text, "meta": _meta}
CORPUS_IDS = list(DOC_BY_ID)
CORPUS_TEXTS = [DOC_BY_ID[i]["text"] for i in CORPUS_IDS]
print(f"Corpus ready: {len(CORPUS_IDS)} chunks from {PAPERS_DIR.name}/")


def paper_of(chunk_id: str) -> str:
    """The filename part of a chunk id such as 'paper.pdf p.3 #1'."""
    return DOC_BY_ID[chunk_id]["meta"]["source"]


# %%
_STOPWORDS = {
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
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


BM25 = BM25Okapi([tokenize(t) for t in CORPUS_TEXTS])


def _bm25_ranked(query: str, k: int) -> list[str]:
    scores = BM25.get_scores(tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [CORPUS_IDS[i] for i in top]


def _vector_ranked(query: str, k: int) -> list[str]:
    return [_chunk_id(d.metadata) for d in DB.similarity_search(query, k=k)]


def baseline_retrieve(query: str, k: int = TOP_K) -> list[tuple[str, float]]:
    """Single-pass hybrid retrieval (the 2.1 / 3.1 baseline): BM25 + vector, fused by RRF.

    Returns (chunk_id, rrf_score) pairs, best first.
    """
    rrf: dict[str, float] = defaultdict(float)
    for ranked in (_bm25_ranked(query, CANDIDATE_POOL), _vector_ranked(query, CANDIDATE_POOL)):
        for rank, chunk_id in enumerate(ranked, 1):
            rrf[chunk_id] += 1.0 / (RRF_K + rank)
    ordered = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)
    return ordered[:k]


# %% [markdown]
# ## Step 2 — Multistep retrieval (adapted from Lab 4.1)
#
# `decompose_query` asks the LLM to split the question into sub-queries.
# `hybrid_graph_retrieve` (Step 4) runs the real hybrid retriever once per sub-query and
# **sums** each chunk's RRF score across all of them, exactly as lab 4.1 does, so a chunk
# relevant to several parts of the question rises to the top. The original question is always one of the sub-queries, so a question that
# does not need splitting is not made worse by it.

# %%
def decompose_query(llm: ChatOpenAI, query: str) -> list[str]:
    messages = [SystemMessage(content=DECOMPOSE_SYSTEM.replace("{TITLE_LIST}", TITLE_LIST)),
                HumanMessage(content=query)]
    raw = llm.invoke(messages).content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("["): raw.rfind("]") + 1] if "[" in raw else raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed) and parsed:
            return parsed[:MAX_SUBQUERIES]
    except (json.JSONDecodeError, ValueError):
        pass
    return [query]


# %% [markdown]
# ## Step 3 — Graph-based retrieval (adapted from Lab 4.2)
#
# The starter's graph had hand-written `links` and `topics`. Here the graph is built from
# the corpus itself, using only what a beginner can read off the data:
#
# - **paper** nodes, one per PDF.
# - **title** nodes: the first 40 letters of page 1, lower-cased and stripped of spaces
#   and punctuation. Two papers that share a title node are two *versions* of the same
#   paper (the corpus holds arXiv and venue copies of several papers). Edge: `same_title`.
# - **author** nodes: the first author's surname, read from the DBLP filename key
#   (`DBLP_conf_iclr_InalaBTS20.pdf` -> `Inala`). Edge: `first_author`.
# - **topic** nodes: 3 to 6 topic phrases extracted from page 1 by the LLM (one call per
#   paper, cached), then resolved so synonymous phrases share one node. Edge: `has_topic`.
#
# Retrieval seeds with the papers already chosen, then checks each seed paper's other
# versions, same-first-author papers and topic siblings as CONTEXT candidates.
# Only a passage that fills a remaining evidence gap may replace a direct hit.

# %%
# Lecture 4.7 section 5: a language model reads each document and extracts its entities and
# concepts, and the mentions are then RESOLVED so the same concept from different papers
# lands on one node. One extraction call per paper, cached in paper_metadata.json, so the
# corpus is read once. Nothing here touches the chunk store.
METADATA_PATH = HERE / "paper_metadata.json"
TOPIC_MAP_PATH = HERE / "paper_topics.json"
TOPIC_MERGE_SIMILARITY = 0.82   # cosine at which two topic phrases become one node (0.88 merged almost nothing; below 0.80 merges opposites)
QUERY_TOPIC_SEEDS = 3           # topic nodes nearest the question that seed PageRank
QUERY_TOPIC_MIN_SIMILARITY = 0.45

EXTRACT_SYSTEM = (
    "You read the first page of a computer-science research paper and return its metadata "
    "as JSON with these fields: title (string or null), first_author (surname, string or "
    "null), authors (list of full names), year (integer or null), venue (string or null), "
    "topics (3 to 6 short lower-case noun phrases a researcher would use to say what the "
    "paper is about; no author names, no generic words such as 'algorithm' or 'method'), "
    "summary (one sentence, at most 25 words).\n"
    "Use only what the text states and return null for anything it does not state. If the "
    "text is publisher front matter or a table of contents rather than a paper, return null "
    "for title and an empty topics list. Return only the JSON object."
)


def _page1_text(paper: str, limit: int = 4000) -> str:
    chunks = sorted((DOC_BY_ID[c] for c in PAPER_IDS[paper] if DOC_BY_ID[c]["meta"].get("page") == 1),
                    key=lambda d: d["meta"].get("chunk", 0))
    return " ".join(d["text"] for d in chunks)[:limit]


def extract_paper_metadata(papers: list[str]) -> dict[str, dict]:
    """{filename: {title, first_author, authors, year, venue, topics, summary}}, cached."""
    cached = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.is_file() else {}
    missing = [p for p in papers if p not in cached]
    if missing:
        llm = make_llm()
        print(f"Extracting metadata for {len(missing)} papers (one LLM call each, cached afterwards)")
        for i, paper in enumerate(missing, 1):
            response = llm.invoke([
                SystemMessage(content=EXTRACT_SYSTEM),
                HumanMessage(content=f"Filename: {paper}\n\nText:\n{_page1_text(paper)}"),
            ]).content
            try:
                data = json.loads(response[response.find("{"):response.rfind("}") + 1])
            except (ValueError, AttributeError):
                data = {}
            cached[paper] = {
                "title": data.get("title") if isinstance(data.get("title"), str) else None,
                "first_author": data.get("first_author") if isinstance(data.get("first_author"), str) else None,
                "authors": [a for a in (data.get("authors") or []) if isinstance(a, str)],
                "year": data.get("year") if isinstance(data.get("year"), int) else None,
                "venue": data.get("venue") if isinstance(data.get("venue"), str) else None,
                "topics": [t.strip() for t in (data.get("topics") or []) if isinstance(t, str) and t.strip()],
                "summary": data.get("summary") if isinstance(data.get("summary"), str) else None,
            }
            if i % 25 == 0 or i == len(missing):
                print(f"  {i}/{len(missing)}", flush=True)
        METADATA_PATH.write_text(json.dumps(cached, indent=1, ensure_ascii=False), encoding="utf-8")
    return cached


def _norm_topic(phrase: str) -> str:
    """Hashing-style resolution: lower case, no punctuation or stop words, crude singular."""
    words = [w for w in re.sub(r"[^a-z0-9 ]", " ", phrase.lower()).split() if w not in _STOPWORDS]
    if words:
        last = words[-1]
        if last.endswith("ies"):
            words[-1] = last[:-3] + "y"
        elif last.endswith("s") and not last.endswith(("ss", "sis", "us", "is")):
            words[-1] = last[:-1]          # 'policies' -> 'policy', but 'synthesis' stays
    return " ".join(words)


def resolve_topics(meta: dict[str, dict]) -> dict[str, str]:
    """{raw topic phrase: canonical topic name}.

    Two passes, as the lecture described: exact match after normalisation (the early
    'hashing' approach), then embedding similarity (what is common now). The more frequent
    phrase gives the merged node its name. Cached in paper_topics.json.
    """
    raw = sorted({t for m in meta.values() for t in m["topics"]})
    if TOPIC_MAP_PATH.is_file():
        saved = json.loads(TOPIC_MAP_PATH.read_text(encoding="utf-8"))
        if set(saved) == set(raw):
            return saved
    import numpy as np
    from collections import Counter
    counts = Counter(_norm_topic(t) for t in raw)
    names = sorted(counts, key=lambda n: (-counts[n], n))
    vectors = np.array(get_embeddings().embed_documents(names))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    sims = vectors @ vectors.T
    canonical: dict[str, str] = {}
    for i, name in enumerate(names):                 # most frequent first
        merged_into = next((names[j] for j in range(i) if sims[i, j] >= TOPIC_MERGE_SIMILARITY), None)
        canonical[name] = canonical[merged_into] if merged_into else name
    topic_map = {t: canonical[_norm_topic(t)] for t in raw}
    TOPIC_MAP_PATH.write_text(json.dumps(topic_map, indent=1, ensure_ascii=False), encoding="utf-8")
    return topic_map


def first_page_text(paper: str) -> str:
    """Page 1, chunk 0 of a paper: the title, authors and start of the abstract."""
    chunk_id = f"{paper} p.1 #0"
    if chunk_id in DOC_BY_ID:
        return DOC_BY_ID[chunk_id]["text"]
    return ""


# Page 1 of a proceedings front-matter PDF, or of a preprint, starts with boilerplate rather
# than a title. Ten such files would otherwise all look like copies of one paper.
BOILERPLATE_STARTS = ("lecturenotesincomputerscience", "toappear", "arxiv", "publishedas")


def title_key(paper: str) -> str:
    """A crude fingerprint of the title so two copies of one paper land on the same node.

    Returns '' when page 1 starts with boilerplate instead of a title (no title edge then).
    """
    text = first_page_text(paper).lower()
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"published as a conference paper at \w+ \d{4}", "", text)
    text = re.sub(r"[^a-z]", "", text)   # also removes the stray spaces in 'V eriﬁable'
    if text.startswith(BOILERPLATE_STARTS):
        return ""
    return text[:40]


def first_author(paper: str) -> str:
    """Surname of the first author from a DBLP key such as 'InalaBTS20' or 'Solar-LezamaRBE05'.

    arXiv keys ('abs-1805-08328') carry no author, so those papers get no author edge.
    """
    key = paper.replace(".pdf", "").split("_")[-1]
    m = re.match(r"([A-Z][a-z]+(?:-[A-Z][a-z]+)?)", key)
    return m.group(1) if m else ""


def _letters(text: str) -> str:
    """Lower-case letters only, ligatures expanded. Used to match text regardless of layout."""
    return re.sub(r"[^a-z]", "", text.lower().replace("ﬁ", "fi").replace("ﬂ", "fl"))


def paper_title(paper: str) -> str:
    """Extracted title when the LLM found one, else '' (front-matter files)."""
    return (META.get(paper, {}).get("title") or "").strip()


def paper_title_key(paper: str) -> str:
    """Fingerprint for same_title and citation edges: extracted title if present, else the
    old page-1 prefix."""
    letters = _letters(paper_title(paper))
    return letters[:40] if len(letters) >= 25 else title_key(paper)


def build_graph(papers: list[str]) -> nx.Graph:
    G = nx.Graph()
    for p in papers:
        G.add_node(f"paper:{p}", node_type="paper")
        if paper_title_key(p):
            G.add_edge(f"paper:{p}", f"title:{paper_title_key(p)}", edge_type="same_title")
        author = (META.get(p, {}).get("first_author") or first_author(p) or "").strip()
        if author:
            G.add_edge(f"paper:{p}", f"author:{author}", edge_type="first_author")
        for topic in PAPER_TOPICS.get(p, []):
            G.add_edge(f"paper:{p}", f"topic:{topic}", edge_type="has_topic")
    # Citation edges, the authoritative structure the lecture says to use before asking an
    # LLM to rediscover it. A paper cites another when the other's title fingerprint appears
    # anywhere in its text (which is where the reference list lives). No reference parsing:
    # one substring test per pair of papers.
    keys = {p: paper_title_key(p) for p in papers}
    full_text = {p: _letters(" ".join(DOC_BY_ID[c]["text"] for c in PAPER_IDS[p])) for p in papers}
    for citing in papers:
        for cited, key in keys.items():
            if citing == cited or len(key) < 25 or keys[citing] == key:
                continue                      # no key, or the two files are copies of one paper
            if key in full_text[citing]:
                G.add_edge(f"paper:{citing}", f"paper:{cited}", edge_type="cites")
    return G


def related_papers(paper: str) -> dict[str, str]:
    """{other paper: relationship} for every paper one hop away: through a shared title,
    author or topic node, or directly along a citation edge."""
    out: dict[str, str] = {}
    me = f"paper:{paper}"
    for hub in GRAPH.neighbors(me):                 # title:, author:, topic: or paper: node
        edge_type = GRAPH.edges[me, hub]["edge_type"]
        if hub.startswith("paper:"):
            out[hub.replace("paper:", "")] = edge_type
            continue
        for other in GRAPH.neighbors(hub):
            if other != me and other.startswith("paper:"):
                name = other.replace("paper:", "")
                # a title or author link is stronger than a topic link, so keep the strongest
                if name not in out or edge_type != "has_topic":
                    out[name] = edge_type
    return out


def query_topics(query: str) -> list[str]:
    """Topic nodes nearest the question by embedding similarity (HippoRAG's query concepts)."""
    if not TOPIC_VECTORS:
        return []
    import numpy as np
    q = np.array(get_embeddings().embed_query(query))
    q /= np.linalg.norm(q)
    scored = sorted(((float(v @ q), t) for t, v in TOPIC_VECTORS.items()), reverse=True)
    return [t for s, t in scored[:QUERY_TOPIC_SEEDS] if s >= QUERY_TOPIC_MIN_SIMILARITY]


def rank_papers_by_graph(seed_papers: list[str], query: str) -> dict[str, float]:
    """Topic-sensitive (personalised) PageRank over the paper graph, lecture 4.7 section 4-6.

    The random walk restarts at the papers already retrieved and at the topic nodes nearest
    the question, so a paper connected to several of those seeds by citation, author, title
    or topic scores higher than one connected to only one. Papers with no path from a seed
    score zero.
    """
    personalization = {f"paper:{p}": 1.0 for p in seed_papers if f"paper:{p}" in GRAPH}
    topics = query_topics(query)
    for topic in topics:
        if f"topic:{topic}" in GRAPH:
            personalization[f"topic:{topic}"] = 1.0
    if topics:
        print(f"  question topics: {topics}")
    if not personalization:
        return {}
    scores = nx.pagerank(GRAPH, alpha=0.85, personalization=personalization)
    return {node.replace("paper:", ""): s for node, s in scores.items()
            if node.startswith("paper:") and s > 0}


# %%
def _is_duplicate(cid: str, chosen: dict[str, str]) -> bool:
    """True when cid's text is the same passage as one already chosen.

    NOT FROM THE LAB OR LECTURE. Kept deliberately as a corpus-specific addition: the
    lab corpus of e-mails holds no duplicates, this corpus does.
    The corpus holds an arXiv and a venue copy of 26 papers and their shared pages are
    byte-identical, so without this the two copies fill the window with one passage twice.
    A passage that DIFFERS between copies (the V1 case) is not a duplicate and is kept.
    """
    # ponytail: exact match on letters only; near-duplicates with a changed number get through,
    # which is what we want for conflicting versions.
    mine = _letters(DOC_BY_ID[cid]["text"])
    return any(_letters(DOC_BY_ID[other]["text"]) == mine for other in chosen)


def hybrid_graph_retrieve(llm: ChatOpenAI, query: str, k: int = TOP_K) -> dict[str, str]:
    """Lecture 4.7 shape: decompose, retrieve per query, explore the graph, filter, answer.

    1. One LLM call splits the question into sub-queries (the original is always kept).
    2. The hybrid retriever runs once per query.
    3. Personalised PageRank from the retrieved papers nominates related papers; the best
       passage in each for the original question joins the candidate pool.
    4. One LLM call (the lecture's optional filtering step) keeps the k most useful
       candidates. Identical duplicate passages never reach it.
    The answer step is separate. Three LLM calls per question in total.
    """
    if k <= 0:
        return {}

    # 1. Decompose
    parts = list(dict.fromkeys(s.strip() for s in decompose_query(llm, query) if s.strip())) or [query]
    print(f"  decomposed into: {parts}")

    # 2. Retrieve for every query
    searches = {query: baseline_retrieve(query, SUBQUERY_POOL)}
    for part in parts:
        if part != query:
            searches[part] = baseline_retrieve(part, SUBQUERY_POOL)

    candidates: dict[str, str] = {}

    def offer(cid: str, label: str) -> None:
        if cid not in candidates and not _is_duplicate(cid, candidates):
            candidates[cid] = label

    # Lab 4.1's merge, unchanged: pool every query's hits and sum each chunk's score over
    # the queries that found it, strongest first. The label names the query that ranked the
    # chunk highest; it is a trace annotation for the log and does not affect selection.
    by_query = {q: dict(ranked) for q, ranked in searches.items()}
    pooled: dict[str, float] = defaultdict(float)
    for ranked in searches.values():
        for cid, score in ranked:
            pooled[cid] += score
    for cid, _ in sorted(pooled.items(), key=lambda item: (-item[1], item[0])):
        if len(candidates) >= k + len(searches):
            break
        best_q = max(by_query, key=lambda q: by_query[q].get(cid, 0.0))
        offer(cid, "PRIMARY (original question)" if best_q == query else f"PRIMARY ({best_q})")

    # 3. Explore the graph
    seeds = list(dict.fromkeys(paper_of(cid) for cid in candidates))
    links: dict[str, tuple[str, str]] = {}
    for seed in seeds:
        for paper, relation in related_papers(seed).items():
            if paper not in seeds and (paper not in links or relation == "same_title"):
                links[paper] = (relation, seed)
    ranked_papers = sorted(
        ((p, s) for p, s in rank_papers_by_graph(seeds, query).items() if p not in seeds),
        key=lambda item: (-item[1], item[0]),
    )[:GRAPH_PAPERS]
    for paper, _ in ranked_papers:
        relation, seed = links.get(paper, ("pagerank", "the retrieved papers"))
        for cid, _ in hybrid_retrieve_in_paper(query, paper, 1):
            offer(cid, f"CONTEXT ({relation} of {seed})")
    print(f"  graph added: {[p for p, _ in ranked_papers]}")

    # 4. Filter
    chosen = select_context(llm, query, candidates, k)
    print(f"  kept {len(chosen)} of {len(candidates)} candidates from "
          f"{len({paper_of(c) for c in chosen})} paper(s)")
    return chosen


def select_context(llm: ChatOpenAI, query: str, candidates: dict[str, str], k: int) -> dict[str, str]:
    """The lecture's filtering step: one LLM call keeps the k most useful candidates.

    Falls back to the first k candidates in pool order if the reply cannot be parsed,
    so a malformed answer never empties the context.
    """
    ordered = list(candidates)
    if len(ordered) <= k:
        return dict(candidates)
    excerpts = [f"{n}. [{cid}] {'CONTEXT' if candidates[cid].startswith('CONTEXT') else 'PRIMARY'}\n"
                f"{DOC_BY_ID[cid]['text'][:1200]}" for n, cid in enumerate(ordered, 1)]
    response = llm.invoke([
        SystemMessage(content=FILTER_SYSTEM.replace("{K}", str(k))),
        HumanMessage(content=f"Question: {query}\n\nCandidates:\n" + "\n\n".join(excerpts)),
    ]).content
    try:
        result = json.loads(response[response.find("{"):response.rfind("}") + 1])
        keep = [n for n in result.get("keep", []) if isinstance(n, int) and 1 <= n <= len(ordered)]
    except (ValueError, AttributeError, TypeError):
        keep = []
    # Top k of the ranking; candidates the model left out are appended in pool order, so the
    # window always holds k passages, the same budget as the baseline.
    keep += [n for n in range(1, len(ordered) + 1) if n not in keep]
    keep = keep[:k]
    return {ordered[n - 1]: candidates[ordered[n - 1]] for n in keep}

def hybrid_retrieve_in_paper(
    query: str, paper: str, k: int = 1,
    query_vector: list[float] | None = None, bm25_scores: Any = None,
) -> list[tuple[str, float]]:
    """BM25 and vector rank fusion within one graph-nominated PDF."""
    ids = PAPER_IDS.get(paper, [])
    if not ids:
        return []
    scores = bm25_scores if bm25_scores is not None else BM25.get_scores(tokenize(query))
    local = sorted(ids, key=lambda cid: scores[CORPUS_INDEX[cid]], reverse=True)[:CANDIDATE_POOL]
    vector_query = query_vector if query_vector is not None else get_embeddings().embed_query(query)
    vector = [_chunk_id(doc.metadata) for doc in DB.similarity_search_by_vector(
        vector_query, k=CANDIDATE_POOL, filter={"source": paper}
    )]
    fused: dict[str, float] = defaultdict(float)
    for ranked in (local, vector):
        for rank, cid in enumerate(ranked, 1):
            fused[cid] += 1.0 / (RRF_K + rank)
    return sorted(
        ((cid, score + min(max(float(scores[CORPUS_INDEX[cid]]), 0.0), 20.0) / 1000)
         for cid, score in fused.items()),
        key=lambda item: (-item[1], item[0]),
    )[:k]


CORPUS_INDEX = {cid: index for index, cid in enumerate(CORPUS_IDS)}
PAPER_IDS: dict[str, list[str]] = defaultdict(list)
for _cid in CORPUS_IDS:
    PAPER_IDS[paper_of(_cid)].append(_cid)

ALL_PAPERS = sorted(PAPER_IDS)
META = extract_paper_metadata(ALL_PAPERS)
TOPIC_MAP = resolve_topics(META)
PAPER_TOPICS = {p: sorted({TOPIC_MAP[t] for t in META[p]["topics"] if t in TOPIC_MAP}) for p in ALL_PAPERS}
_topic_names = sorted({t for ts in PAPER_TOPICS.values() for t in ts})
TOPIC_VECTORS: dict[str, Any] = {}
if _topic_names:
    import numpy as _np
    _vecs = _np.array(get_embeddings().embed_documents(_topic_names))
    _vecs /= _np.linalg.norm(_vecs, axis=1, keepdims=True)
    TOPIC_VECTORS = dict(zip(_topic_names, _vecs))
print(f"Paper metadata: {sum(1 for p in ALL_PAPERS if paper_title(p))} titled papers, "
      f"{sum(len(m['topics']) for m in META.values())} raw topic phrases resolved to {len(_topic_names)} topic nodes")
GRAPH = build_graph(ALL_PAPERS)
print(f"Paper graph: {GRAPH.number_of_nodes()} nodes, {GRAPH.number_of_edges()} edges")


def _title_line(paper: str) -> str:
    """One line per paper for the decomposer: extracted title, year and resolved topics."""
    m = META.get(paper, {})
    year = f" ({m['year']})" if m.get("year") else ""
    topics = ", ".join(PAPER_TOPICS.get(paper, [])[:5])
    return f"{paper_title(paper)}{year}" + (f"; topics: {topics}" if topics else "")


# Lecture 4.7: the more the decomposer knows about the data, the better the decomposition.
# Front-matter files with no extracted title are left out; they help nobody.
TITLE_LIST = "\n".join(f"{p}: {_title_line(p)}" for p in ALL_PAPERS if paper_title(p))
print(f"Decomposer paper list: {TITLE_LIST.count(chr(10)) + 1} papers, {len(TITLE_LIST)} characters")


def write_retrieval_graph(path: Path = HERE / "paper_retrieval_graph.json") -> Path:
    """Save the graph built from the indexed research papers for inspection."""
    data = {
        "papers": len(ALL_PAPERS),
        "nodes": [{"id": node, **attributes} for node, attributes in GRAPH.nodes(data=True)],
        "edges": [{"from": source, "to": target, **attributes}
                  for source, target, attributes in GRAPH.edges(data=True)],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# %% [markdown]
# ## Step 4 — Your advanced-retrieval plan
#
# The plan below is what the code above implements on the Research Paper Navigator corpus.


# %%
def answer_from_docs(llm: ChatOpenAI, query: str, doc_ids: dict[str, str] | list[str]) -> str:
    """Ask the LLM using only the retrieved chunks. Each excerpt is headed
    [filename, p.N] plus PRIMARY or CONTEXT so the report can show the two roles."""
    if isinstance(doc_ids, list):
        doc_ids = {i: "PRIMARY" for i in doc_ids}
    parts = []
    for i, label in doc_ids.items():
        if i not in DOC_BY_ID:
            continue
        meta = DOC_BY_ID[i]["meta"]
        # The model sees only the role, never the sub-query or gap text that nominated the
        # excerpt. On R1-v1 a label reading "PRIMARY (WebAssembly synthesis paper findings)"
        # was quoted back as if the paper had said it, and the answer fabricated from it.
        # The full label stays in the log for the worksheet.
        if label.startswith("CONTEXT"):
            shown = label.split("; fills:")[0].rstrip(")") + ")"
        else:
            shown = "PRIMARY"
        parts.append(f"[{meta['source']}, p.{meta['page']}] {shown}\n{DOC_BY_ID[i]['text']}")
    context = "\n\n---\n\n".join(parts)
    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Paper excerpts:\n{context}\n\nQuestion: {query}"),
    ]
    return llm.invoke(messages).content


# %% [markdown]
# ## The Checkpoint 3.1 metric suite (ported unchanged)
#
# Seven metrics over four criteria: grounding (`citation_valid`, `quote_verbatim`,
# `closed_book`), correctness (`judge`), completeness (`point_coverage`,
# `source_coverage`) and refusal (`refused`). `retrieval_hit` separates a retrieval
# failure from a generation failure. See the 3.1 script for the reasoning behind each.

# %%
def judge(llm: ChatOpenAI, answer_text: str, grading_notes: str) -> tuple[str, str]:
    """CORRECTNESS. Returns (verdict, reason); reason is '' on a pass.

    The verdict is read from a JSON field rather than by searching the reply for the word
    'pass'. Now that the judge explains a failure, a substring test would misread a reason
    such as "does not pass the requirement about the 2009 paper" as a pass.
    """
    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=f"ANSWER:\n{answer_text}\n\nGRADING NOTES:\n{grading_notes}"),
    ]
    response = llm.invoke(messages).content
    try:
        result = json.loads(response[response.find("{"):response.rfind("}") + 1])
        verdict = str(result.get("verdict", "")).strip().lower()
        if verdict in ("pass", "fail"):
            reason = str(result.get("reason", "")).strip()
            return verdict, ("" if verdict == "pass" else reason)
    except (ValueError, AttributeError, TypeError):
        pass
    # Malformed reply: fall back to the old substring read so a parse failure cannot be
    # silently scored as a pass, and say so in the reason.
    flat = response.strip().lower()
    verdict = "pass" if flat.startswith("pass") else "fail"
    return verdict, ("" if verdict == "pass" else f"(judge reply unparseable) {response.strip()[:200]}")


def _norm(text: str) -> str:
    # Every quotation mark, straight or curly, single or double, becomes one symbol. The gold
    # sentence for Q1 has “wake-sleep” and the model wrote ‘wake-sleep’; that alone failed
    # the verbatim check while the judge passed the same answer.
    for a, b in (("ﬁ", "fi"), ("ﬂ", "fl"), ("’", "'"), ("‘", "'"),
                 ("“", "'"), ("”", "'"), ('"', "'"), ("—", "-"), ("–", "-")):
        text = text.replace(a, b)
    text = re.sub(r"[*`]", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _retrieved_pages(chunk_ids: list[str]) -> set[tuple[str, str]]:
    pairs = set()
    for cid in chunk_ids:
        meta = DOC_BY_ID.get(cid, {}).get("meta", {})
        if meta:
            pairs.add((str(meta.get("source", "?")), str(meta.get("page", "?"))))
    return pairs


def cited_pages(answer_text: str) -> list[tuple[str, str]]:
    out = []
    for group in re.findall(r"\[([^\[\]]+)\]", answer_text):
        files = re.findall(r"([A-Za-z0-9_.\-]+\.pdf)", group)
        pages = re.findall(r"p\.\s*(\d+)", group)
        if files and pages:
            out.extend((files[0], page) for page in pages)
    return out


def citation_valid(answer_text: str, chunk_ids: list[str]) -> tuple[bool | None, list[str]]:
    cites = cited_pages(answer_text)
    if not cites:
        return None, []
    available = _retrieved_pages(chunk_ids)
    bad = [f"[{f}, p.{p}]" for f, p in cites if (f, p) not in available]
    return not bad, bad


def quote_verbatim(answer_text: str, item: dict) -> bool | None:
    accepted = item.get("must_quote_any")
    if not accepted:
        return None
    haystack = _norm(answer_text)
    return any(_norm(q) in haystack for q in accepted)


def source_coverage(chunk_ids: list[str], answer_text: str, item: dict) -> tuple[int, int, int] | None:
    gold = item.get("gold_sources") or []
    if not gold:
        return None
    retrieved_files = {f for f, _ in _retrieved_pages(chunk_ids)}
    cited_files = {f for f, _ in cited_pages(answer_text)}
    return (
        len([g for g in gold if g in retrieved_files]),
        len([g for g in gold if g in cited_files]),
        len(gold),
    )


def retrieval_hit(chunk_ids: list[str], item: dict) -> bool | None:
    if not item.get("gold_page") or not item.get("gold_sources"):
        return None
    want = {(g, str(item["gold_page"])) for g in item["gold_sources"]}
    return bool(want & _retrieved_pages(chunk_ids))


_REFUSAL_PATTERNS = (
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
    opening = _norm(answer_text)[:220]
    return any(re.search(p, opening) for p in _REFUSAL_PATTERNS)


POINTS_SYSTEM = (
    "You check whether an answer contains specific facts. You are given an ANSWER and a "
    "numbered list of REQUIRED POINTS. For each point, reply on its own line in the form "
    "'<number>: yes' if the answer supplies that point, or '<number>: no' if it does not. "
    "Judge each point independently. Accept different wording and accept a point that is "
    "clearly implied. Output nothing except those lines."
)


def point_coverage(llm: ChatOpenAI, answer_text: str, item: dict) -> tuple[int, int]:
    points = item["required_points"]
    numbered = "\n".join(f"{n}. {p}" for n, p in enumerate(points, 1))
    reply = llm.invoke([
        SystemMessage(content=POINTS_SYSTEM),
        HumanMessage(content=f"ANSWER:\n{answer_text}\n\nREQUIRED POINTS:\n{numbered}"),
    ]).content
    got = [False] * len(points)
    for n, verdict in re.findall(r"(\d+)\s*:\s*(yes|no)", reply.lower()):
        i = int(n) - 1
        if 0 <= i < len(got):
            got[i] = verdict == "yes"
    return sum(got), len(points)


def load_items(path: Path) -> list[dict]:
    if not path.is_file():
        raise RuntimeError(f"Evaluation file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# %% [markdown]
# ## Step 5b — Comparative evaluation: baseline vs. advanced on the 3.1 set
#
# The closed-book metric is not repeated here. It measures the model without any
# retrieval, so its result cannot change between the two retrievers; the 3.1 number
# (1 of 36 correct with no documents) still stands.

# %%
def _show(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "NO"
    if isinstance(value, tuple):
        if len(value) == 2:
            return f"{value[0]}/{value[1]}"
        return f"{value[0]} of {value[2]} retrieved, {value[1]} cited"
    return str(value)


# --- console formatting -----------------------------------------------------------------
# Output width. 96 keeps a chunk id plus its role on one line without wrapping.
WIDTH = 96


def _rule(char: str = "-", label: str = "") -> str:
    if not label:
        return char * WIDTH
    return f"{char * 3} {label} " + char * max(0, WIDTH - len(label) - 5)


def _wrap(text: str, indent: str = "     ") -> str:
    """Wrap to WIDTH with a HANGING indent: the label prints once, continuations align under
    it. Repeating the label on every wrapped line is what made the old output hard to scan."""
    return textwrap.fill(" ".join(str(text).split()), width=WIDTH,
                         initial_indent=indent, subsequent_indent=" " * len(indent))


def _chunk_lines(retrieved: dict[str, str]) -> list[str]:
    """Retrieved chunks grouped by paper, so a window dominated by one source is obvious."""
    by_paper: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for cid, role in retrieved.items():
        by_paper[paper_of(cid)].append((cid, role))
    lines = []
    for paper, entries in by_paper.items():
        lines.append(f"     {paper}  ({len(entries)} of {len(retrieved)} slots)")
        for cid, role in entries:
            page = cid.split(" ", 1)[1] if " " in cid else cid
            lines.append(f"        {page:<12} {role}")
    return lines


def _verdict_tag(r: dict) -> str:
    """PASS/FAIL plus a marker when the verdict is not what 3.1 predicted."""
    tag = "PASS" if r["correctness"] == "pass" else "FAIL"
    if r["correctness"] != r["expect"]:
        tag += f"  (3.1 predicted {r['expect']})"
    return tag


# Metrics grouped by the criterion each one serves, with a plain-language gloss. The bare
# metric names mean nothing to a reader coming to the log cold.
_METRIC_GROUPS = [
    ("Correctness", [
        ("correctness", "answers the judge accepted"),
        ("predictions_held", "verdicts that matched the 3.1 prediction"),
    ]),
    ("Completeness", [
        ("point_coverage", "required facts supplied"),
        ("gold_files_retrieved", "gold papers that reached the window"),
        ("retrieval_hit", "items whose known gold page was retrieved"),
    ]),
    ("Grounding", [
        ("citation_valid", "answers citing only pages that were in context"),
        ("quote_verbatim", "quotation requests reproduced exactly"),
    ]),
    ("Refusal", [
        ("false_refusals", "refused although the corpus CAN answer"),
        ("fabrications", "answered although the corpus CANNOT"),
    ]),
    ("Context", [
        ("distinct_papers_in_context", "slots holding a distinct paper"),
    ]),
]


def _print_summary(results: list[dict], label: str) -> None:
    """One readable summary block: per-question table, then metrics grouped by criterion."""
    summary = summarize(results)

    print()
    print(_rule("="))
    print(f"  SUMMARY  |  {label}")
    print(_rule("="))

    print(f"\n  {'id':5} {'verdict':8} {'points':>7} {'gold':>6} {'papers':>7}   why it failed")
    print("  " + "-" * (WIDTH - 2))
    for r in results:
        papers = len({paper_of(c) for c in r["retrieved"]})
        sc = r["source_coverage"]
        gold = "n/a" if sc is None else f"{sc[0]}/{sc[2]}"   # compact: the table needs columns
        why = r.get("correctness_reason") or ""
        if len(why) > 38:
            why = why[:35] + "..."
        print(f"  {r['id']:5} {r['correctness'].upper():8} "
              f"{_show(r['point_coverage']):>7} {gold:>6} "
              f"{str(papers) + '/' + str(TOP_K):>7}   {why}")

    for group, metrics in _METRIC_GROUPS:
        print(f"\n  {group}")
        for key, gloss in metrics:
            if key in summary:
                print(f"    {key:28s} {summary[key]:>8s}   {gloss}")

    failed = [r["id"] for r in results if r["correctness"] != "pass"]
    print(f"\n  Still failing: {', '.join(failed) if failed else 'none'}")
    print("  Baseline reference: run the Checkpoint 3.1 script and compare its summary.")
    print(_rule("="))

    log(f"SUMMARY ({label})", "\n".join(f"{m}: {v}" for m, v in summary.items())
        + "\nfailed=" + ",".join(failed))


def summarize(results: list[dict]) -> dict[str, str]:
    """The aggregate line for one system, as {metric: 'x/y'}."""
    n = len(results)
    cited = [r for r in results if r["citation_valid"] is not None]
    quoted = [r for r in results if r["quote_verbatim"] is not None]
    covered = [r for r in results if r["source_coverage"] is not None]
    hits = [r for r in results if r["retrieval_hit"] is not None]
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]
    got = sum(r["point_coverage"][0] for r in results)
    want = sum(r["point_coverage"][1] for r in results)
    files_got = sum(r["source_coverage"][0] for r in covered)
    files_want = sum(r["source_coverage"][2] for r in covered)
    distinct = sum(len({paper_of(c) for c in r["retrieved"]}) for r in results)
    return {
        "correctness": f"{sum(r['correctness'] == 'pass' for r in results)}/{n}",
        "predictions_held": f"{sum(r['correctness'] == r['expect'] for r in results)}/{n}",
        "citation_valid": f"{sum(bool(r['citation_valid']) for r in cited)}/{len(cited)}",
        "quote_verbatim": f"{sum(bool(r['quote_verbatim']) for r in quoted)}/{len(quoted)}",
        "point_coverage": f"{got}/{want}",
        "gold_files_retrieved": f"{files_got}/{files_want}",
        "retrieval_hit": f"{sum(bool(r['retrieval_hit']) for r in hits)}/{len(hits)}",
        "false_refusals": f"{sum(r['refused'] for r in answerable)}/{len(answerable)}",
        "fabrications": f"{sum(not r['refused'] for r in unanswerable)}/{len(unanswerable)}",
        "distinct_papers_in_context": f"{distinct}/{n * TOP_K}",
    }


def evaluate_item(llm: ChatOpenAI, item: dict) -> dict:
    """Score the additional retrieval path with the unchanged 3.1 metrics."""
    chunks = hybrid_graph_retrieve(llm, item["question"], TOP_K)
    chunk_ids = list(chunks)
    ans = answer_from_docs(llm, item["question"], chunks) if chunks else "(no documents retrieved)"
    cite_ok, bad_cites = citation_valid(ans, chunk_ids)
    verdict, why = judge(llm, ans, item["grading_notes"])
    return {
        "id": item["id"],
        "system": "hybrid_graph",
        "question": item["question"],
        "expect": item["expect"],
        "answerable": item["answerable"],
        "retrieved": chunks,
        "answer": ans,
        "correctness": verdict,
        "correctness_reason": why,
        "citation_valid": cite_ok,
        "bad_citations": bad_cites,
        "quote_verbatim": quote_verbatim(ans, item),
        "point_coverage": point_coverage(llm, ans, item),
        "source_coverage": source_coverage(chunk_ids, ans, item),
        "retrieval_hit": retrieval_hit(chunk_ids, item),
        "refused": refused(ans),
    }





def run_evaluation(eval_set: list[dict], label: str) -> None:
    """Score the hybrid + decomposition + graph path on the 3.1 question set.
    """
    llm = make_llm()
    changed = []

    print("\n" + _rule("="))
    print(f"  HYBRID + DECOMPOSITION + GRAPH  |  {label}")
    print(f"  scenario={SCENARIO}  top_k={TOP_K}  model={LLM_MODEL}  temp={TEMPERATURE}")
    print(_rule("="))

    for n, item in enumerate(eval_set, 1):
        after = evaluate_item(llm, item)
        changed.append(after)

        print()
        print(_rule("-", f"{item['id']}  [{n}/{len(eval_set)}]  {_verdict_tag(after)}"))
        print(_wrap(item["question"], "  Q: ").replace("  Q: ", "  Q: ", 1))
        print(f"     points {_show(after['point_coverage']):>7}"
              f"   gold sources {_show(after['source_coverage'])}"
              f"   cites_ok {_show(after['citation_valid'])}"
              f"   quote {_show(after['quote_verbatim'])}"
              f"   refused {_show(after['refused'])}")
        if after["correctness_reason"]:
            print(_wrap(after["correctness_reason"], "     why: "))
        print("     context:")
        for line in _chunk_lines(after["retrieved"]):
            print(line)
        print("     answer:")
        print(_wrap(after["answer"][:600] + ("..." if len(after["answer"]) > 600 else ""),
                    "        "))

        log(f"{item['id']} [hybrid_graph]: {item['question']}", "\n".join([
            f"retrieved={json.dumps(after['retrieved'], ensure_ascii=False)}",
            f"correctness={after['correctness']} (predicted {after['expect']})",
            f"correctness_reason={after['correctness_reason'] or '(pass)'}",
            f"point_coverage={_show(after['point_coverage'])}",
            f"source_coverage={_show(after['source_coverage'])}",
            f"answer={after['answer']}",
        ]))

    _print_summary(changed, f"{label}, hybrid_graph")


# %%
if "--variants" in sys.argv:
    # The 24 paraphrases of the 12 base questions. With a judge that moves the score by two
    # or three questions on its own, 36 questions is what turns a swing into a signal.
    run_evaluation(load_items(HERE / "eval_questions_variants.json"), "24 paraphrase variants")
else:
    run_evaluation(load_items(QUESTIONS_PATH), "12 base questions")

# %% [markdown]
# ## Step 6
#
# The checkpoint deliverable is your completed Capstone Checkpoint 4.1 worksheet, not
# code. Using evidence from your real system, cover:
#
# 1. **System overview** State your scenario and your 2.1 baseline retriever.
# 2. **Failure diagnosis** Describe the meaningful retrieval failure or limitation identified through
#    your Checkpoint 3.1 evaluation and explain why it occurs.
# 3. **Data redesign** Describe the node/edge schema you defined to expose relationships in
#    your corpus (citations/authors/topics, or links/categories/entities), and why you chose it.
# 4. **Advanced retrieval implemented** Provide a query decomposition and/or graph traversal,
#    with code or log evidence. Label primary vs. context documents in the prompt.
# 5. **Comparative evaluation** Re-run your 3.1 evaluation on baseline vs advanced,
#    **comparing a similar number of retrieved documents**, and report before/after.
# 6. **Analysis & reflection** Discuss where it helped, where it hurt (redundant/over-broad
#    context), the cost/latency trade-off, and remaining limitations.

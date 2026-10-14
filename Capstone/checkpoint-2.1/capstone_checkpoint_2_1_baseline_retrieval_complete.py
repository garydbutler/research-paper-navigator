r"""Capstone Checkpoint 2.1 — Retrieval Strategy Design and Baseline Implementation (completed).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code / PyCharm / Jupytext.
"""

# %% [markdown]
# # Capstone Checkpoint 2.1 — Retrieval Strategy Design and Baseline Implementation
# **MO-LLM Module 2 / Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# In Checkpoint 1.1, you showed that a plain LLM can't reliably answer questions about
# your corpus. Now, you will **add retrieval**: Design a retrieval strategy for your scenario
# and build a **baseline retrieval system** that finds the most relevant documents for
# a query, so the model can ground its answers in them.
#
# This mirrors the Module 2 labs — keyword (BM25), vector (semantic), and hybrid
# retrieval — applied to your own capstone corpus. The graded deliverable is the completed 
# Capstone Checkpoint 2.1 worksheet, which includes your written responses and evidence of your 
# retrieval system implementation and testing. This script provides a small working example of 
# baseline retrieval. Use it to understand the retrieval workflow, then adapt the code to implement 
# and test a baseline retriever using your selected capstone dataset.
#
# **Learning outcomes (Module 2):**
# 1. Design a retrieval strategy appropriate for a given dataset and query type.
# 2. Implement and test a baseline retrieval system using structured and/or semantic
#    approaches.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario** you chose in Checkpoint 1.1.
#
# | Scenario | Corpus | Retrieval considerations |
# |---|---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs (`Labs/CapstoneDatasets/ResearchPapers/`) | long documents; you'll likely chunk them; questions often name a specific paper or compare papers. |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles (`Labs/CapstoneDatasets/Wikipedia/`) | many short-to-medium articles; questions name a figure/place or span several articles. |
#
# A good baseline is keyword (BM25), semantic (embeddings + vector search), or a
# hybrid of both — exactly what you built in Labs 1.2–2.2.

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12**
# 2. `pip install langchain-openai langchain-core python-dotenv langchain-chroma langchain-text-splitters rank-bm25 pypdf`
# 3. Use the OpenRouter API key provided for this program. This checkpoint uses
#  the `openai/gpt-5.4-mini` model, with usage covered by the course credits.
# 4. Create a `.env` file at the repository root: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# This runs on a tiny built-in sample corpus, so you do not need to prepare your own
# dataset. It still requires an OpenRouter API key to run the LLM (it is not offline or
# free of API calls). Your real baseline (over your full corpus) is what you describe in
# the writeup.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import re
import shutil
import sys
import textwrap
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterator

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
LLM_MODEL = "openai/gpt-5.4-mini"  # latest small OpenAI model, fast; covered by course credits
EMBEDDING_MODEL = "openai/text-embedding-3-small"  # via OpenRouter (~$0.02/M tokens)
TEMPERATURE = 0.2
TOP_K = 6          # was 3. One chunk rarely holds a whole answer, and six chunks of 1,200
                   # characters is about 7KB of context, which the model handles comfortably.
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
LOG_PATH = HERE / "checkpoint_2_1_retrieval.log"

# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "research_papers"   # "research_papers" or "wikipedia"

# Create the paths to the papers and the chroma vector db
PAPERS_DIR = REPO / "data" / "ResearchPapers"   # one shared copy for every checkpoint
CHROMA_DIR = HERE / "chroma_chunks"

# Chunking. See Step 2a for the reasoning behind each of these.
CHUNK_SIZE = 1200         # characters; large enough for a claim plus its qualifying sentence
CHUNK_OVERLAP = 200       # keeps a definition from being severed from the term it defines
MIN_PAGE_CHARS = 100      # below this a page is a cover, a figure, or a scan artifact

# === SWITCH: which retrievers take part ===
#   "keyword"  BM25 only            (Keyword search from Lab 1.2 baseline)
#   "vector"   Chroma only          (Vector search from Lab 2.1 baseline)
#   "hybrid"   both, fused by RRF   (Hybrid from the Lab 2.2 approach)
# Select the retrieval method for comparisons

RETRIEVAL_MODE = "hybrid"

# CANDIDATE_POOL is how many candidate docs each retriever nominates before fusion; RRF_K is the
# standard reciprocal rank fusion constant.
CANDIDATE_POOL = 20
RRF_K = 60

ANSWER_SYSTEM = (
    "You are a research assistant for a collection of computer-science papers. Answer "
    "the question using ONLY the provided paper excerpts.\n"
    "* Cite the source of every claim formatted as [filename, p.N], using the labels on the "
    "excerpts.\n"
    "* When you quote something, quote verbatim from an excerpt and put it in quotation marks. "
    "Never reconstruct or paraphrase a quotation and present it as verbatim.\n"
    "* If the excerpts do not contain the answer, say so explicitly rather than "
    "guess. A refusal is the best answer when no information is available."
)

# %%
def check_api_key() -> str:
    load_dotenv(REPO / ".env")     # the key lives at the repository root, not beside this file
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Use the OpenRouter API key "
            "provided for this course, put it in a .env file next to this "
            "script, and rerun."
        )
    return key


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


# %% [markdown]
# ## Step 2a — The real corpus, parsed and chunked
#
# Corpus: the 153 research-paper PDFs 
#
# The embedding model takes about 8,191 tokens so the pdf docs will be split into appropriately
# sized chunks that fit within the model's context window. No effort made to optimize chunk sizes.
#

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
    skipped: list[str] = []

    for i, path in enumerate(files, 1):
        try:
            reader = PdfReader(path)
        except Exception as exc:
            skipped.append(f"{path.name}: {exc}")
            continue

        before = len(docs)
        for page_no, text in _page_texts(reader):
            docs.extend(
                Document(
                    page_content=chunk,
                    metadata={"source": path.name, "page": page_no, "chunk": j},
                )
                for j, chunk in enumerate(splitter.split_text(text))
            )
        print(f"  [{i:3d}/{len(files)}] {path.name:52s} {len(docs) - before:5d} chunks", flush=True)

    if skipped:
        print(f"\n  WARNING: {len(skipped)} file(s) could not be parsed:")
        for s in skipped:
            print(f"    {s}")
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

    print(f"Building vector DB from {PAPERS_DIR} (first run — this takes a while)")
    docs = load_and_chunk(PAPERS_DIR)
    print(f"\n  {len(docs)} chunks total. Embedding...")
    db = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=get_embeddings())
    for start in range(0, len(docs), 500):   # batch so a failure loses 500, not everything
        db.add_documents(docs[start:start + 500])
        print(f"    embedded {min(start + 500, len(docs))}/{len(docs)}")
        sys.stdout.flush()
    return db


# %% [markdown]
# ## Step 2b — Load the chunks into memory for keyword search
#
# The vector store holds the embeddings, but BM25 needs the raw text. Rather than re-parse
# 6,108 pages of PDF on every run, the chunks are read back out of Chroma. `DOC_BY_ID` plays
# the same role the starter's version did, keyed by a chunk id of `filename p.N #i`.

# %%
DB = build_or_load_db()
_stored = DB.get(include=["documents", "metadatas"])


def _chunk_id(meta: dict) -> str:
    return f"{meta.get('source', '?')} p.{meta.get('page', '?')} #{meta.get('chunk', 0)}"


DOC_BY_ID = {
    _chunk_id(meta): {"id": _chunk_id(meta), "text": text, "meta": meta}
    for text, meta in zip(_stored["documents"], _stored["metadatas"])
}
CORPUS_IDS = list(DOC_BY_ID)
CORPUS_TEXTS = [DOC_BY_ID[i]["text"] for i in CORPUS_IDS]
print(f"Corpus ready: {len(CORPUS_IDS)} chunks from {PAPERS_DIR.name}/")


# %% [markdown]
# ## Step 2c — The baseline retriever: hybrid BM25 + vector, fused by rank
#
# This replaces the starter's shared-word-count retriever. The design follows from what
# Checkpoint 1.1 got wrong and from what the Module 2 labs showed.
#
# **Why hybrid.** The Checkpoint 1.1 baseline failed in two distinct ways. It invented an
# acronym definition (answering that PAN in the EnCompass paper means "Program-Aware
# Navigation" when the paper says *probabilistic angelic nondeterminism*), and it invented
# verbatim quotations. Both are exact-string problems, which is what keyword search does well
# and embeddings do poorly, because an embedding compresses a passage into a summary of its
# meaning and a rare technical token barely registers in that summary. But the scenario also
# has to answer conceptual questions that span several papers, where different research
# groups describe the same idea in different words, and there keyword search fails and
# semantic search works. Neither retriever alone covers both, so the baseline is hybrid.
#
# **Why fuse by rank rather than by score.** Lab 2.2 combined the two retrievers by scaling
# each one's scores to [0, 1] and taking a weighted sum. Min-max scaling has no notion of
# absolute quality: it stretches whatever candidates a retriever returns across the full
# range whether they are relevant or not. On one lab query the keyword retriever returned
# four irrelevant emails, its best guess was scaled to 1.0 anyway, and three irrelevant
# results displaced the vector retriever's correct hit, so the hybrid answered *worse* than
# its semantic half alone. Reciprocal Rank Fusion scores a chunk as `1/(RRF_K + rank)` from
# each retriever that returned it. Only position matters, never magnitude, so the two scales
# never have to be reconciled. See `../../modules/module-2/LAB_2_FINDINGS.md`.

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
    """Lowercase, split on non-alphanumerics, drop stopwords.

    NOTE: this does no stemming, which is a real limitation and not an oversight. It is
    carried over from the Module 2 labs unchanged so that the evidence run reproduces what
    the worksheet documents. It is the reason query 5 below misses the best paper in the
    corpus: `sketch` never matches `sketching`. Adding a stemmer is the first item in the
    worksheet's improvement list.
    """
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


BM25 = BM25Okapi([tokenize(t) for t in CORPUS_TEXTS])


def _bm25_ranked(query: str, k: int) -> list[str]:
    """Chunk ids in BM25 rank order, best first."""
    scores = BM25.get_scores(tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [CORPUS_IDS[i] for i in top]


def _vector_ranked(query: str, k: int) -> list[str]:
    """Chunk ids in vector-similarity rank order, best first."""
    return [_chunk_id(d.metadata) for d in DB.similarity_search(query, k=k)]


def _ranked_lists(query: str, k: int) -> dict[str, list[str]]:
    """The ranked chunk-id lists that RETRIEVAL_MODE says should take part in fusion.

    With one retriever enabled, fusion is a no-op: RRF is strictly decreasing in rank, so
    fusing a single list returns that retriever's own top-k in its own order. That is what
    makes the switch a genuine comparison rather than three separate code paths.
    """
    lists: dict[str, list[str]] = {}
    if RETRIEVAL_MODE in ("keyword", "hybrid"):
        lists["bm25"] = _bm25_ranked(query, k)
    if RETRIEVAL_MODE in ("vector", "hybrid"):
        lists["vec"] = _vector_ranked(query, k)
    return lists


def retrieve(query: str, k: int = TOP_K) -> list[tuple[str, float]]:
    """Retrieval per RETRIEVAL_MODE, fused by Reciprocal Rank Fusion.

    Returns (chunk_id, rrf_score) pairs, same shape the starter's retrieve() returned.
    """
    rrf: dict[str, float] = defaultdict(float)
    for ranked in _ranked_lists(query, CANDIDATE_POOL).values():
        for rank, chunk_id in enumerate(ranked, 1):
            rrf[chunk_id] += 1.0 / (RRF_K + rank)
    ordered = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)
    return ordered[:k]


def retrieval_detail(query: str, k: int = TOP_K) -> list[tuple[str, float, int | None, int | None]]:
    """Same as retrieve(), plus the rank each retriever gave the chunk (None = not returned).

    The per-retriever ranks are the useful part of the evidence: they show which retriever
    was responsible for each result, rather than only showing that retrieval happened. A
    disabled retriever reports None for every chunk.
    """
    lists = _ranked_lists(query, CANDIDATE_POOL)
    bm_rank = {cid: r for r, cid in enumerate(lists.get("bm25", []), 1)}
    vec_rank = {cid: r for r, cid in enumerate(lists.get("vec", []), 1)}
    return [
        (cid, score, bm_rank.get(cid), vec_rank.get(cid))
        for cid, score in retrieve(query, k)
    ]


def answer(llm: ChatOpenAI, query: str, doc_ids: list[str]) -> str:
    """Ask the LLM using only the retrieved chunks.

    Excerpts are labelled `[filename, p.N]` rather than with the internal chunk id, so the
    citations the model produces point at something a reader can actually go and open.
    """
    parts = []
    for i in doc_ids:
        if i not in DOC_BY_ID:
            continue
        meta = DOC_BY_ID[i]["meta"]
        label = f"{meta.get('source', '?')}, p.{meta.get('page', '?')}"
        parts.append(f"[{label}]\n{DOC_BY_ID[i]['text']}")
    context = "\n\n---\n\n".join(parts)
    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Paper excerpts:\n{context}\n\nQuestion: {query}"),
    ]
    return llm.invoke(messages).content


# %% [markdown]
# ## Step 3 — Your representative queries
#
# Submission item #2 asks for **3-5 representative queries** for your scenario and the
# results your system retrieves for each. Write those queries here. Some good ones include questions that:
#
# - Are answerable from **one** document (tests precision),
# - Need **several** documents (tests recall / aggregation),
# - Have wording that **differs** from the document's wording (i.e., tests whether
#   keyword vs. semantic retrieval matters for your corpus)
#
# Return a list of 3-5 query strings.

# %%
def my_representative_queries() -> list[str]:
    """Five representative queries for the Research Paper Navigator, plus one optional.

    Queries 1 and 2 deliberately re-ask the two questions the Checkpoint 1.1 no-retrieval
    baseline answered with fabrications, so the two checkpoints are directly comparable.
    The rest cover the other retrieval behaviors the worksheet asks about: multi-document
    aggregation, a definition whose wording differs from the source, and a topic the corpus
    does not contain at all.
    """
    return [
        # Single named document, exact term to test precision. 
        "In the paper Computer Aided Verification by Alexandra Silva, what does the abbreviation DNNV stand for?",

        # Info from a single-document, that produces a real quote and not a fabricated one. 
        "In the paper 'Is Self-Repair a Silver Bullet for Code Generation?', what does the "
        "paper say is the bottleneck for self-repair, and how much did human feedback improve the "
        "repair rate?",

        # Multi-document aggregation. Tests recall: no single paper is the answer.
        "How do these papers reduce the search space in program synthesis, and what does each approach sacrifice to do so?",

        # Broad and exploratory. No passage contains the answer; it has to be assembled.
        "What are the main challenges in synthesizing programs from input-output examples?",

        # Wording differs from the source. The corpus contains "The Sketching Approach to
        # Program Synthesis", but this query says "sketch-based", and the tokenizer does no
        # stemming. Tests whether keyword and semantic retrieval diverge on this corpus.
        "What is sketch-based program synthesis and how does a sketch constrain the search "
        "space?",

        # Optional sixth: not in the corpus. Tests refusal. A confident survey here would be
        # a worse failure than retrieving nothing.
        "What do these papers conclude about applying deep learning to diagnostic medical "
        "imaging?",

        "In the paper sk_p: a neural program corrector for MOOCs, quote the sentence "
        "where the term MOOC is defined.",

        "What is the main contribution of the NeurIPS 2024 paper Retrieval-Augmented "
        "Reasoning for Frontier Language Models?"
    ]


# %% [markdown]
# ## Step 4 — Run the baseline and capture the evidence
#
# This runs each query through the baseline retriever and the LLM, printing the
# retrieved document ids/scores and the grounded answer, and logging everything to
# `checkpoint_2_1_retrieval.log`. The retrieved documents from the output are the rest of the evidence for
# submission item #2.

# %%
# Terminal presentation. Colour is switched off when stdout is not a terminal, so piping
# the run to a file gives clean text, and NO_COLOR is honoured for anyone who wants it off.
# Windows consoles default to cp1252, which cannot encode the box-drawing rules below, so
# stdout is reconfigured to UTF-8 rather than left to raise UnicodeEncodeError mid-run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
BOLD, DIM, CYAN, GREEN, YELLOW = "1", "2", "36", "32", "33"
WIDTH = min(shutil.get_terminal_size((100, 24)).columns, 100)


def paint(text: str, *codes: str) -> str:
    """Wrap text in ANSI codes, or return it unchanged when colour is off."""
    return f"\033[{';'.join(codes)}m{text}\033[0m" if _COLOR else text


def field(label: str, value: str, pad: int = 11) -> str:
    """One aligned `label   value` line, with the value wrapped under itself."""
    body = textwrap.fill(value, width=WIDTH - pad - 2,
                         subsequent_indent=" " * (pad + 2))
    return f"  {paint(label.ljust(pad), DIM)}{body}"


# %%
def run() -> None:
    llm = make_llm()
    queries = my_representative_queries()
    n_pdfs = sum(1 for f in PAPERS_DIR.iterdir() if f.suffix.lower() == ".pdf")
    fusion = f", RRF (k={RRF_K})" if RETRIEVAL_MODE == "hybrid" else ""

    print(paint("═" * WIDTH, CYAN))
    print(paint("  Checkpoint 2.1 · baseline retrieval", BOLD, CYAN))
    print(paint("═" * WIDTH, CYAN))
    print(field("scenario", SCENARIO))
    print(field("corpus", f"{len(CORPUS_IDS):,} chunks from {n_pdfs} PDFs"))
    print(field("retrieval", f"{RETRIEVAL_MODE}{fusion}, pool {CANDIDATE_POOL} per "
                             f"retriever, top {TOP_K} to the LLM"))

    for i, query in enumerate(queries, 1):
        hits = retrieval_detail(query, TOP_K)
        print()
        print(paint("─" * WIDTH, CYAN))
        # The label is padded to a fixed width so a two-digit query number does not shift
        # the wrapped continuation lines out of alignment with the first one.
        label = f"QUERY {i}".ljust(9)
        print(f"  {paint(label, BOLD, CYAN)} " +
              textwrap.fill(query, width=WIDTH - 12,
                            subsequent_indent=" " * 12).lstrip())
        print(paint("─" * WIDTH, CYAN))
        if not hits:
            print(paint("  nothing matched. Note this in your writeup.", YELLOW))
            continue

        # Report the per-retriever ranks alongside the fused score. A chunk both retrievers
        # nominated is far better supported than one only a single retriever saw, and that
        # distinction is invisible if you print the fused score alone. Those rows are also
        # tinted green, so consensus is visible before anyone reads the numbers.
        metas = [DOC_BY_ID[cid]["meta"] for cid, _, _, _ in hits]
        src_w = max(len(m["source"]) for m in metas)
        head = (f"  {'#':>2s}  {'fused':>7s}  {'bm25':>4s}  {'vec':>4s}  "
                f"{'source'.ljust(src_w)}  {'page':>4s}  {'chunk':>5s}")
        print(paint(head, DIM))
        print(paint(f"  {'─' * 2}  {'─' * 7}  {'─' * 4}  {'─' * 4}  "
                    f"{'─' * src_w}  {'─' * 4}  {'─' * 5}", DIM))
        for rank, ((cid, score, br, vr), meta) in enumerate(zip(hits, metas), 1):
            row = (f"  {rank:2d}  {score:7.5f}  {str(br or '·'):>4s}  "
                   f"{str(vr or '·'):>4s}  {meta['source'].ljust(src_w)}  "
                   f"{meta['page']:4d}  {meta['chunk']:5d}")
            print(paint(row, GREEN) if br and vr else row)

        both = sum(1 for h in hits if h[2] and h[3])
        papers = {m["source"] for m in metas}
        print(paint(f"  {both} of {len(hits)} found by both retrievers "
                    f"in {len(papers)} paper(s)", DIM))

        ans = answer(llm, query, [cid for cid, _, _, _ in hits])
        print()
        print(f"  {paint('ANSWER', BOLD, GREEN)}")
        print(textwrap.fill(ans, width=WIDTH - 4,
                            initial_indent="  ", subsequent_indent="  "))

        detail = "\n".join(
            f"  {r:2d}. rrf={s:.5f} bm25={br or '-'} vec={vr or '-'}  {cid}"
            for r, (cid, s, br, vr) in enumerate(hits, 1)
        )
        log(f"[{RETRIEVAL_MODE}] QUERY {i}: {query}",
            f"retrieved:\n{detail}\nfound_by_both={both}/{len(hits)} "
            f"papers={len(papers)}\n\nanswer={ans}")

    print()
    print(paint("═" * WIDTH, CYAN))
    print(field("done", f"Evidence written to {LOG_PATH.name}. The retrieved results and "
                        f"the per-retriever ranks above are the evidence for worksheet "
                        f"items 2 and 3.", pad=8))


run()

# %% [markdown]
# ## Step 5 — Your written submission (the graded deliverable)
#
# Use your completed retrieval implementation and test results to complete the Capstone Checkpoint 2.1
# worksheet. In the worksheet, you will document your retrieval approach, provide evidence that your 
# system is functioning, include 3–5 representative queries and retrieved results, and reflect on where
# your approach performs well and where it struggles.  
 
# Save your completed Python file in the appropriate checkpoint folder in your GitHub repository. 
# Upload the completed worksheet only to the learning platform as your graded submission.

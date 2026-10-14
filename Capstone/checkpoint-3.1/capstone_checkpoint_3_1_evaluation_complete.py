r"""Capstone Checkpoint 3.1 — Evaluation Infrastructure and Baseline Diagnosis (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code / PyCharm / Jupytext.

This demonstration system is not your capstone system and 
does not use the Research Paper Navigator or Wikipedia corpus.
"""

# %% [markdown]
# # Capstone Checkpoint 3.1 — Evaluation Infrastructure and Baseline Diagnosis
# **MO-LLM Module 3 / Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# #
# This mirrors **Lab 3.1** (an LLM-judge that scores answers pass/fail against grading
# notes), applied to your capstone system. 
# You have a baseline retrieval system from Checkpoint 2.1. In this checkpoint, 
# you will use a structured evaluation approach to measure baseline performance, diagnose strengths 
# and weaknesses, and examine whether the evaluation framework detects problematic outputs.
# The starter script includes a small demonstration corpus and retriever to illustrate the 
# evaluation workflow. Apply the same evaluation approach to your selected capstone scenario and 
# baseline retrieval system. The graded deliverable is the completed Capstone Checkpoint 3.1 worksheet.
#
# **Learning outcomes (Module 3):**
# 1. Define evaluation metrics that reflect the real-world performance requirements of an LLM-powered retrieval system.
# 2. Identify key variables that influence system performance during evaluation and development.
# 3. Use language models to support evaluation tasks while avoiding common pitfalls.
# 4. Evaluate the performance of a retrieval-augmented system during development using a structured evaluation framework.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario and baseline retriever** from Checkpoints 1.1 and 2.1.
#
# | Scenario | Corpus |
# |---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs (`Labs/CapstoneDatasets/ResearchPapers/`) |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles (`Labs/CapstoneDatasets/Wikipedia/`) |

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12.**
# 2. `pip install langchain-openai langchain-core python-dotenv`
# 3. Use the OpenRouter API key provided for this program.
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# Runs on a tiny built-in sample corpus, so you do not need to prepare your own dataset.
# It still requires an OpenRouter API key to run the LLM (it is not offline or free of API
# calls). You apply the same evaluation approach to your real system for the report.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import os
import re
import sys
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
EMBEDDING_MODEL = "openai/text-embedding-3-small"  # ported from 2.1; via OpenRouter
TEMPERATURE = 0.2
TOP_K = 6          # was 3 in the starter. 2.1's baseline retrieves six chunks, and this
                   # checkpoint measures that baseline, so the value has to match.
LOG_PATH = Path.cwd() / "checkpoint_3_1_evaluation.log"

SCENARIO = "research_papers"   # "research_papers" or "wikipedia"

# --- Ported from Checkpoint 2.1: corpus, chunking and retrieval settings ---------------
# It ingests the shared PDFs in ../../data/ResearchPapers and builds
# its own vector store in ./chroma_chunks. Every value below is the 2.1 value unchanged,
# because 3.1 measures the 2.1 baseline as it stands. Change one and the numbers stop being
# a baseline that Checkpoint 4.1 can be compared against.
HERE = Path(__file__).resolve().parent
SHARED_DATA_DIR = HERE.parent.parent / "data"    # one copy of the PDF corpus for every checkpoint
PAPERS_DIR = SHARED_DATA_DIR / "ResearchPapers"
CHROMA_DIR = HERE / "chroma_chunks"
QUESTIONS_PATH = HERE / "eval_questions.json"
VARIANTS_PATH = HERE / "eval_questions_variants.json"

CHUNK_SIZE = 1200         # characters; large enough for a claim plus its qualifying sentence
CHUNK_OVERLAP = 200       # keeps a definition from being severed from the term it defines
MIN_PAGE_CHARS = 100      # below this a page is a cover, a figure, or a scan artifact

# "keyword" = BM25 only, "vector" = Chroma only, "hybrid" = both fused by RRF.
RETRIEVAL_MODE = "hybrid"
CANDIDATE_POOL = 20       # candidates each retriever nominates before fusion
RRF_K = 60                # standard reciprocal rank fusion constant

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

JUDGE_SYSTEM = (
    "You are a strict evaluator. You are given an ANSWER and GRADING NOTES describing "
    "what a correct answer must contain. Reply with exactly one word: 'pass' if the "
    "answer satisfies the grading notes, or 'fail' if it does not."
)


# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Use the OpenRouter API key "
            "provided for this program, put it in a .env file next to this "
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
# ## The capstone corpus and the Checkpoint 2.1 baseline retriever (ported)
#
# The starter shipped six invented sentences and a shared-word-count retriever to
# demonstrate the evaluation workflow without needing a corpus. Both are replaced here by
# the real thing, ported unchanged from
# `../checkpoint-2.1/capstone_checkpoint_2_1_baseline_retrieval_complete.py`.
#
# The pipeline, in the order it runs:
#
# 1. `load_and_chunk()` reads each PDF page by page and splits every page into overlapping
#    1,200-character chunks tagged with source filename, page number and chunk index.
# 2. `build_or_load_db()` embeds those chunks and persists them to `./chroma_chunks`. On
#    every run after the first it loads the store instead of re-embedding.
# 3. The chunk text is read back out of Chroma so BM25 can index it, rather than re-parsing
#    6,000-odd pages of PDF on every run.
# 4. `retrieve()` runs BM25 and vector search, each nominating 20 candidates, and fuses the
#    two ranked lists with Reciprocal Rank Fusion, returning the top 6.
#
# Fusion is by rank, not by score. Min-max scaling a retriever's scores has no notion of
# absolute quality: it stretches whatever candidates come back across the full range whether
# they are relevant or not. RRF scores a chunk as `1/(RRF_K + rank)` from each retriever that
# returned it, so only position matters and the two scales never have to be reconciled.

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

    if not PAPERS_DIR.is_dir():
        raise RuntimeError(
            f"No corpus found. Expected the research-paper PDFs in {PAPERS_DIR}. "
            "Copy them there and rerun."
        )

    print(f"Building vector DB from {PAPERS_DIR} (first run - this takes a while)")
    docs = load_and_chunk(PAPERS_DIR)
    print(f"\n  {len(docs)} chunks total. Embedding...")
    db = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=get_embeddings())
    for start in range(0, len(docs), 500):   # batch so a failure loses 500, not everything
        db.add_documents(docs[start:start + 500])
        print(f"    embedded {min(start + 500, len(docs))}/{len(docs)}")
        sys.stdout.flush()
    return db


# %% [markdown]
# The vector store holds the embeddings, but BM25 needs the raw text, so the chunks are read
# back out of Chroma. `DOC_BY_ID` plays exactly the role the starter's version did, keyed by
# a chunk id of `filename p.N #i`.

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

    NOTE: this does no stemming, which is a real limitation carried over from 2.1 unchanged
    rather than an oversight. It is why a question saying 'sketch' never matches a paper
    saying 'sketching' on the keyword side.
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
    """The ranked chunk-id lists that RETRIEVAL_MODE says should take part in fusion."""
    lists: dict[str, list[str]] = {}
    if RETRIEVAL_MODE in ("keyword", "hybrid"):
        lists["bm25"] = _bm25_ranked(query, k)
    if RETRIEVAL_MODE in ("vector", "hybrid"):
        lists["vec"] = _vector_ranked(query, k)
    return lists


def retrieve(query: str, k: int = TOP_K) -> list[tuple[str, float]]:
    """Retrieval per RETRIEVAL_MODE, fused by Reciprocal Rank Fusion.

    Returns (chunk_id, rrf_score) pairs, the same shape the starter's retrieve() returned,
    so nothing downstream in this script had to change.
    """
    rrf: dict[str, float] = defaultdict(float)
    for ranked in _ranked_lists(query, CANDIDATE_POOL).values():
        for rank, chunk_id in enumerate(ranked, 1):
            rrf[chunk_id] += 1.0 / (RRF_K + rank)
    ordered = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)
    return ordered[:k]


def retrieval_detail(query: str, k: int = TOP_K) -> list[tuple[str, float, int | None, int | None]]:
    """Same as retrieve(), plus the rank each retriever gave the chunk (None = not returned).

    The per-retriever ranks are what turn 'the right page was not retrieved' into 'the right
    page was ranked 9th by BM25 and 14th by vector search, so fusion could not reach it'.
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
# ## Step 2 — The evaluation metric (provided)
#
# A simple **LLM-judge** that returns pass/fail by checking an answer against grading
# notes — the same idea as Lab 3.1's DiscreteMetric, written directly here so the
# checkpoint needs no extra packages. Tuning this metric (stricter notes, a 'partial'
# level, a stronger judge model) is part of the diagnosis.

# %%
def judge(llm: ChatOpenAI, answer_text: str, grading_notes: str) -> str:
    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=f"ANSWER:\n{answer_text}\n\nGRADING NOTES:\n{grading_notes}\n\nVerdict (pass/fail):"),
    ]
    verdict = llm.invoke(messages).content.strip().lower()
    return "pass" if "pass" in verdict else "fail"


# %% [markdown]
# ## Step 3b - The rest of the metric suite
#
# `judge()` above measures correctness and nothing else. It never sees what was retrieved,
# so it cannot tell an answer grounded in the excerpts from one that cites a plausible page
# the model was never shown. Six more metrics cover the other three criteria.
#
# Four of the six are ordinary Python. Whether a cited page was in the context window is a
# fact, and asking a language model about a fact adds cost, latency and a source of error
# for no benefit. Only the two that require judgement call the model.
#
# | Criterion | Metric | Type |
# |---|---|---|
# | Grounding | `citation_valid` | rule |
# | Grounding | `quote_verbatim` | rule |
# | Grounding | `closed_book_delta` | judge |
# | Correctness | `correctness` (above) | judge |
# | Completeness | `point_coverage` | judge |
# | Completeness | `source_coverage` | rule |
# | Refusal | `abstention_accuracy` | rule |

# %%
def _norm(text: str) -> str:
    """Collapse whitespace and fold the typographic characters PDF extraction introduces.

    The stored chunks contain ligatures (ﬁ, ﬂ) and curly quotes, and a chunk wraps a
    sentence over several lines. Comparing raw strings would report a genuine verbatim
    quotation as a mismatch.
    """
    for a, b in (("ﬁ", "fi"), ("ﬂ", "fl"), ("’", "'"), ("‘", "'"),
                 ("“", '"'), ("”", '"'), ("—", "-"), ("–", "-")):
        text = text.replace(a, b)
    text = re.sub(r"[*`]", "", text)          # markdown emphasis, not content
    return re.sub(r"\s+", " ", text).strip().lower()


def _retrieved_pages(chunk_ids: list[str]) -> set[tuple[str, str]]:
    """The (filename, page) pairs actually placed in the context window."""
    pairs = set()
    for cid in chunk_ids:
        meta = DOC_BY_ID.get(cid, {}).get("meta", {})
        if meta:
            pairs.add((str(meta.get("source", "?")), str(meta.get("page", "?"))))
    return pairs


def cited_pages(answer_text: str) -> list[tuple[str, str]]:
    """Every (filename, page) the answer claims to be citing.

    A single bracket can carry several pages, as in `[paper.pdf, p.16; p.17]`, so the
    filename found in a bracket applies to every page number in that same bracket.
    """
    out = []
    for group in re.findall(r"\[([^\[\]]+)\]", answer_text):
        files = re.findall(r"([A-Za-z0-9_.\-]+\.pdf)", group)
        pages = re.findall(r"p\.\s*(\d+)", group)
        if files and pages:
            out.extend((files[0], page) for page in pages)
    return out


def citation_valid(answer_text: str, chunk_ids: list[str]) -> tuple[bool | None, list[str]]:
    """GROUNDING. Does every citation name a page that was actually in context?

    Returns (verdict, bad citations). None means the answer cited nothing, which is not a
    failure: a refusal has nothing to cite.
    """
    cites = cited_pages(answer_text)
    if not cites:
        return None, []
    available = _retrieved_pages(chunk_ids)
    bad = [f"[{f}, p.{p}]" for f, p in cites if (f, p) not in available]
    return not bad, bad


def quote_verbatim(answer_text: str, item: dict) -> bool | None:
    """GROUNDING. Did the answer reproduce one of the accepted sentences word for word?

    None where the item does not ask for a quotation. Passing requires only ONE of the
    accepted sentences, since a question may have more than one correct answer.
    """
    accepted = item.get("must_quote_any")
    if not accepted:
        return None
    haystack = _norm(answer_text)
    return any(_norm(q) in haystack for q in accepted)


def source_coverage(chunk_ids: list[str], answer_text: str, item: dict) -> tuple[int, int, int] | None:
    """COMPLETENESS. On a question spanning several papers, how many were reached?

    Returns (gold files retrieved, gold files cited, gold files total). None where the item
    has no gold sources, which is every question the corpus cannot answer.
    """
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
    """Did the gold page reach the context window, and if so at what rank?

    Scored only where a single known page holds the answer. This is what separates a
    retrieval failure from a generation failure.
    """
    if not item.get("gold_page") or not item.get("gold_sources"):
        return None
    want = {(g, str(item["gold_page"])) for g in item["gold_sources"]}
    return bool(want & _retrieved_pages(chunk_ids))


_REFUSAL_PATTERNS = (
    r"do(es)? not (contain|include|provide|mention|specify|state|identify|give)",
    r"don'?t (contain|include|provide|mention)",
    r"cannot (determine|answer|find|identify|state|tell)",
    r"can'?t (determine|answer|find|identify|state|tell)",
    r"not (present|found|included|mentioned|described) in",
    r"no (information|mention|description) (about|of|on|in)",
    r"contains? no ", r"not supported by", r"unable to (determine|answer|find)",
    r"i'?m sorry",
)


def refused(answer_text: str) -> bool:
    """REFUSAL. Did the answer decline rather than answer?

    NOTE: this reads only the opening of the answer, deliberately. A long answer that does
    answer the question often contains a phrase like "the excerpts do not state what
    CRUXEval gives up" partway through, and scanning the whole text would misread that as a
    refusal. Judging the first sentence or two matches how a reader decides. The ceiling is
    an answer that buries its refusal at the end, which nothing here would catch.
    """
    opening = _norm(answer_text)[:220]
    return any(re.search(p, opening) for p in _REFUSAL_PATTERNS)


# %%
POINTS_SYSTEM = (
    "You check whether an answer contains specific facts. You are given an ANSWER and a "
    "numbered list of REQUIRED POINTS. For each point, reply on its own line in the form "
    "'<number>: yes' if the answer supplies that point, or '<number>: no' if it does not. "
    "Judge each point independently. Accept different wording and accept a point that is "
    "clearly implied. Output nothing except those lines."
)

CLOSED_BOOK_SYSTEM = (
    "You are a research assistant. Answer the question from your own knowledge. Be specific "
    "and give figures where you know them. If you do not know, say so plainly."
)


def point_coverage(llm: ChatOpenAI, answer_text: str, item: dict) -> tuple[int, int, list[bool]]:
    """COMPLETENESS. How many of the required facts did the answer supply?

    One model call per item. Asking for a verdict per point rather than one verdict for the
    whole answer gives the judge a smaller job, which makes it a steadier judge, and it
    distinguishes a half-answer from a wrong answer.
    """
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
    return sum(got), len(points), got


def closed_book(llm: ChatOpenAI, item: dict) -> tuple[str, str]:
    """GROUNDING. Answer the question with no documents at all, then score it.

    An item the model answers correctly from memory is an item this evaluation cannot claim
    credit for, because a pass there is not evidence that retrieval worked. The neutral
    system prompt matters: reusing ANSWER_SYSTEM would order the model to refuse whenever no
    excerpts are supplied, which would measure the instruction rather than the model.
    """
    ans = llm.invoke([
        SystemMessage(content=CLOSED_BOOK_SYSTEM),
        HumanMessage(content=item["question"]),
    ]).content
    return judge(llm, ans, item["grading_notes"]), ans


# %% [markdown]
# ## Step 3 — The evaluation set, loaded from file
#
# The questions live in `eval_questions.json` rather than in this script, so the test set
# can be edited, reviewed and version-controlled without touching code, and so the same set
# can be re-run against a changed system in Checkpoint 4.1.
#
# Each item carries the two fields the harness needs (`question` and `grading_notes`) plus
# the fields the metric suite consumes:
#
#   required_points   the grading notes broken into individually checkable facts -> point_coverage
#   must_quote_any    exact sentences from the corpus, any ONE of which is accepted -> quote_verbatim
#   gold_sources      the file(s) that actually hold the answer -> source_coverage
#   gold_page         the page the answer sits on, where exactly one page holds it
#   answerable        whether the corpus can answer this at all -> abstention_accuracy
#   expect            the pass/fail prediction recorded BEFORE the first run, so a surprise
#                     reads as a surprise instead of being explained away afterwards
#
# Every fact in the grading notes was checked against the stored chunk text of this corpus
# before the note was written. Notes derived from the system's own earlier answers can only
# ever confirm those answers.
#
# `eval_questions_variants.json` holds five paraphrases of each question, generated with an
# LLM in the manner of Lab 3.2 and then reviewed by hand. Each paraphrase inherits its
# original's grading notes, required points, gold sources and prediction, so the same answer
# must still be correct. A paraphrase that scores differently from its original is measuring
# brittleness rather than capability, which is the point of running them. Pass `--variants`
# to include them; they roughly triple the cost of a run.

# %%
def _load_items(path: Path, what: str) -> list[dict]:
    if not path.is_file():
        raise RuntimeError(f"{what} file not found: {path}")
    items = json.loads(path.read_text(encoding="utf-8"))
    required = {"id", "question", "grading_notes", "required_points",
                "gold_sources", "answerable", "expect"}
    for item in items:
        missing = required - item.keys()
        if missing:
            raise RuntimeError(f"{path.name}: item {item.get('id', '?')} is missing {sorted(missing)}")
    return items


def my_eval_set() -> list[dict]:
    """The five hand-written evaluation questions, read from eval_questions.json."""
    return _load_items(QUESTIONS_PATH, "Evaluation questions")


def variant_eval_set() -> list[dict]:
    """The paraphrases, read from eval_questions_variants.json.

    Returns an empty list when the file is absent, so a normal run does not depend on it.
    """
    if not VARIANTS_PATH.is_file():
        return []
    return _load_items(VARIANTS_PATH, "Question variants")


# %% [markdown]
# Run the demonstration code to understand the evaluation workflow. Then apply the same evaluation 
# design to your own capstone baseline system, using its retrieval and answer-generation functions. 
# Record results from your capstone system in the worksheet.
#
# ## Step 4 — Run the baseline evaluation
#
# Answers each question with the baseline retriever, scores it with the judge, and
# reports the pass rate. This is your baseline diagnosis: the failures are what you
# analyse in the report.

# %%
def evaluate_item(llm: ChatOpenAI, item: dict) -> dict:
    """Score one evaluation item on all seven metrics."""
    hits = retrieve(item["question"], TOP_K)
    chunk_ids = [cid for cid, _ in hits]
    ans = answer(llm, item["question"], chunk_ids) if hits else "(no documents retrieved)"

    cite_ok, bad_cites = citation_valid(ans, chunk_ids)
    got, total, _ = point_coverage(llm, ans, item)
    cb_verdict, cb_answer = closed_book(llm, item)

    return {
        "id": item["id"],
        "question": item["question"],
        "expect": item["expect"],
        "answerable": item["answerable"],
        "retrieved": hits,
        "answer": ans,
        "correctness": judge(llm, ans, item["grading_notes"]),
        "citation_valid": cite_ok,
        "bad_citations": bad_cites,
        "quote_verbatim": quote_verbatim(ans, item),
        "point_coverage": (got, total),
        "source_coverage": source_coverage(chunk_ids, ans, item),
        "retrieval_hit": retrieval_hit(chunk_ids, item),
        "refused": refused(ans),
        "closed_book": cb_verdict,
        "closed_book_answer": cb_answer,
    }


def _show(value) -> str:
    """Render a metric for the console. None means the metric does not apply to this item."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "NO"
    if isinstance(value, tuple):
        return f"{value[0]}/{value[-1]}" if len(value) == 2 else f"{value[0]} of {value[2]} retrieved, {value[1]} cited"
    return str(value)


def run_evaluation(eval_set: list[dict] | None = None, label: str = "baseline") -> list[dict]:
    """Score every item, print the per-item detail, then the aggregate for each criterion."""
    llm = make_llm()
    eval_set = eval_set if eval_set is not None else my_eval_set()
    results = []

    print(f"Checkpoint 3.1 - {label} evaluation  |  scenario: {SCENARIO}")
    print(f"retrieval={RETRIEVAL_MODE}  top_k={TOP_K}  answer/judge model={LLM_MODEL}\n")

    for r in (evaluate_item(llm, item) for item in eval_set):
        results.append(r)
        surprise = "" if r["correctness"] == r["expect"] else "   <-- not what was predicted"
        print("=" * 78)
        print(f"{r['id']}: {r['question']}")
        print(f"  correctness    = {r['correctness'].upper()}  (predicted {r['expect']}){surprise}")
        print(f"  point_coverage = {_show(r['point_coverage'])}")
        print(f"  citation_valid = {_show(r['citation_valid'])}"
              + (f"  bad: {r['bad_citations']}" if r["bad_citations"] else ""))
        print(f"  quote_verbatim = {_show(r['quote_verbatim'])}")
        print(f"  source_coverage= {_show(r['source_coverage'])}")
        print(f"  retrieval_hit  = {_show(r['retrieval_hit'])}")
        print(f"  refused        = {_show(r['refused'])}   (corpus can answer: {r['answerable']})")
        print(f"  closed_book    = {r['closed_book'].upper()}  (correct with NO documents?)")
        print(f"  retrieved      = {[cid for cid, _ in r['retrieved']]}")
        print(f"  answer: {r['answer']}")
        log(f"{r['id']}: {r['question']}", "\n".join([
            f"retrieved={r['retrieved']}",
            f"correctness={r['correctness']} (predicted {r['expect']})",
            f"point_coverage={_show(r['point_coverage'])}",
            f"citation_valid={_show(r['citation_valid'])} bad={r['bad_citations']}",
            f"quote_verbatim={_show(r['quote_verbatim'])}",
            f"source_coverage={_show(r['source_coverage'])}",
            f"retrieval_hit={_show(r['retrieval_hit'])}",
            f"refused={r['refused']} answerable={r['answerable']}",
            f"closed_book={r['closed_book']}",
            f"answer={r['answer']}",
            f"closed_book_answer={r['closed_book_answer']}",
        ]))

    # ---- aggregates, grouped by the criterion each metric serves -----------------------
    n = len(results)
    passes = sum(r["correctness"] == "pass" for r in results)
    cited = [r for r in results if r["citation_valid"] is not None]
    quoted = [r for r in results if r["quote_verbatim"] is not None]
    covered = [r for r in results if r["source_coverage"] is not None]
    hits = [r for r in results if r["retrieval_hit"] is not None]
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]
    got = sum(r["point_coverage"][0] for r in results)
    want = sum(r["point_coverage"][1] for r in results)

    print("\n" + "=" * 78)
    print("BASELINE SUMMARY\n")
    print("Grounding")
    print(f"  citation_valid     {sum(r['citation_valid'] for r in cited)}/{len(cited)} answers that cited "
          f"anything used only pages that were in context")
    print(f"  quote_verbatim     {sum(bool(r['quote_verbatim']) for r in quoted)}/{len(quoted)} "
          f"quotation request(s) reproduced an accepted sentence exactly")
    print(f"  closed_book_delta  {sum(r['closed_book'] == 'pass' for r in results)}/{n} correct with NO "
          f"documents (a pass here means retrieval cannot claim the credit)")
    print("\nCorrectness")
    print(f"  correctness        {passes}/{n} pass")
    print(f"  predictions held   {sum(r['correctness'] == r['expect'] for r in results)}/{n}")
    print("\nCompleteness")
    print(f"  point_coverage     {got}/{want} required points supplied across all items")
    for r in covered:
        rt, ct, tot = r["source_coverage"]
        print(f"  source_coverage    {r['id']}: {rt} of {tot} gold file(s) retrieved, {ct} cited")
    print(f"  retrieval_hit      {sum(bool(r['retrieval_hit']) for r in hits)}/{len(hits)} "
          f"item(s) with a known gold page reached it")
    print("\nRefusal (abstention_accuracy - two rates, never one)")
    fr = sum(r["refused"] for r in answerable)
    fab = sum(not r["refused"] for r in unanswerable)
    print(f"  false refusals     {fr}/{len(answerable)} on questions the corpus CAN answer")
    print(f"  fabrications       {fab}/{len(unanswerable)} on questions the corpus CANNOT answer")

    log("BASELINE SUMMARY", "\n".join([
        f"correctness={passes}/{n}",
        f"predictions_held={sum(r['correctness'] == r['expect'] for r in results)}/{n}",
        f"citation_valid={sum(r['citation_valid'] for r in cited)}/{len(cited)}",
        f"quote_verbatim={sum(bool(r['quote_verbatim']) for r in quoted)}/{len(quoted)}",
        f"closed_book_pass={sum(r['closed_book'] == 'pass' for r in results)}/{n}",
        f"point_coverage={got}/{want}",
        f"retrieval_hit={sum(bool(r['retrieval_hit']) for r in hits)}/{len(hits)}",
        f"false_refusals={fr}/{len(answerable)}",
        f"fabrications={fab}/{len(unanswerable)}",
    ]))
    return results


def run_variants(baseline: list[dict]) -> list[dict]:
    """Score the paraphrases and report how stable each original question's verdict is.

    A paraphrase preserves the meaning of its original, so the same answer is still correct
    and the verdict should not move. Where it does move, the pass rate measured on one
    phrasing was overstating how much the system actually knows.
    """
    variants = variant_eval_set()
    if not variants:
        print("\nNo variants file found; skipping the paraphrase pass.")
        return []

    print("\n\n" + "#" * 78)
    print(f"# PARAPHRASE PASS - {len(variants)} rewordings of {len(baseline)} questions")
    print("#" * 78)
    results = run_evaluation(variants, label="paraphrase variants")

    print("\n" + "=" * 78)
    print("PARAPHRASE STABILITY\n")
    base = {r["id"]: r for r in baseline}
    total_agree = total_seen = 0
    for original in baseline:
        kids = [r for r in results if r["id"].startswith(original["id"] + "-v")]
        if not kids:
            continue
        agree = sum(k["correctness"] == original["correctness"] for k in kids)
        total_agree += agree
        total_seen += len(kids)
        verdicts = " ".join(k["correctness"][0].upper() for k in kids)
        flag = "" if agree == len(kids) else "   <-- unstable"
        print(f"  {original['id']}  original={original['correctness'].upper():4s} "
              f"variants=[{verdicts}]  agree {agree}/{len(kids)}{flag}")
    print(f"\n  overall agreement with the original verdict: {total_agree}/{total_seen}")
    print("  Every disagreement is the same system answering the same question, reworded.")
    log("PARAPHRASE STABILITY", f"agreement={total_agree}/{total_seen}")
    return results


# %% [markdown]
# ## Step 5 — Validate the framework: can it catch a manipulated answer? (provided)
#
# A good evaluation framework must FAIL a wrong answer, not just pass good ones. This
# takes a question your corpus can answer, produces a correct answer, then feeds the
# judge a deliberately manipulated (false) answer — and checks that the judge flags it.
# This is your "evaluation framework validation" evidence for the report.

# %%
def validate_framework() -> None:
    llm = make_llm()
    # The starter asked "What is BM25?", which suited its six-sentence demo corpus. This
    # corpus never defines BM25 - it appears only as an example of something that performs
    # poorly on large codebases - so the control answer was a correct refusal and the
    # validation proved nothing. Sketching is used instead: the corpus defines it plainly,
    # on a page we know retrieval reaches.
    q = "What is sketching in program synthesis, and what does the programmer supply?"
    notes = (
        "States that sketching is a form of localized program synthesis in which the "
        "programmer supplies a partial program (a sketch) describing the implementation "
        "strategy, and an automated synthesizer fills in the low-level details."
    )
    good = answer(llm, q, [doc_id for doc_id, _ in retrieve(q)])
    manipulated = (
        "In sketching, the synthesizer writes the complete program from scratch and the "
        "programmer afterwards checks it by hand. A sketch places no restriction on the "
        "space of programs the synthesizer considers."
    )
    good_verdict = judge(llm, good, notes)
    manip_verdict = judge(llm, manipulated, notes)
    print("\n--- Framework validation ---")
    print(f"  correct answer   -> {good_verdict.upper()}   (expected PASS)")
    print(f"  manipulated answer -> {manip_verdict.upper()}   (expected FAIL)")
    print("  The framework works if it PASSES the correct answer and FAILS the manipulated one.")
    log("FRAMEWORK VALIDATION", f"good={good_verdict} manipulated={manip_verdict}")


# %%
# Pass --variants to add the paraphrase pass (5 rewordings per question, ~3x the cost).
_baseline = run_evaluation()
validate_framework()
if "--variants" in sys.argv:
    run_variants(_baseline)

# %% [markdown]
# ## Step 6 — Your written responses in the Capstone Checkpoint 3.1 worksheet
#
# Complete the Capstone Checkpoint 3.1 worksheet using evidence from your capstone evaluation.
# Address the seven sections: system overview, evaluation design, testing approach, baseline results, 
# performance analysis, evaluation framework validation, and reflection and next steps.
#
# 1. **System overview** — your scenario and your 2.1 baseline retriever.
# 2. **Evaluation design** — your criteria and metric (what "correct" means; how the
#    judge decides pass/fail; any thresholds).
# 3. **Testing approach** — how you built your evaluation set and ran it.
# 4. **Baseline results** — the pass/fail outcomes and where the system falls short.
# 5. **Performance analysis** — what the results reveal about strengths, weaknesses,
#    and failure modes.
# 6. **Evaluation framework validation** — show your framework detects a degraded or
#    manipulated output (use the Step 5 result, or your own).
# 7. **Reflection and next steps** — limitations of your evaluation and what you'll
#    improve (this motivates the advanced retrieval in Checkpoint 4.1).

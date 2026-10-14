"""Research Paper Navigator - chat app (Capstone 8.1).

A chat front end for the 8.1 system in capstone_checkpoint_8_1_improved.py. Every question
goes through answer_question() with all the 7.1 safeguards and the 8.1 improvements on.

Run from this folder:
    ..\\..\\.venv\\Scripts\\streamlit.exe run app.py
"""
import html
import re
import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

EXAMPLE_QUESTIONS = [
    "How many human design sequences are in the Fusion 360 Gallery reconstruction dataset?",
    "For the computeDeriv assignment in the 6.00x course, how many student attempts were "
    "incorrect, and for what share of them did the tool produce a correction?",
    "How did the correctness reference that a programmer supplies alongside a sketch change "
    "between the 2005 StreamBit paper and the 2009 overview of the sketching approach?",
    "How many problems does LiveCodeBench contain, and what period do they cover?",
]

VERDICT_LABELS = {
    "supports": "✅ supported by this page",
    "contradicts": "❌ contradicted by this page",
    "says_nothing": "⚠️ this page does not state it",
    "page not retrieved": "⚠️ page was not retrieved",
}

st.set_page_config(page_title="Research Paper Navigator", page_icon="📄", layout="centered")


@st.cache_resource(show_spinner="Loading the 153 papers (the first start takes a minute)...")
def load_navigator():
    """Import the 8.1 system once and switch every 8.1 layer on."""
    import capstone_checkpoint_8_1_improved as nav
    nav.use_config(nav.CONFIG_8_1)
    nav.LOG_PATH = HERE / "chat_app.log"      # chats stay out of the evaluation log
    llm = nav.make_llm(nav.LLM_MODEL)
    return nav, llm


def paper_title(nav, filename: str) -> str:
    """The paper's title from paper_metadata.json, or the file name if there is none."""
    title = nav.META.get(filename, {}).get("title")
    if not title:
        return filename
    return str(title)


def number_citations(nav, answer: str) -> tuple[str, list[tuple[str, str]]]:
    """The answer with each [file, p.N] group replaced by numbers like [1][2].

    Returns (the answer as safe HTML-escaped markdown, the sources in number order as
    (file, page) pairs).
    """
    sources = []                       # (file, page), in the order they are first cited
    pieces = []
    position = 0
    for match in re.finditer(r"\[([^\[\]]+)\]", answer):
        parts = nav.citation_parts(match.group(1))
        if len(parts) == 0:
            continue                   # brackets that are not a citation stay as they are
        pieces.append(html.escape(answer[position:match.start()], quote=False))
        numbers = []
        for pair in parts:
            if pair not in sources:
                sources.append(pair)
            numbers.append(str(sources.index(pair) + 1))
        pieces.append("<sup>[" + ", ".join(numbers) + "]</sup>")
        position = match.end()
    pieces.append(html.escape(answer[position:], quote=False))
    return "".join(pieces), sources


def verdict_for(checks: list, filename: str, page: str) -> str:
    """The weakest Jev verdict for one cited page ("" if Jev did not check it)."""
    order = ["contradicts", "says_nothing", "page not retrieved", "supports"]
    worst = ""
    for check in checks:
        if check["file"] != filename or str(check["page"]) != str(page):
            continue
        if check["verdict"] not in order:
            continue
        if worst == "" or order.index(check["verdict"]) < order.index(worst):
            worst = check["verdict"]
    return worst


def whole_page_text(nav, filename: str, page: str) -> str:
    """Every stored chunk of one page. A cache hit has no passages of its own, because
    the cache keeps only the answer text."""
    texts = []
    for chunk in nav.CHUNKS.values():
        meta = chunk["meta"]
        if str(meta.get("source")) == filename and str(meta.get("page")) == str(page):
            texts.append(chunk["text"])
    return "\n\n".join(texts)


def build_reply(nav, result: dict) -> dict:
    """Everything the chat needs to show one answer."""
    cached = result["path"] == "cache-hit"
    body, sources = number_citations(nav, result["answer"])
    source_lines = []
    for number, (filename, page) in enumerate(sources, 1):
        line = f"**[{number}]** *{html.escape(paper_title(nav, filename))}*, page {page}  \n`{filename}`"
        verdict = verdict_for(result.get("citation_checks", []), filename, page)
        if verdict:
            line = line + "  \n" + VERDICT_LABELS.get(verdict, verdict)
        if cached:
            passage = whole_page_text(nav, filename, page)
        else:
            passage = nav.page_text(filename, page, result["chunk_ids"])
        source_lines.append({"line": line, "passage": passage})
    details = (f"Path: {result['path']} · model calls: {result['llm_calls']} · "
               f"Jev calls: {result.get('jev_calls', 0)} · {result['seconds']} s")
    if result.get("removed"):
        details = details + " · part of the question was removed by the safety filter"
    cache_note = ""
    if cached:
        cache_note = (f"♻️ **Cached answer.** This question matches one answered earlier "
                      f"(similarity {result['similarity']:.2f}), so the saved answer is shown and "
                      f"no model was called. The cache keeps only the answer text, so the sources "
                      f"below show the whole cited page and have no Jev citation check.")
    return {"role": "assistant", "body": body, "sources": source_lines, "details": details,
            "cache_note": cache_note}


def show_message(message: dict) -> None:
    """Draw one chat message, with its sources under an assistant answer."""
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(html.escape(message["body"], quote=False))
            return
        if message.get("cache_note"):
            st.info(message["cache_note"])
        st.markdown(message["body"], unsafe_allow_html=True)
        if len(message["sources"]) > 0:
            with st.expander(f"Sources ({len(message['sources'])})"):
                for source in message["sources"]:
                    st.markdown(source["line"])
                    if source["passage"]:
                        st.caption(source["passage"][:1500])
                    st.divider()
        st.caption(message["details"])


def ask(nav, llm, question: str) -> None:
    """Send one question through answer_question() and add both turns to the chat."""
    st.session_state.messages.append({"role": "user", "body": question})
    # answer_question() uses the cache only for the first question of a chat
    follow_up = len(st.session_state.history) > 0
    jev_before = nav.JEV_CALLS
    result = nav.answer_question(llm, question, list(st.session_state.history))
    # Jev does some of the work the small models did in 7.1, so show both counts
    result["jev_calls"] = nav.JEV_CALLS - jev_before
    # like ChatSession.send(): the history keeps the SANITIZED question (Lab 7.1)
    if result["path"] != "rejected":
        st.session_state.history.append({"role": "user", "content": result["question_used"]})
        st.session_state.history.append({"role": "assistant", "content": result["answer"]})
    reply = build_reply(nav, result)
    if follow_up and result["path"] != "rejected":
        reply["details"] = reply["details"] + " · cache not used: follow-up question in this chat"
    st.session_state.messages.append(reply)


nav, llm = load_navigator()

if "messages" not in st.session_state:
    st.session_state.messages = []       # what the chat shows
    st.session_state.history = []        # what the system gets as the conversation

with st.sidebar:
    st.header("Research Paper Navigator")
    st.write("Answers questions about 153 computer-science papers, with a citation for "
             "every claim. It says so when the papers do not contain the answer.")
    if nav.jev_client() is not None:
        st.success(f"Jev checks are on ({nav.JEV_MODEL})")
    else:
        st.warning(f"Jev checks are off: {nav.JEV_STATUS}")
    if st.button("New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()
    st.subheader("Try a question")
    for example in EXAMPLE_QUESTIONS:
        if st.button(example, use_container_width=True):
            st.session_state.pending = example
            st.rerun()

st.title("📄 Research Paper Navigator")
if len(st.session_state.messages) == 0:
    st.caption("Ask a question about the papers, or pick an example in the sidebar.")

for message in st.session_state.messages:
    show_message(message)

typed = st.chat_input("Ask about the papers...")
question = ""
if typed:
    question = typed
elif st.session_state.get("pending"):
    question = st.session_state.pending
    st.session_state.pending = ""

if question:
    show_message({"role": "user", "body": question})
    with st.spinner("Searching the papers..."):
        ask(nav, llm, question)
    st.rerun()

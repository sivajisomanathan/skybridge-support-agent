"""
Policy knowledge base: PDF extraction, section-based chunking, embedding,
and Pinecone-backed retrieval. Logic unchanged from Phase 4/5 -- moved into
a shared module so both the FastAPI app and any notebook import the exact
same implementation, per the Problem Framing Document's architecture note.
"""
import re
from pathlib import Path

from agent.config import client, pc, EMBED_MODEL, INDEX_NAME, EMBED_DIM, SIMILARITY_THRESHOLD

HEADER_PATTERN = re.compile(r"^SkyBridge Airlines.*Support Agent Use$")
FOOTER_PATTERN = re.compile(r"^Document Code:.*Page \d+$")
SECTION_START_PATTERN = re.compile(r"^(SB-POL-\d{3}|Scope Note — Topics Not Covered)\b", re.MULTILINE)

_index = None  # lazily initialized -- see get_index()


def extract_pdf_text(path: str, skip_first_page: bool = True) -> str:
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        pages = pdf.pages[1:] if skip_first_page else pdf.pages
        all_lines = []
        for page in pages:
            page_text = page.extract_text() or ""
            for line in page_text.split("\n"):
                line = line.strip()
                if not line or HEADER_PATTERN.match(line) or FOOTER_PATTERN.match(line):
                    continue
                all_lines.append(line)
    return "\n".join(all_lines)


def chunk_by_section(full_text: str) -> list[dict]:
    marked = SECTION_START_PATTERN.sub(lambda m: "\n<<<SPLIT>>>" + m.group(0), full_text)
    raw_sections = marked.split("<<<SPLIT>>>")
    chunks = []
    for section in raw_sections:
        section = section.strip()
        if not section or not SECTION_START_PATTERN.match(section):
            continue
        header_line, _, body = section.partition("\n")
        match = SECTION_START_PATTERN.match(header_line)
        code_token = match.group(1)
        if code_token.startswith("SB-POL"):
            section_id = code_token
            title = header_line[len(code_token):].strip()
        else:
            section_id = "SCOPE-NOTE"
            title = header_line.strip()
        body_clean = re.sub(r"(?m)^l ", "- ", body.strip())
        chunks.append({"section_id": section_id, "title": title, "text": body_clean})
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def build_index(pdf_path: str) -> None:
    """Extracts, chunks, embeds, and upserts the policy handbook. Called once
    at FastAPI startup, not per-request -- embedding on every request would
    be slow and wasteful since the handbook doesn't change at runtime."""
    global _index
    full_text = extract_pdf_text(pdf_path)
    chunks = chunk_by_section(full_text)
    if not chunks:
        raise RuntimeError(f"No policy sections found in {pdf_path} -- check the PDF is the expected handbook.")

    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        from pinecone import ServerlessSpec
        pc.create_index(name=INDEX_NAME, dimension=EMBED_DIM, metric="cosine",
                         spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    _index = pc.Index(INDEX_NAME)

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    vectors = [{"id": c["section_id"], "values": emb, "metadata": {"title": c["title"], "text": c["text"]}}
               for c, emb in zip(chunks, embeddings)]
    _index.upsert(vectors=vectors)
    return chunks


def get_index():
    if _index is None:
        raise RuntimeError("KB index not initialized -- call build_index() at startup first.")
    return _index


def retrieve_grounded(query: str, top_k: int = 3) -> dict:
    index = get_index()
    query_embedding = embed_texts([query])[0]
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    matches = [{"section_id": m["id"], "title": m["metadata"]["title"],
                "text": m["metadata"]["text"], "score": m["score"]} for m in results["matches"]]
    if not matches or matches[0]["score"] < SIMILARITY_THRESHOLD:
        return {"grounded": False, "matches": []}
    return {"grounded": True, "matches": matches}

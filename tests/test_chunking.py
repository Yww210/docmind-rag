from app.rag.chunking import chunk_text

def test_chunking_nonempty():
    chunks = chunk_text("a" * 2000, chunk_size=500, overlap=50)
    assert len(chunks) >= 3
    assert all(len(c) > 0 for c in chunks)

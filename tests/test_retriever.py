from app.rag.pipeline import RAGPipeline

def test_retrieve_after_ingest(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "index"))
    p = RAGPipeline()
    p.ingest_text("hello world about neural networks", source="t1")
    out = p.answer("neural networks", use_hyde=False)
    assert out.retrieved is not None

import argparse
from pathlib import Path
from app.rag.pipeline import RAGPipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Folder containing .txt/.md files")
    args = ap.parse_args()

    pipeline = RAGPipeline()
    folder = Path(args.path)
    files = list(folder.rglob("*.txt")) + list(folder.rglob("*.md"))

    total = 0
    for fp in files:
        text = fp.read_text(encoding="utf-8", errors="ignore")
        n = pipeline.ingest_text(text, source=str(fp))
        print(f"Ingested {n:4d} chunks from {fp}")
        total += n

    print(f"Done. Total chunks indexed: {total}")

if __name__ == "__main__":
    main()

import argparse, json, time
from app.rag.pipeline import RAGPipeline

def overlap_score(pred: str, gold: str) -> float:
    pt = set(pred.lower().split())
    gt = set(gold.lower().split())
    if not gt:
        return 0.0
    return len(pt & gt) / len(gt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="JSONL with {question, answer}")
    ap.add_argument("--use_hyde", default="false", choices=["true","false"])
    args = ap.parse_args()
    use_hyde = args.use_hyde == "true"

    pipeline = RAGPipeline()

    latencies = []
    scores = []

    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            q = ex["question"]
            gold = ex["answer"]

            t0 = time.time()
            out = pipeline.answer(q, use_hyde=use_hyde)
            dt = time.time() - t0

            pred = out.answer
            s = overlap_score(pred, gold)

            latencies.append(dt)
            scores.append(s)

            print(f"Q: {q}\nscore={s:.3f} latency={dt:.3f}s\n")

    if latencies:
        print("==== Summary ====")
        print(f"n={len(latencies)}")
        print(f"avg_latency={sum(latencies)/len(latencies):.3f}s")
        print(f"avg_overlap={sum(scores)/len(scores):.3f}")

if __name__ == "__main__":
    main()

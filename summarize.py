"""Benchmark v1.1 summary (2026-10-29): score each distinct prompt + sentence pair once.

FLEURS has several recordings of many sentences, so the 82 items are 64 distinct (prompt, target sentence) pairs:
some pairs appear 2-3 times with a different real recording each time. v1.0 averaged over the 82 items, which counted
those pairs more than once. v1.1 first averages within a pair (CER: errors and characters; voice match and UTMOS:
the mean), then over the 64 pairs. CER stays character-weighted, as in v1.0. The v1.0 per-item numbers are kept
under "per_item".

Needs no models: it reads the per-item rows score.py wrote.
Usage: python summarize.py [--data DIR]   (rewrites results/*.json headline numbers and results/leaderboard.json)"""
import argparse
import glob
import json
import os
from collections import defaultdict


def pair_key(it):
    return (it["prompt_audio"], it["target_text"])


def summarize(rows, items):
    pair = {it["id"]: pair_key(it) for it in items}
    g = defaultdict(list)
    for r in rows:
        g[pair[r["id"]]].append(r)
    mean = lambda rs, k: sum(r[k] for r in rs) / len(rs)
    return {
        "n_pairs": len(g),
        "cer": round(100 * sum(mean(rs, "errors") for rs in g.values()) / sum(mean(rs, "chars") for rs in g.values()), 2),
        "voice_match": round(sum(mean(rs, "sim") for rs in g.values()) / len(g), 3),
        "utmos": round(sum(mean(rs, "utmos") for rs in g.values()) / len(g), 2),
    }


def per_item(rows):
    n = len(rows)
    return {"n": n, "cer": round(100 * sum(r["errors"] for r in rows) / sum(r["chars"] for r in rows), 2),
            "voice_match": round(sum(r["sim"] for r in rows) / n, 3), "utmos": round(sum(r["utmos"] for r in rows) / n, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    a = ap.parse_args()
    items = [json.loads(l) for l in open(os.path.join(a.data, "items.jsonl"), encoding="utf-8")]
    board = {}
    for f in sorted(glob.glob("results/*.json")):
        if f.endswith("leaderboard.json"):
            continue
        res = json.load(open(f, encoding="utf-8"))
        res.update(summarize(res["rows"], items))
        res["per_item"] = per_item(res["rows"])
        res["version"] = "1.1"
        json.dump(res, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        board[res["name"]] = {k: res[k] for k in ("name", "track", "n_pairs", "cer", "voice_match", "utmos", "per_item", "version")}
        print(f"{res['name']:18} [{res['track']}]  CER {res['cer']:5.2f}  voice match {res['voice_match']:.3f}  UTMOS {res['utmos']:.2f}"
              f"   (v1.0 per item: {res['per_item']['cer']:.2f} / {res['per_item']['voice_match']:.3f} / {res['per_item']['utmos']:.2f})")
    json.dump(board, open("results/leaderboard.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

"""How much answer F1 is left once the gold pages are already in the dossier."""

from __future__ import annotations

import json

import open_retrieve as ore
import run_open as ro


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def main() -> None:
    rows = ro.load_hotpot()
    _, test_idx, _ = ro.split_indices(rows)
    by_id = {rows[i]["id"]: rows[i] for i in test_idx}
    retrieved = {}
    with (ro.RESULT_DIR / "action_retrieve_predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row["id"] in by_id:
                    retrieved[row["id"]] = row
    answers = {}
    with (ro.RESULT_DIR / "action_answer_3b_predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                answers[row["id"]] = row
    db = ore.WikiDB()
    contain = both = n = 0
    f1_both = []
    f1_contain = []
    try:
        for qid, row in by_id.items():
            packed = retrieved[qid]
            pages = db.fetch(packed["rowids"])
            text = norm(" ".join(pages[i].title + " " + pages[i].text for i in packed["rowids"] if i in pages))
            gold = row["answers"][0]
            hit = norm(gold) in text
            f1 = answers[qid]["f1"]
            n += 1
            if packed["ce"] == 1:
                both += 1
                f1_both.append(f1)
            if hit:
                contain += 1
                f1_contain.append(f1)
    finally:
        db.close()
    print(f"n {n} both {both} answer_in_pages {contain}", flush=True)
    print(f"f1_when_both {sum(f1_both)/len(f1_both):.3f}" if f1_both else "no both", flush=True)
    print(f"f1_when_answer_present {sum(f1_contain)/len(f1_contain):.3f}" if f1_contain else "no contain", flush=True)


if __name__ == "__main__":
    main()

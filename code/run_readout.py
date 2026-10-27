"""Off-the-shelf readers over the frozen cross-encoder pages.

No threshold is fit on the test slice. Bridge and comparison questions use the
same rule: the highest-scoring extractive span, and a separate Flan answer.
"""

from __future__ import annotations

import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, AutoTokenizer

import open_retrieve as ore
import run_experiment as rx
import run_open as ro


def spans(tokenizer, model, questions, contexts):
    out = []
    for start in range(0, len(questions), 8):
        qs = questions[start : start + 8]
        cs = contexts[start : start + 8]
        batch = tokenizer(
            qs,
            cs,
            padding=True,
            truncation="only_second",
            max_length=384,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = batch.pop("offset_mapping")
        encodings = batch.encodings
        with torch.inference_mode():
            pred = model(**batch)
        start_logits = pred.start_logits.float()
        end_logits = pred.end_logits.float()
        for b in range(len(qs)):
            allowed = [i for i, sid in enumerate(encodings[b].sequence_ids) if sid == 1]
            if not allowed:
                out.append(("", 0.0))
                continue
            s_logit = start_logits[b].clone()
            e_logit = end_logits[b].clone()
            ban = torch.full_like(s_logit, -1e4)
            ban[allowed] = 0.0
            s_logit = s_logit + ban
            e_logit = e_logit + ban
            s_idx = int(torch.argmax(s_logit))
            e_idx = int(torch.argmax(e_logit[s_idx:])) + s_idx
            score = float(torch.softmax(s_logit, dim=0)[s_idx] * torch.softmax(e_logit, dim=0)[e_idx])
            s_char = int(offsets[b, s_idx, 0])
            e_char = int(offsets[b, e_idx, 1])
            out.append((cs[b][s_char:e_char].strip(), score))
    return out


def main() -> None:
    rows = ro.load_hotpot()
    _, test_idx, _ = ro.split_indices(rows)
    chosen = [rows[i] for i in test_idx]
    saved = {}
    with (ro.RESULT_DIR / "action_retrieve_predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                saved[row["id"]] = row
    db = ore.WikiDB()
    questions = []
    contexts = []
    owners = []
    page_text = {}
    try:
        for row in chosen:
            packed = saved[row["id"]]
            pages = db.fetch(packed["rowids"])
            ordered = [pages[i] for i in packed["rowids"] if i in pages]
            page_text[row["id"]] = "\n".join(f"{page.title}. {page.text}" for page in ordered)
            for page in ordered:
                questions.append(row["question"])
                contexts.append(page.title + ". " + page.text)
                owners.append(row["id"])
    finally:
        db.close()
    del db
    print(f"span pairs {len(questions)}", flush=True)
    tok = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2", local_files_only=True)
    model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2", local_files_only=True)
    model.eval()
    found = spans(tok, model, questions, contexts)
    del model
    best = {}
    for qid, (text, score) in zip(owners, found):
        prev = best.get(qid)
        if prev is None or score > prev[1]:
            best[qid] = (text, score)
    span_f1 = []
    span_em = []
    for row in chosen:
        em, f1 = rx.best_em_f1(best[row["id"]][0], row["answers"])
        span_em.append(em)
        span_f1.append(f1)
    print(f"squad em {np.mean(span_em):.3f} f1 {np.mean(span_f1):.3f}", flush=True)

    ftok = AutoTokenizer.from_pretrained("google/flan-t5-base", local_files_only=True)
    fmodel = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", local_files_only=True)
    fmodel.eval()
    flan_f1 = []
    flan_em = []
    for row in chosen:
        prompt = (
            "Answer the question with the shortest span from the pages. "
            "If it is a yes/no question, answer yes or no.\n\n"
            + page_text[row["id"]]
            + "\n\nQuestion: "
            + row["question"]
            + "\nAnswer:"
        )
        enc = ftok(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.inference_mode():
            out = fmodel.generate(**enc, max_new_tokens=16, do_sample=False)
        pred = ftok.decode(out[0], skip_special_tokens=True).strip()
        em, f1 = rx.best_em_f1(pred, row["answers"])
        flan_em.append(em)
        flan_f1.append(f1)
    print(f"flan em {np.mean(flan_em):.3f} f1 {np.mean(flan_f1):.3f}", flush=True)
    payload = {
        "n": len(chosen),
        "squad_em": float(np.mean(span_em)),
        "squad_f1": float(np.mean(span_f1)),
        "flan_em": float(np.mean(flan_em)),
        "flan_f1": float(np.mean(flan_f1)),
    }
    (ro.RESULT_DIR / "action_reader_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

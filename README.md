# CoalitionRAG

CoalitionRAG selects evidence for multi-hop questions by modeling how passages support one another. Its current implementation learns a distribution over evidence forests, computes passage membership probabilities, and uses selective closure and retrieval expansion to assemble useful evidence sets.

Start with `code_v2/`. The earlier retrieval and reader experiments remain in `code/` so their implementations can be inspected and reused.

## Code map

| Path | Purpose |
| --- | --- |
| `code_v2/forest.py` | Evidence-forest training, probabilities and marginals |
| `code_v2/closure.py` | Selective evaluation of passage dependencies |
| `code_v2/expand.py` | Candidate recruitment through additional retrieval |
| `code_v2/first_stage.py` | Initial retrieval and embedding utilities |
| `code_v2/prep_train.py` | Training examples from benchmark splits |
| `code_v2/prep_open_train.py` | Retrieval-generated training candidate pools |
| `code_v2/qa.py` | Answer generation from selected evidence |
| `code_v2/test_forest.py` | Exact enumeration check of the forest equations |
| `code/` | Earlier retrieval policies, readers and diagnostic experiments |

## Setup and paths

Use Python 3.10 or newer with a suitable PyTorch installation:

```bash
python -m pip install -r requirements.txt
python code_v2/test_forest.py
```

`CRAG_ROOT` selects the experiment root; it defaults to this repository. Data, models and outputs are read from `data/`, `models/` and `results/` beneath that root. For Bash launchers, `env.sh` establishes these paths and creates output directories. Set `PYTHON` to use a particular interpreter.

The language-model reader scripts additionally require vLLM and a compatible GPU environment. The optional HippoRAG comparison imports its upstream implementation from `repos/HippoRAG`; obtain that project separately if you run this comparison. Core forest computations do not require that checkout.

## Prepare benchmark inputs

Obtain HotpotQA, 2WikiMultihopQA and MuSiQue from their original distributors. The current loader expects two JSON files per evaluation dataset in `data/`: `<dataset>.json` for questions and `<dataset>_corpus.json` for the passage corpus. `code_v2/common.py::load_bench` defines the expected fields and evidence matching.

For training, place the original files under:

```text
data/raw/hotpot_qa/distractor/train-*.parquet
data/raw/2WikiMultihopQA/data/train-*.parquet
data/raw/MuSiQue/musique_ans_v1.0_train.jsonl
```

Prepare the evaluation question files before building training pools: `prep_train.py` uses their IDs to exclude overlaps. Then run:

```bash
python code_v2/prep_train.py --n 4000 --seed 0
python code_v2/prep_open_train.py
python code_v2/forest.py --name forest --train train_open.jsonl --epochs 1 --seed 0
```

Pool construction uses a GPU and `BAAI/bge-large-en-v1.5`. Forest evaluation also needs the candidate pools identified by `--pool_file`; training alone does not generate those evaluation pools. Inspect `first_stage.py`, `expand.py` and `forest.py --help` when preparing a new experiment. Keep the same question IDs, corpus and pools across comparisons.

## Evaluation and checks

The enumeration test compares forest probabilities and marginals against all valid parent assignments for a small passage pool. Retrieval and question-answering runners write JSON outputs under `results/`; the analysis scripts consume those outputs.

The Bash queues retain multi-GPU experiment schedules. They include process management and assume prepared inputs, so use the individual Python entry points for an initial run and adapt a queue before unattended execution.

This repository contains source code and configuration only. Manuscripts, publication figures, generated results, model checkpoints, dataset copies and private connection settings are excluded. `RELEASE_MANIFEST.json` lists the released files and their hashes.

# Thesis Work 2

This repository includes related codes and files for the Thesis Work titled:

**Evaluating the Evaluators: A Study of Anomalies in LLM-Based Assessment Systems**

The project studies reliability issues in LLM-based evaluation via two experiments:

- **Translation Game:** tests whether a model follows or resists state-changing instructions such as stop/continue commands during translation.
- **Ranking Task:** tests whether LLM evaluators consistently rank candidate translations against a ground-truth sentence.

## Setup

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

Set the API token for the backend you want to use:

```bash
export API_KEY=your_openai_key
export HF_TOKEN=your_huggingface_token
```

The scripts currently support `openai` and `llama` backends. Backend selection is configured in each script's `main()` function.

## Project Structure

```text
.
├── data/
│   ├── input/                 # Source sentences and prepared translation candidates
│   └── output/                # Generated experiment results and analysis reports
├── ranking/                   # Translation ranking experiments and analysis
├── translation_game/          # Translation instruction-following experiments
├── requirements.txt
```

## Translation Game

Run the translation experiments:

```bash
python translation_game/translation_game.py
```

This script uses `data/input/1000_sentences.txt`, runs temperature-based sentence-level and multi-turn translation-state tests, and writes JSON results to:

```text
data/output/translation_game/{backend}/
```

Optional data generation:

```bash
python translation_game/generate_data.py
```

Inspect selected result slices:

```bash
python translation_game/translation_game_analysis.py
```

The system prompt for the complex translation-state test is stored in `translation_game/config.yaml`.

## Ranking Task

Run the ranking experiments:

```bash
python ranking/ranking.py
```

The script reads:

```text
data/input/translation_versions_final.csv
```

It generates triplet and pairwise ranking results, with optional confidence scoring controlled by `with_confidence` in `ranking/ranking.py`.

Outputs are written to:

```text
data/output/ranking/{backend}/
```

Analyze ranking stability and mitigation strategies:

```bash
python ranking/ranking_analysis.py
```

This produces stability reports, permutation/run stability summaries, and mitigation result CSVs.

## Data Preparation

If rebuilding the ranking input data from raw translation candidates:

```bash
python ranking/clean_translations_csv.py
python ranking/calc_bert_score.py
```

`calc_bert_score.py` selects best, middle, and worst alternatives using BERTScore and writes `data/input/translation_versions_final.csv`.

## Notes

- Run commands from the repository root.
- `ranking/calc_bert_score.py` uses `device="mps"` by default; change it to `cuda` or `cpu` if needed.
- Some scripts contain experiment parameters directly in `main()`.

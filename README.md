# Critique-Resilient Benchmarking

A multi-stage evaluation pipeline for LLM-generated questions, answers, critiques, debates, and automated judgments. The repository produces dataset artifacts plus analysis outputs (stats and a bipartite Bradley–Terry Elo fit) to compare models as both questioners and answerers.

## Quickstart

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set provider keys in your environment (or in a `.env` file if you prefer to load it manually):

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
export OPENROUTER_API_KEY=...
# Optional for some Gemini batch workflows
export GOOGLE_CLOUD_PROJECT=...
```

## Configure a run

- `configs/models.json`: model registry, roles, and provider routing.
- `configs/runs.json`: mapping of run_id to topic slug.
- `configs/topic_info.json`: topic metadata for generation.
- `configs/parsing.json` (optional): JSON parsing repair settings.
- Prompt templates live in `prompt_library.py` and `prompt.md`.

## Run the full pipeline

The recommended entry point is the orchestrator:

```bash
python run_full_pipeline.py
```

`docs/params.md` lists all CLI parameters and defaults.

### Run steps individually

```bash
python generate_benchmark.py --runs-file configs/runs.json --topic-info-file configs/topic_info.json --config configs/models.json --output-dir benchmarks
python generate_answers.py --config configs/models.json --benchmark-dir benchmarks --output-dir answers
python generate_critiques.py --mode contradictor --config configs/models.json --benchmark-dir benchmarks --answers-dir answers --output-dir critiques
python generate_critiques.py --mode evaluator --config configs/models.json --benchmark-dir benchmarks --answers-dir answers --output-dir critiques
python debate.py --mode ill-posed --config configs/models.json --benchmark-dir benchmarks --answers-dir answers --critiques-dir critiques --output-dir debates
python debate.py --mode critique --config configs/models.json --benchmark-dir benchmarks --answers-dir answers --critiques-dir critiques --output-dir debates
python automated_judge.py --mode all --config configs/models.json --benchmark-dir benchmarks --answers-dir answers --critiques-dir critiques --debates-dir debates --output-dir automated_evaluations
```

## Analyze results

```bash
python stats.py --disable-disagreement-logs
python compute_bt_elo.py --disable-disagreement-logs
```

Analysis tables and figures are typically stored under `analysis/`.

## Data layout (artifacts)

- `benchmarks/`: generated questions + reference answers by question author.
- `answers/`: answers per questioner/answerer pair.
- `critiques/`: critique records (contradictor + evaluator modes).
- `debates/`: debate transcripts for disputed critiques or ill-posed claims.
- `automated_evaluations/`: automated judge decisions.
- `evaluations/`: optional human labels.

## Human labeling

There is a Streamlit-based labeler for disputes:

```bash
./run_labeller.sh YOUR_USERNAME
```

Or run the CLI helper directly:

```bash
python label.py
```

## Documentation

- `docs/question_flow.md`: end-to-end walkthrough with code pointers.
- `docs/writeup.md`: experiment writeup + current dataset stats.
- `docs/params.md`: full pipeline CLI parameters and defaults.


### Bayesian Prompt Optimization (LLM-as-a-Judge)

This repository contains a small framework to optimize and validate LLM-as-a-Judge prompts for two metrics:

- Code Quality
- Faithfulness (adherence of the code to a stated requirement)

It uses DSPy to define judge signatures, optimize them with a Bayesian-inspired setup, and then validate the optimized prompts on a holdout dataset against human annotations.

#### Key Features
- DSPy signatures for `EvaluateCodeQuality` and `EvaluateFaithfulness`.
- Programmatic optimization of prompts using `MIPROv2` with a continuous-score metric.
- Validation runner that compares Original vs. Normal vs. Bayesian optimized prompts, alongside human annotators.
- Reproducible artifacts written to the `artefacts/` folder.

---

### Repository Structure
- `code/`
  - `optimization.py` — builds training datasets and optimizes prompt programs via DSPy.
  - `validation.py` — loads prompts (original and optimized) and evaluates on a holdout set with human scores.
  - `code_quality_signature.py`, `faithfulness_signature.py` — DSPy signatures defining inputs/outputs for the judge.
- `prompts/` — saved/loaded prompt programs in JSON (outputs of optimization).
- `data/` — benchmark and human-annotation datasets (train and holdout).
- `artefacts/` — produced metrics and validation outputs.
- `settings.py` — environment-driven configuration for the LLM client.
- `pyproject.toml` — project configuration and dependencies.
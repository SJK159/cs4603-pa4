# Bonus A — CI/CD with GitHub Actions: Student Guide

This guide explains the concepts behind Bonus A and gives step-by-step guidelines for
building the `lint → test → deploy` pipeline in `.github/workflows/deploy.yml`.

---

## 1. What is a "pipeline" and why do we need one?

A **pipeline** is an automated sequence of steps that runs every time your code changes.
Instead of manually running `ruff`, then tests, then a deploy script on your laptop (and
forgetting a step, or deploying broken code), the pipeline does it for you — the same way,
every time.

Two terms you'll hear:

- **CI (Continuous Integration):** automatically **lint + test** every change so bugs are
  caught early.
- **CD (Continuous Deployment/Delivery):** automatically **ship** the code (here: log the
  model, register a Unity Catalog version, update the serving endpoint) once tests pass.

Bonus A asks you to build a `lint → test → deploy` pipeline. The golden rule:
**deploy only runs if lint and test pass.**

---

## 2. GitHub Actions vocabulary

GitHub Actions is GitHub's built-in CI/CD engine. Your pipeline lives in a YAML file under
`.github/workflows/`. Five concepts explain 90% of it:

| Concept | What it is | In `deploy.yml` |
|---|---|---|
| **Workflow** | The whole automated process (one YAML file) | `name: deploy` |
| **Event / Trigger** | *When* it runs | `on: push`, `pull_request`, `workflow_dispatch` |
| **Job** | A group of steps running on one fresh machine | `lint-and-test`, `deploy` |
| **Step** | A single command or reusable Action | `uses: actions/checkout@v4`, `run: ruff check` |
| **Runner** | The throwaway VM GitHub gives you | `runs-on: ubuntu-latest` |

Key ideas students trip over:

- **Every job gets a brand-new, empty machine.** That's why the first step is always
  `actions/checkout@v4` (pull your code onto the runner) and why you must **re-install
  dependencies** in each job — nothing persists between jobs.
- `uses:` runs a **pre-built Action** someone published (e.g. checkout, setup-uv).
  `run:` runs a **shell command** you write.
- Jobs run **in parallel** by default. Use `needs:` to force ordering
  (`deploy` `needs: lint-and-test`).

---

## 3. How the triggers map to the assignment requirements

```yaml
on:
  push:
    branches: [main]     # deploy path
  pull_request:          # lint+test on PRs, but NOT deploy
  workflow_dispatch:     # the "manual trigger" the assignment requires
```

- `push` to `main` → the real deploy.
- `pull_request` → runs lint+test so reviewers see green/red before merging, but must
  **not** deploy.
- `workflow_dispatch` → adds a "Run workflow" button in the GitHub UI for ad-hoc deploys.

The deploy job enforces "main only" with an `if` condition:

```yaml
deploy:
  needs: lint-and-test
  if: github.ref == 'refs/heads/main' && github.event_name != 'pull_request'
```

---

## 4. Secrets — never hardcode credentials

`DATABRICKS_HOST` and `DATABRICKS_TOKEN` must **not** be written into the YAML. Store them
in GitHub:

**Repo → Settings → Secrets and variables → Actions → New repository secret.**

Then reference them in the workflow, which injects them as environment variables:

```yaml
    env:
      DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
      DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
```

This is the same `DATABRICKS_TOKEN` env var described in the assignment's auth note — the
pipeline runs headless, so it reads the token from a secret instead of a CLI profile.

---

## 5. Step-by-step guidelines

The repo already gives you a skeleton in `.github/workflows/deploy.yml` with `TODO`s.
Fill them in:

1. **`lint-and-test` job — fill the TODO:**
   - `actions/checkout@v4` (already there).
   - Install `uv` (use the `astral-sh/setup-uv` action), then `uv sync` to install deps.
   - Run `uv run ruff check agent/ client/`.
   - Run `uv run pytest -q` — this executes `tests/test_smoke.py`, which imports and
     compiles the graph offline (no Databricks, no network). **You must implement that
     smoke test** (its TODO) so this step is meaningful.

2. **`deploy` job — fill the TODO:**
   - `needs: lint-and-test` + the `if` guard (already there) enforce the ordering and
     main-only rule.
   - Checkout, install deps again (new machine!).
   - Pass the two secrets as `env`.
   - `run: uv run python deployment/deploy.py` — this logs the model to MLflow, registers
     a UC version, and updates the endpoint.
   - Finish by **printing the deployed model version and endpoint status** (a requirement)
     — either `echo` it or have `deploy.py` print it.

3. **Test it before submitting:**
   - Push to a **feature branch** first → confirm lint+test run and deploy is **skipped**.
   - Open a PR → confirm lint+test run.
   - Merge/push to `main` → confirm the full deploy runs and prints the version + status.
   - Watch runs live under the repo's **Actions** tab; click a red X to read logs.

---

## 6. Answering the analysis questions

- **Q1 (why deploy only on `main`?):** `main` is the single source of truth that's
  protected and reviewed. Feature branches are experimental and may be broken; deploying
  from them would push half-finished or untested work to the live endpoint and let
  concurrent developers overwrite each other's deployments. Merging to `main` is the
  deliberate "this is ready" signal.
- **Q2 (a quality gate before deploy):** Add an **evaluation gate** between test and
  deploy — run the new model against a held-out eval set, compute a metric (e.g. answer
  accuracy / faithfulness), and compare it to the currently-serving version's score. If the
  new score is worse (or below a threshold), fail the job so the deploy step never runs.
  Use MLflow's `mlflow.evaluate` and compare against the production version's logged metrics.

---

## Common mistakes to avoid

- Forgetting to re-install dependencies in the `deploy` job (fresh machine).
- Hardcoding the host/token instead of using `${{ secrets.* }}`.
- Leaving `pytest` passing trivially because the smoke test wasn't implemented.
- Letting deploy run on PRs or feature branches (missing/incorrect `if`).
- YAML indentation errors — GitHub Actions is strict; use the Actions tab error messages.

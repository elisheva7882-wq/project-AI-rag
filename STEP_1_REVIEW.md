# KingdomAI — Step 1 review

Date: 2026-11-04 (Asia/Jerusalem).

Scope: project preparation only, as defined in section 14 of the supplied plan. Step 2 requires the user's approval. No other projects were modified.

## Inputs inspected

- The entire `CODEX_PLAN_KingdomAI_RAG.md` and the archive's `00_קראו_תחילה.md` were read before implementation.
- The workspace was empty, with no applicable `AGENTS.md` found in its ancestor directories.
- ZIP directory inspection found 19 original Markdown documents (`ALT-01`–`ALT-19`), 19 TXT duplicates, two evaluation files, an archive introduction and a manifest.
- No archive files were extracted or copied into `knowledge_base/` or `eval/` in Step 1.
- The plan's 512/64/5 configuration takes precedence over the archive introduction's alternative 512/50/4 suggestion.

## Files created

| File | Purpose |
|---|---|
| `CODEX_PLAN_KingdomAI_RAG.md` | Unmodified copy of the supplied plan. |
| `requirements.txt` | Nine required direct dependencies pinned to installed, import-verified versions. |
| `requirements.lock.txt` | Snapshot of 141 installed distributions for reproducing this Windows / Python 3.12.9 environment. |
| `.env.example` | Plan defaults and a placeholder API key. |
| `.gitignore` | Excludes secrets, environments, caches, logs and local index storage. |
| `.streamlit/config.toml` | Initial theme and disabled Streamlit usage statistics. |
| `config.py` | Validated settings with separate knowledge/evaluation/index paths, optional credentials and prerequisite reporting. |
| `src/__init__.py` | RAG package boundary for subsequent steps. |
| `scripts/__init__.py` | Command-line utilities package. |
| `scripts/check_setup.py` | Offline dependency imports, versions, settings and prerequisite diagnostics. |
| `pytest.ini` | Project test discovery and import path. |
| `tests/test_config.py` | Settings validation, UTF-8, missing prerequisites, secret representation and contamination prevention tests. |
| `knowledge_base/.gitkeep` | Empty source-document directory. |
| `eval/.gitkeep` | Empty evaluation directory, separate from sources. |
| `storage/.gitkeep` | Empty local storage directory. |
| `README.md` | Initial Hebrew documentation and working PowerShell commands for the current step. |
| `STEP_1_REVIEW.md` | This inventory, verification results and approval boundary. |

Generated local environment: `.venv/` (ignored), including dependency installation output in `.venv/dependency-install.log`. No real `.env` was created and no API key was added to any file.

## Verification actually performed

| Check | Actual result |
|---|---|
| Available interpreter | Python 3.12.9; `python` works, while the `py` launcher reports no installed interpreters. |
| Clean virtual environment installation | The nine plan dependencies and their transitive dependencies installed successfully into the new `.venv`, exit 0. Sandbox networking initially blocked pip; the authorized network-enabled retry succeeded. |
| Installed dependency consistency | `.venv\Scripts\python.exe -m pip check`: `No broken requirements found.`, exit 0. |
| Required dependency/adapter imports | All nine distributions and required imports passed in `scripts.check_setup`, including `GoogleGenAI`, `GoogleGenAIEmbedding`, `ChromaVectorStore` and `SentenceSplitter`. No clients were instantiated. |
| Normal setup diagnostic | `.venv\Scripts\python.exe -X utf8 -m scripts.check_setup` returned expected exit code 1: no API key, no Markdown source documents and no index manifest. |
| Source syntax | `python -m compileall -q config.py src scripts tests` exited 0. |
| Configuration tests | `.venv\Scripts\python.exe -m pytest -q`: **36 passed in 0.24s**, exit 0. |
| Missing-dependency diagnostic | `.venv\Scripts\python.exe -S -m scripts.check_setup` reported the nine unavailable distributions and returned the intended exit code 2 without a traceback. |
| Plan copy integrity | Both original and copied file have SHA-256 `613f93302fbd394c2870dd5bdfaa4f27cdeb8b7d483331016fa1175d5ed327cf`. |
| Data boundary | `knowledge_base/`, `eval/` and `storage/` each contain only `.gitkeep`. |
| Final file audit | All nine direct pins match installed metadata and the lock snapshot; Streamlit TOML parses with usage statistics disabled; no real `.env` exists. The review inventory contains 17 project files plus the ignored local environment/caches. |

Verified direct versions: `streamlit==1.66.1`, `python-dotenv==1.2.4`, `chromadb==1.5.9`, `llama-index-core==0.14.25`, `llama-index-vector-stores-chroma==0.6.0`, `llama-index-llms-google-genai==0.11.3`, `llama-index-embeddings-google-genai==0.6.0`, `google-genai==2.34.0`, `pytest==9.2.0`. Python versions other than 3.12.9 have not been tested here.

An independent read-only review identified and resolved an early dotenv import in the setup checker and missing Chroma collection-name validation. The follow-up review reported no remaining material findings; it did not run tests.

## External documentation checks

Public Google documentation lists the specified [Gemini 3.8 Flash model](https://aistudio.google.com/docs/models/gemini-3.8-flash) and [gemini-embedding-001](https://aistudio.google.com/docs/embeddings). Neither account access nor quotas have been verified. No model was substituted, no Gemini API requests were made and no billing settings were changed.

The required adapters and explicit model/API-key parameters were checked against the official [LlamaIndex LLM](https://developers.llamaindex.ai/python/framework-api-reference/llms/google_genai/) and [embedding](https://developers.llamaindex.ai/python/framework-api-reference/embeddings/google_genai/) references. Collection-name validation follows the current [Chroma Rust validator](https://raw.githubusercontent.com/chroma-core/chroma/main/rust/types/src/validators.rs).

These are implementation references only. They will not be indexed or used as knowledge for answers about Altheria.

## Not yet implemented or tested

- Archive preparation, full document ID/content validation and evaluation-file copying (Step 2).
- LlamaIndex chunking, Gemini embeddings, Chroma persistence/rebuild and index manifest (Step 3).
- Retrieval, grounded answers, conflict handling and source citation checks (Step 4).
- Streamlit chat and its runtime/error states (Step 5).
- Corpus/RAG acceptance tests and the 45 supplied evaluation questions (Step 6).
- Live model availability, Gemini calls, account entitlement and quotas.

No end-to-end or answer-quality pass is claimed. Step 1 prepares the foundation; the bot is not runnable yet. The next action after approval is Step 2: safely prepare only the 19 original Markdown documents and keep the evaluation materials separate.

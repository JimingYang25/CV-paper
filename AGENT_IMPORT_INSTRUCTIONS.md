# Import instructions for another agent

Target GitHub repository: https://github.com/JimingYang25/CV-paper

## Contents

This archive contains:
- 4 **complete** original Markdown lessons, copied from accessible exports dated 2026-10-13 through 2026-10-16.
- Root `README.md`, category indexes, `CURRICULUM.md`, and `NOTE_TEMPLATE.md`.
- A checksum manifest in `MANIFEST.json`.

The remaining earlier lessons are listed in the curriculum but **not** included as full Markdown documents.

## Import safely

1. Extract the ZIP. Its top-level folder is `CV-paper/`.
2. Merge the **contents of** `CV-paper/` into the root of the GitHub repository, not a nested `CV-paper/CV-paper/` path.
3. Verify existing repository contents before overwriting any file. Prefer a new branch and pull request if the repository is not empty.
4. Do not accidentally commit the `.zip` inside the repo or add unrelated temp data.
5. Preserve note contents and references; review linked research and numbers before publishing as authoritative material.
6. Add future lessons under the matching category and update category `README.md`, root `README.md`, and `CURRICULUM.md`.

Suggested commands (after manual download and extraction):

```bash
unzip CV-paper_Categorized_VLM_Notes.zip
# From a local clone of the target repo:
cp -a /path/to/unzipped/CV-paper/. /path/to/your/CV-paper-clone/
cd /path/to/your/CV-paper-clone
git status
git add README.md CURRICULUM.md AGENT_IMPORT_INSTRUCTIONS.md NOTE_TEMPLATE.md MANIFEST.json VLM/
git commit -m "docs: add categorized VLM deep-dive lessons"
git push origin main
```

**CAUTION:** `git push` is only appropriate after confirming branch policy and authorization. The user has delegated uploading to another agent; this package itself makes no repo changes.

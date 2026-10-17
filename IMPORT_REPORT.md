# Categorized VLM notes import — 2026-10-17

## Source and contents

Source archive: `CV-paper_Categorized_VLM_Notes.zip` supplied by the repository owner.
The archive contains 19 files, including four complete lessons dated October
13–16, ten category indexes, a curriculum, a note template, import instructions,
a root README, and a checksum manifest. Twelve earlier topics are listed in the
curriculum without complete exported notes; this import does not reconstruct them.

## Verification performed

- ZIP integrity check passed for every entry.
- All 18 entries in the source checksum manifest matched their SHA-256 hashes
  and recorded byte sizes before import.
- The four complete lesson files, including their mathematics, code blocks,
  and source links, are preserved byte-for-byte from the archive.
- Category indexes, curriculum, template, instructions, and the source manifest
  are preserved. The repository root README is merged with the existing CV index.
- A `VLM/README.md` navigation page is added, with links adjusted for its directory.
- Existing CV lesson, code, illustration, and validation files are retained.
- No archive file or unrelated local files are committed.

The source `MANIFEST.json` remains the original export manifest. Its
`offline_export_not_pushed` field describes the ZIP's creation state; it is not
the current publication status. Its root `README.md` checksum describes the
original archive README, which was merged during import, rather than this
repository's combined README. It is not a checksum inventory of the whole repository.

## Scope of validation

This import checks provenance, file integrity, repository integration, and local
navigation. It does not independently revalidate research claims or performance
numbers in the four source notes and does not execute their embedded examples.
Executed checks for the October 17 CV example remain documented in
[its validation record](examples/deformable-attention/VALIDATION.md).

## Full source lessons

- [2026-10-13 — Exact Multimodal Speculative Decoding](VLM/Inference-and-Decoding/2026-10-13_Exact-Multimodal-Speculative-Decoding.md)
- [2026-10-14 — SLO-Aware Multimodal Serving](VLM/Serving-and-Infrastructure/2026-10-14_SLO-Aware-Multimodal-Serving.md)
- [2026-10-15 — ColPali / MaxSim / Vision-First Multimodal RAG](VLM/Multimodal-RAG-and-Retrieval/2026-10-15_ColPali-MaxSim-Vision-First-RAG.md)
- [2026-10-16 — Verifier-Guided Multi-Hop Multimodal RAG](VLM/Multimodal-RAG-and-Retrieval/2026-10-16_Verifier-Guided-Multi-Hop-Multimodal-RAG.md)

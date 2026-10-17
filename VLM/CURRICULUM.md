# VLM Daily Deep Dive — Curriculum Map

This map tracks the sequence discussed in ChatGPT. **Only the last four entries have complete Markdown exports inside this ZIP.** Earlier material exists in the conversation but is not included as a verified full text file.

| Day | Topic | Suggested category | Full file in archive |
|---:|---|---|---|
| 0 | SigLIP and pairwise sigmoid image–text contrastive training | Representation-Learning | No |
| 1 | Q-Former vs. MLP Projector vs. Perceiver Resampler | Multimodal-Connectors | No |
| 2 | Native Dynamic Resolution and visual-token budgeting | Quantization-and-Compression | No |
| 3 | Visual token pruning and merging | Quantization-and-Compression | No |
| 4 | KV-cache compression, eviction and reuse | Inference-and-Decoding | No |
| 5 | Spatial grounding as language/coordinate tokens | Grounding | No |
| 6 | Flow-matching action experts for VLA | VLA-and-Embodied | No |
| 7 | Multimodal preference optimization (mDPO) | Alignment-and-Post-Training | No |
| 8 | Set-valued temporal grounding with Wasserstein rewards | Video-and-Temporal | No |
| 9 | MotionVLA and motion-consistent 4D memory | VLA-and-Embodied | No |
| 10 | Action-conditioned world models and CEM-MPC | VLA-and-Embodied | No |
| 11 | SAFE conformal failure detection for VLA | VLA-and-Embodied | No |
| 12 | Exact multimodal speculative decoding | Inference-and-Decoding | **Yes** |
| 13 | SLO-aware multimodal serving and chunked prefill | Serving-and-Infrastructure | **Yes** |
| 14 | ColPali, MaxSim, vision-first multimodal RAG | Multimodal-RAG-and-Retrieval | **Yes** |
| 15 | Verifier-guided multi-hop multimodal RAG | Multimodal-RAG-and-Retrieval | **Yes** |

## Curation rules

1. Prefer preserving full original lesson text, math, code and source links.
2. Distinguish **paper-established results**, **engineering heuristics**, and **interpretation**.
3. Keep dates in filenames and factual citations in the lesson body.
4. Place each topic by technical theme rather than date alone.
5. Never label an abbreviated reconstruction as a verbatim archived lesson.
6. Inspect and independently verify cited papers and performance figures before presenting the repository as an authoritative public reference.

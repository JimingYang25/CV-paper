# Daily VLM Deep Dive — Day 14
## ColPali, MaxSim, and Scalable Vision-First Retrieval: Building a Multimodal RAG System That Can Actually Find a Figure

**15 October 2026 | Advanced VLM Engineering | Retrieval, Training, Systems, Deployment**

**Central claim:** Instead of converting every PDF page into OCR text and then searching that text, encode the *rendered page itself* into many visual feature vectors. Match each query token against the most relevant page vector with late interaction, then give the original page to a generative VLM for grounded answering.

**Research anchor:** Faysse et al., *ColPali: Efficient Document Retrieval with Vision Language Models* (submitted June 2024; ICLR 2025). Current extensions: PULSAR (28 August 2026) and MIDR (1 September 2026). Sources and reproducibility links appear at the end.

## 1. Motivation: OCR destroys information that matters

Imagine searching 20,000 robotics and endoscopic-imaging papers for a diagram that compares tactile sensor gel thickness against contact sensitivity. The relevant evidence might appear as a plot with axes, colored curves, a legend, and a caption. A text extractor may recover the caption but miss the curve-to-legend association. OCR may read every word correctly and still lose *which value belongs to which row or curve*. Conventional text RAG often retrieves the wrong page because the visual relationship was discarded before indexing.

The classical pipeline is `PDF -> layout detector -> OCR -> tables/captions -> text chunks -> text embeddings -> vector search -> LLM`. The vision-first pipeline is `PDF -> rendered page -> visual retrieval embeddings -> MaxSim search -> original page image -> VLM answer`.

The retrieval model is **not** the answer generator. Retrieval finds evidence; the answer VLM interprets it. A perfect retriever cannot fix a hallucinating generator, and a perfect generator cannot answer from a page that was never retrieved.

## 2. Core intuition: a page is not one semantic vector

A single-vector image encoder collapses a page to `d in R^D`. That vector must represent headings, diagrams, tables, numbers, captions, and layout simultaneously. ColPali retains a *set* of contextualized vectors. Each query token can independently find its strongest supporting patch or contextualized page token. This is the ColBERT late-interaction idea transferred from text documents to rendered document images.

For the query “Which plot shows decreasing insertion force after lubricant application?”, the vectors for “plot”, “insertion”, “force”, “decreasing”, and “lubricant” may match different regions of the same page. MaxSim sums the best local match for each query vector. The matches need not all land in one image patch, and their independent maxima do not by themselves establish the causal relationship expressed in the question. That is why a second-stage VLM should verify the evidence.

## 3. Mathematical formulation and tensor shapes

Let `q` be a text query and `I_p` be a rasterized image of page `p`. A query encoder and page encoder (two input modes of a retrieval-adapted VLM) produce contextual hidden states:

`H_q = F_theta(q) in R^(B x N_q x d_h)` and `H_p = F_theta(I_p) in R^(B x N_p x d_h)`.

`B` is batch size; `N_q` is query-token count, including any learned expansion tokens; `N_p` is the number of page-associated visual/contextual tokens retained by the processor; `d_h` is the backbone hidden dimension. Actual `N_p` depends on model, image resolution, and token filtering.

A trainable projection `W in R^(d_h x D)` maps hidden states into retrieval space. Normalize each valid row to unit length:

`Q = normalize(H_q W, dim=-1) in R^(B x N_q x D)`

`P = normalize(H_p W, dim=-1) in R^(B x N_p x D)`.

For original ColPali, `D = 128`; newer ColQwen variants can use different dimensions (e.g., 320 for the PULSAR backbone). A normalized dot product is cosine similarity. Query and page padding tokens must be excluded from scoring.

For one query-page pair:

`A = Q P^T in R^(N_q x N_p)`,

`score(q,p) = sum_(i=1)^(N_q) max_(j=1)^(N_p) A[i,j]`.

This is **MaxSim late interaction**. Each query token picks its highest-similarity page token, and the score sums these maxima. The max operation is differentiable almost everywhere; gradients flow to the selected maxima (or a valid subgradient at ties). A query may match multiple terms to the same page patch, so this score is not proof of phrase-level or spatial relational correctness.

### A numerical example

Suppose three query tokens and four page patches yield:

`A = [[0.8, 0.1, 0.2, 0.0], [0.0, 0.7, 0.2, 0.1], [0.1, 0.0, 0.3, 0.9]]`.

The per-query-token maxima are `[0.8, 0.7, 0.9]`; therefore `score = 2.4`. If a second page has maxima `[0.7, 0.2, 0.5]`, its score is `1.4`. The first page ranks higher. These are illustrative numbers, not model outputs.

### Why not mean-pool?

Mean-pooling produces `q_bar in R^D`, `p_bar in R^D`, and one dot product. This is compact but loses the ability to match multiple independent visual facts. Conversely, a cross-encoder can attend jointly to a query and a page, but doing that for millions of pages is too expensive. Late interaction is the compromise: encode pages offline, encode the query once, and retain a cheap(er) query-time token interaction.

## 4. Architecture and complete data flow

**Offline indexing**

`PDF -> page rasterization -> page image preprocessing -> vision-language backbone -> token projection + normalization -> optional token pooling/quantization -> multi-vector index + page-image store + page metadata`.

**Online retrieval**

`user query -> query encoder -> query vectors -> candidate generation -> MaxSim rerank -> top-k page IDs -> high-resolution page images -> generative VLM with explicit evidence instructions -> answer + page citations`.

The original ColPali adapts PaliGemma-3B, whose SigLIP image features are contextualized in the language backbone. It projects output token states to 128-dimensional vectors. It is inaccurate to describe the complete ColPali retrieval representation as simply the raw, frozen SigLIP patches: the language-model contextualization and retrieval projection are important.

A modern production architecture usually separates **retrieval resolution** from **answer resolution**. You can index a lower-resolution rendering and store a higher-resolution page for reading small annotations after retrieval. This saves indexing and retrieval costs without forcing the answer VLM to interpret illegible images.

## 5. Training objective and optimization details

Training requires positive `(query, relevant_page)` pairs and suitable negatives. Let a batch contain `B` paired queries and pages. Define `S_ij = score(q_i,p_j)`. In the original ColPali objective, the hardest in-batch negative for query `i` is:

`s_i^- = max_(j != i) S_ij`, and the positive score is `s_i^+ = S_ii`.

The published pairwise objective is:

`L = (1/B) sum_i softplus(s_i^- - s_i^+)`, with `softplus(x)=log(1+exp(x))`.

This pushes each positive score above the hardest in-batch negative. It differs from full in-batch InfoNCE:

`L_InfoNCE = -(1/B) sum_i log[ exp(S_ii/tau) / sum_j exp(S_ij/tau) ]`.

Here `tau > 0` is a temperature. InfoNCE is a useful alternative, but it is **not the exact original ColPali loss**. In-batch negatives may be false negatives when two pages legitimately answer the same query; a relevance-aware mask is needed for such datasets.

**Original paper configuration:** 118,695 query-page pairs; 63% from academic datasets and 37% synthetic PDF/question pairs; BF16; LoRA rank 32 and alpha 32 on language-model Transformer layers; trainable retrieval projection; paged 8-bit AdamW; learning rate 5e-5 with linear decay and 2.5% warmup; one epoch; reported batch size 32; five learned query-augmentation tokens. These are historical paper settings, not automatically optimal settings for a new corpus.

**Engineering heuristics for domain adaptation:** Start with frozen visual backbone and LoRA on the language component; construct document-level train/validation/test splits; mine visually similar hard negatives (same anatomy, different figure or caption); mix table, diagram, caption, and cross-page queries; verify synthetic questions are actually answerable from the labeled page. Fine-tune only after establishing a strong zero-shot baseline.

## 6. PyTorch-style implementation: MaxSim and hard-negative loss

```python
import torch
import torch.nn.functional as F


def maxsim_matrix(q, d, q_mask=None, d_mask=None):
    """
    q: [B, Nq, D] normalized query vectors
    d: [B, Nd, D] normalized page vectors
    q_mask: [B, Nq] bool, valid query tokens
    d_mask: [B, Nd] bool, valid document tokens
    Returns: [B, B] query-page scores
    """
    B, Nq, D = q.shape
    assert d.shape[0] == B and d.shape[-1] == D
    if q_mask is None:
        q_mask = torch.ones(B, Nq, dtype=torch.bool, device=q.device)
    if d_mask is None:
        d_mask = torch.ones(B, d.shape[1], dtype=torch.bool, device=d.device)
    assert q_mask.any(dim=1).all() and d_mask.any(dim=1).all()

    # [B(query), B(page), Nq, Nd]
    sim = torch.einsum('aqd,bpd->abqp', q.float(), d.float())
    sim = sim.masked_fill(~d_mask[None, :, None, :], float('-inf'))
    best_patch = sim.amax(dim=-1)                  # [B, B, Nq]
    best_patch = best_patch.masked_fill(~q_mask[:, None, :], 0.0)
    return best_patch.sum(dim=-1)                  # [B, B]


def colpali_hard_negative_loss(q, d, q_mask, d_mask):
    scores = maxsim_matrix(q, d, q_mask, d_mask)
    B = scores.size(0)
    assert B >= 2
    positive = scores.diag()
    negatives = scores.masked_fill(
        torch.eye(B, device=scores.device, dtype=torch.bool),
        float('-inf'),
    )
    hardest = negatives.max(dim=1).values
    return F.softplus(hardest - positive).mean()


# Unit test: score is invariant to adding masked page padding.
torch.manual_seed(7)
q = F.normalize(torch.randn(2, 3, 8), dim=-1)
d = F.normalize(torch.randn(2, 4, 8), dim=-1)
qm = torch.ones(2, 3, dtype=torch.bool)
dm = torch.ones(2, 4, dtype=torch.bool)
base = maxsim_matrix(q, d, qm, dm)
padded = torch.cat([d, F.normalize(torch.randn(2, 2, 8), dim=-1)], dim=1)
padmask = torch.cat([dm, torch.zeros(2, 2, dtype=torch.bool)], dim=1)
assert torch.allclose(base, maxsim_matrix(q, padded, qm, padmask))
```

**Important:** This transparent reference implementation materializes a `[B,B,Nq,Nd]` tensor. For large training batches, use a fused MaxSim kernel, chunk pair scoring, or an optimized late-interaction library. The public ColPali repository documents fused kernels that avoid materializing the full pairwise tensor.

### Using a pretrained retriever

As of October 2026, the original `colpali-engine` repository marks that package **deprecated** for new projects and recommends Sentence Transformers v6 `MultiVectorEncoder`. A documented interface is:

```python
from sentence_transformers import MultiVectorEncoder
from PIL import Image

retriever = MultiVectorEncoder("vidore/colqwen2.5-v0.2")
queries = ["Which figure shows force decreasing after lubrication?"]
pages = [Image.open("page_001.png").convert("RGB"),
         Image.open("page_002.png").convert("RGB")]
q_vecs = retriever.encode_query(queries)
p_vecs = retriever.encode_document(pages)
scores = retriever.similarity(q_vecs, p_vecs)
print(scores)
```

Check the installed version and model compatibility before deployment; the model is far too large to assume it will fit on an 8GB edge device without a measured quantization/offloading strategy. For a first end-to-end prototype, use a remote GPU for indexing and a smaller retriever or stored embeddings for local search.

## 7. Computational complexity, memory, and latency

For one query-page score, dense MaxSim takes `O(Nq * Np * D)` arithmetic and an `Nq x Np` similarity matrix in a naive implementation. Query encoding costs one backbone pass; page encoding is performed offline. Full brute-force scoring over `M` pages costs `O(M * Nq * Np * D)` and is often unsuitable for large collections.

**Concrete memory example:** with `Np=1024`, `D=128`, FP16, each page consumes `1024*128*2 = 262,144 bytes = 256 KiB` for raw embeddings. At 100,000 pages, that is about **24.4 GiB** before metadata, ANN structures, caches, and any duplication. One million pages require about **244 GiB** just for the raw vectors. This is why multi-vector compression is not optional at enterprise scale.

For `Nq=16`, `Np=1024`, `D=128`, one exact score requires approximately `2.1 million` dot-product multiply-accumulate terms; doing that for 100,000 pages is roughly `210 billion` terms, ignoring index acceleration. These are arithmetic estimates, not measured latencies.

**Index-time versus query-time trade-off:** Rendering pages and generating embeddings may be costly but is amortized across queries. Query-time MaxSim is relatively cheap on a small shortlist, not necessarily across the entire corpus. A production system typically uses pooled or quantized candidate search followed by higher-fidelity MaxSim reranking.

**Compression choices:** FP16 to INT8 reduces raw embedding bytes by about 2x; pooling from 1024 vectors to 256 gives 4x; combining both gives about 8x *raw vector storage* reduction. These are mathematical ratios, not guaranteed total system memory savings or accuracy preservation. Quantization requires correct scale/zero-point handling, and pooling can remove tiny figure labels or important rare patches.

**Deployment checklist:** measure page-indexing time, query-encoding time, candidate retrieval, MaxSim rerank, page-image fetch, VLM prefill/decode, P95/P99 latency, peak VRAM, CPU RAM, index size, and retrieval recall. Distinguish throughput from interactive latency.

## 8. Modern scaling: PULSAR and MIDR (2026)

**PULSAR (28 August 2026)** reports a production vision-first retrieval system with a frozen ColPali-style backbone. It pools row/column vectors for fast first-stage candidate search, uses a finer hierarchical-pooled representation for MaxSim reranking, and stores higher-resolution page images for downstream answer generation. Its authors report **15.1x lower median vector-search latency** than their unpooled baseline with less than 0.01 absolute loss in nDCG@10 and Recall@10, and about **88x higher concurrent QPS** in the evaluated setup. Their deployed corpus contained approximately **2.4 million pages**. These are authors' measurements from a particular enterprise corpus and stack, not universal speedup factors.

PULSAR also separates rasterization resolution: **150 DPI for retrieval embeddings** and **500 DPI for answer-time page images**, selected by their ablations. It uses document/page hashes for incremental re-indexing. Its lesson is that retrieval and reading can use different visual fidelity.

**MIDR (1 September 2026)** provides a contrasting systems design. It asks a multimodal model to create *verified structured textual enrichment at indexing time*, then searches with BM25F and optional dense fusion. On its ViDoRe V3 evaluation, its authors report approximately **9x smaller index memory** and **2x lower query latency** than ColQwen2.5, with performance depending on domain. This is an important counterexample to the claim that vision-first late interaction is always best. When the corpus changes rarely and query volume is high, paying the multimodal reasoning cost once at ingestion may be attractive.

**Emerging research question:** Should multimodal understanding happen during indexing, query-time retrieval, or final answer generation? The best placement depends on corpus update frequency, query mix, model cost, retrieval fidelity, and the cost of incorrect answers.

## 9. Failure modes and debugging checks

| Failure | Symptom | Check |
|---|---|---|
| OCR-like failure persists | Correct page not retrieved when text is tiny | Render resolution, patch coverage, small-font stress test |
| False positives from shared vocabulary | Many pages with similar captions rank high | Hard negatives from same document and same chart type |
| Wrong table row/column | Relevant page retrieved but answer wrong | Separate retrieval relevance from answer grounding; crop table and verify |
| Query padding contamination | Score changes with batch padding | Mask padded query vectors and document vectors |
| Background-patch dominance | White margins show high similarity | Inspect per-token MaxSim heatmaps; don't assume visual attention equals evidence |
| Length bias | Long queries receive larger raw scores | Compare calibrated/rank-normalized variants while keeping evaluation fair |
| Duplicate/near-duplicate leakage | Validation looks unrealistically strong | Split by whole PDF, author/report series, and source |
| False negatives | Training loss fights genuinely relevant pages | Use multi-positive labels or mask known relevant negatives |
| Quantization ranking flips | FP16 and INT8 retrieve different top-k | Measure top-k overlap, nDCG, Recall@k, not only embedding MSE |
| Model preprocessing mismatch | Embeddings degrade after export | Golden-sample test: RGB, resize, normalization, token masks, outputs |
| Slow retrieval | Brute-force all pages dominates | Candidate prefetch, pooling, ANN, MaxSim only on shortlist |
| Hallucinated answers | Retrieved page is relevant but answer fabricated | Require page-level evidence, image inspection, abstention on missing facts |

A crucial subtlety: a MaxSim similarity map indicates *which embeddings scored highly*, not that the system has proven a causal or numeric statement. For high-stakes biomedical use, the downstream VLM must explicitly inspect the retrieved original image and cite page/figure locations; human verification remains necessary for clinical decisions.

## 10. Integrated end-to-end example: a surgical-robotics paper assistant

**Task:** “Find the experiment that compares elastomer thickness with tactile contact localization error, and tell me which thickness performs best.”

**Corpus:** 8,000 PDFs covering optical tactile sensors, robot manipulation, and endoscopic imaging. Assume 10 pages each, or about 80,000 pages. This is a hypothetical example, not a benchmark result.

**Ingestion:** Render each PDF page at a controlled retrieval resolution; record `document_id`, `page_number`, checksum, source, and page-image path. Run a ColPali-family retriever offline and store multivectors. At 1024 vectors/page, 128 dimensions, FP16, the raw index would require roughly **19.5 GiB** for 80,000 pages. Use pooling and a remote storage/index service if needed.

**Query encoding:** The text query becomes `Q in R^(1 x Nq x D)`. For illustration, `Nq=24`, `D=128`.

**Candidate retrieval:** Use pooled vectors or hybrid lexical search to select 100 candidate pages. For scientific diagrams, combine visual retrieval with text retrieval over captions, titles, and bibliography; this helps rare terminology and exact numeric identifiers.

**Reranking:** Compute full or higher-fidelity MaxSim over the 100 candidates. Inspect per-token heatmaps for “thickness”, “localization”, “error”, and “best” to understand the ranker. Return top 5 pages with document IDs and page numbers.

**Grounded answer:** Supply the five *original high-resolution page images* to a separate generative VLM. Prompt it to identify axes, units, legend colors, error bars, and experimental conditions; require citations such as `paper_0142.pdf, p. 7, Fig. 4`. If the figure is unreadable or the comparison is not actually present, respond with uncertainty rather than fabricate a thickness value.

**Verification:** For numeric claims, compare the generated answer against figure labels and, when available, the original data table. If the graph contains only curves without precise values, report approximate readings as estimates.

**Serving architecture:** `PDF ingest worker -> visual embedding GPU -> pooled ANN index + page store -> retrieval API -> reranker -> VLM server -> cited response`. Keep an MCU/robot-control pathway independent: this is research-document assistance, not direct autonomous surgery.

**Metrics:** retrieval Recall@1/5/20, nDCG@5, mean reciprocal rank, answer-fact recall, citation correctness, numeric error, query P95 latency, indexing pages/s, index size, and GPU peak memory. A page can be relevant without supporting the exact requested conclusion, so evaluate evidence sufficiency separately.

## 11. Connections to earlier VLM/VLA topics

**SigLIP and contrastive pretraining:** SigLIP helps create useful vision-language representations, but a pooled image-text similarity score is not enough for document page retrieval. ColPali adds contextualized multivectors and task-specific late-interaction training.

**Multimodal projectors:** A retrieval projection `W: d_h -> D` is not the same as the projector that maps visual tokens into an LLM's hidden dimension. One serves cross-modal fusion; the other produces compact vectors for retrieval scoring.

**Visual token compression and dynamic resolution:** They now have a second role: reducing *index storage* and MaxSim cost, not only generative VLM prefill cost. Aggressive compression can disproportionately damage rare small labels.

**KV-cache and SLO-aware serving:** Retrieval has its own latency/memory budget. Saving 100 ms in the VLM decoder may not help if visual retrieval takes 800 ms. Profile the entire request path.

**Grounding:** Retrieval ranks *pages*, while grounding identifies the *specific figure, table cell, or region* supporting a claim. Add a grounding/crop stage for fine-grained evidence.

**VLA bridge:** The same idea can retrieve historical demonstrations, failure cases, or manuals for a robot's supervisory VLM. But similarity is not a physical safety certificate and should not directly bypass local controllers.

## 12. Research protocol and self-test

**Ablation A:** OCR+BM25 vs OCR+dense embeddings vs vision-first MaxSim vs hybrid retrieval, using identical PDFs and query splits.

**Ablation B:** One pooled page vector vs 64/128/256/1024 page vectors. Plot Recall@5 and index bytes/page.

**Ablation C:** FP16 vs INT8 embeddings and pooled+INT8. Plot rank agreement, nDCG@5, and P95 latency.

**Ablation D:** Generic negatives vs same-document hard negatives. Track training instability and false-negative frequency.

**Ablation E:** Retrieval resolution vs answer-time resolution. Separate page-retrieval errors from VLM reading errors.

**Self-test 1:** Why can MaxSim retrieve a page containing all query terms but fail to establish that the terms have the requested spatial relationship?

**Self-test 2:** Calculate raw index size for 500,000 pages, each with 768 FP16 vectors of dimension 128. Answer: `500000*768*128*2 = 98,304,000,000 bytes`, about **91.6 GiB**.

**Self-test 3:** Why is `softplus(s_negative - s_positive)` not equivalent to full InfoNCE? Answer: it compares the positive with the *single hardest* negative, not a normalized sum over all negatives.

**Self-test 4:** Why might a 4x vector pooling improvement fail to deliver 4x end-to-end latency improvement? Answer: encoder, ANN, page fetch, and VLM generation can dominate; pooling also changes index overhead and recall.

**Self-test 5:** If a retriever returns the correct figure but the answer VLM confuses the two legend colors, which component failed? Answer: grounding/interpretation, not necessarily page retrieval.

## 13. Facts, heuristics, and interpretation

**Established facts:** ColPali uses VLM-derived multi-vector page embeddings and a MaxSim late-interaction score. The original paper uses a hardest-negative softplus training objective and reports 81.3 average nDCG@5 on its ViDoRe V1 evaluation, versus 67.0 for its strongest OCR/captioning baseline. The original paper and the newer ViDoRe versions are not directly interchangeable benchmarks.

**Reported 2026 research, not independently verified deployment guarantees:** PULSAR's large-scale pooling and two-resolution results; MIDR's index-time enrichment trade-offs. Treat speedups as relative to each paper's measured baselines.

**Engineering heuristics:** Start with an existing ColQwen/ColSmol-style model; use document-level splits; preserve high-resolution evidence for answer generation; combine visual and lexical retrieval when rare terminology matters; compress only after measuring recall and citation correctness.

**Interpretation:** The most important insight is *where* to spend multimodal computation. Vision-first indexing preserves visual information; late interaction lets query terms inspect that information; pooled two-stage search controls query-time cost; a separate generative VLM performs evidence-based reasoning. A serious VLM engineer must optimize all four without confusing retrieval relevance with answer truth.

## 14. Research links (accessed 15 October 2026)

- Original ColPali paper (ICLR 2025): https://arxiv.org/abs/2407.01449
- Original paper HTML, including exact loss: https://arxiv.org/html/2407.01449v6
- ColPali/ColQwen reference repository (now recommends Sentence Transformers v6 for new work): https://github.com/illuin-tech/colpali
- Sentence Transformers multi-vector documentation: https://www.sbert.net/docs/multi_vector_encoder/usage/usage.html
- ViDoRe Benchmark V2 (May 2025): https://arxiv.org/abs/2505.17166
- PULSAR (28 August 2026): https://arxiv.org/abs/2608.28572
- MIDR (1 September 2026): https://arxiv.org/abs/2609.01316
- VisDocAgentBench (18 August 2026; multi-hop retrieval limitations): https://arxiv.org/abs/2608.17889

**One-sentence takeaway:** In multimodal RAG, retrieval quality is determined not just by what your model understands, but by what information your indexing representation chooses to preserve—and how efficiently it can be searched.

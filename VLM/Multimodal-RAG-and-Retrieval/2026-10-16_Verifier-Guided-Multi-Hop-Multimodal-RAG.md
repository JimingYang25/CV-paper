# Daily VLM Deep Dive — Day 15
## Verifier-Guided Multi-Hop Multimodal RAG: Evidence Graphs, Adaptive Retrieval, and Grounded Synthesis
**October 16, 2026 | Advanced VLM Systems | Research-to-Production**

**Central idea:** Last lesson's ColPali/MaxSim can retrieve a visually relevant *page*. Today's problem is harder: an answer may require *several pages, figures, tables, and documents*, with a logical relationship that no single retrieved page establishes. We will build an iterative, evidence-aware retrieval system and learn exactly where the algorithms are established versus proposed engineering designs.

**Literature anchors:**
- ViDoRAG (EMNLP 2025): https://aclanthology.org/2025.emnlp-main.464/
- Doc-Researcher (October 2025 preprint): https://arxiv.org/abs/2510.21603
- VLD-RAG (July 2026 preprint): https://arxiv.org/abs/2607.24748
- VisDoM (NAACL 2025): https://manansuri.com/papers/visdom/

## 1. Motivation: the missing-evidence problem

Consider a scientific question: “Which tactile elastomer thickness gives the lowest localization error at 2 N, and does a second paper independently establish that it is optimal?” The first answer may be encoded only in a plotted curve on page 7 of paper A; paper B may contain a single result for that thickness but no thickness comparison. A one-shot top-5 page retriever can find plausible pages and still fail the reasoning task: the evidence chain is incomplete, or the model confuses *agreement on a measurement* with *independent confirmation of an optimum*.

The distinction is fundamental:

- **Page relevance:** a page concerns the subject.
- **Claim support:** a specific page region entails a specific statement.
- **Evidence completeness:** every required subclaim has supporting evidence.
- **Cross-page validity:** comparisons use compatible units, conditions, and metrics.

A high ColPali MaxSim score establishes none of the last three by itself. This is why we need retrieval, decomposition, evidence tracking, and verification as separate modules.

## 2. Core intuition: retrieval is a sequential decision process

Instead of retrieving once and asking a VLM to improvise, decompose the question into atomic claims; search for each; preserve source provenance; inspect the evidence; and issue another search only for what is missing. Stop when the evidence is sufficient or the cost budget is exhausted.

```
Question
  -> Claim / dependency decomposition
  -> Hybrid text + visual candidate retrieval
  -> Region/page reranking
  -> Evidence ledger + relation graph
  -> Verifier: supported? contradicted? missing?
      -> missing: targeted query reformulation -> retrieve again
      -> complete: high-resolution VLM reading -> cited answer
      -> budget exhausted: partial answer + explicit uncertainty
```

**Established research:** ViDoRAG (EMNLP 2025) uses hybrid retrieval and iterative exploration, summarization, and reflection. Doc-Researcher (2025) studies multi-granular parsing, planning, and iterative research across documents. VLD-RAG (2026 preprint) proposes dual visual/text indexes and verifier-guided query refinement. **Our evidence-graph and coverage-based stopping formulation below is an engineering synthesis, not an equation copied from those papers.**

## 3. Mathematical problem statement

Let a corpus contain M documents, with each document holding pages and regions. Denote an evidence unit by

\[e_j=(d_j,p_j,b_j,m_j,x_j),\]

where d_j is the document ID, p_j the page number, b_j a bounding box or region identifier, m_j the modality (text/table/figure/image), and x_j the content or a reference to it. The original question is q. A planner produces m atomic requirements:

\[C(q)=\{c_1,c_2,\ldots,c_m\}.\]

For example: c1 = “paper A reports localization error at 2 N for each thickness”; c2 = “the minimum is at thickness h*”; c3 = “paper B tests h* at 2 N”; c4 = “paper B compares h* with alternatives.” The fourth is essential if the user asks whether B establishes *optimality*.

Let E_t be the accumulated evidence after retrieval round t. Let a_{ij} in [0,1] denote a verifier's estimated support score for claim c_i from evidence e_j. This is a learned score, **not a calibrated probability unless separately validated**.

A soft coverage score is

\[F(E)=\frac{\sum_{i=1}^m w_i\max_{e_j\in E} a_{ij}}{\sum_{i=1}^m w_i},\]

where w_i >= 0 reflects the importance of requirement i and max over an empty set is zero. For strict, human-auditable completion, use a thresholded support matrix and require every necessary claim to have a verified citation:

\[\operatorname{Complete}(E)=\bigwedge_{i=1}^m\left[\exists e_j\in E: a_{ij}\ge \tau_i \ \land\ \operatorname{ValidSource}(e_j)\right].\]

This formula is a *proposed engineering rule*. A source validity check can require a real document/page, a valid bounding box, readable values, and consistent units. A separate contradiction check is needed: evidence can support an isolated claim yet contradict another claim.

### 3.1 Why selecting a good evidence set is a coverage problem

Suppose the VLM has room for at most K evidence units. We want

\[E^*=\arg\max_{E\subseteq\mathcal E,\ |E|\le K}F(E).\]

With fixed nonnegative a_{ij}, F(E) is monotone submodular: adding another page gives diminishing returns once its claims are already covered. Under a simple cardinality constraint, the standard greedy algorithm that repeatedly selects the evidence unit with greatest marginal coverage achieves the classical (1 - 1/e) approximation to the optimal F. This mathematical statement applies to this *coverage objective*, not automatically to arbitrary graph-reasoning scores or the real VLM's answer accuracy.

A cost-aware heuristic instead selects

\[e^*=\arg\max_{e\notin E}\frac{F(E\cup\{e\})-F(E)}{\operatorname{cost}(e)},\]

where cost(e) can represent estimated visual tokens, VLM reading time, or retrieval latency. The ratio rule is useful but does not inherit the simple cardinality guarantee under arbitrary costs and extra constraints.

### 3.2 The evidence graph

Construct G=(V,R), where V contains claim nodes and evidence nodes. A typed edge can be `supports`, `contradicts`, `compares-with`, or `depends-on`. A dependency edge is not proof: the verifier must still inspect both endpoints. In a scientific comparison, store normalized units, test force, metric definition, sample size, and measurement uncertainty as attributes. This prevents an attractive but invalid cross-paper comparison.

## 4. Hybrid retrieval and query reformulation

Text retrieval is valuable for rare identifiers, author names, equations, exact units, and part numbers. Visual retrieval is valuable for graphs, layout, and figures that OCR cannot fully describe. Use both, but **do not add unnormalized BM25 and cosine scores directly**.

A robust rank-based baseline is reciprocal rank fusion (RRF):

\[RRF(e)=\sum_{r\in\{\mathrm{text},\mathrm{visual}\}}\frac{1}{k_0+\operatorname{rank}_r(e)},\]

where k_0>0 dampens rank differences; a missing candidate contributes zero. RRF avoids assuming raw modality scores share a scale. A visual branch might use ColPali MaxSim to rerank a small candidate set:

\[S(q,p)=\sum_{i=1}^{N_q}\max_{j\le N_p}Q_i^\top P_j,\]

with Q in R^{N_q x D} and P in R^{N_p x D}. Query vectors and page vectors must use compatible embedding dimensions D. The multimodal query can be a subquestion rather than the entire original question.

At iteration t, define unresolved requirements

\[U_t=\{c_i: c_i\text{ lacks valid evidence in }E_t\}.\]

Then choose a new query q_{t+1}=Rewrite(q,U_t,E_t). Example: instead of repeating “best tactile sensor,” search “Paper B Table 2 2 N 2 mm localization error; comparison with 1 mm and 3 mm.” A verifier should generate a *missing-evidence specification*, not simply say “search again.”

**Important:** If no relevant page is found, that is not proof that the claim is false. It is an evidence-retrieval failure or an unknown, depending on the corpus and search coverage.

## 5. Training objectives and optimization

This architecture can be deployed using frozen retrievers and a frozen VLM. No end-to-end training is required to demonstrate the pipeline. For domain adaptation, however, three trainable modules are useful.

### 5.1 Claim-conditioned retriever

Create labeled pairs (c_i,e_j,y_{ij}) where y_{ij}=1 if evidence unit e_j supports requirement c_i and 0 if it is an explicitly judged negative. Train a reranker with binary cross entropy:

\[\mathcal L_{\rm ret}=-\sum_{i,j}\left[y_{ij}\log\sigma(s_{ij})+(1-y_{ij})\log(1-\sigma(s_{ij}))\right].\]

Here s_{ij} is the learned relevance logit and sigma is the logistic sigmoid. Use hard negatives: same sensor but wrong force, same graph style but wrong metric, or matching caption but wrong table. **Do not label unannotated relevant pages as negatives by default.**

### 5.2 Evidence verifier

For a claim and an image crop plus text context, train a classifier over `supports`, `contradicts`, and `insufficient`. A three-way cross-entropy is

\[\mathcal L_{\rm verify}=-\sum_n\log p_\psi(z_n\mid c_n,e_n),\]

where z_n is the human-annotated evidence relation and psi are verifier parameters. Calibrate the threshold tau on a held-out set, and evaluate by domain, evidence modality, and claim type. An LLM-based verifier without supervised calibration should not be presented as a probabilistic guarantee.

### 5.3 Planner and stopping policy

A planner can initially be rule-based or prompted. Later it can be supervised using annotated evidence chains: input (question, evidence ledger) -> next missing claim/query. A possible research objective is minimizing expected cost plus missed-evidence penalty:

\[\mathcal J(\pi)=\mathbb E_\pi\left[\lambda_c C_{\rm compute}+\lambda_m N_{\rm missing}+\lambda_u N_{\rm unsupported}\right].\]

This is a *proposed objective*, not the original ViDoRAG or VLD-RAG loss. Training a planner through reinforcement learning is optional and much harder than a frozen-model baseline.

## 6. Implementation: a minimal iterative evidence engine

The following PyTorch-style Python skeleton is deliberately model-agnostic. The `search_*` and `verify_*` functions are interfaces to real services; it does not pretend to provide pretrained weights.

```python
from dataclasses import dataclass
from collections import defaultdict

@dataclass(frozen=True)
class Evidence:
    doc_id: str
    page: int
    region: str
    modality: str
    source_uri: str


def rrf_fuse(text_hits, visual_hits, k0=60):
    # Each hit list contains Evidence in descending relevance order.
    scores = defaultdict(float)
    for hits in (text_hits, visual_hits):
        for rank, ev in enumerate(hits, start=1):
            scores[ev] += 1.0 / (k0 + rank)
    return sorted(scores, key=scores.get, reverse=True)


def greedy_pack(claims, candidates, support, cost, budget):
    """support[(claim, evidence)] in [0,1]; cost[evidence] > 0."""
    selected, spent = [], 0.0
    covered = {c: 0.0 for c in claims}
    remaining = list(candidates)
    while remaining:
        best, best_ratio = None, 0.0
        for e in remaining:
            if spent + cost[e] > budget:
                continue
            gain = sum(
                max(0.0, support.get((c, e), 0.0) - covered[c])
                for c in claims
            )
            ratio = gain / cost[e]
            if ratio > best_ratio:
                best, best_ratio = e, ratio
        if best is None:
            break
        selected.append(best)
        spent += cost[best]
        remaining.remove(best)
        for c in claims:
            covered[c] = max(covered[c], support.get((c, best), 0.0))
    return selected, covered


def research(question, max_rounds=3, top_k=12):
    claims = decompose_into_atomic_requirements(question)
    ledger = {}  # (claim, evidence) -> verified relation + provenance
    seen = set()
    for round_idx in range(max_rounds):
        missing = [c for c in claims if not has_valid_support(c, ledger)]
        if not missing and not has_unresolved_contradiction(ledger):
            break
        subqueries = formulate_queries(question, missing, ledger)
        for subquery in subqueries:
            candidates = rrf_fuse(
                search_text(subquery, top_k),
                search_visual(subquery, top_k),
            )
            for evidence in candidates[:top_k]:
                if (subquery, evidence) in seen:
                    continue
                seen.add((subquery, evidence))
                # Inspect full-res crop, text, metadata, and units.
                for claim in missing:
                    ledger[(claim, evidence)] = verify_claim(claim, evidence)
    selected = select_evidence_under_token_budget(claims, ledger)
    answer = generate_with_citations(question, selected)
    result = validate_answer_claims(answer, selected)
    if not result.supported:
        return make_partial_answer_with_unknowns(claims, ledger, result)
    return answer
```

`has_valid_support` must not equate a high retrieval score with entailment. `validate_answer_claims` must check that cited page/region identifiers exist and that numbers, units, and comparisons are traceable. The system must never manufacture a missing citation to satisfy an output schema.

## 7. Complexity, memory, latency, and deployment

Let M be the number of indexed pages, R the number of retrieval rounds, K the candidate pages reranked per round, N_q query tokens, N_p page vectors, and D the embedding dimension. A brute-force late-interaction rerank costs approximately

\[O(R K N_q N_p D).\]

An ANN or sparse index can reduce candidate-generation work, but it does not eliminate the cost of verifying high-resolution page images. If VLM reading is invoked on H evidence units per round, the total online latency is approximately

\[T\approx\sum_{t=1}^R(T_{\rm plan,t}+T_{\rm search,t}+T_{\rm rerank,t}+T_{\rm verify,t})+T_{\rm answer}.\]

An approximate visual-token budget is

\[B_{\rm vision}=\sum_{e\in E}N_v(e),\]

where N_v(e) depends on crop size, dynamic resolution, and the chosen VLM. Reading five full pages can be much more expensive than reading one page and four localized crops.

**Memory example:** 80,000 pages x 1,024 vectors/page x 128 dimensions x 2 bytes (FP16) is 20,971,520,000 bytes, approximately 19.5 GiB before metadata and indexing overhead. This makes a full multi-vector corpus unsuitable for keeping entirely in an 8GB Jetson's GPU memory. Store indexes on a host/server or compressed CPU-backed storage; use the Jetson for selective inference or robotics orchestration.

**Engineering heuristics:** (a) start with R=2-3 rounds and hard caps on search calls; (b) reuse parsed text, page embeddings, and crops; (c) first retrieve at document/page level, then zoom to regions; (d) prioritize a new page only if it covers a missing requirement; (e) send the answer VLM a small set of high-resolution crops plus exact citations rather than dozens of full pages. These are tuning starting points, not universally optimal values.

## 8. Common failure modes and debugging checks

| Failure | Why it happens | Diagnostic |
|---|---|---|
| Retrieval succeeds but answer fails | Relevant pages lack the required relation | Compare page recall vs full evidence-chain recall |
| “Confirmation” from a single measurement | A replication value is mistaken for an optimum comparison | Encode the missing comparison as a separate claim |
| Same fact counted twice | Duplicate pages or derivative citations | Deduplicate by source lineage and experiment ID |
| False contradiction | Units, force, or error definitions differ | Normalize metadata before comparing |
| Wrong chart values | Small legend, color confusion, or cropping | Use high-res region crops and numerical extraction checks |
| Infinite retrieval loop | Query rewriting retrieves same pages | Track visited queries/evidence and marginal coverage gain |
| Overconfident verifier | Entailment score is uncalibrated | Held-out reliability curves and abstention evaluation |
| Evidence laundering | Generated captions are treated as original measurements | Preserve original image and extraction provenance |
| Poor latency | Too many large VLM verification calls | Batch/rerank cheaply; verify only promising missing claims |
| Citation hallucination | Generator invents page or region IDs | Validate IDs against immutable evidence ledger |
| Domain leakage | Pages from the same paper cross train/test | Split by document, experiment, or research group |

## 9. Integrated end-to-end example: visual-tactile research assistant

**All numbers in this example are synthetic.** A robotics researcher asks: “At 2 N, which elastomer thickness gives the lowest localization error, and does Paper B independently verify it is optimal?”

The planner creates four requirements: locate the experiment in A; extract all thickness-error pairs at 2 N; locate B's comparable measurement; determine whether B tests alternative thicknesses under matching conditions.

Round 1 uses hybrid retrieval. Text search finds A's methods paragraph; visual MaxSim retrieves A page 7, Figure 4. The VLM reads the plotted 2 N curve and extracts: 1 mm -> 2.1 mm error, 2 mm -> 1.3 mm, 3 mm -> 1.9 mm. The smallest reported error is 1.3 mm at 2 mm thickness. This supports A's within-experiment optimum. A separate search finds B page 4, Table 2: 2 mm -> 1.5 mm at 2 N, using a nominally similar localization metric.

The verifier checks the fourth requirement: does B compare 1, 2, and 3 mm? No. Round 2 searches B's experimental design, appendix, and thickness-ablation terms. No additional thickness comparison is found within the available corpus. The correct answer is: “Paper A reports the lowest error for 2 mm at 2 N. Paper B reports a 2 mm result but does not independently establish that 2 mm is optimal, because its available experiment does not compare thickness alternatives.” Cite A Figure 4 and B Table 2. Do **not** say B disproves the optimum; missing evidence is not a negative result.

For a deployment evaluation, record: evidence-chain Recall@K, exact citation accuracy, numerical extraction error, unsupported-claim rate, abstention accuracy, rounds per question, P95 latency, GPU peak memory, and cost per successfully grounded answer. Ablate (1) visual-only vs text-only vs hybrid retrieval, (2) single-shot vs iterative retrieval, (3) verifier on/off, (4) full-page vs region-crop reading, and (5) fixed K vs coverage-based evidence selection.

## 10. Published evidence and critical reading

- **ViDoRAG, EMNLP 2025:** introduces the ViDoSeek benchmark, GMM-based hybrid retrieval, and iterative exploration/summarization/reflection; authors report over 10% improvement against evaluated baselines. This is not a claim that its precise architecture equals our evidence graph.
- **Doc-Researcher, October 2025 preprint:** introduces M4DocBench (158 expert-annotated questions across 304 documents) and reports 50.6% accuracy in its strongest evaluated configuration. Its ablations report a 6-8 percentage-point drop without planning in several model configurations, and retrieval improvements that diminish after a few iterations. These are authors' benchmark results, not universal production numbers.
- **VLD-RAG, July 2026 preprint:** describes dual visual/text indexing, hybrid retrieval, and verifier-driven refinement on LongDocURL and MMLongBench-Doc. Its experiments are *closed-document* (retrieval restricted to the question's document), so do not treat them as evidence of unrestricted multi-document search performance. The paper's reported ranking tables should be read metric by metric rather than assuming every metric is best.

## 11. Connections to the broader VLM/VLA curriculum

**ColPali/MaxSim:** provides page-level visual similarity; today's system turns page retrieval into complete evidence-chain discovery. **Multimodal projectors and dynamic resolution:** determine how much visual detail survives the reading stage. **Visual token compression:** trades evidence fidelity for context/latency; tiny axis labels can be disproportionately important. **KV caching and SLO-aware serving:** reduce repeated prompt and image-processing overhead when the agent conducts multiple retrieval rounds. **Speculative decoding:** can accelerate answer generation but cannot compensate for missing evidence. **Conformal failure detection:** is a runtime risk monitor, while our verifier judges documentary claim support; neither is a physical safety certificate. **VLA bridges:** the same retrieval machinery can fetch relevant manipulation demonstrations, calibration procedures, or failure logs, but retrieved similarity must not directly authorize physical actions.

## 12. Research exercises and self-test

1. Prove that F(E)=sum_i w_i max_{e in E} a_i(e) is monotone submodular for nonnegative a and w. State why adding arbitrary pairwise graph rewards can break the guarantee.
2. Implement RRF and compare it with raw BM25-plus-cosine fusion. Explain why score scales are incomparable without calibration.
3. Construct 100 multi-hop visual-tactile questions requiring at least two pages, with annotated evidence chains and deliberately misleading near-match pages.
4. Compare answer accuracy and **complete evidence-chain recall** for one, two, three, and five retrieval rounds; plot latency against accuracy.
5. Design an adversarial case where two papers use the same unit (mm) but different localization-error definitions. Ensure your verifier abstains from an invalid comparison.

**Answer sketches:** (1) the maximum of fixed nonnegative per-item support scores has diminishing marginal gain as the selected set grows; pairwise synergy can instead create increasing returns. (2) RRF fuses ranks, not raw scores. (3) split by whole papers and underlying experiments. (4) stop when new evidence adds little coverage; benchmark the threshold. (5) preserve metric definitions and conditions as explicit graph attributes.

## 13. Established facts, engineering heuristics, interpretation

**Established facts:** Single-page retrieval is insufficient for many multi-page/multi-document reasoning tasks; hybrid visual/text retrieval and iterative search have published evidence in ViDoRAG, Doc-Researcher, and VLD-RAG. The mathematical greedy guarantee applies to the specifically defined monotone submodular coverage objective.

**Engineering heuristics:** Use 2-3 retrieval rounds initially, hard token/call budgets, RRF as a simple fusion baseline, and immutable document/page/region citations. Verify numbers and relationships using original high-resolution evidence.

**My interpretation:** The next major frontier in multimodal RAG is not simply increasing retriever recall. It is making the system *aware of what it has not yet established*. An engineer should optimize evidence-chain completeness per unit of compute, not merely top-k page similarity.

**One-line takeaway:** A strong multimodal research agent does not stop when it finds relevant pages; it stops when it can justify every important claim—or explicitly explain which claim remains unproven.

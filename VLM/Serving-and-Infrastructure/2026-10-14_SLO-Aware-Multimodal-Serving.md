# Daily VLM Deep Dive — Day 13
## SLO-Aware Multimodal Serving: Chunked Prefill, Continuous Batching, and Encoder-Aware Scheduling
**October 14, 2026 | Advanced VLM inference systems**

**Central question:** How can one GPU serve a long document-VLM request, several short camera queries, and an ongoing streamed response without making the ongoing response freeze?

Yesterday's topic, exact multimodal speculative decoding, accelerated *serial output generation*. Today's topic optimizes a different layer: *which requests and tokens get GPU time, when, and with what cache state*. A perfect decoding kernel can still deliver terrible user experience if the scheduler allows a 4,000-token visual prefill to block a live stream.

**Evidence base:** Sarathi-Serve (2024), DistServe (2024), and the vLLM V1 scheduler, multimodal encoder-cache, prefix-cache, and production-metrics documentation. The concrete numbers below are illustrative unless explicitly attributed to a paper.

## 1. Motivation: two different kinds of GPU work

A decoder-only VLM typically runs (A) vision encoding, (B) LLM **prefill** of image embeddings plus text tokens, then (C) autoregressive **decode**. A request with 1,024 visual tokens and 256 text tokens has an effective LLM prompt length of about 1,280 tokens, subject to the architecture's tokenization and packing rules.

Prefill processes many tokens in parallel and is often compute intensive. Decode usually adds one token per active request per iteration; at low batch sizes, repeated weight movement and KV-cache reads can dominate. A naive scheduler may execute a long prefill in one iteration and delay all existing decodes. This raises **inter-token latency (ITL)** for the existing streams. Alternatively, always prioritizing decode can starve new long-prefill requests, increasing **time to first token (TTFT)**.

Sarathi-Serve's key idea is to split prefill into chunks and combine bounded amounts of prefill work with ongoing decode work. Its published paper reports up to 2.6× serving-capacity improvement for Mistral-7B on a single A100 and up to 3.7× for Yi-34B on two A100s versus the evaluated vLLM baseline; these are not VLM-specific Jetson measurements.

## 2. Core intuition: schedule a budget, not an entire prompt

Suppose a GPU iteration can process at most **C = 512 scheduled tokens**. Three existing conversations each need one decode token. A new request has 1,500 unprocessed prompt tokens.

**Unchunked:** the new prefill can monopolize a large GPU iteration and create a long gap between streamed tokens.

**Chunked:** first reserve 3 token slots for the existing decodes, then allocate up to 509 slots to the new prefill. Next iteration, repeat. After approximately three such chunks, the new request finishes prefill and begins decoding. The example is simplified: real schedulers also enforce memory, multimodal, concurrency, and fairness constraints.

The conceptual policy is **decode-first within a bounded per-iteration token budget**, with a starvation-prevention mechanism for pending prefills.

## 3. Mathematical formulation and tensor flow

For request i, let:

- I_i: image or frame set; x_i: text tokens; q_i: instruction.
- E_v(I_i) ∈ R^{B_i × N_{v,i} × d_v}: visual encoder output.
- P(E_v(I_i)) ∈ R^{B_i × N_{v,i} × d}: projected visual embeddings.
- E_t(x_i) ∈ R^{B_i × N_{t,i} × d}: language embeddings.
- Z_i = pack(P(E_v(I_i)), E_t(x_i)) ∈ R^{B_i × S_i × d}: packed LLM input, where S_i depends on the model's image-token expansion and layout.
- p_i: number of prompt tokens already processed; r_i = S_i - p_i: remaining prefill tokens.
- g_i: number of tokens already generated; K_i: resident KV-cache blocks.
- c_i ∈ {0,1,...,r_i}: number of new prefill tokens scheduled this iteration.
- d_i ∈ {0,1}: whether an active request gets its next decode token this iteration.

With per-iteration token budget C:

    Σ_i (c_i + d_i) ≤ C.

With memory budget M_KV:

    Σ_i memory(K_i plus newly allocated blocks) ≤ M_KV.

And for each request, prefill and decode are mutually exclusive during a given transition: d_i = 0 while its prompt is unfinished. In a production scheduler, other limits include maximum concurrent sequences, encoder-token budget, model-specific atomic multimodal spans, and admission control.

**Latency metrics:** for request i arriving at a_i, first output at f_i, and final output at e_i:

    TTFT_i = f_i - a_i
    E2E_i  = e_i - a_i
    TPOT_i = (e_i - f_i)/(L_i - 1), for L_i > 1 output tokens.

ITL is the time gap between successive streamed output events. It need not equal TPOT, especially when an event contains multiple tokens (for example under speculative decoding).

**Goodput** at arrival rate λ is the rate of completed requests meeting a specified service-level objective (SLO), e.g. both TTFT_i ≤ τ_first and TPOT_i ≤ τ_token. A useful optimization target is:

    maximize  λ × Pr(TTFT ≤ τ_first AND TPOT ≤ τ_token)

subject to GPU-memory, power, fairness, and model-correctness constraints. This is a serving optimization objective, **not a neural-network training loss**.

## 4. Architecture and end-to-end data flow

    Camera / Document / Text requests
                  |
             CPU preprocessing
                  |
        Multimodal encoder work queue
                  |
        Image encoder + encoder cache
                  |
         Packed image/text embeddings
                  |
       Admission and KV-block allocation
                  |
        Token-budget scheduler (C)
              /            \
       decode tokens    prefill chunks
              \            /
          One batched target-model step
                  |
          Update KV blocks and cursors
                  |
          Stream completed output tokens
                  |
       Metrics, backpressure, eviction

**Important distinction:** an *encoder cache* stores image-encoder outputs for repeated multimodal items. A *prefix KV cache* stores LLM attention keys/values for identical processed prefixes. These are different caches with different keys, eviction conditions, and memory costs.

For images, matching placeholder tokens alone is insufficient for prefix reuse: the cache identity must incorporate the actual image content or a trustworthy image fingerprint, along with model, preprocessing, and other relevant context. vLLM's prefix-caching design explicitly discusses multimodal hashes.

## 5. Multimodal-specific complication: atomic visual spans

Suppose the packed sequence is:

    [TEXT × 120] [IMAGE EMBEDDINGS × 576] [TEXT × 20].

A generic chunk boundary at token 256 would cut into the image span. Whether this is supported depends on the model and runtime. Some runtimes schedule multimodal input items as indivisible units. The vLLM V1 configuration includes `disable_chunked_mm_input`, which, when enabled, prevents partially scheduling a multimodal item; its documentation illustrates scheduling text first and the complete image span next.

**Correctness requirement:** the same image embeddings, position IDs, attention mask, and prefix state must be produced whether prefill is whole or chunked. Chunking should change the execution schedule, not the model's intended conditioning. Do not assume every VLM or attention implementation supports arbitrary chunk boundaries.

A practical scheduler may reserve the entire next image span, delay it until sufficient token budget is available, or use a model-supported subspan strategy. This creates a real trade-off between small iteration budgets and the ability to admit large image spans.

## 6. Scheduling objective, optimization, and training details

There is **no new neural-network training objective** in ordinary chunked prefill. The model weights θ stay fixed. The optimizer is a runtime scheduler selecting c_i, d_i, and cache allocations. A more detailed formulation is:

    minimize  α·P99(ITL) + β·P95(TTFT) + γ·E[queue_delay]

subject to:

    Σ scheduled_tokens ≤ C,
    KV_used ≤ KV_capacity,
    multimodal_atomicity = satisfied,
    priority / fairness constraints,
    no future-token leakage.

This is a conceptual multi-objective formulation; actual implementations often use rules or heuristics rather than solving it globally. Decode-first scheduling improves streaming smoothness but can starve long requests if no capacity remains. Add age-based priority, prefill quotas, or bounded waiting. Do not claim that reducing C always improves throughput: smaller chunks increase scheduling and kernel overhead and can hurt TTFT.

**If learning a scheduler:** a policy network could optimize a reward such as `goodput - latency_penalty - memory_penalty`, but this is a research extension, not necessary to deploy chunked prefill. Training it requires representative traces and careful protection against pathological starvation.

## 7. Implementation: scheduler pseudocode

```python
from dataclasses import dataclass

@dataclass
class Request:
    request_id: str
    remaining_prefill: int
    decode_ready: bool
    next_atomic_span: int = 1  # model-dependent, e.g. image item
    age: int = 0


def schedule_iteration(active, waiting, token_budget: int):
    """Illustrative policy, not vLLM source code."""
    selected = []
    budget = token_budget

    # Keep existing streams responsive.
    for req in active:
        if req.decode_ready and budget >= 1:
            selected.append((req.request_id, "decode", 1))
            budget -= 1

    # Fairness: order pending requests by age, not just arrival order.
    pending = sorted(
        [r for r in active + waiting if r.remaining_prefill > 0],
        key=lambda r: -r.age,
    )
    for req in pending:
        if budget <= 0:
            break
        # Do not split a model-specific atomic multimodal span.
        if req.next_atomic_span > budget:
            continue
        n = min(req.remaining_prefill, budget)
        # Real implementation must choose n at a valid span boundary.
        n = valid_chunk_boundary(req, n)  # model-specific helper
        if n <= 0:
            continue
        selected.append((req.request_id, "prefill", n))
        budget -= n

    return selected
```

`valid_chunk_boundary` is deliberately unspecified because its correct behavior depends on the model's multimodal packing contract. A real implementation must also track KV-block reservations, cache hits, admission, preemption, asynchronous encoder readiness, and token-level output bookkeeping. The pseudocode is a scheduling sketch, not a runnable VLM server.

**Practical runtime starting point:** for a supported vLLM V1 model, examine `--max-num-batched-tokens`, `--max-num-seqs`, and the documented chunked-prefill settings. A CLI illustration, to be validated against the installed version and model:

```bash
vllm serve YOUR_SUPPORTED_VLM_MODEL \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 8
```

In vLLM V1, chunked prefill is normally enabled where supported. Do not copy generic high-end GPU settings onto an 8GB Jetson without measuring memory and compatibility.

## 8. Complexity, memory, latency, and deployment

For a dense Transformer layer with total processed context S and hidden width d, full causal prefill has roughly O(S·d² + S²·d) arithmetic; constants and attention implementation matter. With KV caching, decoding one token has attention work scaling approximately O(S·d) plus weight-projection/MLP work O(d²) per layer. Chunking does **not** eliminate the need to attend to prior context; with correct KV reuse it partitions work and changes scheduling, not the underlying model distribution.

For L layers, H_kv KV heads, head dimension d_h, b bytes per element, and S cached positions:

    M_KV ≈ 2 · L · S · H_kv · d_h · b.

Illustrative example: L=24, S=2048, H_kv=8, d_h=128, BF16 b=2 gives 201,326,592 bytes ≈ 192 MiB **per request**. Eight such requests could require ≈1.5 GiB of KV storage, before model weights, image encoder, runtime workspace, and fragmentation. Real paged allocators reserve blocks and can share identical prefixes; observed memory differs.

For an 8GB edge GPU, the crucial trade-off is concurrency versus resident KV. Quantized model weights do not imply quantized KV. Reduce `max_num_seqs`, context limits, visual-token expansion, or KV precision when supported; measure P95/P99 latency rather than only average tokens/s.

**Multi-GPU extension:** DistServe separates prefill and decode onto different GPU pools and transfers KV state. It reported up to 7.4× more request capacity or 12.6× tighter SLOs in its tested LLM settings, while keeping >90% of requests within latency constraints. Such disaggregation is attractive in a cluster but usually not the first step for a single 8GB device, where transfer and operational overhead may dominate.

## 9. Failure modes and debugging checks

1. **Decode stalls:** P99 ITL spikes when a large image request arrives. Correlate per-iteration prefill tokens with ITL and inspect token budget.
2. **Prefill starvation:** streaming clients consume all scheduling slots. Measure maximum waiting age and enforce a fairness bound.
3. **Visual span fragmentation:** image embeddings are scheduled with invalid boundaries. Assert exact position IDs, attention masks, encoder-item boundaries, and output parity against unchunked prefill.
4. **KV exhaustion / preemption thrash:** too many concurrent long prompts. Track KV occupancy, preemptions, queue delay, and cache-eviction churn.
5. **False prefix-cache hits:** image placeholder tokens match but image content differs. Include robust multimodal identity in cache keys; never reuse cross-user sensitive context incorrectly.
6. **Encoder cache confusion:** LLM prefix cache hits do not guarantee vision encoder cache hits, or vice versa. Measure separately.
7. **Misleading aggregate throughput:** tokens/s improves while interactive requests violate TTFT or ITL targets. Report goodput under explicit SLOs.
8. **Incorrect benchmarking:** compare runs with different image resolution, generated length, concurrency, or warm-cache state. Fix seeds where feasible and publish workload distributions.
9. **Network masquerading as GPU latency:** distinguish capture, JPEG/transport, queue, encoder, prefill, decode, and return time.
10. **Version-specific flag mismatch:** consult the installed runtime's `--help`; scheduler behavior and multimodal support evolve.

## 10. Integrated system example: OpenFox remote vision server

Consider OpenFox with a local MCU/MPU, camera, and remote Jetson-class VLM server. Three workloads arrive concurrently:

- **A:** 15 Hz short camera-status requests, each asking for a short JSON answer.
- **B:** one operator request describing a detailed scene, generating 150 output tokens.
- **C:** a long visual instruction with multiple images and substantial prefill.

The MCU must remain autonomous for safety-critical motor control. The VLM is supervisory, not the real-time torque controller.

**Pipeline:** camera capture → timestamp → optional compression → network → VLM input preprocessing → visual encoder → multimodal encoder cache → LLM prefix/KV cache → SLO-aware chunked scheduler → streamed answer → schema validation → ROS2 message → local freshness check → safe supervisory action.

**SLO proposal (engineering hypothesis, not a paper result):** for A, target P95 end-to-end ≤ 500 ms; for B, target P95 TTFT ≤ 1 s and P95 ITL ≤ 80 ms. Set C as best-effort with a maximum waiting-age limit. These targets may be unattainable on an 8GB Jetson for a given model; measure before committing to them.

**Experiment:** collect 10-minute timestamped traces under idle, A-only, A+B, and A+B+C loads. Compare (1) no chunking if supported, (2) chunked prefill C=512, (3) C=1024, (4) C=2048, with consistent cache conditions. Measure P50/P95/P99 TTFT, ITL, E2E, per-class goodput, peak VRAM, preemptions, vision-encoder time, prefix-cache hit rate, and energy/request. Run correctness tests on identical prompts and images; scheduling should not change intended model outputs except for numerical nondeterminism and sampling variability.

For a production deployment, separate a short-deadline status queue from long-form analysis if fairness and memory limits cannot meet both SLOs on one model instance. Use a deterministic local fallback when responses are stale.

## 11. Connections to related VLM/VLA methods

**Speculative decoding (yesterday):** reduces serial target-model decode steps. **Chunked prefill (today):** changes scheduling of prompt work relative to decode work. The two are complementary, but speculative verification may output multiple tokens per event, complicating ITL interpretation.

**Visual token pruning / dynamic resolution:** reduces N_v and hence prompt compute and KV residency. Chunked prefill does not reduce the token count; it schedules it more smoothly. First ask whether tokens can be removed without losing OCR or grounding accuracy, then schedule the remaining tokens efficiently.

**Prefix caching:** reuses attention state for exact shared prefixes; **multimodal encoder caching:** reuses image embeddings. Both can reduce repeated computation but require careful cache identity, eviction, and tenant isolation.

**Disaggregated serving:** DistServe places prefill and decode on separate workers, whereas chunked prefill time-shares one pool. The former needs KV transfer; the latter needs carefully controlled iteration budgets.

**VLA control:** streaming language output is not a hard real-time actuator command. Model serving should feed a supervisory planner, with independent low-level control and a deadline-aware execution monitor.

## 12. Established facts, heuristics, interpretation, and exercises

**Established facts:** Sarathi-Serve proposed chunked prefills and stall-reducing schedules; vLLM V1 documents decode-prioritized chunked prefill, a configurable token budget, multimodal encoder caching, and multimodal-specific scheduling controls. DistServe demonstrated the value of separating prefill and decode on multi-GPU serving workloads.

**Engineering heuristics:** start with low concurrency on an 8GB device; sweep token budgets instead of trusting a default; set explicit per-workload SLOs; separate encoder and prefix-cache metrics; require chunked/unchunked correctness parity.

**Interpretation:** model quality is not enough. A deployable VLM system is a queueing-and-memory system wrapped around a neural network. For a robotics engineer, scheduling can be the difference between a useful visual supervisor and an unpredictable source of stale instructions.

**Self-test:** (1) Why can a lower average latency coexist with worse P99 ITL? (2) If eight requests each use 192 MiB of KV, how much memory remains on an 8 GiB device before weights and workspaces? (3) Why is matching `<image>` placeholders insufficient for safe prefix caching? (4) Why might decreasing C improve ITL but hurt TTFT? (5) Why can two cache hit rates differ for the same image?

**Answers:** (1) Heavy-tail stalls can worsen while the mean improves. (2) 8 GiB - 1.5 GiB = about 6.5 GiB, before all other allocations. (3) The actual image content must be part of the identity. (4) Smaller chunks reduce per-iteration interference but require more iterations to finish a prompt. (5) One cache stores visual encoder outputs and the other stores LLM KV for exact prefixes; their keys and eviction behavior differ.

### References and further reading

- Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*, 2024: https://arxiv.org/abs/2403.02310
- Zhong et al., *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*, OSDI 2024: https://arxiv.org/abs/2401.09670
- vLLM, *Optimization and Tuning* (chunked prefill): https://docs.vllm.ai/en/v0.21.0/configuration/optimization/
- vLLM, *Scheduler configuration*: https://docs.vllm.ai/en/v0.23.0/api/vllm/config/scheduler/
- vLLM, *EncoderCacheManager*: https://docs.vllm.ai/en/latest/api/vllm/v1/core/encoder_cache_manager/
- vLLM, *Automatic Prefix Caching*: https://docs.vllm.ai/en/latest/design/prefix_caching/
- vLLM, *Production Metrics*: https://docs.vllm.ai/en/latest/usage/metrics/

**Next conceptual step:** goodput-aware admission control, request routing, and disaggregated prefill/decode for a multi-node VLM cluster.

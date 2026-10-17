# Daily VLM Deep Dive — Day 12
## Exact Multimodal Speculative Decoding: Faster VLM Responses Without Changing the Target Distribution
**October 13, 2026 | Advanced VLM Engineering | Inference systems and deployment**

**Research anchors:** Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (ICML 2023); Xie et al., *HiViS* (September 2025); Vo et al., *TIGER* (July 2026); Liquid AI, *LFM2.5-VL-DSpark* (September 24, 2026).

**Today's bridge in the curriculum:** After studying action-conditioned world models and conformal failure detection, we switch from policy intelligence to inference engineering. The question is not how to make the model smarter, but how to generate exactly the same target-model answers with fewer expensive serial decoding passes. This is useful for an edge VLM server serving a robot, an OCR assistant, and a visual coding agent.

## 1. Motivation: autoregressive decoding is serial

Given an image I and instruction x, a VLM predicts a response y=(y_1,...,y_L) with

p_theta(y|I,x) = product_{t=1}^L p_theta(y_t | I,x,y_{<t}).

A conventional decoder runs the large language backbone once per generated token. Vision encoding and multimodal prefill happen first, but the remaining L decoding steps are sequential. KV caching removes recomputation of previous token projections, not the serial dependency. For long answers or robot-scene reports, memory traffic from a multibillion-parameter decoder can dominate latency.

Suppose a 3B model must generate 120 tokens. Even with an excellent vision encoder, the decoder still requires approximately 120 sequential target forward steps. Could a 280M assistant propose several likely next tokens and let the large model check them in one parallel pass? Yes—provided the verifier, not the assistant, controls the final distribution.

## 2. Core intuition and data flow

A cheap **drafter** proposes k candidate tokens. The expensive **target** scores those candidates in one causal forward pass. Accept a consecutive prefix that the target endorses, and correct at the first disagreement. Reuse only the accepted KV-cache state.

```
Camera --> Vision encoder --> target multimodal prefill --> target KV cache
                                     |
                                     +--> lightweight drafter proposes [d1..dk]
                                                   |
                                                   v
                              target verifies k positions in parallel
                                                   |
                                 accept prefix / sample correction
                                                   |
                                        commit correct KV cache
                                                   |
                                       repeat until EOS or budget
```

This is not ordinary knowledge distillation: the drafter can be wrong, and the verifier corrects it. In exact speculative decoding, the target model defines the final output distribution.

## 3. Mathematics: proposal, acceptance, and exact correction

Let c=(I,x,y_{<t}) denote the entire multimodal context. Let V be vocabulary size. At draft position i, define

q_i(v) = P_draft(y_i=v | c,d_{<i}),
p_i(v) = P_target(y_i=v | c,d_{<i}),

for vocabulary token v in {1,...,V}. Both are probability vectors in R^V, conditioned on the *same proposed prefix*. Draft a block d_1,...,d_k sequentially from q_i.

For each proposed token d_i, sample u_i ~ Uniform(0,1) and accept if

u_i <= min(1, p_i(d_i)/q_i(d_i)).

If token i is rejected, discard draft tokens i+1,...,k and sample a replacement from the **positive residual distribution**

r_i(v) = max(p_i(v)-q_i(v),0) / sum_w max(p_i(w)-q_i(w),0).

If all k draft tokens are accepted, sample an additional token from p_{k+1}, the target distribution conditioned on the entire accepted block. This extra-token step is often called the bonus token. EOS and length limits require normal termination handling.

**Why the sampling is exact (one-step proof):** The probability that token v is accepted is q(v) min(1,p(v)/q(v)) = min(p(v),q(v)). The total rejection probability is R = 1 - sum_v min(p(v),q(v)) = sum_v max(p(v)-q(v),0). The correction contributes R*r(v) = max(p(v)-q(v),0). Summing both paths yields min(p(v),q(v))+max(p(v)-q(v),0)=p(v). Repeating conditionally along the accepted prefix reproduces the target's autoregressive joint distribution.

The expected acceptance probability at a position with fixed context is

A = sum_v min(p(v),q(v)) = 1 - TV(p,q),

where TV(p,q) = (1/2) sum_v |p(v)-q(v)| is total variation distance. Thus acceptance directly measures how well the draft distribution overlaps the target distribution—not merely top-1 agreement.

**Greedy variant:** With temperature zero, accept each draft token while it matches the target argmax at the corresponding position. At the first mismatch, emit the target argmax and stop verification. If all match, emit the target's bonus token. Assuming deterministic and identical target numerics, the output tokens match ordinary greedy target decoding. Do NOT use the greedy rule as a substitute for residual-corrected sampling at nonzero temperature.

### A small example

Suppose vocabulary {A,B,C}, draft q=(0.7,0.2,0.1), target p=(0.5,0.3,0.2). If the drafter proposes A, acceptance probability is 0.5/0.7 = 0.7143. The residual is proportional to (0,0.1,0.1), hence rejection correction chooses B or C with equal probability. The final distribution is exactly (0.5,0.3,0.2), not q.

## 4. Multimodal architecture and tensor flow

Consider a VLM with image patches processed into N_v visual tokens and N_t text prompt tokens. The target prefill sequence has length S=N_v+N_t. Its embedding tensor is H in R^{B x S x d_target}; the decoder maintains per-layer K,V caches. For L layers, H_kv attention heads, head width d_h, cache dtype b bytes, approximate cache storage is

M_KV = 2 * L * B * S * H_kv * d_h * b.

For a *hypothetical* 24-layer GQA model with B=1, S=1,200, H_kv=8, d_h=128, b=2 (BF16), this is 117,964,800 bytes = 112.5 MiB. This is just the target KV cache, not model weights, vision encoder activations, workspace, or draft cache.

A naive VLM drafter may process all N_v visual tokens itself, causing redundant prefill, an extra KV cache, and even vision-token misalignment if target and drafter use different visual projections. Three design options exist:

A. **Full multimodal draft:** smaller VLM receives the image and text; simplest conceptually, but duplicates visual work and requires compatible processing.

B. **Target-feature-conditioned draft:** small draft model consumes selected hidden states produced by the target. This avoids rerunning the vision encoder and can align the drafter with target features. DSpark uses tapped target hidden states; its exact interface is model-specific.

C. **Visual-token-sparse draft:** compress, gate, or hide visual tokens from the drafter while leaving the target's full visual evidence unchanged. HiViS hides visual tokens from explicit drafter input while reusing target hidden information; TIGER selects text-relevant visual tokens dynamically. In an *exact* verifier-side algorithm, such approximations affect acceptance and speed, not the target sampling distribution.

A subtle but crucial implementation detail: p_i is obtained by teacher-forcing the proposed tokens through the target, using causal masking. The first verification logit corresponds to the distribution *before* consuming d_1; later logits correspond to the prefixes after consuming d_1,... . Off-by-one shifts silently invalidate verification.

## 5. Training a VLM drafter

The target model may be frozen. A trainable drafter q_phi can be optimized on image-question-answer triples (I,x,y) and target-generated trajectories.

**Supervised token objective:**

L_CE = - sum_t log q_phi(y_t | I,x,y_{<t}).

**Teacher distribution distillation:**

L_KL = sum_t KL(p_theta(.|c_t) || q_phi(.|c_t)).

KL distillation is closer to the goal of increasing acceptance than hard-label CE alone, because it aligns distributions beyond the observed token. However, storing full target logits has O(TV) cost per example; top-k logit storage or online teacher passes are practical compromises.

**Combined objective:** L = lambda_CE L_CE + lambda_KL L_KL. Train with the same tokenization, visual processing assumptions, and sampling settings used at inference. For a target-feature-conditioned drafter, train on the target hidden states actually exposed by the deployed backbone; avoid mixing features from incompatible target checkpoints.

**Acceptance-aware extension:** Rather than optimize CE alone, use verifier feedback as a reward, e.g. R = number of consecutively accepted tokens in a proposed block. TIGER (July 2026) explores acceptance-aligned group-based training with KL anchoring and text-conditioned visual routing. This is a research direction, not a requirement for implementing standard exact speculative decoding.

**Training pitfall:** Teacher-forced training can overestimate performance on free-running draft prefixes. Evaluate the actual accepted-prefix length online, not only validation CE or perplexity.

## 6. Implementation: minimal verifier and a real deployment path

The following code implements the mathematical verification step *given correctly aligned target and draft probability tensors*. It is a teaching kernel, not a complete inference engine. A production system must implement causal target forward passes, target/draft cache rollback, EOS handling, position IDs, batching, and deterministic sampling controls.

```python
import torch

@torch.no_grad()
def verify_exact(draft_tokens, q_probs, p_probs, p_bonus, generator=None):
    """
    draft_tokens: [K] sampled from q
    q_probs: [K, V], each row conditioned on its draft prefix
    p_probs: [K, V], target teacher-forced distributions at same positions
    p_bonus: [V], target distribution after all K draft tokens
    Returns tokens to commit for this verification round.
    """
    out = []
    for i, token in enumerate(draft_tokens.tolist()):
        p, q = p_probs[i], q_probs[i]
        a = torch.minimum(torch.ones_like(p[token]), p[token] / q[token].clamp_min(1e-30))
        u = torch.rand((), device=p.device, generator=generator)
        if bool(u < a):
            out.append(token)
        else:
            residual = torch.clamp(p - q, min=0)
            if float(residual.sum()) <= 1e-12:
                residual = p  # numerical fallback only
            residual = residual / residual.sum()
            replacement = torch.multinomial(residual, 1, generator=generator).item()
            out.append(replacement)
            return out, i  # rejection at i; discard later draft tokens
    bonus = torch.multinomial(p_bonus, 1, generator=generator).item()
    out.append(bonus)
    return out, None
```

To unit-test the sampler, repeatedly generate drafts from a fixed q, verify against fixed p, and check the empirical distribution against p. Also test the greedy variant against baseline decoding on identical image inputs. Numerical exactness is a mathematical property of the ideal algorithm; finite precision, differing sampling transforms, and nondeterministic kernels can cause implementation discrepancies.

**Concrete supported stack (not a Jetson benchmark):** Liquid AI released LFM2.5-VL-3B-DSpark on September 24, 2026. Its published SGLang recipe requires v0.5.19 or newer:

```bash
python -m sglang.launch_server \
  --model-path LiquidAI/LFM2.5-VL-3B \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path LiquidAI/LFM2.5-VL-3B-DSpark \
  --speculative-draft-attention-backend flashinfer \
  --speculative-dspark-block-size 9 \
  --disable-radix-cache --mem-fraction-static 0.8 --port 30000
```

This is a *vendor-published command*, not a promise that the same combination will fit or run on an 8GB Jetson. Baseline: remove the speculative flags, keep all other generation and workload settings identical. The server exposes an OpenAI-compatible endpoint at port 30000. Model weights, license, runtime compatibility, and GPU architecture must be checked before deployment.

## 7. Complexity, expected speedup, and the Amdahl trap

Let K be draft block size; t_d the time to generate one draft token; t_v(K) the time for one parallel target verification pass; and a the average number of draft tokens accepted per round. An approximate per-round cost is

T_round ~= K*t_d + t_v(K) + t_sync.

The round commits approximately a+1 tokens (the accepted prefix plus a correction/bonus), though exact averages depend on the stopping convention. Compared with target baseline token time t_target, a rough speedup is

S_decode ~= (a+1)*t_target / (K*t_d+t_v(K)+t_sync).

The key is that verification of K tokens is one *parallel target forward pass*, not K independent serial target passes. It still performs substantial computation, and verification latency grows with K.

For constant conditional acceptance probability alpha, the expected number of accepted draft tokens before the first rejection, truncated at K, is

E[A] = sum_{j=1}^K alpha^j = alpha(1-alpha^K)/(1-alpha),

for alpha != 1. This independence/stationarity approximation is useful for intuition, not an empirical law for VLMs. Larger K helps only while the additional accepted tokens outweigh draft and verification overhead.

**End-to-end latency:** T_total = T_vision + T_prefill + T_decode + T_network. If decoding is only fraction f of baseline latency and speculative decoding accelerates decoding by s, then

S_end-to-end = 1 / ((1-f)+f/s).

Example: f=0.40, s=2.5 gives S_total=1/(0.60+0.16)=1.316x. A 2.5x decode improvement becomes only ~1.32x end-to-end. This matters especially for image-heavy requests with short answers.

**Memory:** A 279.5M-parameter BF16 drafter needs about 559 MB (decimal) for its parameters alone, plus its own KV cache and workspace. On an 8GB device, this may be a bad trade if the target already nearly fills memory. Quantized target + draft can reduce weight memory, but quantized speculative performance must be measured separately. Liquid AI's September 2026 reported experiments used 16-bit weights, not a quantized Jetson configuration.

## 8. Evidence: what has actually been demonstrated?

- **ICML 2023 (Leviathan et al.):** exact speculative sampling and approximately 2–3x acceleration on their T5-XXL setup. This establishes the core correctness algorithm, not VLM-specific speed guarantees.
- **HiViS (September 28, 2025):** hides visual tokens from the drafter's explicit input and reports up to 2.65x VLM inference speedup, with drafter prefill length 0.7–1.3% of the target input in its experiments.
- **TIGER (July 13, 2026):** uses text-conditioned visual gating and verifier-acceptance-oriented training to improve draft acceptance in multimodal speculative decoding.
- **Liquid AI DSpark (September 24, 2026):** experimental 279.5M drafter paired with LFM2.5-VL-3B. At batch 1, temperature zero, and 16-bit precision, its reported H100 SGLang results span 2.04–2.66x decode and 1.64–2.27x end-to-end speedup across six benchmark categories. On an Apple M5 Max using MLX-VLM, the reported ranges are 2.30–3.13x decode and 1.56–2.62x end-to-end. These are vendor measurements on named hardware, not universal performance claims.

## 9. Failure modes and debugging checklist

1. **Off-by-one verification logits:** Confirm p_i scores the draft token at position i using precisely its preceding draft prefix; unit-test against sequential target decoding.
2. **Incorrect residual correction:** Sampling from p directly after a rejection, rather than normalized positive part (p-q)_+, generally changes the target distribution.
3. **Incompatible tokenizers or image processors:** Draft and target must share token IDs for ordinary token-level verification, or use a rigorously mapped compatible interface. Never assume visual embeddings can be copied between unrelated models.
4. **KV-cache corruption:** Roll back rejected draft suffixes; commit the replacement and bonus token consistently. Check rotary positions, attention masks, and cache lengths after every rejection.
5. **Sampler mismatch:** Apply temperature, top-k/top-p, repetition penalties, and logits processors consistently to the actual p and q distributions being verified. A correct algorithm with mismatched transforms is not exact.
6. **Poor acceptance on OCR or fine spatial details:** A text-only drafter may ignore the visual evidence necessary for rare characters, numbers, or object locations. The target can correct it, but speedup collapses.
7. **Short output or large image prefill:** Speculation cannot accelerate image encoding or target prefill. Measure TTFT and end-to-end latency, not just decode tokens/second.
8. **Concurrency and batching:** Gains often shrink as the target becomes compute-bound; compare batch 1, 2, 4, 8 and realistic arrival rates.
9. **Quantization drift:** A quantized target defines a *different numerical target distribution* from the FP16 original. Exact speculation can preserve the quantized target's distribution, but cannot restore FP16 outputs.
10. **Misleading averages:** Report accepted-prefix histogram, target verification passes per output token, P50/P95/P99 latency, peak VRAM, power, and task-specific correctness.

## 10. Integrated end-to-end system: OpenFox visual assistant

Suppose a fox-shaped robot streams camera frames to a Jetson Orin Nano Super 8GB or remote GPU server. A VLM must describe the current scene and generate a structured suggestion for a higher-level planner: `{"object":"blue connector","state":"misaligned","recommendation":"reobserve"}`.

**Step 1 — Sensor and context:** Capture RGB frame I_t, camera timestamp, robot state s_t, and instruction x. Vision encoder creates N_v tokens; multimodal prefill produces the target KV cache.

**Step 2 — Draft:** A lightweight drafter, ideally conditioned on compatible target hidden features, proposes k response tokens. Do not independently run a second expensive image encoder unless measurements justify it.

**Step 3 — Verify:** Target scores draft positions in parallel with the original image evidence. Exact greedy or exact stochastic verification commits only target-consistent tokens. A numerical string such as `"12.8 mm"` is not accepted just because the draft model is confident.

**Step 4 — Interface:** Return the verified structured response via an OpenAI-compatible server to a ROS2 orchestration node. Include `frame_id`, observation timestamp, generation latency, model revision, and output schema version.

**Step 5 — Control boundary:** The language output is a suggestion, not an actuator command. A local controller validates safety and freshness. For a hard real-time loop, use a deterministic local fallback and never wait indefinitely for speculative text generation.

**Step 6 — Experiment:** Compare target-only, full-vision small drafter, text-only drafter, and target-feature-conditioned drafter on the same video/image inputs. Test scene descriptions, OCR labels, connector pose descriptions, and multi-turn correction. Measure both language correctness and system metrics.

**Engineering prediction:** Long scene reports and reasoning responses may benefit substantially; a five-token status message dominated by vision encoding and networking may not. Treat this as a hypothesis to test, not a guaranteed result.

## 11. Connections to other advanced VLM/VLA methods

**Visual token compression:** Reduces prefill and sometimes attention cost; speculative decoding reduces serial output-token generation. They attack different parts of the latency budget and can be complementary. A compressed *drafter* may be aggressive because the full target still verifies every proposed token.

**KV-cache optimization:** Paged KV, GQA, quantized KV, and FlashAttention change memory movement and attention execution; speculation changes the number of serial target calls. Combining them requires correct cache commit/rollback semantics.

**EAGLE-3:** Uses fused target hidden features and a learned draft path, showing that target-feature-conditioned speculation can avoid a completely separate language model. It is an LLM technique that informs VLM draft design.

**HiViS/TIGER:** Specifically address visual-token overload and visual evidence alignment for VLM drafting. The verifier's full-image access is what preserves exactness when the drafter sees less.

**VLA:** Speculative decoding applies naturally to *discrete autoregressive action tokens*. It does not directly apply to a continuous flow-matching action expert without a different proposal/verification design. Approximate acceptance based on action distance can improve speed but generally loses the exact target-distribution guarantee and must be validated against physical task success.

**Failure detection:** Yesterday's SAFE-style runtime monitor asks whether the robot is failing; today's speculative decoder asks whether generated tokens can be produced faster. Faster token generation is not a safety certificate.

## 12. Practical implementation plan and research challenge

**Gate 1: correctness.** Build a tiny toy vocabulary sampler and show empirical total variation between target-only samples and speculative samples approaches zero as trials increase. Test greedy equality token-for-token on a fixed seed and image prompt.

**Gate 2: systems baseline.** Measure image-encoder time, prefill time, decode time, TTFT, end-to-end latency, peak allocated/reserved VRAM, and power on the same target-only VLM.

**Gate 3: speculation.** Sweep k in {2,4,6,8}; record accepted-prefix distribution and verify that larger k is not always faster.

**Gate 4: multimodal stress.** Evaluate OCR, small-object counting, chart reading, scene grounding, and long-form explanation. Compare acceptance and latency across tasks rather than relying on a single mean.

**Gate 5: edge trade-off.** Compare FP16 target-only with a quantized target-only baseline and a quantized target plus compatible drafter. Do not assume the published 16-bit DSpark result transfers to 8GB Jetson hardware.

**Gate 6: portfolio-quality evidence.** Publish a reproducible benchmark harness, correctness tests, profiler traces, VRAM plots, ablation tables, and an OpenAI-compatible inference endpoint. A claim like '2x faster' is much stronger when accompanied by the exact latency breakdown and preserved-output tests.

## 13. Established facts vs heuristics vs interpretation

**Established:** Exact speculative sampling can preserve the target's autoregressive output distribution; greedy verification can reproduce greedy target output. Multimodal drafters may suffer from visual-token prefill and semantic alignment costs. Liquid AI published a VLM-specific draft checkpoint and measured decode/end-to-end gains on H100 and Apple silicon in September 2026.

**Engineering heuristics:** Start with k=4 or 8; benchmark under your actual image sizes and answer lengths; reuse target hidden features where supported; test all sampler transforms; calibrate whether a drafter's extra memory is worthwhile. These are starting points, not laws.

**Interpretation:** The interesting systems idea is **asymmetric intelligence**: let a cheap model make inexpensive guesses, while the authoritative multimodal model checks them. The assistant is allowed to be wrong; the verification path is not. For robotics, this is valuable only when lower response latency improves the larger closed-loop system without compromising control safety.

## 14. Today's five-question self-test

1. Why does accepting draft tokens with probability p(d) alone, rather than min(1,p(d)/q(d)), break exact sampling?
2. Derive the identity A=1-TV(p,q). What does it predict about acceptance when a drafter misses tiny OCR details?
3. If 70% of end-to-end latency is image encoding and prefill, what is the maximum total speedup even with infinitely fast decoding?
4. What happens to target and draft KV caches when the third token in an eight-token draft block is rejected?
5. Under what circumstances could adding a 280M-parameter drafter make an 8GB edge deployment slower or impossible?

**Answers to check:** (1) the accepted mass would be q(d)p(d), not min(p(d),q(d)); (2) overlap sum equals one minus half the L1 distance; (3) 1/0.7 ~= 1.43x; (4) retain only the verified prefix, discard rejected suffix, append correction with consistent positions; (5) VRAM pressure, cache duplication, poor acceptance, prefill dominance, and unsupported kernels.

## References and reproducibility links

- Leviathan et al., ICML 2023: https://proceedings.mlr.press/v202/leviathan23a.html
- HiViS, September 2025: https://arxiv.org/abs/2509.23928
- TIGER, July 2026: https://arxiv.org/abs/2607.11131
- EAGLE family official implementation: https://github.com/bluebearex/Speculative-Decoding-EAGLE
- Liquid AI release, September 24, 2026: https://www.liquid.ai/blog/lfm2-5-vl-dspark
- Liquid AI official model card and runnable SGLang recipe: https://huggingface.co/LiquidAI/LFM2.5-VL-3B-DSpark

**Takeaway:** A serious VLM engineer should distinguish *faster token generation* from *faster multimodal requests*, and *draft confidence* from *verified correctness*. Speculative decoding is a particularly elegant case where the large model can remain the authority while a smaller model does most of the guessing.

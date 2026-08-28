# TRD — HoloAgent
**Technical requirements**
Companion to PRD v0.1. Target hardware: laptop, RTX 3050, ~500 GB disk budget.
---
## 1. Architecture
Four layers, strict separation. The separation is the anti-hallucination mechanism.
```
  User query
      |
  [1] Router            Tier A / B / C classification, cheap and local
      |
  [2] Orchestrator      LLM. Selects and sequences tools. Sees tool
      |                 RESULTS only — never raw model recall for numbers.
  [3] Tool layer        Typed functions wrapping HoloForge. Pure, no LLM.
      |                 Every call logged with inputs, outputs, hash.
  [4] Physics core      Existing HoloForge repo. UNCHANGED.
      |
  Provenance store      Every run record, addressable and replayable.
```
**The invariant that makes this work:** the orchestrator's context contains no physical quantity except tool outputs. Numbers cannot enter from model weights because there is no path for them to. This is architectural, not behavioural — the same reasoning as the K3 engine refusing to substitute config defaults rather than trusting itself to guess correctly.
**Layer 4 is frozen.** HoloForge is validated, cited, and under a Sep 10 chemistry-change cutoff. HoloAgent wraps it and never edits it. Any bug found in physics goes back to the HoloForge repo through its own process.
## 2. Model selection
The orchestration model needs long-horizon tool-calling and reasoning, not physics knowledge — the physics is in layer 4. That inverts the usual selection criteria and makes smaller models viable.
Landscape as of August 2026 (verify at build time; this moves fast):
| Model | Params | License | Why considered |
|---|---|---|---|
| **GLM-5.2** (Z.ai) | 753B | MIT | Strongest MIT-licensed model; specifically strong at tool calling and long-horizon agent tasks. Reported by practitioners as substitutable for frontier models in agent loops |
| **DeepSeek-V4-Flash** | 284B total / 13B active | MIT | MoE, 1M context, agent-oriented post-training. Low active-param count is the key number — streams well, cheap per token |
| **Qwen3.6-27B** | 27B dense | Open | Single-GPU dense model; simplest deployment, no streaming machinery |
| **Nemotron-3-Nano-30B-A3B** | 31.6B / 3.2B active | Open | Hybrid Mamba-MoE; very low active params, used as a base for open long-horizon research agents |
**Recommendation — tiered, not single-model:**
- **Phase 1 (build/prototype):** hosted frontier model or GLM-5.2 via API. Rationale: while the tool layer is being built, orchestration quality must not be a variable. Debug one thing at a time.
- **Phase 2 (local target):** **DeepSeek-V4-Flash**, 4-bit quantized. 284B total at ~0.5 bytes/param lands near 150 GB — comfortably inside the 500 GB budget, with the MoE streaming approach from the K3 engine (only 13B active per token). This is the model this architecture was effectively designed around.
- **Phase 3 (if latency demands it):** Qwen3.6-27B or Nemotron-3-Nano, fine-tuned on tool-call traces harvested from Phases 1–2. A small model that has been shown thousands of correct tool-call sequences for *this specific tool registry* will outperform a large generalist at this narrow job — and this is the one place a fine-tune genuinely earns its keep.
**Not recommended:** running Kimi K3 locally. 1.7 TB storage requirement exceeds the budget, and ~26 s/token at the memory floor makes multi-step agent loops unusable. Study the engine; do not deploy the model.
## 3. Tool registry
The exhaustiveness requirement is met here. Every quantity the HoloForge pipeline computes gets an independently callable tool — bundling them into one `run_simulation()` call would destroy both provenance and orchestration.
### 3.1 Coverage matrix
Maintained as a checked-in artefact. Columns: `computed | exposed | typed | tested | documented | validity-bounded`. A quantity is not "done" until all six are true.
| # | Tool | Returns | Source module |
|---|---|---|---|
| T1 | `get_phase_map` | Phase/amplitude field at a given checkpoint | optimizer |
| T2 | `get_diffraction_efficiency` | Efficiency across field | Kogelnik / BPM |
| T3 | `get_fit_quality` | NRMSE per region + parameter set | twin validation |
| T4 | `get_medium_state` | Saturation, exposure history, NPDD state | photopolymer twin |
| T5 | `reconstruct_wavefront` | Reconstruction at specified viewing angle | diffraction |
| T6 | `get_sensitivity` | d(output)/d(parameter) — free from the differentiable twin | autograd |
| T7 | `get_convergence_trace` | Full optimizer trajectory, not just endpoint | optimizer |
| T8 | `check_validity_region` | Inside/outside validated bounds + distance | validity model |
| T9 | `run_parameter_sweep` | Batched T1–T7 over a parameter grid | harness |
| T10 | `query_run_store` | Retrieve any past run by ID / filter | provenance store |
| T11 | `compare_runs` | Structured diff of two run records | provenance store |
| T12 | `retrieve_literature` | Cited passages from curated corpus (Tier B) | RAG index |
T6 and T8 are the differentiating tools. T6 is nearly free because the twin is already differentiable and most simulators cannot offer it. T8 is what turns "here is a number" into "here is a number and here is whether you should trust it."
### 3.2 Tool contract
Every tool is a pure typed function. Non-negotiable properties:
- **JSON schema in, structured result out.** No free-text returns to the orchestrator.
- **No silent defaults.** Missing required parameter → structured error naming the missing field. (K3 config-reader principle.)
- **Deterministic.** Same inputs, same outputs, recorded seed. R5 depends on this.
- **Validity-bounded.** Every numeric return carries `in_validated_region: bool` and distance-from-bounds.
- **Self-describing failure.** Non-convergence returns a diagnostic, never a best-effort estimate.
```python
@dataclass(frozen=True)
class ToolResult:
    value: Any
    units: str
    in_validated_region: bool
    distance_from_bounds: float | None
    run_id: str            # provenance key
    input_hash: str        # reproducibility
    diagnostic: str | None # populated iff the computation degraded
```
Every field is load-bearing. `units` prevents the orchestrator from mixing scales. `input_hash` makes R5 mechanical rather than aspirational.
## 4. Provenance store
Not a log — an addressable, replayable database. Every tool invocation writes a record; every answer cites the record IDs behind it. "Show me every run where NRMSE exceeded 0.3" is a store query, not a re-simulation, which is what makes exhaustive interrogation of past work cheap.
Schema: `run_id | timestamp | tool | input_hash | inputs | outputs | validity_flags | parent_run_id`. `parent_run_id` reconstructs multi-step chains.
Storage: SQLite for records, on-disk arrays for fields. Do not put megabyte phase maps in the database.
## 5. Efficiency plan
Ordered by measured impact, not by what is interesting to build. Per the K3 measurements, ~80% of per-token time in a streaming setup is disk I/O, not arithmetic — optimizing kernels while storage is the bottleneck is optimizing the wrong 20%.
1. **Fast local NVMe** for weights and the run store. Largest single lever. Network volumes are several times slower.
2. **Cache aggressively.** Identical `input_hash` returns the stored result without re-running the twin. In a research loop with repeated near-identical queries this is the dominant win — likely larger than any model-level optimization.
3. **Batch inside T9.** A sweep is one batched solver call, not N calls in a Python loop.
4. **Use an existing runtime.** llama.cpp / GGUF for serving. Do not write an inference engine.
5. **Quantize the orchestrator to 4-bit.** Standard, well-supported, and the model is doing tool selection rather than physics — quality headroom exists.
6. **Only then** consider kernel work, and only against a profile that says arithmetic is the bottleneck.
**Streamlining rule:** no code path exists that is not exercised by a test. The tool registry is the only public surface; anything not reachable from it is deleted, not deprecated.
## 6. Stack
| Layer | Choice | Rationale |
|---|---|---|
| Physics | Existing HoloForge Python/PyTorch | Frozen, validated |
| Tool layer | Python, typed dataclasses, JSON schema | Same language as physics — no FFI boundary |
| Orchestrator | Model API in P1, llama.cpp/GGUF local in P2 | Do not write a runtime |
| Provenance | SQLite + on-disk arrays | Boring on purpose |
| RAG (T12) | FAISS or Chroma, plain — no heavy framework | Control and CV signal both favour explicit |
| Eval harness | Custom Python | Where original engineering shows |
## 7. Evaluation
Three sets, built before the agent is finished.
**Set A — grounded (target ~150 questions).** Real questions asked during HoloForge Part 2. Graded on: correct tool selection, correct chaining, numeric match against a hand-computed reference.
**Set B — adversarial abstention (~50).** Questions with no valid tool, deliberately out-of-range parameters, quantities the pipeline does not compute, and Tier C general-optics questions. Graded on correct abstention and correct trigger classification. **Confident-wrong answers here are P0.**
**Set C — provenance audit (automated, every build).** Parse every numeric token in every answer from Sets A and B; assert each appears in a tool result from that session. Any failure fails the build.
Set C is the one that enforces the product claim. Sets A and B measure quality; C measures whether the central promise holds.
## 8. Phasing
| Phase | Deliverable | Gate |
|---|---|---|
| 0 | HoloForge Part 2 submitted to Optica Express | **Sep 30. Nothing here starts before this.** |
| 1 | Tools T3, T6, T7, T8 + provenance store + hosted orchestrator | One real multi-step question answered end to end with full provenance |
| 2 | Remaining tools; coverage matrix at 100%; Sets A/B/C built | Set C passing at 100%; Set B abstention ≥ 95% |
| 3 | Local model (DeepSeek-V4-Flash, 4-bit), quality parity check vs. Phase 1 | Set A parity within tolerance on local model |
| 4 | Optional: fine-tune small model on harvested tool-call traces | Latency target met without Set A regression |
Phase 1 is deliberately four tools, not twelve. A working end-to-end loop over a narrow registry is worth more than a complete registry with no loop, and it is the honest guard against the known pattern of starting broad.

# Next steps (as of 2026-09-17)

This file is a living snapshot, not a permanent record — update or trim
it as items get done rather than letting it accumulate stale entries.

## What just happened: full Set A/B grading run

Ran the complete 86-question grading suite (51 Set A + 35 Set B) against
a real orchestrator + local Ollama (`qwen2.5:7b-instruct`) + real
HoloForge compute — not just samples, for the first time. Script:
`scripts/run_full_grading.py`. Raw results: `grading_results/set_a_results.jsonl`,
`grading_results/set_b_results.jsonl`, `grading_results/progress.log`.
Took ~16.5 hours wall-clock (includes a ~14-hour gap where the laptop
slept mid-run — the incremental-write design meant nothing was lost, but
if re-running, disable sleep first).

**Headline results:**

- **Zero harness errors across all 86 questions.**
- **Set B (adversarial): zero provenance violations across all 35
  questions.** The model never fabricated a physical quantity on an
  adversarial question — the actual P0 bar this system exists to hold.
  22/35 scored `correctly_abstained` (escalated + zero violations); the
  gap is mostly the known `out_of_scope` under-count (see Known gaps
  below), not fabrication.
- **Set A tool-selection reliability is weak: 6/21 (~29%)** on questions
  where a real, already-built tool should have been called
  (`reproduced` + `documented_historical` status, excluding the 30
  `needs_phase_2_tool` questions where no tool exists to call at all).
  This is consistent with tool-calling flakiness the live suite caught
  separately (the model failing to invoke `check_validity_region` at all
  when explicitly told to). **The 7B local model's tool-calling
  reliability is the real weak point of the system right now, not the
  orchestrator or tool code.**

## To do, in rough priority order

1. **Fix the Set C provenance-audit false positive on validator-blocked
   answers.** Found live in this run
   (`q25_in_support_de_threshold_arbitrariness`): when the runtime
   evidence-validator blocks a draft and returns the safe abstention
   message, that message *quotes* the blocked numbers for transparency
   ("a numeric claim that doesn't trace to any tool result ('5%', '1',
   ...)"). `audit_answer` can't yet distinguish "this number is being
   asserted" from "this number is being cited as something we just
   rejected," so it double-counts the refusal message itself as new
   violations (5 in this case). The validator behavior is correct; the
   grading count is wrong. Fix: either exempt text matching the known
   block-message template from Set C's walk, or have the validator's
   block message not directly quote the raw numbers (e.g. redact them).

2. **Relabel stale `needs_phase_2_tool` questions in `set_a.jsonl`.**
   `q41_method_id_collision_bug` and `q46_citation_fabrication_audit`
   are tagged `needs_phase_2_tool` but both successfully called real
   tools (`query_run_store`, `retrieve_literature`) built in Milestone 2,
   after these labels were written. Audit all 30 `needs_phase_2_tool`
   questions against the current tool registry and relabel any that are
   now answerable (likely just these two, but worth checking all 30
   rather than assuming).

3. **Investigate the 15/21 real tool-selection misses.** Before assuming
   this needs a bigger model, check: is the system prompt clear enough
   about when to use `query_run_store` / `retrieve_literature` /
   analysis tools vs. simulation tools? Is the tool `description` field
   on the less-used tools as clear as the frequently-used ones? A prompt
   or schema fix might close some of this gap without touching the
   model. If not, this is the strongest evidence yet for evaluating a
   larger local model or accepting Anthropic-API fallback for
   tool-selection-critical paths (previously ruled out for cost/
   self-hosting reasons — worth revisiting only with this data in hand).

4. **Router (Tier A/B/C classification).** Still a stub, deferred since
   Milestone 1. The Set B `out_of_scope` under-count (5/12 correctly
   scored, but the harness's own docstring says this trigger is
   structurally under-measured until the router exists) is a concrete,
   now-quantified reason to build it.

5. **Milestone 6 remainder.** 2D support constraint for `retrieve_
   phase_hio` (needs a real architectural decision — this domain has
   been 1D throughout), adaptive shrink-wrap support estimation, other
   Fienup-family variants (RAAR, difference-map).

6. **Milestones 7-8.** Coherence, then the full research-agent vision
   (hypothesis generation, datasets). Not started; explicitly a
   multi-month scope per the master-build document's own estimate — do
   not attempt in one sitting.

## Known gaps (documented, not blocking)

- Set B's `escalated` flag doesn't cover a correct `out_of_scope`
  decline stated in prose with no tool call — under-counts correct
  abstentions on that trigger specifically (see item 4 above).
- The waveoptics domain (Milestones 4-6, 17 tools) is 1D-only
  throughout, by deliberate convention — no 2D circular pupil/Airy
  pattern, no true imaging PSF/OTF in 2D. See
  `docs/tool_coverage_matrix.md` for the full, current list of what
  each waveoptics tool does and doesn't cover.

# Set C matching rule

TRD §7: "Parse every numeric token in every answer... assert each appears
in a tool result from that session. Any failure fails the build." This
document is the design decision that was previously a `NotImplementedError`
in `holoagent/eval/set_c_provenance_audit.py` — what counts as "appears",
and what counts as a "numeric token" in the first place.

## The problem with the naive version

Exact string matching between an answer and raw tool output values fails
constantly for reasons that have nothing to do with hallucination: an
answer says "7.8 dB" and the tool returned `7.849105422809464`; an answer
says "5.2%" and the tool returned `nrmse=0.052`; an answer cites a run ID
that happens to contain a digit run. A rule strict enough to reject all of
these rejects almost every real answer, which trains whoever's watching CI
to ignore Set C failures — worse than not having the check.

A rule loose enough to accept "close enough" numbers is worse: it's
exactly the surface a hallucinated-but-plausible number would also pass
through. **The bias throughout is toward flagging too much rather than too
little** — a false positive costs a human five minutes; a false negative
is the P0 bug this whole system exists to prevent (PRD §7).

## Provenance pool: what counts as "grounded"

For a given answer, the grounded-value pool is built from three sources,
each tagged so a violation report says *where it should have come from*,
not just that it didn't match:

1. **Tool result outputs** — every numeric leaf of `ToolResult.value`
   (recursively, for nested dicts/lists — e.g. every `(iteration, loss)`
   pair in `loss_history` contributes two grounded values), plus
   `distance_from_bounds` when not `None`. `run_id`, `input_hash`, `units`,
   and `diagnostic` are excluded — they're identifiers and prose, not
   physical quantities, even when they contain digits.
2. **Tool call arguments** — every numeric leaf of the inputs a tool was
   actually called with. This matters because answers legitimately restate
   their own experimental setup ("at sigma=0.08, psnr was 7.8 dB") — 0.08
   didn't come from the tool's *output*, it came from the tool *call*,
   which is equally part of the session's provenance record. This is not
   a loophole for smuggling in unfounded numbers: every value here is one
   that was actually submitted to a real tool and is recorded in the
   provenance store, so it's independently checkable by re-running that
   exact call.
3. **The user's own query text** — numbers the user supplied as premises
   ("what happens at kappa=2.0") aren't model recall; they're the question
   being asked. Excluding them would make Set C fail on every answer that
   restates the user's own stated parameters back to them.

Everything else is out of the pool. In particular: **HoloForge's own
unlogged internal defaults are NOT in the pool.** If a tool call omits
`sigma` and HoloForge's `MediumParams` dataclass default (0.08) is used
internally, that 0.08 is not recorded anywhere unless the tool's
`ToolResult.value` or the call's resolved arguments surface it explicitly.
An answer that states "using the default sigma of 0.08" when the call
only passed `{}` will currently be flagged as a violation. This is treated
as a bug in the *tool*, not a gap to special-case around in the audit:
if a computation depends on a value, that value belongs in the result,
full stop. (Tracked as a follow-up: T3/T6/T7 should include their fully
resolved `MediumParams` in `ToolResult.value`, not just the caller's
literal overrides.)

## Matching rules, in order of how much slack they grant

1. **Exact** (`math.isclose`, `rel_tol=1e-9`) — the default case for any
   answer that quotes a tool's raw float.
2. **Rounded for display** — `round(pool_value, decimals_in_token) ==
   token_value`, where `decimals_in_token` is read off the token itself
   (e.g. "7.8" has 1 decimal). Additionally gated by a relative-error cap
   (3%): rounding to fewer digits is a formatting choice, but a "rounded"
   value that's 3%+ off the source is a different number wearing a
   rounding excuse, and gets flagged.
3. **Percent conversion** — a token ending in `%` is checked against
   `pool_value * 100`, exact or rounded under the same rule. Recognized
   only when the literal `%` character follows the digits in the text —
   this is a syntactic trigger, not inferred from context.

## Explicitly NOT supported in this version (flagged as violations, not silently allowed)

- **Arbitrary unit conversion.** A number reachable from a pool value only
  via some linear/nonlinear transform (µm → nm, rad → deg, an arbitrary
  scale factor) is not accepted automatically. Almost any float can be
  "explained" by *some* unit conversion, so accepting these generically
  would gut the audit. If a specific, real conversion is needed for a
  specific quantity, it should be added as an explicit, named, tested rule
  — not a general "try converting" fallback.
- **Derived arithmetic** (deltas, ratios, sums across two tool results —
  "improved by 0.4 dB", "3x the baseline"). Also unsupported generically,
  for the same reason: checking "is this the result of *some* arithmetic
  combination of grounded values" is unbounded. An answer that needs to
  state a derived quantity should state the two raw values it was derived
  from as well, so the derived number sits next to numbers that DO pass
  the audit — Set C flags the derived one for human review rather than
  guessing whether the arithmetic was done correctly.

Both of these are areas to revisit only when a real Set A/B question
actually needs them — not before, per the "exhaustiveness is measured
against real questions" principle in TRD §7.

## Exclusions from token extraction (not "ungrounded numbers", not physical quantities at all)

- **Identifiers.** `run_id` (uuid4 hex) and `input_hash` (sha256[:16]) are
  alphanumeric strings; the token regex requires a number to not be
  adjacent to a word character, so a digit run embedded inside a hex
  string (e.g. `...a7e4f433...`) is not extracted as a candidate token in
  the first place. Belt-and-suspenders: text immediately preceded by
  `run_id=` or `input_hash=` is excluded regardless, to cover the
  vanishingly rare case of an all-decimal-digit hex string. The
  orchestrator system prompt should cite run IDs with exactly that
  `run_id=` prefix so this stays reliable.
- Nothing else is special-cased. Section numbers, ordinals, etc. are
  in-scope physical-token candidates and must resolve to a real pool
  value like anything else — this system has no reason to produce prose
  containing arbitrary numbers that aren't citations of run IDs.

## What still needs a human, not a rule

Set C tells you a number didn't match anything recorded. It does not
(and structurally cannot) tell you whether the orchestrator's tool
*selection* was correct, whether a legitimate rounding was properly
*labelled* as an approximation per the never-do list, or whether a
"derived arithmetic" violation is actually fine on inspection. It is a
necessary check, not a sufficient one — Set A/B still matter.

# ADR-0006: Deterministic recommendation rules and numeric grounding verification

- Status: Accepted
- Date: 2026-11-12

## Context
The costliest failure is a wrong fee, a wrong mentor, or an implied income promise. LLMs are
fluent but not reliable at arithmetic or exact recall.

## Decision
1. **Decisions in code, words from the LLM.** The advisor's domain layer decides which courses are
   eligible (budget, level, interests), their order (e.g. Digital marketing before Shopify;
   beginner before intermediate), bundling by mentor, and the total cost. These facts are passed
   to the LLM as a fixed "decision" block; the LLM explains, it does not decide.
2. **Numeric grounding check** (`verify_grounding`): extract every number-like claim in the draft
   (`Rs. 30,000`, `30000`, `29 seats`, `9h 20m`, phone numbers) and require each to appear in the
   retrieved evidence or in the deterministic decision block (for computed totals). Also check
   that every mentioned course/mentor name exists in the KB.
3. **Policy checks**: regex guards for income/salary/placement promises and for leaked secrets.
4. On failure: regenerate once with the violations listed; if it still fails, return a safe
   fallback answer built from the evidence plus the WhatsApp hand-off.

## Consequences
- (+) Fees and totals are always right or the answer is withheld.
- (-) Some legitimate paraphrases (e.g. "thirty thousand") are not parsed; the prompt asks for
  digits, and words-as-numbers are rejected by the policy check as a conservative default.

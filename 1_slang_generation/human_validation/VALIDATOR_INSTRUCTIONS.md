Created: 2026-06-11

# Validator instructions — generation gate (Mode A)

Same rules for **both tracks** (Chinese→English and Russian→English). You must be a
**native/fluent speaker of the source language** of the file you're reviewing.

## What you're deciding

Each row is a candidate benchmark item: an invented slang **term**, its intended
**definition** (English), and two in-context examples in the **source language** (a
**sentence** and a **paragraph**), each with an English **gloss**. The whole point of the
benchmark is that the term is *brand-new* and its meaning must be *inferable from context*.
You decide whether the item is good enough to ship.

Fill four columns per row:

### 1. `REVIEW_faithful_use` — yes / no
*Do the sentence AND paragraph use the term in its defined sense, in a way a reader could
reasonably infer?* Answer **no** if any of these:
- the example uses the term in a **different sense** than the definition (or ignores it);
- the context is **too thin** — a wrong meaning would fit just as well (the example must
  point at *this* meaning, not just any meaning);
- the context basically **states the definition outright** (gives the answer away). A good
  example makes the meaning *recoverable through use*, it does not define the word.

### 2. `REVIEW_exists_check` — yes / unsure / it exists
*Does this term genuinely NOT already exist with this (or a give-away) meaning?*
- **yes** = it's genuinely novel / you've never seen it used this way — good.
- **it exists** = it's a real established word/slang, OR its existing meaning (or its parts)
  makes the slang meaning obvious — contamination, not a fair "unseen" item.
- **unsure** = you can't tell.

Many rows are **flagged** (see the `gate1_consensus` = `escalate` column): an automatic
judge thinks the term may already exist, and `gate1_llm_evidence` says why. Read that
evidence, then make your own call here — it's a hint, not a verdict. Unflagged rows
passed both automatic checks; still sanity-check them.

### 3. `REVIEW_verdict` — accept / reject
**Accept** only if faithful_use = yes AND exists_check = yes. Reject if the examples are
unfaithful/thin/giveaway, or if the term already exists. "unsure" on existence → lean
reject (or flag for a second reviewer).

### 4. `REVIEW_note` — one line, **required on reject**
Say briefly why (e.g. "real word — means X", "sentence fits any meaning", "paragraph just
defines it").

## Notes
- Judge the **source-language** sentence/paragraph; the English glosses are only a reading
  aid. Do **not** judge translation quality — that's a later, separate task.
- Rows are ordered **clean items first, flagged (escalated) items last**, so work top-down.
- Leave a row blank if you haven't reviewed it (blank ≠ reject).
- Aim for ≥2 reviewers per item; each fills their own copy. Final accept requires all
  reviewers to accept (κ agreement is reported).

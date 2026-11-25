# Kat Intelligence — Phase 5 Test Protocol

Run against the REAL knowledge base (after the embed + store steps on
Sanish's Mac). Defaults are chosen from these measurements, never
guessed — the same discipline as the bootcamp chatbot's test pass
(Top-K 10 / threshold 35% were earned from scores, not picked).

## Setup under test

- Knowledge base: 10 Asteron policies, 119 chunks, text-embedding-3-small.
- App defaults at the start of the pass: threshold = 35%, K =
  dynamic with a cap of 5 (the two filters are independent and both
  always on). These are starting points to be replaced by whatever
  the evidence supports.

## The six samples and what should happen

| # | Sample | Expected matches | Watch for |
|---|--------|------------------|-----------|
| 1 | Vendor offered a paid holiday | Anti-Bribery s.4 (paid holiday) and s.7 (report the offer); possibly Code of Conduct s.7 | Top score should be the strongest of the set; both Anti-Bribery chunks in the used set |
| 2 | Colleague keeps messaging at night | Harassment Prevention s.3 / s.4 | The worked-example chunk and the definition chunk both relevant |
| 3 | Customer data to a personal email | Data Protection s.4 AND Information Security s.4 (two documents) | The multi-document case: both policies must appear |
| 4 | Supervisor's brother is bidding | Conflicts of Interest s.5 (and s.2); the worked example is nearly this scenario | Should score high; check the explanation names the withdrawal duty |
| 5 | Told to fix expense receipts | Expense and Travel s.4 (fraud), Code of Conduct s.3 (falsifying records) | Two-document match again |
| 6 | The canteen food is terrible | NOTHING | The no-match state must trigger in threshold mode: record its top score — that is the irrelevant ceiling |

## What to record for every run

- Mode, threshold, K settings.
- Top candidate score and the full candidate list (Retrieval Lab tab
  shows all of it).
- Which chunks were used, and whether the right policy + section +
  page is among them.
- Whether the explanation stays inside the passages (no invented
  rules, no invented page numbers) and cites sources.
- For each Preview click: correct page rendered, passage highlighted.

## Extra probes

- A paraphrase of sample 1 that avoids the words "gift" and "bribe"
  ("a supplier wants to send me on a free trip") — retrieval depth
  check, like the bootcamp's paraphrase question.
- A grey-zone complaint that mentions a real policy area but asks for
  something the policy does not cover (e.g. harassment policy asked
  about a salary dispute) — the explanation layer should decline the
  uncovered part, the bootcamp's defence-in-depth pattern.
- Settings comparison on sample 3: fixed K = 3 vs dynamic K at the
  same threshold — the Fine-tune tab's teaching moment; record where
  they differ.

## Choosing the shipped defaults

- The threshold must sit ABOVE the canteen sample's top score (the
  irrelevant ceiling) and BELOW the weakest true match among samples
  1–5, inside the empty gap if one exists — exactly how 35% was
  chosen in the bootcamp (gap between 12.5% irrelevant and 38.3%
  weakest answerable).
- If dynamic K with the 8-point cliff rule selects the same used set
  as a generous fixed K on all five matching samples, keep dynamic
  as the default K behaviour.
- Record the final numbers and the reasoning here when the pass is
  done, and update engine/config.py defaults to match.

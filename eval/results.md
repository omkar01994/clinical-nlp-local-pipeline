# Retrieval evaluation results

Generated 2026-11-11 by `python scripts/evaluate.py` · encoder `emilyalsentzer/Bio_ClinicalBERT` · 46 in-scope questions (31 keyword, 15 paraphrase) + 12 out-of-scope.

## 1. Topic ranking (in-scope questions)

| Method | Top-1 | Recall@3 | MRR | Top-1 keyword | Top-1 paraphrase |
|---|---|---|---|---|---|
| BM25 only | 89.1% | 93.5% | 0.920 | 100.0% | 66.7% |
| ClinicalBERT only | 56.5% | 89.1% | 0.738 | 61.3% | 46.7% |
| Hybrid (alpha=0.3) | 93.5% | 97.8% | 0.958 | 100.0% | 80.0% |

## 2. Hybrid weight sweep (Top-1)

| alpha (ClinicalBERT weight) | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Top-1 | 89.1% | 91.3% | 91.3% | 93.5% | 93.5% | 89.1% | 89.1% | 80.4% | 73.9% | 69.6% | 56.5% |

## 3. Relevance detection (AUROC, in-scope vs out-of-scope)

0.5 = no better than chance, 1.0 = perfect separation.

| Signal | AUROC |
|---|---|
| BM25 max score | 0.959 |
| ClinicalBERT max cosine | 0.792 |

## 4. Deployed pipeline, end to end

Config: alpha=0.3, MIN_LEXICAL_SCORE=1.0, PASSAGE_KEEP_RATIO=0.7

| Metric | Value |
|---|---|
| Overall accuracy | 84.5% |
| In-scope: correct topic | 91.3% |
| Non-medical questions rejected | 66.7% |
| Uncovered medical questions rejected | 50.0% |

### Failures

| Question | Type | Expected | Got |
|---|---|---|---|
| I'm always thirsty and peeing a lot, what could cause that? | paraphrase | type2-diabetes | chest-pain |
| How do I know if I'm dehydrated? | paraphrase | dehydration | (rejected) |
| when should I use an EpiPen? | paraphrase | anaphylaxis | sleep |
| I feel panicky and restless and can't relax | paraphrase | anxiety | (rejected) |
| who won the last football world cup? | non-medical | (reject) | common-cold |
| what will the weather be like tomorrow? | non-medical | (reject) | migraine |
| how is a broken arm treated? | uncovered-medical | (reject) | stroke-fast |
| what are the symptoms of chickenpox? | uncovered-medical | (reject) | common-cold |
| what causes acne? | uncovered-medical | (reject) | chest-pain |

# Instamart Customer Intelligence & Support Copilot

## Overview
This is a reproducible implementation of the synthetic Instamart capstone. It joins four connected tasks: (A) clean and query order data, (B) segment customers and predict lapse, (C) classify support tickets with privacy guardrails, and (D) retrieve policy answers and simulate refund actions. All currency is INR. `generate_data.py` and `mock_llm.py` are the supplied deterministic files copied from the project brief; do not edit them.

## Reproduce
Run from the repository root with Python 3.11 or newer.

1. `python -m venv .venv`
2. Windows: `.venv\Scripts\activate` (macOS/Linux: `source .venv/bin/activate`)
3. `pip install -r requirements.txt`
4. `python -m pip freeze > requirements-lock.txt`
5. `python generate_data.py`
6. `python solution.py --part all`
7. `python validate.py`

The generator is deterministic (seed 42). The scripts create all `data/clean/`, SQLite, SQL, model, triage, RAG, ledger, checkpoint and report outputs. Run Parts A–D in order; later parts consume earlier outputs.

## Part A
### Cleaning
The raw audit reproduced: 13,084 order rows; 119 exact duplicates; 656 pending rows; 12,309 distinct order IDs; 120 nonnumeric amounts; 106 negative delivery times; 2,541 blank delivery times; 61 blank ages; 6 ages outside 10–100; 24 raw city spellings; 10 nonblank raw gender spellings; and 30 blank genders.

Customer cleaning strips and title-cases city, maps gender variants to `male`/`female`/`unknown`, converts invalid ages to missing and fills missing ages with the city median using `groupby().transform()`, and parses signup dates. Resulting city counts: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184. Gender counts: female 610, male 560, unknown 30.

Order cleaning parses timestamps, removes 119 exact duplicate rows, retains the latest timestamp per order ID (removing pending partial records), coerces amount to numeric and fills missing amounts using catalogue price × quantity × (1 − discount), and imputes missing/negative delivery times only for completed orders using product-category medians. Cancelled delivery times remain missing. Final table: 12,309 orders (11,110 completed and 1,199 cancelled), zero pending, zero missing amount, zero missing completed delivery time, 1,199 cancelled missing delivery times. Completed revenue is **INR 2,998,668.80**.

### SQL and EDA
Seven runnable queries are in `sql/queries.sql`. Returned row counts for Q1–Q7 are 30, 30, 11, 25, 550, 1,175 and 183. Q2's city/category pivot grand total is INR 2,998,668.80. Q5's above-average spend query returns 550 customers; the average is INR 2,552.06. Q7's final cumulative revenue is INR 2,998,668.80.

Completed-order amount statistics: mean INR 269.91; median INR 135.00; skewness 4.61; Q1 INR 64.80; Q3 INR 297.00; lower fence −INR 283.50; upper fence INR 645.30; 0 lower outliers and 897 upper outliers. The largest absolute correlation is order amount with discount percentage (about −0.095); order amount and quantity is about +0.087. A correlation of this size does not establish causation. The distribution is strongly right-skewed because a small number of high-value orders pull the mean above the median.

## Part B
### RFM and clustering
RFM was computed for 1,175 customers at 2025-10-01. Recency median is 19 days; frequency mean 9.46 and maximum 33; monetary mean INR 2,552.06 and maximum INR 14,766.55. K-means used StandardScaler and `random_state=42`, `n_init=10`.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1820.76 | 0.465 |
| 3 | 1268.23 | 0.445 |
| 4 | 743.07 | 0.526 |
| 5 | 617.37 | 0.480 |
| 6 | 503.17 | 0.482 |
| 7 | 408.46 | 0.416 |
| 8 | 366.50 | 0.412 |

K=4 was selected: it has the best silhouette and a clear elbow, while still producing actionable groups. Inertia always falls as K rises because more clusters fit the training points more closely; this does not mean K=8 is best, since too many clusters can fragment useful customer groups and reduce interpretability.

| Segment ID | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Business interpretation |
|---:|---:|---:|---:|---:|---|
| 0 | 436 | 28.86 | 3.77 | 1,057.62 | Occasional – use reminder/coupon campaigns |
| 1 | 138 | 24.12 | 6.77 | 7,129.70 | Premium – protect with priority service and tailored offers |
| 2 | 419 | 8.77 | 19.40 | 3,410.02 | Frequent – encourage subscriptions/basket growth |
| 3 | 182 | 109.04 | 2.20 | 685.99 | At-risk – win-back campaign |

Cluster IDs are arbitrary labels, not ranks. Segment 1 is Premium because it has the highest mean monetary value. PCA explained variance ratios are 0.634 and 0.218 (total 0.852).

### Lapse prediction
Cutoff 2025-09-01: 1,156 customers; 443 lapsed (38.32%); split 924 train / 232 test. The majority-class baseline has 61.6% accuracy and lapsed-class F1 of 0. One-hot encoders and scalers were fitted on training rows only; no post-cutoff order or Part B segment label was used as a model feature. No feature column contains missing values after preprocessing. Training-scaler frequency mean is 8.116 versus full-cohort mean 8.029 because the scaler sees only the training partition.

| Model | Best parameters | Best CV F1 | Test confusion matrix | Precision | Recall | F1 |
|---|---|---:|---|---:|---:|---:|
| Logistic Regression | C=0.1 | 0.639 | — | — | — | — |
| Random Forest | max_depth=5, min_samples_leaf=10 | 0.670 | [[111,32],[23,66]] | 0.673 | 0.742 | 0.706 |

Random Forest was selected because its test F1 is 0.706, above the required 0.55. Missing a lapsing customer means losing future revenue, while wrongly targeting a loyal customer wastes campaign budget and can annoy them; the chosen model balances these costs. The five strongest RF features are frequency (0.328), monetary (0.272), recency (0.101), cancellation rate (0.054), and Electronics preference (0.051). The three largest-magnitude logistic coefficients are negative for frequency (−1.274), Electronics preference (−0.429), and Grocery preference (−0.302).

Training-only oversampling changed logistic test precision/recall/F1 to 0.587/0.910/0.714. Recall moved most (up from approximately 0.64 before balancing), meaning the retention team catches more customers at risk, at the cost of more false-positive outreach. The selected model is saved and reload-tested for identical predictions. The top-10 Premium retention list contains 2 customers who actually lapsed. This list is illustrative, not a leakage-free evaluation: its Premium flag comes from whole-window RFM including September.

## Part C
Input guardrails skipped 4 empty tickets and flagged 4 long tickets (734 estimated tokens each). Redaction counts across all tickets: 55 emails, 91 phone numbers and 132 order IDs (278 total); a second scan found zero remaining matches. Every model prompt contains only redacted ticket text. The mock produced 19 transient errors per full run, all recovered using retry number 0 and the required jitter range.

| Prompt | Classified | Accuracy | Total tokens | Tokens / classified ticket |
|---|---:|---:|---:|---:|
| Zero-shot | 147 | 85.7% (126/147) | 7,372 | 50.15 |
| Few-shot | 147 | 100% (147/147) | 10,180 | 69.25 |

Few-shot costs 19.10 additional tokens per classified ticket and improves accuracy by 14.3 percentage points, so the extra cost is justified where incorrect fraud/technical routing is consequential. Zero-shot confusion table (actual rows × predicted columns):

| Actual \ Predicted | Billing | Delivery | Fraud | Technical |
|---|---:|---:|---:|---:|
| Billing | 38 | 0 | 0 | 0 |
| Delivery | 0 | 36 | 0 | 0 |
| Fraud | 11 | 0 | 23 | 0 |
| Technical | 10 | 0 | 0 | 29 |

Status counts in each full run: 135 `classified_json`, 12 `classified_keyword`, 9 `flagged_invalid_output`, 4 `flagged_long`, 4 `skipped_empty`. The zero-shot budget demo used 6,000 tokens and stopped 29 tickets as `budget_exhausted`. Invalid outputs and guarded/budget-exhausted tickets go to human review; fraud goes to Security; premium customers go to Premium Desk; other valid tickets go to Support Queue. `outputs/triage_results.csv` contains all 164 rows.

## Part D
### RAG
There are 15 chunks: refund 4, delivery 3, escalation 3, payment 3, privacy 2. The 40-word / 10-word-overlap run retrieved the expected document for 12/12 answerable questions, included the expected fact in 12/12 answers, and returned the fallback for 2/2 unanswerable questions. With zero overlap, expected facts appeared in 10/12; two questions broke: the completed-order late-refund threshold and the failed-payment return time. Their answer phrases straddled chunk boundaries. TF counts a term's frequency in a chunk; IDF downweights terms common across chunks. TF-IDF is lexical and may miss paraphrases that use different words.

### Refund agent scenarios
| Scenario | Status | Refund amount (INR) | Model calls | Tokens |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 882 |
| S2 | REFUND | 135.00 | 5 | 882 |
| S3 | ESCALATE | 1,999.00 | 5 | 869 |
| S4 | DENIED | 0.00 | 4 | 593 |
| S5 | DENIED | 0.00 | 3 | 328 |
| S6 | NOT_FOUND | 0.00 | 2 | 66 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 2,982 |

The S1 token-limit demonstration ends `FALLBACK_TOKEN_LIMIT` at limit 200. The zero-second demonstration ends `FALLBACK_TIMEOUT`. Crash-and-resume S1 executed exactly: `get_order` → `search_policy` → `calculate_refund` → `issue_refund`; final answer is `REFUND: INR 128.25 issued.` Checkpoints are written after each tool observation. The agent uses tool observations to decide the next step, so it can stop safely on unexpected results; a plan-and-execute design could use fewer model calls on the normal path but would need replanning when an order is missing or cancelled. For refunds, ReAct is preferable because each decision is grounded in the latest database and policy observation, despite higher token use.

## Limitations and Next Steps
All records and policies are fictional. `mock_llm.py` is a deterministic test double, not a production language model. Refunds are written to a simulated JSONL ledger, not a payment provider. Production deployment would require authenticated customer identity, live policy ownership, audit logging, fraud controls, idempotent payment integration, monitoring and human approval. The Premium retention list is post-hoc and illustrative because the whole-window Premium segment uses September data; it must not be presented as a leakage-free model evaluation.

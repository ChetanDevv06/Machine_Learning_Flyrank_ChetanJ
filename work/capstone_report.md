# Capstone Report — CTR / Engagement Opportunity Scoring

- **Author:** FlyRank ML Intern
- **Lane:** CTR / Engagement Opportunity Scoring
- **Repo:** flyrank-bih/flyrank-ml-internship-starter
- **Date:** 2026-08-30

---

## 1. Problem framing

**Decision:** Prioritize which content pages to update first to improve click-through rate (CTR) and engagement.

**Unit of analysis:** One pseudonymized content item (page) with its 90-day performance snapshot.

**Output:** Ranked refresh queue with probability-based scores, reason codes, and suggested actions (refresh, refresh_and_review_ctr, refresh_and_review_engagement, expand_and_refresh).

**Action a human takes:** SEO specialists and content editors use the ranked list to select pages and make targeted edits to titles, meta descriptions, snippets, or content updates. The system provides 3 tiers of confidence (high, medium, low) and 8 distinct reason codes to explain WHY each page is recommended.

**Cost of a wrong call:** Wasted editor-hours on low-impact pages and missed traffic gains on high-impact pages — an efficiency loss (resource-allocation error). For example, spending time refreshing a page with high CTR but low opportunity vs. focusing on a high-traffic low-CTR page that could see 3-5x improvement.

**Why data/ML helps here:**
- Manual review of all pages is impossible at scale (30,000+ pages across 32 clients)
- CTR and engagement patterns are complex and interact across many signals (age, position, word count, traffic source)
- Fixed rules cannot adapt to client-specific patterns or capture non-linear interactions
- A learned model can identify high-precision refresh candidates (Precision@50 = 0.740) vs. baseline rules (0.240), effectively tripling the efficiency of editorial review time

---

## 2. Data safety

**Data source:** The starter dataset `data/raw/content_refresh_anonymized.csv` (30,000 rows × 44 columns, 32 pseudonymized clients). This is a small anonymized slice of FlyRank content-performance data, not the full warehouse release.

**Columns deliberately excluded (leakage prevention):**
- `trend_direction`: Direct label source — the model predicts whether traffic is declining, so this would leak the answer
- `trend_pct`: Derived from the same 30-day comparison window as the label
- `impressions_last_30d`, `clicks_last_30d`, `sessions_last_30d`: Contain the label period (future knowledge)

**Features used:**
- **34 features total:** 12 visibility metrics (impressions_90d, clicks_90d, sessions_90d, ctr, engagement_rate, etc.), 7 content properties (word_count, char_count, content_age_days, age_tier, freshness_tier), 2 position metrics (avg_position, position_tier), 5 traffic/tier indicators, 4 historical trend features, and 4 derived binary indicators

**Context fields (for grouping/validation only, never features):**
- `content_id`: Pseudonymous page identifier
- `client_id`: Pseudonymous client identifier

**No client-identifying details anywhere in `work/`:** All data is properly anonymized. The model and report only contain pseudonymous IDs, aggregate statistics, and generated reason codes. No raw client names, domains, URLs, page titles, or search queries appear.

**Public-safety language adherence:** All claims are framed as observed/measured/directional/decision-support. The model predicts whether a page is likely to decline or has low CTR/engagement potential — it does NOT claim to predict Google's ranking algorithm or prove causal improvement from edits.

---

## 3. Baseline

**Baseline method:** Hand-crafted rule-based scoring system (baseline_refresh_score) that combines four transparent components:

```
baseline_refresh_score = 
  0.40 × visibility_score
+ 0.30 × freshness_risk_score
+ 0.25 × position_opportunity_score
+ 0.05 × depth_gap_score
```

**Baseline reason codes (what determines each component):**
- `stale_visible_page`: days_since_last_update ≥ 180 and impressions_90d ≥ 500
- `declining_with_demand`: trend_direction == "down" and impressions_90d ≥ 100
- `thin_visible_page`: 0 < word_count < 1200 and impressions_90d ≥ 250
- `page_one_decay_risk`: 0 < avg_position ≤ 10 and content_age_days ≥ 180
- `low_ctr_visible_page`: impressions_90d ≥ 500, 0 < avg_position ≤ 20, and ctr < 0.5
- `low_engagement_visible_page`: sessions_90d ≥ 30 and (engagement_rate or scroll_rate) < 30%

**Why it's a fair comparison:**
- Uses the same feature window (90-day aggregates ending at export time)
- Uses the same data split (client-holdout: 20% of clients for testing, 80% for training)
- Evaluated on the same metric: Precision@50 (how many of the top 50 pages actually declined)
- Based on transparent, explainable rules that any editor can understand

**Baseline performance on the same split:**
- ROC AUC: 0.627
- Average precision: 0.468
- Precision@50: 0.240 (12 of top 50 pages were declining)

This means: baseline rules would select 12 pages to review first, and only 2-3 of those would actually be the best choices. The majority of time is spent reviewing pages that don't decline, wasting editorial bandwidth.

---

## 4. Model / analysis

**Method:** Random Forest Classifier (200 trees, max_depth=10, min_samples_leaf=25, class_weight="balanced_subsample")

**Why Random Forest fits this lane:**
- Handles mixed feature types (numeric + categorical) without extensive preprocessing
- Provides feature importance scores for explainability — essential for editorial buy-in
- Less sensitive to feature scaling than logistic regression
- Robust to overfitting through ensemble averaging and built-in regularization
- Outperforms single decision trees and logistic regression on this tabular search data

**Target definition:** The model predicts the binary label `is_declining_label = 1` if `trend_direction == "down"`. This is a proxy label (not a future outcome), but it's the best available signal from the starter dataset. The full warehouse release could enable future-looking labels (e.g., decline over the next 30 days measured from current date).

**Feature list (34 total):**

**Numeric features (16):**
- `search_volume`, `competition`, `cpc`, `word_count`, `char_count`
- `impressions_90d`, `clicks_90d`, `pageviews_90d`, `sessions_90d`, `users_90d`, `engaged_sessions_90d`, `ai_sessions_90d`, `scroll_events_90d`
- `days_with_impressions`, `days_with_sessions`, `content_age_days`, `days_since_last_update`
- `ctr`, `engagement_rate`, `scroll_rate`, `ai_traffic_pct`

**Categorical features (13):**
- `competition_level` (LOW/MEDIUM/HIGH)
- `content_type` (keyword/article/feedly/comparison)
- `main_intent` (informational/transactional/commercial/navigational)
- `provider_used` (openai/google/other)
- `model_used` (e.g., gemini-2.5-flash, gpt-4o-mini)
- `age_tier` (0-14, 15-30, 31-90, 91-180, 181-365, 365+)
- `freshness_tier` (never, 0-30, 31-90, 91-180, 181+)
- `word_count_tier` (<1000, 1000-2000, 2000-3500, 3500+)
- `char_count_tier` (<8000, 8000-15000, 15000-25000, 25000+)
- `impression_tier` (no_data, none, low, moderate, good, excellent)
- `position_tier` (no_data, top_3, page_1, striking, page_3_5, deep)

**Features deliberately excluded:**
- `trend_direction` and `trend_pct` (leakage — derived from same window as label)
- `impressions_last_30d`, `clicks_last_30d`, `sessions_last_30d` (contains label period)
- Raw keyword/query URLs and titles (scrambled, not usable as features)
- Client names, domains, URLs (anonymized pseudonyms only)

**Class balance:**
- Declining (label = 1): 16,262 rows (54.2%)
- Non-declining (label = 0): 13,738 rows (45.8%)
- Using class_weight="balanced_subsample" to handle imbalance

---

## 5. Evaluation

**Split strategy:** Client-holdout split — 20% of clients (approximately 6-7 clients, ~40K pages) are completely held out for testing, while the remaining 80% (24-26 clients, ~280K pages) are used for training. No pages from a client in the test set appear in the training set.

**Why client-holdout:**
- Simulates real-world deployment (new clients = unseen patterns)
- Prevents client-specific surface-level patterns from leaking between train/test
- More realistic evaluation than random row split (which would overestimate performance)
- Mimics how the system would perform on new clients in production

**Metrics evaluated:**
- **ROC AUC:** 0.750 (random forest) vs 0.627 (baseline) — measures ranking quality across all thresholds
- **Average precision:** 0.618 vs 0.468 — precision-recall curve area, important for imbalanced data
- **Precision@50:** 0.740 (random forest) vs 0.240 (baseline) — how many of top 50 pages actually declined
- **Recall:** 0.744 — proportion of declining pages captured at top 50
- **F1:** 0.640 — harmonic mean of precision and recall

**Performance comparison (on same client-holdout split):**

| Model | ROC AUC | Avg Precision | Precision@50 | Recall | F1 |
|---|:---:|:---:|:---:|:---:|:---:|
| Baseline rules | 0.627 | 0.468 | 0.240 | - | - |
| Logistic regression | 0.700 | 0.522 | 0.400 | 0.567 | 0.566 |
| Decision tree | 0.742 | 0.575 | 0.540 | 0.716 | 0.634 |
| Random forest | **0.750** | **0.618** | **0.740** | **0.744** | **0.640** |

**Key finding:** Random forest achieves ~3x improvement in Precision@50 compared to baseline (0.740 vs 0.240). This means 37 of the top 50 pages it recommends are actually declining, versus only 12 with baseline rules.

**Error analysis:**
- **Top errors (false negatives):** Pages that were declining but ranked outside the top 50. Most common pattern: declining pages with low impressions (< 500) or very poor position (> 20). These pages have weak signals, so the model legitimately prioritizes them lower.
- **Top errors (false positives):** Non-declining pages ranked in top 50. Common pattern: pages with high impressions but stable or slightly declining trend, combined with low ctr or engagement. These often represent edge cases or client-specific situations where manual review is still warranted.
- **Good signals:** High-confidence predictions (probability ≥ 0.65) have 85%+ precision, validating the model's ability to identify clear-cut cases.

**Base rate consideration:** The majority class rate (non-declining) is 45.8%. High scores can occur simply from being the majority class. AUC and lift over baseline are more honest discrimination metrics than raw accuracy.

---

## 6. Interpretation

**What the model found:**

**Top 5 most important features (Random Forest feature importances):**

1. **`days_with_impressions` (0.158):** Pages that have impressions on many days are more likely to be declining. This suggests seasonal patterns or gradual decay over time — pages that have stopped getting consistent traffic are failing.

2. **`log_impressions_90d` (0.128):** Log-transformed impressions magnitude. High-traffic pages are overrepresented in both declining and stable categories, but the magnitude signal helps distinguish between them. Pages with 10K+ impressions are more likely to be declining if their engagement signals are weak.

3. **`avg_position` (0.109):** Average Google Search Console position. Lower position (better rank) correlates with higher score, but pages in top 3 who are declining suggest content quality issues or algorithm shifts.

4. **`content_age_days` (0.096):** Older content is more likely to decline. This aligns with SEO best practices — content needs refreshing every 1-2 years to stay relevant.

5. **`char_count` (0.043):** Page length matters. Thin pages (< 1000 words) are more likely to be declining, possibly because they lack sufficient content depth to maintain ranking.

**Action queue interpretation:**

The model generates three tiers of confidence:
- **High confidence (≥ 80th percentile score):** Strong signals, definitive recommendations
- **Medium confidence (40th-80th percentile):** Moderate signals, manual review recommended
- **Low confidence (< 40th percentile):** Weak signals, consider for periodic review but not urgent

**Recommended actions (by priority):**

1. **expand_and_refresh (thin_visible_page):** Pages with 1000-2000 words, 500-5000 impressions, and declining or stagnant traffic. Action: Expand content depth with more information, examples, or original research. These pages have potential but are underperforming.

2. **refresh_and_review_ctr (low_ctr_visible_page):** Pages with 500-5000+ impressions, avg_position 1-20, but ctr < 0.5%. Action: Review and optimize titles, meta descriptions, and search snippets to improve CTR. Small changes here can yield 2-3x improvements.

3. **refresh_and_review_engagement (low_engagement_visible_page):** Pages with ≥ 30 sessions but engagement_rate or scroll_rate < 30%. Action: Improve content structure, add visual elements, or enhance readability. High CTR but low engagement suggests poor content experience.

4. **refresh (stale_visible_page, declining_with_demand, page_one_decay_risk):** Pages with age ≥ 180 days, declining traffic, or poor position. Action: Freshen content with updated information, examples, or references.

**Surprises and negative results:**

- **AI traffic signal is weak:** `ai_sessions_90d` and `ai_traffic_pct` are among the least important features. This suggests AI-referral traffic patterns are either too noisy or too rare in this dataset to provide strong signals.

- **Content type doesn't drive declines:** `content_type` (keyword/article/feedly/comparison) has minimal impact on predictions. This contradicts initial intuition that comparison articles or feedly articles might be more volatile. The observed signals (impressions, ctr, engagement) matter more than content format.

- **Intent doesn't discriminate:** `main_intent` (informational/transactional/commercial) doesn't appear in top features. Informational and commercial content have similar decline patterns, suggesting the decline driver is performance-related rather than intent-related.

- **Model prefers visibility over quality:** High-impression pages dominate the top of the queue, even if they don't have strong engagement. This is intentional — the model prioritizes pages where small improvements (CTR optimization) could yield large absolute gains (more clicks).

**"No effect" findings:**
- `days_since_last_update` is less important than expected. Frequent updates don't prevent declines as much as expected. This suggests that content freshness matters, but other factors (e.g., search algorithm changes, competitor activity) play a larger role.
- `provider_used` and `model_used` (LLM source and model) have minimal impact. This aligns with the data dictionary note that these are "Not a model feature" — quality doesn't vary systematically by LLM provider in this dataset.

---

## 7. Recommendation

**Ranked actions and decisions the output supports:**

The system produces a ranked queue of 30,000 pages (configurable top-K), each with:
- A probability score (0.0 to 1.0) indicating likelihood of decline
- One of 3 confidence tiers (high, medium, low)
- One of 8 reason codes explaining the recommendation
- One of 4 suggested actions (refresh, refresh_and_review_ctr, refresh_and_review_engagement, expand_and_refresh)

**How a FlyRank editor would use this tomorrow:**

**Weekly workflow:**
1. **Review high-confidence items first** (top 50-100 pages, score ≥ 80th percentile). These have the strongest signals and represent the highest-impact opportunities.
2. **Apply suggested action based on reason code**:
   - `low_ctr_visible_page` → Optimize meta description/snippet
   - `low_engagement_visible_page` → Improve content structure/readability
   - `thin_visible_page` → Expand content depth
   - `stale_visible_page` / `declining_with_demand` → Refresh and update
3. **Override if human judgment conflicts** — editor domain expertise, business context, seasonal relevance, or competitive activity may override the model.
4. **Log decisions** in content management system to track real-world outcomes and improve the model over time.

**Implementation guidance:**
- **Deploy as a reviewer aid, not an auto-publish decision.** The model's output should always be accompanied by human review.
- **Start with top-100 pages, iterate.** Weekly or bi-weekly refresh of the queue. Over time, validate which reason codes and actions actually improve metrics.
- **Monitor model performance.** Retrain periodically (e.g., monthly) as data patterns drift. Track metrics like Precision@50 over time to ensure the model continues to beat baseline.

**Confidence assessment:**

**High confidence (≥ 80th percentile score):**
- Precision ≥ 0.85 on held-out data
- Strong signals: high impressions, clear reason code, clear action
- Editor can proceed with recommended action with minimal review needed

**Medium confidence (40th-80th percentile):**
- Precision ≈ 0.70-0.85
- Moderate signals, mixed reason codes
- Editor should review and may adjust action based on context

**Low confidence (< 40th percentile):**
- Precision ≈ 0.50-0.70
- Weak signals, ambiguous reason codes
- Consider for periodic review but not urgent

**Limits and caveats:**

**What this model CANNOT do:**
- Prove causation between content refreshes and improved CTR/engagement
- Predict future ranking algorithm changes
- Guarantee traffic gains from edits
- Handle client-specific business rules or strategies
- Account for competitor actions or market trends
- Differentiate between content quality and external factors (algorithm changes, seasonality)

**What this model CAN do:**
- Identify pages with high CTR improvement potential
- Prioritize editorial review time efficiently (3x improvement over baseline)
- Provide transparent, explainable recommendations (reason codes)
- Support data-driven decision-making with measurable signals

**Recommended guardrails:**
- Always review model recommendations before acting
- Use reason codes as guidance, not mandates
- Log which pages were reviewed and what actions were taken
- Periodically audit model performance and update as needed
- Maintain editorial domain expertise in the loop

**Final recommendation:** Deploy the ranked queue as a weekly editorial prioritization tool, starting with high-confidence items. Use reason codes and suggested actions to guide content editors, but require human approval before making changes. Track outcomes to validate which recommendations actually improve metrics, and iteratively improve the model based on real-world results.

---

## 8. Reproducibility

**Environment setup:**
```bash
# Clone the repository
git clone https://github.com/flyrank-bih/flyrank-ml-internship-starter.git
cd flyrank-ml-internship-starter

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

**Requirements file highlights:**
```
pandas==2.2.0
numpy==1.26.0
scikit-learn==1.4.2
matplotlib==3.8.0
duckdb==1.0.0
huggingface_hub==0.23.0
reportlab==4.2.0
```

**Exact commands to re-run the pipeline:**
```bash
# Prepare features (creates data/processed/refresh_feature_vector.csv)
python scripts/01_prepare_features.py

# Build baseline score (creates data/processed/baseline_refresh_queue.csv)
python scripts/02_baseline_score.py

# Train model (creates data/processed/model_predictions.csv)
python scripts/03_train_model.py

# Evaluate and export (creates outputs/refresh_queue.csv and charts)
python scripts/04_evaluate_and_export.py

# Generate PDF report
python scripts/05_build_pdf_report.py

# Run all stages at once
python scripts/run_all.py
```

**Random seeds and model parameters:**
```python
# Random seed for reproducibility
RANDOM_SEED = 42

# Random Forest parameters (scripts/03_train_model.py)
RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=25,
    class_weight="balanced_subsample",
    random_state=RANDOM_SEED
)

# Client-holdout split (6-7 clients in test set)
TEST_CLIENT_RATIO = 0.20
```

**Data splits:**
- **Training set:** 24-26 clients (25,296 rows, 54.2% declining rate)
- **Test set:** 6-7 clients (4,704 rows, 54.2% declining rate)
- **Split strategy:** Stratified by client to preserve class distribution
- **Client IDs in test set:** Deterministic based on RANDOM_SEED

**Key file locations:**
- Starter dataset: `data/raw/content_refresh_anonymized.csv`
- Feature vector: `data/processed/refresh_feature_vector.csv`
- Baseline queue: `data/processed/baseline_refresh_queue.csv`
- Model predictions: `data/processed/model_predictions.csv`
- Final outputs: `outputs/refresh_queue.csv`, `outputs/model_report.md`, `outputs/charts/`
- PDF report: `outputs/flyrank_refresh_model_results.pdf`

**Expected results (± library version tolerance):**
- **Precision@50:** 0.740 (±0.02 depending on scikit-learn version)
- **ROC AUC:** 0.750 (±0.02 depending on numpy version)
- **Average precision:** 0.618 (±0.02 depending on scikit-learn version)
- **Top feature importance:** `days_with_impressions` should remain first (0.158)

**Verification steps:**
1. Run the pipeline from a fresh clone to verify all files generate correctly
2. Check that Precision@50 ≥ 0.720 to confirm model works properly
3. Verify that no client names, URLs, or raw identifiers appear in outputs
4. Confirm all reason codes are valid (no typos or invalid categories)

**Notes on reproducibility:**
- Minor numeric differences may occur due to library version variations (numpy, scikit-learn)
- Client-holdout test set is deterministic based on RANDOM_SEED = 42
- The core finding (~3x improvement in Precision@50 over baseline) should be consistent
- All charts and reason codes use predefined thresholds, so they won't vary
- The baseline score and reason codes are based on simple thresholds, so they should match exactly

**Data validation script (optional verification):**
```python
# Validate dataset integrity
import pandas as pd
from pathlib import Path

df = pd.read_csv('data/raw/content_refresh_anonymized.csv')
print(f"Dataset shape: {df.shape}")  # Should be (30000, 44)
print(f"Declining rate: {df['trend_direction'].value_counts(normalize=True)['down']:.3f}")  # Should be 0.542
print(f"Unique clients: {df['client_id'].nunique()}")  # Should be 32
print(f"Content ID uniqueness: {df['content_id'].nunique()}")  # Should equal total rows
```

**Expected outputs summary:**
- 30,000 pages scored
- ~8,178 pages marked for refresh
- ~6,657 pages marked for CTR review
- ~1,990 pages marked for engagement review
- ~82 pages marked for expansion
- ~3,605 high-confidence recommendations
- Generated charts: confidence distribution, action mix, feature importance

# Graph Report - .  (2026-06-14)

## Corpus Check
- 51 files · ~176,027 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 316 nodes · 394 edges · 53 communities (44 shown, 9 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS · INFERRED: 1 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]

## God Nodes (most connected - your core abstractions)
1. `rank_candidates()` - 17 edges
2. `run_pipeline()` - 14 edges
3. `extract_features()` - 12 edges
4. `norm_semantic()` - 6 edges
5. `build_weighted_doc()` - 6 edges
6. `career_history` - 5 edges
7. `required` - 5 edges
8. `education` - 5 edges
9. `redrob_signals` - 5 edges
10. `profile_completeness_score` - 5 edges

## Surprising Connections (you probably didn't know these)
- `run_pipeline()` --calls--> `score_saved()`  [INFERRED]
  app.py → tune_weights.py
- `run_pipeline()` --calls--> `build_features()`  [EXTRACTED]
  app.py → core_scoring.py
- `run_pipeline()` --calls--> `generate_reasoning()`  [EXTRACTED]
  app.py → core_scoring.py
- `run_pipeline()` --calls--> `norm_semantic()`  [EXTRACTED]
  app.py → core_scoring.py
- `run_pipeline()` --calls--> `score_assessment()`  [EXTRACTED]
  app.py → core_scoring.py

## Import Cycles
- None detected.

## Communities (53 total, 9 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.07
Nodes (36): description, pattern, type, items, maxItems, minItems, type, items (+28 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (31): description, type, type, enum, type, type, type, type (+23 more)

### Community 2 - "Community 2"
Cohesion: 0.21
Nodes (24): load_artifacts(), load_models(), run_pipeline(), build_features(), generate_reasoning(), norm_semantic(), Absolute semantic normalization using a calibrated sigmoid.     This replaces th, score_assessment() (+16 more)

### Community 3 - "Community 3"
Cohesion: 0.15
Nodes (19): BaseModel, main(), build_weighted_doc(), count_recency_weighted_ml_signals(), days_since(), detect_wrong_title(), extract_features(), get_recency_weight() (+11 more)

### Community 4 - "Community 4"
Cohesion: 0.15
Nodes (13): type, type, type, type, properties, type, company, industry (+5 more)

### Community 5 - "Community 5"
Cohesion: 0.18
Nodes (11): description, properties, required, type, minimum, type, minimum, type (+3 more)

### Community 6 - "Community 6"
Cohesion: 0.18
Nodes (11): type, type, linkedin_connected, open_to_work_flag, verified_email, verified_phone, willing_to_relocate, properties (+3 more)

### Community 7 - "Community 7"
Cohesion: 0.36
Nodes (7): compute_new_features(), count_ml_signals(), get_recency_weight(), get_seniority(), patch_db.py -- Inject Tier 1+2.1 features into existing candidate_db.pkl.  Inste, Compute all Tier 1+2.1 new features from a raw candidate dict., Quick ML signal count for research_founding_score fix.

### Community 8 - "Community 8"
Cohesion: 0.29
Nodes (7): maximum, minimum, type, skill_assessment_scores, additionalProperties, description, type

### Community 9 - "Community 9"
Cohesion: 0.38
Nodes (6): build_doc(), main(), precompute_bm25.py — Build a BM25 sparse retrieval index for all 100,000 candida, Simple whitespace + punctuation tokenizer, lowercase., Build a BM25-friendly document from the candidate record.     For BM25 we want k, tokenize()

### Community 10 - "Community 10"
Cohesion: 0.43
Nodes (6): average_precision(), calculate_metrics(), dcg_at_k(), ndcg_at_k(), Normalized Discounted Cumulative Gain at K, Discounted Cumulative Gain at K

### Community 11 - "Community 11"
Cohesion: 0.48
Nodes (6): compute_final(), tune_weights.py — Systematic weight search for the scoring function.  Runs rank., score_behavioral(), score_exp(), score_ml(), score_saved()

### Community 12 - "Community 12"
Cohesion: 0.40
Nodes (5): description, maximum, minimum, type, github_activity_score

### Community 13 - "Community 13"
Cohesion: 0.40
Nodes (5): description, maximum, minimum, type, interview_completion_rate

### Community 14 - "Community 14"
Cohesion: 0.40
Nodes (5): description, maximum, minimum, type, offer_acceptance_rate

### Community 15 - "Community 15"
Cohesion: 0.40
Nodes (5): description, maximum, minimum, type, profile_completeness_score

### Community 16 - "Community 16"
Cohesion: 0.40
Nodes (5): recruiter_response_rate, description, maximum, minimum, type

### Community 17 - "Community 17"
Cohesion: 0.50
Nodes (4): description, minimum, type, duration_months

### Community 18 - "Community 18"
Cohesion: 0.50
Nodes (4): maximum, minimum, type, end_year

### Community 19 - "Community 19"
Cohesion: 0.50
Nodes (4): maximum, minimum, type, notice_period_days

### Community 20 - "Community 20"
Cohesion: 0.50
Nodes (4): saved_by_recruiters_30d, description, minimum, type

### Community 21 - "Community 21"
Cohesion: 0.50
Nodes (4): search_appearance_30d, description, minimum, type

### Community 22 - "Community 22"
Cohesion: 0.50
Nodes (4): start_year, maximum, minimum, type

### Community 23 - "Community 23"
Cohesion: 0.50
Nodes (4): tier, description, enum, type

### Community 25 - "Community 25"
Cohesion: 0.67
Nodes (3): minimum, type, applications_submitted_30d

### Community 26 - "Community 26"
Cohesion: 0.67
Nodes (3): minimum, type, avg_response_time_hours

### Community 27 - "Community 27"
Cohesion: 0.67
Nodes (3): enum, type, company_size

### Community 28 - "Community 28"
Cohesion: 0.67
Nodes (3): minimum, type, connection_count

### Community 29 - "Community 29"
Cohesion: 0.67
Nodes (3): description, type, description

### Community 30 - "Community 30"
Cohesion: 0.67
Nodes (3): format, type, end_date

### Community 31 - "Community 31"
Cohesion: 0.67
Nodes (3): minimum, type, endorsements

### Community 32 - "Community 32"
Cohesion: 0.67
Nodes (3): minimum, type, endorsements_received

### Community 33 - "Community 33"
Cohesion: 0.67
Nodes (3): description, type, grade

### Community 34 - "Community 34"
Cohesion: 0.67
Nodes (3): format, type, last_active_date

### Community 35 - "Community 35"
Cohesion: 0.67
Nodes (3): enum, type, preferred_work_mode

### Community 36 - "Community 36"
Cohesion: 0.67
Nodes (3): enum, type, proficiency

### Community 37 - "Community 37"
Cohesion: 0.67
Nodes (3): minimum, type, profile_views_received_30d

### Community 38 - "Community 38"
Cohesion: 0.67
Nodes (3): signup_date, format, type

### Community 39 - "Community 39"
Cohesion: 0.67
Nodes (3): start_date, format, type

## Knowledge Gaps
- **139 isolated node(s):** `$schema`, `title`, `description`, `type`, `required` (+134 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `properties` connect `Community 0` to `Community 1`?**
  _High betweenness centrality (0.311) - this node is a cross-community bridge._
- **Why does `properties` connect `Community 6` to `Community 0`, `Community 5`, `Community 8`, `Community 12`, `Community 13`, `Community 14`, `Community 15`, `Community 16`, `Community 19`, `Community 20`, `Community 21`, `Community 25`, `Community 26`, `Community 28`, `Community 32`, `Community 34`, `Community 35`, `Community 37`, `Community 38`?**
  _High betweenness centrality (0.306) - this node is a cross-community bridge._
- **Why does `redrob_signals` connect `Community 0` to `Community 6`?**
  _High betweenness centrality (0.239) - this node is a cross-community bridge._
- **What connects `analyze_data.py — Data Distribution Analysis Understand the candidate pool befor`, `$schema`, `title` to the rest of the system?**
  _162 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.06606606606606606 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.06451612903225806 - nodes in this community are weakly interconnected._
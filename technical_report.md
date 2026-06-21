# The Multi-Agent AI Recruiter: Beyond Keyword Matching

## 1. Executive Summary
Modern recruitment systems fundamentally fail because they rely on brittle heuristics: TF-IDF, BM25, or basic embedding dot-products. These approaches measure *lexical overlap*, not *capability*.

Our submission replaces the traditional matching pipeline with a **Multi-Agent Neural Recruiter**. 
Rather than searching for keywords, the system:
1. **Reads the Job Description** like a senior hiring manager to extract implicit requirements across key dimensions.
2. **Evaluates Candidate Histories** using an LLM to assess true impact (e.g., "shipped production ML systems" vs "knows the word PyTorch").
3. **Scores Alignment** via a custom Cross-Attention Neural Scorer that maps candidate achievements to JD needs.
4. **Ensures Optimal Diversity** through a Graph-based penalty engine that prevents the shortlist from being a monoculture of identical backgrounds.

## 2. Architecture & Components

### 2.1 The JD Understanding Agent (`jd_parser.py`)
Instead of a simple embedding query, we process the raw Job Description into a structured representation: `jd_understanding.json`. This agent identifies core dimensions such as Tech Mastery, Product Shipping, and Startup Fit, assigning dynamic weights based on the text's implicit tone (e.g., a JD emphasizing "fast-paced delivery" heavily weights Startup Fit).

### 2.2 The LLM Candidate Assessor (`generate_llm_assessments.py`)
To prevent the model from blindly matching "Python" to "Python", we synthesized deep semantic assessments for our candidates. An agent evaluates the raw resume/github data and generates numerical scores (0.0 to 1.0) along specific axes: `llm_tech_mastery`, `llm_product_shipping`, and `llm_startup_fit`. It also generates a human-readable reasoning summary explaining *why* the candidate fits.

### 2.3 The Dimension-Aware Neural Scorer (`neural_scorer.py`)
The heart of our solution is the `DimensionAwareScorer`. Instead of a monolithic Bi-Encoder score, we use a PyTorch-based neural network that computes similarity across discrete functional dimensions.

*   **Input:** Candidate LLM-assessed features (Tech, Product, Startup) and dense embeddings.
*   **Mechanism:** A cross-attention layer aligns the candidate's functional features against the JD's required dimension weights.
*   **Advantage:** If a JD desperately needs "Product Shipping", a candidate with extreme technical depth but zero shipping history will score lower than a balanced candidate with proven delivery capabilities.

### 2.4 The Graph Diversity Engine (`ranking_pipeline.py`)
A mathematically perfect scoring model often yields a highly homogenous top 100 list (e.g., 90 engineers from Google with identical skillsets). 

Our `graph_diversity_select` algorithm operates on the top candidates:
*   It models candidates as nodes in a graph.
*   Edge weights are determined by skill and company overlap (calculated via Jaccard similarity and shared company history).
*   During selection, when a candidate is added to the final shortlist, all unselected candidates with high similarity to the chosen candidate receive a penalty to their score.
*   **Result:** A maximally diverse shortlist covering different companies, educational tiers, and unique skill combinations, avoiding the "monoculture trap."

## 3. Fairness and Bias Audit
We implemented an automated fairness audit (`fairness_audit.py`) to measure demographic and pedigree parity.
Our analysis shows:
*   **Pedigree Independence:** The model does not aggressively filter out Tier-2/3 candidates. It successfully promotes candidates with high 'Product Shipping' and 'Startup Fit' scores regardless of their university.
*   **Counterfactual Robustness:** Altering a candidate's university tier or geographic origin while maintaining their GitHub/Product metrics results in negligible score variance (`< 0.05`). 
*   **Diversity Impact:** The Graph Diversity Engine actively forces the inclusion of non-traditional backgrounds by penalizing over-represented archetypes in the top quartile.

## 4. The Interactive 'Killer Demo'
To visualize this multi-agent process, we built a React/Vite frontend powered by a FastAPI backend.
The demo allows a hiring manager to:
1. Input a raw Job Description.
2. View the resulting Top 100 shortlist in real-time.
3. Examine the specific Neural Dimension breakdown for each candidate.
4. Read the LLM-generated reasoning explaining *why* the AI Recruiter selected them.

## 5. Conclusion
By shifting the paradigm from *Information Retrieval* to *Agentic Reasoning*, we have built a system that recruits based on capability, potential, and objective alignment—solving the exact pain points that traditional Boolean and Vector search systems create in the talent market.

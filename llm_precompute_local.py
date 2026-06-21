"""
llm_precompute_local.py — Generate LLM features using a LOCAL model on CPU.
Downloads Qwen2.5-1.5B-Instruct (~3GB) and runs it on your CPU.

Expected runtime: ~10-14 hours for 1,500 candidates.
Run overnight. It saves checkpoints every 50 candidates.

Requires: transformers, torch, accelerate
  pip install transformers torch accelerate
"""

import os
import json
import pickle
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path(__file__).parent

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────

# Model: Qwen2.5-1.5B-Instruct
# - ~3GB download
# - Good at structured JSON output
# - Fits in 16GB RAM easily
# - Reasonable speed on CPU
LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# How many candidates to process
# 1,500 = ~10-14 hours on CPU (overnight run)
# 500  = ~3-5 hours (faster test)
# 100  = ~45-60 minutes (quick test)
MAX_BULK_CANDIDATES = 1500

# Premium reasoning for top N
MAX_PREMIUM_REASONING = 200

# Save checkpoint every N candidates
CHECKPOINT_INTERVAL = 50

# Device: "cpu" only (challenge constraint)
DEVICE = "cpu"

# Model cache directory (saves to project folder)
CACHE_DIR = str(BASE / "models_cache")

# ──────────────────────────────────────────────────────────────────────
# MODEL LOADING (happens once, cached)
# ──────────────────────────────────────────────────────────────────────

_model = None
_tokenizer = None


def load_model():
    """Load model and tokenizer. First run downloads ~3GB."""
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    print(f"Loading {LOCAL_MODEL}...")
    print(f"  This downloads ~3GB on first run. Cached afterwards.")
    print(f"  Cache dir: {CACHE_DIR}")

    start = time.time()

    _tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL,
        trust_remote_code=True,
        cache_dir=CACHE_DIR
    )

    # Load in float16 to save RAM (16GB machine)
    _model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        device_map="cpu",  # Force CPU
        low_cpu_mem_usage=True,
    )
    _model.eval()

    elapsed = time.time() - start
    print(f"  Model loaded in {elapsed:.1f}s")

    # Estimate RAM usage
    param_bytes = sum(p.numel() * p.element_size() for p in _model.parameters())
    print(f"  RAM used: ~{param_bytes / 1e9:.1f} GB")

    return _model, _tokenizer


# ──────────────────────────────────────────────────────────────────────
# LOCAL GENERATION
# ──────────────────────────────────────────────────────────────────────

def local_generate(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.05,
) -> Optional[str]:
    """Generate text using local model on CPU."""
    model, tokenizer = load_model()

    # Format for Qwen chat model
    messages = [
        {"role": "system", "content": "You are a senior recruiter. Output only structured JSON or plain text as requested."},
        {"role": "user", "content": prompt}
    ]

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        # Fallback
        text = f"System: You are a senior recruiter.\nUser: {prompt}\nAssistant:"

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            top_p=0.9,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only new tokens
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()

    return response


# ──────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATES (optimized for 1.5B model)
# ──────────────────────────────────────────────────────────────────────

# NOTE: 1.5B models need SHORT, STRUCTURED prompts. Less context = better JSON.

JD_UNDERSTANDING_PROMPT = """Extract a JSON hiring model from this JD. Return ONLY JSON.

JD:
{jd_text}

JSON schema:
{{
  "role_summary": "brief summary",
  "dimensions": {{
    "technical_depth": {{
      "must_have": ["skills"],
      "must_avoid": ["wrong signals"],
      "weight": 0.30
    }},
    "product_shipping": {{
      "must_have": ["shipping signals"],
      "must_avoid": ["research-only"],
      "weight": 0.25
    }},
    "startup_fitness": {{
      "must_have": ["startup signals"],
      "must_avoid": ["big-corp comfort"],
      "weight": 0.20
    }},
    "experience_range": {{
      "min_years": 5,
      "max_years": 9,
      "ideal": 7,
      "weight": 0.15
    }},
    "location": {{
      "preferred": ["Pune", "Noida"],
      "relocation_ok": true,
      "weight": 0.10
    }}
  }},
  "implicit_culture": "culture in 1 sentence",
  "disqualifying_patterns": ["patterns to avoid"]
}}

JSON:"""


CANDIDATE_ASSESSMENT_PROMPT = """Rate candidate for: {role_summary}

Return ONLY JSON with scores 0.0-1.0:
{{
  "technical_depth": {{"score": 0.0, "evidence": "..."}},
  "product_shipping": {{"score": 0.0, "evidence": "..."}},
  "startup_fitness": {{"score": 0.0, "evidence": "..."}},
  "experience_alignment": {{"score": 0.0, "evidence": "..."}},
  "location_fit": {{"score": 0.0, "evidence": "..."}},
  "overall_fit": 0.0,
  "red_flags": ["concerns"],
  "unique_selling_points": ["special traits"]
}}

Candidate: {current_title} at {current_company}, {years_exp} years
Career: {career_text}

JSON:"""


PREMIUM_REASONING_PROMPT = """Write 2-3 sentences as a recruiter assessing this candidate for: {role_summary}

Candidate: {current_title} at {current_company}, {years} years
Profile: {career_text}

Mention specific evidence, connect to role needs, be honest about concerns."""


# ──────────────────────────────────────────────────────────────────────
# JSON EXTRACTION (handles imperfect LLM output)
# ──────────────────────────────────────────────────────────────────────

def extract_json(text: str) -> Optional[dict]:
    """Extract JSON from LLM response, handling various formats."""
    if not text:
        return None

    # Try to find JSON block
    start = text.find('{')
    end = text.rfind('}') + 1

    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    # Try to find JSON array
    start = text.find('[')
    end = text.rfind(']') + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    return None


# ──────────────────────────────────────────────────────────────────────
# STEP 1: JD UNDERSTANDING
# ──────────────────────────────────────────────────────────────────────

def generate_jd_understanding(jd_text: str) -> dict:
    prompt = JD_UNDERSTANDING_PROMPT.format(jd_text=jd_text[:3000])
    print(f"  Generating JD understanding with local {LOCAL_MODEL}...")

    start = time.time()
    response = local_generate(prompt, max_tokens=1024, temperature=0.05)
    elapsed = time.time() - start
    print(f"    Generated in {elapsed:.1f}s")

    if not response:
        print("    Failed. Using fallback.")
        return _fallback_jd_understanding()

    result = extract_json(response)
    if result:
        print(f"    JD understanding extracted")
        return result

    print(f"    JSON parse failed. Using fallback.")
    print(f"    Raw: {response[:300]}")
    return _fallback_jd_understanding()


def _fallback_jd_understanding() -> dict:
    return {
        "role_summary": "Senior AI Engineer with IR/ML focus at Series A startup",
        "dimensions": {
            "technical_depth": {
                "must_have": ["embeddings", "retrieval", "ranking", "LLMs"],
                "nice_to_have": ["vector search", "MLOps", "fine-tuning"],
                "must_avoid": ["pure research", "theoretical only"],
                "weight": 0.30
            },
            "product_shipping": {
                "must_have": ["production deployment", "shipped", "MVP"],
                "must_avoid": ["research paper only", "no production"],
                "weight": 0.25
            },
            "startup_fitness": {
                "must_have": ["startup", "early-stage", "0-to-1", "founding"],
                "must_avoid": ["Google", "Meta", "well-scoped", "defined ladder"],
                "weight": 0.20
            },
            "experience_range": {"min_years": 5, "max_years": 9, "ideal": 7, "weight": 0.15},
            "location": {"preferred": ["Pune", "Noida"], "relocation_ok": True, "weight": 0.10}
        },
        "implicit_culture": "Tilt toward shipper over researcher. Comfortable with ambiguity.",
        "disqualifying_patterns": ["consulting-only", "title-chaser", "keyword-stuffer"]
    }


# ──────────────────────────────────────────────────────────────────────
# STEP 2: BULK CANDIDATE ASSESSMENT
# ──────────────────────────────────────────────────────────────────────

def assess_single_candidate(meta: dict, jd_understanding: dict) -> Optional[dict]:
    """Assess one candidate with the local LLM."""
    career_text = meta.get("doc_text", "")[:1200]  # Shorter for speed

    prompt = CANDIDATE_ASSESSMENT_PROMPT.format(
        role_summary=jd_understanding.get("role_summary", "Senior AI Engineer"),
        current_title=meta.get("current_title", "Unknown"),
        current_company=meta.get("current_company", "Unknown"),
        years_exp=meta.get("years_exp", 0),
        career_text=career_text
    )

    response = local_generate(prompt, max_tokens=512, temperature=0.05)
    if not response:
        return None

    return extract_json(response)


def run_bulk_assessment(metadata: list, jd_understanding: dict, max_candidates: int = 1500):
    """Process candidates in bulk."""
    # Select top candidates by heuristic
    eligible = []
    for m in metadata:
        if m.get("honeypot") or m.get("wrong_title"):
            continue
        score = (
            float(m.get("ml_signal_count", 0)) * 10 +
            float(m.get("years_exp", 0)) * 2 +
            float(m.get("location_score", 0))
        )
        eligible.append((score, m))

    eligible.sort(reverse=True, key=lambda x: x[0])
    selected = [m for _, m in eligible[:max_candidates]]

    print(f"\nProcessing {len(selected)} candidates with local {LOCAL_MODEL}...")
    print(f"  Estimated time: ~{len(selected) * 25 / 3600:.1f} hours (CPU, ~25s/candidate)")
    print(f"  This will run overnight. Checkpoints saved every {CHECKPOINT_INTERVAL}.")
    print(f"  Press Ctrl+C to interrupt (checkpoint will be saved).")

    processed = 0
    failed = 0
    total_time = 0.0

    try:
        for i, meta in enumerate(selected):
            start = time.time()
            assessment = assess_single_candidate(meta, jd_understanding)
            elapsed = time.time() - start
            total_time += elapsed

            if assessment:
                meta["llm_assessment"] = assessment
                meta["llm_overall_fit"] = float(assessment.get("overall_fit", 0.5))
                processed += 1
            else:
                meta["llm_assessment"] = None
                meta["llm_overall_fit"] = 0.5
                failed += 1

            # Progress report
            if (i + 1) % 10 == 0:
                avg_time = total_time / (i + 1)
                remaining = (len(selected) - i - 1) * avg_time
                print(f"  {i+1}/{len(selected)} | avg {avg_time:.1f}s | "
                      f"ETA {remaining/3600:.1f}h | failed: {failed}")

            # Checkpoint
            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                _save_checkpoint(metadata, f"checkpoint_local_{i+1}.pkl")

    except KeyboardInterrupt:
        print(f"\n  Interrupted by user. Saving checkpoint...")
        _save_checkpoint(metadata, f"checkpoint_local_interrupted.pkl")
        print(f"  Resume later by loading this checkpoint.")
        return metadata

    print(f"\n  Bulk assessment complete: {processed} success, {failed} failed")
    return metadata


# ──────────────────────────────────────────────────────────────────────
# STEP 3: PREMIUM REASONING
# ──────────────────────────────────────────────────────────────────────

def generate_premium_reasoning(meta: dict, jd_understanding: dict) -> str:
    """Generate reasoning for top candidates."""
    prompt = PREMIUM_REASONING_PROMPT.format(
        role_summary=jd_understanding.get("role_summary", "Senior AI Engineer"),
        current_title=meta.get("current_title", "Unknown"),
        current_company=meta.get("current_company", "Unknown"),
        years=meta.get("years_exp", 0),
        career_text=meta.get("doc_text", "")[:1500]
    )

    response = local_generate(prompt, max_tokens=256, temperature=0.2)
    return response or "No LLM reasoning available."


def run_premium_reasoning(metadata: list, jd_understanding: dict, top_n: int = 200):
    """Generate reasoning for top candidates."""
    ranked = sorted(
        [m for m in metadata if m.get("llm_assessment")],
        key=lambda x: x.get("llm_overall_fit", 0),
        reverse=True
    )

    selected = ranked[:top_n]
    print(f"\nGenerating premium reasoning for top {len(selected)}...")
    print(f"  Estimated time: ~{len(selected) * 15 / 60:.0f} minutes")

    for i, meta in enumerate(selected):
        reasoning = generate_premium_reasoning(meta, jd_understanding)
        meta["llm_reasoning"] = reasoning

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(selected)} done")
            _save_checkpoint(metadata, f"checkpoint_local_premium_{i+1}.pkl")

    print(f"  Premium reasoning complete")
    return metadata


# ──────────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────────

def _save_checkpoint(metadata: list, filename: str):
    """Save checkpoint to avoid losing work."""
    path = BASE / filename
    with open(path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"  Checkpoint saved: {filename}")


def resume_from_checkpoint(checkpoint_path: Path) -> Optional[list]:
    """Resume from a checkpoint file."""
    if not checkpoint_path.exists():
        return None

    print(f"Resuming from checkpoint: {checkpoint_path.name}")
    with open(checkpoint_path, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("TALENT TERMINAL — LLM PRECOMPUTATION (LOCAL CPU)")
    print("=" * 60)
    print(f"Model: {LOCAL_MODEL}")
    print(f"Device: {DEVICE}")
    print(f"Candidates: {MAX_BULK_CANDIDATES}")
    print(f"Reasoning: top {MAX_PREMIUM_REASONING}")
    print("\nNOTE: First run downloads ~3GB of model weights.")
    print("After download, runs completely offline.\n")

    # Check for existing checkpoints to resume
    checkpoint_files = sorted(BASE.glob("checkpoint_local_*.pkl"))
    if checkpoint_files:
        latest = checkpoint_files[-1]
        print(f"Found checkpoint: {latest.name}")
        response = input("Resume from checkpoint? (y/n): ").strip().lower()
        if response == 'y':
            metadata = resume_from_checkpoint(latest)
            if metadata:
                print(f"  Resumed with {len(metadata):,} candidates\n")
            else:
                print("  Failed to load checkpoint. Starting fresh.")
                metadata = None
        else:
            metadata = None
    else:
        metadata = None

    # Load fresh data if not resuming
    if metadata is None:
        meta_path = BASE / "candidate_meta.pkl"
        if not meta_path.exists():
            print(f"ERROR: {meta_path} not found. Run precompute.py first.")
            return

        print(f"Loading {meta_path}...")
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        print(f"  Loaded {len(metadata):,} candidates")

    # Load JD
    jd_path = BASE / "job_desc.txt"
    if not jd_path.exists():
        print(f"ERROR: {jd_path} not found.")
        return

    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read()
    print(f"  JD loaded ({len(jd_text)} chars)")

    # Check if JD understanding already exists
    jd_out = BASE / "jd_understanding.json"
    if jd_out.exists():
        print(f"\nFound existing JD understanding: {jd_out.name}")
        with open(jd_out, "r", encoding="utf-8") as f:
            jd_understanding = json.load(f)
        print(f"  Loaded JD understanding")
    else:
        # Step 1: JD Understanding
        print("\n" + "=" * 60)
        print("STEP 1: JD Understanding")
        print("=" * 60)
        jd_understanding = generate_jd_understanding(jd_text)

        with open(jd_out, "w", encoding="utf-8") as f:
            json.dump(jd_understanding, f, indent=2)
        print(f"  Saved to {jd_out.name}")

    # Step 2: Bulk Assessment (skip if already done)
    n_with_assessment = sum(1 for m in metadata if m.get("llm_assessment"))
    if n_with_assessment >= MAX_BULK_CANDIDATES * 0.9:
        print(f"\n  Skipping bulk assessment ({n_with_assessment} already done)")
    else:
        print("\n" + "=" * 60)
        print("STEP 2: Bulk Candidate Assessment")
        print("=" * 60)
        metadata = run_bulk_assessment(metadata, jd_understanding, MAX_BULK_CANDIDATES)

    # Step 3: Premium Reasoning
    n_with_reasoning = sum(1 for m in metadata if m.get("llm_reasoning"))
    if n_with_reasoning >= MAX_PREMIUM_REASONING * 0.9:
        print(f"\n  Skipping premium reasoning ({n_with_reasoning} already done)")
    else:
        print("\n" + "=" * 60)
        print("STEP 3: Premium Reasoning Generation")
        print("=" * 60)
        metadata = run_premium_reasoning(metadata, jd_understanding, MAX_PREMIUM_REASONING)

    # Final save
    print("\n" + "=" * 60)
    print("SAVING FINAL RESULTS")
    print("=" * 60)
    with open(BASE / "candidate_meta.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print(f"  Saved to candidate_meta.pkl")

    n_with_llm = sum(1 for m in metadata if m.get("llm_assessment"))
    n_with_reasoning = sum(1 for m in metadata if m.get("llm_reasoning"))
    print(f"\n  Candidates with LLM assessment: {n_with_llm:,}")
    print(f"  Candidates with premium reasoning: {n_with_reasoning:,}")
    print("\nDone! Now run: python rank.py")


if __name__ == "__main__":
    main()

import pickle
import random
from pathlib import Path
from tqdm import tqdm

BASE = Path(__file__).parent
META_PATH = BASE / "candidate_meta.pkl"

def generate_mock_assessment(meta):
    """
    Generate realistic synthetic LLM assessments based on candidate metadata.
    This simulates an LLM parsing the candidate profile.
    """
    yrs = meta.get("years_exp", 0) or 0
    ml = meta.get("ml_signal_count", 0) or 0
    loc = meta.get("location_score", 0) or 0
    
    # 1. Technical Depth
    tech_score = min(0.3 + (ml * 0.2), 0.95)
    if meta.get("github_score", 0) > 50:
        tech_score += 0.1
    tech_score = min(tech_score, 1.0)
    
    # 2. Product Shipping
    product_score = 0.5
    if meta.get("has_product_company"):
        product_score += 0.3
    if yrs > 4:
        product_score += 0.1
    product_score = min(product_score, 0.95)
    
    # 3. Startup Fitness
    startup_score = 0.4
    if meta.get("consulting_only"):
        startup_score -= 0.3
    if yrs < 10:
        startup_score += 0.2
    
    # 4. Experience Alignment
    exp_score = 1.0 - abs(7 - yrs) / 7.0
    exp_score = max(0.1, min(exp_score, 0.95))
    
    # 5. Location Fit
    loc_score = min((loc / 5.0) + 0.2, 0.95) if meta.get("willing_to_relocate") else (loc / 5.0)

    assessment = {
        "technical_depth": {
            "score": round(tech_score, 2),
            "evidence": f"Shows {ml} strong ML signals and {yrs:.1f} years of experience."
        },
        "product_shipping": {
            "score": round(product_score, 2),
            "evidence": "Product company background." if meta.get("has_product_company") else "No clear product track record."
        },
        "startup_fitness": {
            "score": round(startup_score, 2),
            "evidence": "Consulting background might be a culture risk." if meta.get("consulting_only") else "Good tenure band for startup agility."
        },
        "experience_alignment": {
            "score": round(exp_score, 2),
            "evidence": f"{yrs:.1f} years falls within acceptable bounds."
        },
        "location_fit": {
            "score": round(loc_score, 2),
            "evidence": f"Location score {loc}/5. Relocation: {meta.get('willing_to_relocate', False)}."
        }
    }
    
    red_flags = []
    if meta.get("consulting_only"):
        red_flags.append("Entire career in consulting/services.")
    if meta.get("title_chaser"):
        red_flags.append("Frequent job hops with rapid title inflation.")
    
    usps = []
    if tech_score > 0.85 and product_score > 0.8:
        usps.append("Rare blend of deep technical ML expertise AND strong product shipping track record.")
        
    reasoning = (
        f"This candidate demonstrates a {'strong' if tech_score > 0.7 else 'moderate'} technical depth "
        f"with {yrs:.1f} years of experience. "
    )
    if product_score > 0.7:
        reasoning += "They possess the critical 'shipper' mentality required for a Series A startup. "
    else:
        reasoning += "Their ability to ship products rapidly in an unstructured environment is unproven. "
        
    if red_flags:
        reasoning += f"However, concerns include: {red_flags[0]} "
    
    return assessment, red_flags, usps, reasoning

def main():
    if not META_PATH.exists():
        print(f"Error: {META_PATH} not found.")
        return
        
    print("Loading candidate_meta.pkl...")
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
        
    print(f"Generating LLM assessments for {len(metadata)} candidates...")
    for meta in tqdm(metadata):
        assess, red_flags, usps, reasoning = generate_mock_assessment(meta)
        meta["llm_assessment"] = assess
        meta["llm_red_flags"] = red_flags
        meta["llm_unique_selling_points"] = usps
        meta["llm_reasoning"] = reasoning
        
    print("Saving updated candidate_meta.pkl...")
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)
        
    print("Done! LLM assessments added.")

if __name__ == "__main__":
    main()

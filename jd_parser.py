import json
from pathlib import Path

def load_jd_understanding():
    """Loads the Recruiter's Mental Model from jd_understanding.json."""
    jd_path = Path(__file__).parent / "jd_understanding.json"
    if not jd_path.exists():
        return None
    with open(jd_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_dimension_weights():
    """Returns a dictionary of dimension weights or defaults."""
    data = load_jd_understanding()
    if not data or "role_dimensions" not in data:
        return {
            "technical_depth": 0.30,
            "product_shipping": 0.25,
            "startup_fitness": 0.20,
            "experience_band": 0.15,
            "location": 0.10
        }
    dims = data["role_dimensions"]
    return {
        "technical_depth": dims.get("technical_depth", {}).get("weight", 0.30),
        "product_shipping": dims.get("product_shipping", {}).get("weight", 0.25),
        "startup_fitness": dims.get("startup_fitness", {}).get("weight", 0.20),
        "experience_band": dims.get("experience_band", {}).get("weight", 0.15),
        "location": dims.get("location", {}).get("weight", 0.10)
    }

if __name__ == "__main__":
    weights = get_dimension_weights()
    print("JD Dimension Weights:", weights)

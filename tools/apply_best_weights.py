"""Apply weights from best_weights.json to config.yaml (without re-tuning)."""

import json
from pathlib import Path

import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ranking_pipeline import RankingConfig

BASE = Path(__file__).parent.parent


def main():
    src = BASE / "best_weights.json"
    if not src.exists():
        raise FileNotFoundError("best_weights.json not found. Run tune_weights.py first.")

    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    cfg = RankingConfig(
        w_semantic=data["w_semantic"],
        w_experience=data["w_experience"],
        w_company=data["w_company"],
        w_ml=data["w_ml"],
        w_behavioral=data["w_behavioral"],
        w_saved=data["w_saved"],
        w_github=data["w_github"],
        w_education=data["w_education"],
        w_location=data["w_location"],
        w_ml_ratio=data["w_ml_ratio"],
        rrf_k=data["rrf_k"],
        rrf_weight=data["rrf_weight"],
    )

    with open(BASE / "config.yaml", "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    yaml_data["weights"].update({
        "W_SEMANTIC": round(cfg.w_semantic, 2),
        "W_EXPERIENCE": round(cfg.w_experience, 2),
        "W_COMPANY_TYPE": round(cfg.w_company, 2),
        "W_ML_SIGNALS": round(cfg.w_ml, 2),
        "W_BEHAVIORAL": round(cfg.w_behavioral, 2),
        "W_SAVED_RECRUITERS": round(cfg.w_saved, 2),
        "W_GITHUB": round(cfg.w_github, 2),
        "W_EDUCATION": round(cfg.w_education, 2),
        "W_LOCATION": round(cfg.w_location, 2),
        "W_ML_RATIO": round(cfg.w_ml_ratio, 2),
        "RRF_K": int(cfg.rrf_k),
        "RRF_WEIGHT": round(cfg.rrf_weight, 2),
    })

    with open(BASE / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print("Applied weights from best_weights.json to config.yaml")
    print("Re-run: python rank.py")

    from core_scoring import reload_config
    reload_config()


if __name__ == "__main__":
    main()

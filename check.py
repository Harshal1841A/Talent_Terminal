import pandas as pd
import pickle

df = pd.read_csv('submission.csv')
with open('candidate_db.pkl', 'rb') as f:
    db = pickle.load(f)

meta_dict = {m['candidate_id']: m for m in db['metadata']}

print("--- TOP 5 CANDIDATES ---")
for i, row in df.head(5).iterrows():
    c_id = row['candidate_id']
    score = row['score']
    m = meta_dict.get(c_id, {})
    
    title = m.get('current_title', 'UNKNOWN')
    exp = m.get('years_exp', 'N/A')
    skills = m.get('skill_count', 'N/A')
    tier = m.get('tier', 'None')
    jd_bonus = m.get('jd_term_bonus', '0')
    lgbm = m.get('lgbm_score', '0')
    
    print(f"{i+1}. {c_id}")
    print(f"    Total Score : {score:.2f}")
    print(f"    Title       : {title}")
    print(f"    Experience  : {exp} years")
    print(f"    Skills      : {skills}")
    print(f"    Tier        : {tier}")
    print(f"    JD Match    : {jd_bonus}")
    print(f"    LGBM Score  : {lgbm}")
    print()

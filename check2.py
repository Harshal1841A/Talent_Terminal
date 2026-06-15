import pandas as pd
df = pd.read_csv('submission.csv')
print('--- TOP 5 REASONING ---')
for i, row in df.head(5).iterrows():
    print(f"{i+1}. {row['candidate_id']} | Score: {row['score']:.2f}")
    print(f"   Reasoning: {row['reasoning']}\n")

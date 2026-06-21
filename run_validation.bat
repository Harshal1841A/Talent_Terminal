@echo off
cd /d "c:\Users\Admin\OneDrive\Desktop\Talent Terminal\[PUB] India_runs_data_and_ai_challenge\Talent Terminal"
python -u rank.py
python -u validate_submission.py "Team Rocket.csv"
python -u eval_metrics.py
python -u fairness_audit.py

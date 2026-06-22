import os
from huggingface_hub import HfApi

print("Starting direct upload to Hugging Face Spaces...")
try:
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.upload_folder(
        folder_path=r"d:\Talent Terminal\[PUB] India_runs_data_and_ai_challenge\Talent Terminal",
        repo_id="NeuralHU/Talent_Terminal",
        repo_type="space",
        ignore_patterns=[
            "candidate_db.pkl",
            "candidates.jsonl",
            "frontend.zip",
            ".git/*",
            ".agents/*",
            "models/*",
            "*.csv",
            "frontend/node_modules/*",
            "__pycache__/*",
            "tools/*",
            "dashboard.html",
            "dashboard_template.html",
            "test_rank.py",
            "upload_hf.py",
            "upload_missing_files.py",
            "upload_small.py"
        ]
    )
    
    # Re-upload ignored massive files explicitly so the sync doesn't delete them
    print("Forcing upload of large data files ignored by .gitignore...")
    base = r"d:\Talent Terminal\[PUB] India_runs_data_and_ai_challenge\Talent Terminal"
    for file_name in ["candidate_meta.pkl", "faiss_index.bin", "lgbm_reranker.pkl"]:
        print(f"Uploading {file_name}...")
        api.upload_file(
            path_or_fileobj=os.path.join(base, file_name),
            path_in_repo=file_name,
            repo_id="NeuralHU/Talent_Terminal",
            repo_type="space"
        )
    print("Upload completed successfully!")
except Exception as e:
    print(f"Error during upload: {e}")

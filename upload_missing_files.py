import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ.get("HF_TOKEN"))

repo_id = "NeuralHU/Talent_Terminal"
base_dir = r"d:\Talent Terminal\[PUB] India_runs_data_and_ai_challenge\Talent Terminal"

files_to_upload = [
    "candidate_meta.pkl",
    "faiss_index.bin",
    "lgbm_reranker.pkl"
]

for file_name in files_to_upload:
    print(f"Uploading {file_name}...")
    api.upload_file(
        path_or_fileobj=os.path.join(base_dir, file_name),
        path_in_repo=file_name,
        repo_id=repo_id,
        repo_type="space"
    )

print("Uploading bm25_index folder...")
api.upload_folder(
    folder_path=os.path.join(base_dir, "bm25_index"),
    path_in_repo="bm25_index",
    repo_id=repo_id,
    repo_type="space"
)

print("All missing models and indices uploaded successfully!")

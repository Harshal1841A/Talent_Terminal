import os
from huggingface_hub import HfApi

print("Uploading updated files...")
api = HfApi(token=os.environ.get("HF_TOKEN"))

base = r"d:\Talent Terminal\[PUB] India_runs_data_and_ai_challenge\Talent Terminal"
files_to_upload = ["api_server.py", "Dockerfile", "requirements.txt"]

for f in files_to_upload:
    print(f"Uploading {f}...")
    api.upload_file(
        path_or_fileobj=os.path.join(base, f),
        path_in_repo=f,
        repo_id="NeuralHU/Talent_Terminal",
        repo_type="space"
    )
print("Done!")

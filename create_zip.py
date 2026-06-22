import zipfile
import os

os.chdir(r"d:\Talent Terminal\[PUB] India_runs_data_and_ai_challenge\Talent Terminal")

exclude_dirs = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', 'models', '.pytest_cache'}
exclude_exts = {'.bin', '.pkl', '.safetensors', '.h5', '.index', '.onnx', '.zip', '.pt'}

with zipfile.ZipFile('talent_terminal_source.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if any(file.endswith(ext) for ext in exclude_exts):
                continue
            file_path = os.path.join(root, file)
            zipf.write(file_path, os.path.relpath(file_path, '.'))
print('ZIP created successfully!')

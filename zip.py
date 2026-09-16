import os
import sys
import zipfile

OUTPUT_ZIP = "Malu.zip"
EXCLUDE_DIRS = {"__pycache__", ".git", "checkpoints", ".ipynb_checkpoints", "data", ".venv", "venv"}
EXCLUDE_FILES = {"Malu.zip", "zip.py", "make_zip.py"}
EXCLUDE_EXT = {".pyc", ".pyo"}

def should_skip(rel_path):
    parts = rel_path.replace("\\", "/").split("/")
    if any(p in EXCLUDE_DIRS for p in parts):
        return True
    name = parts[-1]
    if name in EXCLUDE_FILES:
        return True
    if any(name.endswith(ext) for ext in EXCLUDE_EXT):
        return True
    return False

def make_zip():
    project_root = os.path.dirname(os.path.abspath(__file__))
    project_name = os.path.basename(project_root)
    zip_path = os.path.join(project_root, OUTPUT_ZIP)

    print(f"Projeto: {project_root}")
    print(f"Nome:    {project_name}")
    print(f"Saida:   {zip_path}")

    if os.path.exists(zip_path):
        os.remove(zip_path)

    count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(project_root):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, project_root)
                arcname = os.path.join(project_name, rel)
                if should_skip(arcname):
                    continue
                z.write(full, arcname)
                count += 1
                print(f"  + {arcname}")

    if count == 0:
        print("ERRO: nenhum arquivo foi adicionado. ZIP vazio.")
        sys.exit(1)

    size = os.path.getsize(zip_path)
    print(f"\nOK: {count} arquivos, {size} bytes")

if __name__ == "__main__":
    make_zip()

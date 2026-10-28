"""Garde-Fil : vérifie que tous les prérequis de l'étape 0 sont en place."""
import importlib
import shutil
import subprocess
import sys

results = []


def check(label: str, ok: bool, detail: str = "") -> None:
    """Affiche une ligne OK/KO et mémorise le résultat."""
    print(f"{'✅' if ok else '❌'} {label} {detail}")
    results.append(ok)


# 1. Python du venv (et pas celui du système)
check("Python >= 3.10", sys.version_info >= (3, 10), sys.version.split()[0])
check("venv actif", sys.prefix != sys.base_prefix, f"({sys.prefix})")

# 2. Binaires système appelés par le pipeline
for binary in ("ffmpeg", "ffprobe", "tesseract"):
    check(f"binaire {binary}", shutil.which(binary) is not None)

# 3. ffmpeg doit savoir incruster du texte (drawtext) et encoder en H.264 (libx264)
if shutil.which("ffmpeg"):
    filters = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                             capture_output=True, text=True).stdout
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                              capture_output=True, text=True).stdout
    check("ffmpeg filtre drawtext", " drawtext " in filters)
    check("ffmpeg encodeur libx264", " libx264 " in encoders)

# 4. Tesseract doit disposer du pack français
if shutil.which("tesseract"):
    out = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True)
    langs = (out.stdout + out.stderr).split()  # selon la version, la liste sort sur l'un ou l'autre
    check("Tesseract langue 'fra'", "fra" in langs)

# 5. Modules Python (nom d'import, pas nom du paquet pip)
for module in ("faster_whisper", "pytesseract", "PIL", "sentence_transformers",
               "sklearn", "datasets", "pandas", "streamlit", "yaml"):
    try:
        importlib.import_module(module)
        check(f"module {module}", True)
    except ImportError as err:
        check(f"module {module}", False, f"({err})")

# 6. Torch + accélération GPU Apple (MPS) : utile mais pas obligatoire
try:
    import torch
    check("torch", True, f"{torch.__version__} | GPU Apple (MPS) : {torch.backends.mps.is_available()}")
except ImportError:
    check("torch", False)

print("\nEnvironnement prêt." if all(results) else "\nCorrige les ❌ avant la phase 1.")
sys.exit(0 if all(results) else 1)

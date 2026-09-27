import json
from pathlib import Path

nb_path = Path("notebooks/stage1_colab_runner.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code" and any("DRIVE_RAW_TRAIN" in line for line in cell["source"]):
        cell["source"] = [
            "from google.colab import drive\n",
            "drive.mount('/content/drive')\n",
            "\n",
            "# Base Google Drive directories\n",
            "DRIVE_BASE = Path('/content/drive/MyDrive/entity-resolution')\n",
            "DRIVE_RAW_TRAIN = DRIVE_BASE / 'data/raw/train'\n",
            "DRIVE_RAW_TEST = DRIVE_BASE / 'data/raw/test'\n",
            "DRIVE_PROC_TRAIN = DRIVE_BASE / 'data/processed/train'\n",
            "DRIVE_PROC_TEST = DRIVE_BASE / 'data/processed/test'\n",
            "DRIVE_CANDIDATES = DRIVE_BASE / 'data/candidates'\n",
            "\n",
            "# Create local & Drive directories\n",
            "!mkdir -p data/raw/train data/raw/test data/processed/train data/processed/test data/candidates data/cache\n",
            "DRIVE_PROC_TRAIN.mkdir(parents=True, exist_ok=True)\n",
            "DRIVE_PROC_TEST.mkdir(parents=True, exist_ok=True)\n",
            "DRIVE_CANDIDATES.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "# Copy raw datasets from Drive if available\n",
            "if DRIVE_RAW_TRAIN.exists():\n",
            "    print(f\"Copying raw train datasets from {DRIVE_RAW_TRAIN}...\")\n",
            "    !cp -n \"{DRIVE_RAW_TRAIN}\"/*.tsv data/raw/train/ 2>/dev/null || true\n",
            "\n",
            "if DRIVE_RAW_TEST.exists():\n",
            "    print(f\"Copying raw test datasets from {DRIVE_RAW_TEST}...\")\n",
            "    !cp -n \"{DRIVE_RAW_TEST}\"/*.tsv data/raw/test/ 2>/dev/null || true\n",
            "\n",
            "# Copy preprocessed datasets from Drive if available (skips re-running preprocessing!)\n",
            "if DRIVE_PROC_TRAIN.exists():\n",
            "    print(f\"Copying preprocessed train datasets from {DRIVE_PROC_TRAIN}...\")\n",
            "    !cp -n \"{DRIVE_PROC_TRAIN}\"/*.tsv data/processed/train/ 2>/dev/null || true\n",
            "\n",
            "if DRIVE_PROC_TEST.exists():\n",
            "    print(f\"Copying preprocessed test datasets from {DRIVE_PROC_TEST}...\")\n",
            "    !cp -n \"{DRIVE_PROC_TEST}\"/*.tsv data/processed/test/ 2>/dev/null || true\n",
            "\n",
            "print(\"\\n--- Local Datasets Detected ---\")\n",
            "print(\"Raw Train (data/raw/train):\")\n",
            "for f in sorted(Path(\"data/raw/train\").glob(\"*.tsv\")):\n",
            "    print(f\"  - {f.name} ({f.stat().st_size / (1024 * 1024):.2f} MB)\")\n",
            "\n",
            "print(\"\\nProcessed Train (data/processed/train):\")\n",
            "proc_train = list(sorted(Path(\"data/processed/train\").glob(\"*.tsv\")))\n",
            "if proc_train:\n",
            "    for f in proc_train:\n",
            "        print(f\"  - {f.name} ({f.stat().st_size / (1024 * 1024):.2f} MB)\")\n",
            "else:\n",
            "    print(\"  (Empty - need to copy from Drive or run Step 4 Preprocessing)\")\n",
        ]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated stage1_colab_runner.ipynb")

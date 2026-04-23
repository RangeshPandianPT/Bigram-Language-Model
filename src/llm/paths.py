from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
ONNX_DIR = ARTIFACTS_DIR / "onnx"
TOKENIZER_DIR = ARTIFACTS_DIR / "tokenizer"

REPORTS_DIR = ROOT_DIR / "reports"

INPUT_TEXT_PATH = RAW_DATA_DIR / "input.txt"
TRAIN_BIN_PATH = PROCESSED_DATA_DIR / "train.bin"
VAL_BIN_PATH = PROCESSED_DATA_DIR / "val.bin"

CHAT_TRAIN_BIN_PATH = PROCESSED_DATA_DIR / "chat_train.bin"
CHAT_VAL_BIN_PATH = PROCESSED_DATA_DIR / "chat_val.bin"

TOKENIZER_PREFIX = TOKENIZER_DIR / "bpe"

MODEL_PATH = CHECKPOINTS_DIR / "model.pth"
MODEL_BEST_PATH = CHECKPOINTS_DIR / "model_best.pth"
LORA_WEIGHTS_PATH = CHECKPOINTS_DIR / "lora_weights.pth"
ONNX_MODEL_PATH = ONNX_DIR / "model.onnx"

EVALUATION_REPORT_PATH = REPORTS_DIR / "evaluation_report.md"
LEAKAGE_REPORT_PATH = REPORTS_DIR / "leakage_report.txt"


def ensure_project_dirs() -> None:
    for directory in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        CHECKPOINTS_DIR,
        ONNX_DIR,
        TOKENIZER_DIR,
        REPORTS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

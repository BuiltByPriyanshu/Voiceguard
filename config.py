"""Central config: paths, thresholds, device selection.

Keep every tunable here so src/ and backend/ never hardcode a magic number.
"""
import os

# --- Paths -------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
DEMO_CLIPS_DIR = os.path.join(ROOT_DIR, "demo_clips")
CHECKPOINT_PATH = os.path.join(ARTIFACTS_DIR, "model.pth")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")

# Kaggle-attached dataset roots (adjust slugs once datasets are attached
# in the Kaggle notebook -- see notebooks/train_kaggle.ipynb).
KAGGLE_INPUT_DIR = "/kaggle/input"
ASVSPOOF2019_LA_DIR = os.path.join(KAGGLE_INPUT_DIR, "asvspoof-2019-la")
ASVSPOOF2021_DF_DIR = os.path.join(KAGGLE_INPUT_DIR, "asvspoof-2021-df")
IN_THE_WILD_DIR = os.path.join(KAGGLE_INPUT_DIR, "in-the-wild")

# --- Audio ---------------------------------------------------------------
SAMPLE_RATE = 16_000          # wav2vec2 / XLS-R required rate
CLIP_SECONDS = 4.0            # fixed-length training clip
WINDOW_SECONDS = 4.0          # streaming inference window
HOP_SECONDS = 1.0             # streaming inference hop
EMA_ALPHA = 0.6               # running risk score smoothing

# --- Model -----------------------------------------------------------
SSL_MODEL_NAME = "facebook/wav2vec2-xls-r-300m"   # multilingual front-end
# Lighter fallback if compute/time is tight tonight:
SSL_MODEL_NAME_LIGHT = "facebook/wav2vec2-base"
FREEZE_SSL = True
HIDDEN_DIM = 256
NUM_EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 1e-4

# --- Risk engine / alerting -------------------------------------------
# Risk score is 0-100. Below is the default alert threshold; can be
# overridden per-call via POST /config for "high-value transaction" scenarios.
DEFAULT_ALERT_THRESHOLD = 70
HIGH_VALUE_ALERT_THRESHOLD = 50   # stricter for sensitive transactions

ALERT_REASON = (
    "Likely synthetic voice — recommend call-back / MFA before approving."
)

# --- Labels -----------------------------------------------------------
LABEL_BONAFIDE = 0
LABEL_SPOOF = 1

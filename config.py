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
CLIP_SECONDS = 4.0            # fixed-length training clip (raw-audio path only)
# 2s window / 1s hop matches exactly how eminkorkut/deepfakevoice-wac2vec-4datasets
# built its embeddings (2s segments, 1s overlap) -- keeping inference-time
# pooling on the same timescale as training data reduces train/inference skew.
WINDOW_SECONDS = 2.0           # streaming inference window
HOP_SECONDS = 1.0              # streaming inference hop
EMA_ALPHA = 0.6                # running risk score smoothing

# --- Model -----------------------------------------------------------
# Matched to the precomputed-embedding dataset used for tonight's training
# (eminkorkut/deepfakevoice-wac2vec-4datasets: 768-dim wav2vec2 embeddings,
# 2s segments, mean-pooled) -- the head is trained on those embeddings, so
# inference must extract features the same way, hence this must stay
# wav2vec2-base (hidden_size=768), not XLS-R (1024) or wav2vec2-large (1024).
SSL_MODEL_NAME = "facebook/wav2vec2-base"
# Multilingual front-end -- NOT used tonight (would mismatch the precomputed
# training embeddings) but worth revisiting post-hackathon for Indian-accent
# robustness if retraining from raw audio becomes an option.
SSL_MODEL_NAME_XLSR = "facebook/wav2vec2-xls-r-300m"
EMBEDDING_DIM = 768   # matches the wav2vec2-base hidden size / precomputed embedding width
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

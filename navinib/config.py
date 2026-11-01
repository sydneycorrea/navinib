"""Shared settings. Everything Navinib uses is synthetic or public data."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "navinib.db"
CHROMA_PATH = ROOT / "chroma_db"

SEED = 42                 # fixed so every demo and evaluation run is the same world
DEMO_DATE = "2026-10-27"  # the scripted demo day (a Tuesday)
N_PROVIDERS = 40

# Placeholder drive-time model until the OSRM matrix lands: straight-line
# distance times a road factor, at an average urban speed.
ROAD_FACTOR = 1.3
AVG_SPEED_KMH = 40

# Provider scoring weights (sum to 1). Long-term memory tunes these later.
SCORE_WEIGHTS = {
    "product_use": 0.35,   # Product X and competitor volume, synthetic Part D-like
    "goal_gap": 0.30,      # calls still owed this month for the provider's tier
    "lab_alerts": 0.20,    # open alert count, provider level only
    "recency": 0.15,       # days since the last visit
}
CALL_GOALS = {"A": 4, "B": 2, "C": 1}   # calls per month by tier
CALL_CREDIT = {"face_to_face": 1.0, "phone": 0.5}   # how much each completed call counts toward goals
RECENCY_CAP_DAYS = 60                    # 60+ days since a visit scores as fully overdue

# Lab alerts show provider-level counts only; small counts are suppressed so a
# count can never point to an individual patient.
ALERT_SUPPRESS_BELOW = 5

# Route planning
VISIT_MIN = 30          # default length of an in-person call
PREP_BUFFER_MIN = 60    # time kept free before a booked call to prep and drive (Checkpoint 2.1)

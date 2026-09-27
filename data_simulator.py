"""
data_simulator.py
------------------
Synthetic data generation for "The Watcher" - Child Digital Activity &
Behavioral Safety Monitoring Dashboard.

This module produces realistic, fully synthetic activity streams so the
dashboard can be demoed and tested without any real user data or external
database connection.
"""

import random
import datetime as dt

import numpy as np
import pandas as pd

APP_CATEGORIES = ["Games", "Social Media", "Learning", "Messaging", "Video Streaming"]

APP_NAMES = {
    "Games": ["Fortnite", "Roblox", "Minecraft", "Among Us", "Call of Duty Mobile"],
    "Social Media": ["Instagram", "TikTok", "Snapchat", "X", "Discord"],
    "Learning": ["Khan Academy", "Duolingo", "Google Classroom", "Quizlet"],
    "Messaging": ["WhatsApp", "Messenger", "iMessage", "Discord DM"],
    "Video Streaming": ["YouTube", "Netflix", "Twitch"],
}

# Small curated phrase list used only to FLAG concerning messages in the
# synthetic "bullying" demo scenario. This is for pattern-matching /
# classification demo purposes only.
CONCERNING_PHRASES = [
    "kill yourself", "worthless", "hate you", "ugly loser", "nobody likes you",
    "stupid idiot", "kys", "you should die", "everyone hates you", "loser",
]

NEUTRAL_PHRASES = [
    "let's play after school", "did you finish the homework", "nice goal in the match",
    "see you tomorrow", "can you send me the notes", "that movie was great",
    "happy birthday!", "good luck on the test", "want to study together",
]

SEARCH_QUERIES_NORMAL = [
    "how photosynthesis works", "best soccer drills", "funny cat videos",
    "homework help algebra", "new movie trailers", "how to draw anime",
]

SEARCH_QUERIES_CONCERNING = [
    "how to hide app usage from parents", "is it normal to feel empty all the time",
    "how to stay awake all night gaming", "why does everyone hate me",
]

PRESETS = {
    "Normal Teen": "normal",
    "Addictive Gaming Pattern": "addictive_gaming",
    "Bullying/Toxic Keywords Detected": "bullying",
}


def _timestamp_series(n, minutes_step=30, end=None):
    end = end or dt.datetime.now()
    start = end - dt.timedelta(minutes=n * minutes_step)
    return [start + dt.timedelta(minutes=i * minutes_step) for i in range(n)]


def generate_screen_time_series(days=14, pattern="normal", seed=None):
    """Daily screen-time (hours) for the last `days` days."""
    rng = np.random.default_rng(seed)
    dates = [dt.date.today() - dt.timedelta(days=i) for i in range(days)][::-1]

    if pattern == "addictive_gaming":
        base = rng.normal(6.5, 0.8, size=days) + np.linspace(0, 2.5, days)
        base = np.clip(base, 3, 12)
    elif pattern == "bullying":
        base = np.clip(rng.normal(4.0, 1.0, size=days), 1, 8)
    else:
        base = np.clip(rng.normal(2.8, 0.6, size=days), 0.5, 5)

    return pd.DataFrame({"date": dates, "screen_time_hours": np.round(base, 2)})


def generate_app_usage(pattern="normal", seed=None):
    """Category usage share (%) for the pie chart."""
    rng = np.random.default_rng(seed)
    if pattern == "addictive_gaming":
        shares = {"Games": 62, "Social Media": 15, "Learning": 5, "Messaging": 10, "Video Streaming": 8}
    elif pattern == "bullying":
        shares = {"Games": 15, "Social Media": 30, "Learning": 10, "Messaging": 35, "Video Streaming": 10}
    else:
        shares = {"Games": 20, "Social Media": 20, "Learning": 25, "Messaging": 15, "Video Streaming": 20}

    noise = rng.normal(0, 2, size=len(shares))
    vals = np.clip(np.array(list(shares.values())) + noise, 2, None)
    vals = np.round(vals / vals.sum() * 100, 1)
    return pd.DataFrame({"category": list(shares.keys()), "usage_pct": vals})


def generate_message_logs(pattern="normal", n=25, seed=None):
    """Synthetic message log entries with sentiment / concern flags."""
    rng = random.Random(seed)
    timestamps = _timestamp_series(n, minutes_step=45)
    rows = []
    for ts in timestamps:
        is_concerning = pattern == "bullying" and rng.random() < 0.35
        text = rng.choice(CONCERNING_PHRASES) if is_concerning else rng.choice(NEUTRAL_PHRASES)
        sentiment = round(rng.uniform(-1.0, -0.4), 2) if is_concerning else round(rng.uniform(-0.1, 0.9), 2)
        rows.append({
            "timestamp": ts,
            "app": rng.choice(APP_NAMES["Messaging"] + APP_NAMES["Social Media"]),
            "message_snippet": text,
            "sentiment_score": sentiment,
            "concern_flag": is_concerning,
        })
    return pd.DataFrame(rows)


def generate_search_queries(pattern="normal", n=10, seed=None):
    rng = random.Random(seed)
    timestamps = _timestamp_series(n, minutes_step=90)
    concern_rate = 0.4 if pattern == "bullying" else 0.05
    rows = []
    for ts in timestamps:
        query = rng.choice(SEARCH_QUERIES_CONCERNING) if rng.random() < concern_rate else rng.choice(SEARCH_QUERIES_NORMAL)
        rows.append({"timestamp": ts, "query": query})
    return pd.DataFrame(rows)


def generate_activity_log(n=40, pattern="normal", seed=None):
    """Per-event app-session log used by the agentic scoring pipeline."""
    rng = random.Random(seed)
    timestamps = _timestamp_series(n, minutes_step=20)
    if pattern == "addictive_gaming":
        weights = [65, 15, 5, 10, 5]
    elif pattern == "bullying":
        weights = [15, 30, 10, 35, 10]
    else:
        weights = [20, 20, 25, 15, 20]

    rows = []
    for ts in timestamps:
        category = rng.choices(APP_CATEGORIES, weights=weights)[0]
        app = rng.choice(APP_NAMES[category])
        duration = max(2, int(rng.gauss(25, 12)))
        hour = ts.hour
        rows.append({
            "timestamp": ts,
            "app": app,
            "category": category,
            "duration_min": duration,
            "late_night": bool(hour >= 22 or hour <= 5),
        })
    return pd.DataFrame(rows)


def load_preset(name, seed=42):
    """Load a full mock dataset bundle for one of the named demo scenarios."""
    pattern = PRESETS.get(name, "normal")
    return {
        "preset_name": name,
        "pattern": pattern,
        "screen_time": generate_screen_time_series(pattern=pattern, seed=seed),
        "app_usage": generate_app_usage(pattern=pattern, seed=seed),
        "messages": generate_message_logs(pattern=pattern, seed=seed),
        "searches": generate_search_queries(pattern=pattern, seed=seed),
        "activity_log": generate_activity_log(pattern=pattern, seed=seed),
    }


def new_manual_entry(app, category, duration_min, message_text=None, timestamp=None):
    """Build a single manual activity-log row (used by the manual entry form)."""
    timestamp = timestamp or dt.datetime.now()
    hour = timestamp.hour
    return {
        "timestamp": timestamp,
        "app": app,
        "category": category,
        "duration_min": duration_min,
        "late_night": bool(hour >= 22 or hour <= 5),
        "message_snippet": message_text or "",
    }

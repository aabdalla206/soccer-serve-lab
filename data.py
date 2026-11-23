"""Load a StatsBomb open-data match and turn it into a long text timeline plus a question list.

StatsBomb open data is free but requires attribution if you publish anything built on it.
Default match: 2018 World Cup final (competition_id=43, season_id=3, match_id=8658).
"""
import warnings

import pandas as pd

warnings.filterwarnings("ignore")  # statsbombpy warns about missing credentials for open data

DEFAULT_MATCH_ID = 8658

# Event types worth keeping. Passes are kept only when they set up a shot or goal.
KEEP_TYPES = {
    "Shot", "Foul Committed", "Foul Won", "Substitution", "Bad Behaviour",
    "Goal Keeper", "Interception", "Clearance", "Dribble", "Block",
    "Half Start", "Half End", "Starting XI",
}

QUESTIONS = [
    "Who scored the goals, and in which minutes?",
    "Which team took more shots, and how many?",
    "Who took the most shots in the match?",
    "Summarize the first half in three sentences.",
    "Summarize the second half in three sentences.",
    "Which fouls happened in the last 20 minutes?",
    "List every substitution with the minute it happened.",
    "Were there any yellow or red cards? Who received them?",
    "Which goalkeeper actions stood out?",
    "Who created the most chances?",
    "Describe the build-up to the first goal.",
    "Which team looked more dangerous after the 60th minute?",
    "How many dribbles were attempted, and who attempted them?",
    "What were the key clearances late in the match?",
    "Which player appears most often in the timeline?",
    "Describe the final ten minutes.",
    "Which team committed more fouls?",
    "Name three players who had a strong game and why.",
    "What happened right before halftime?",
    "Give a one-paragraph match report.",
]


def load_events(match_id=DEFAULT_MATCH_ID):
    from statsbombpy import sb
    return sb.events(match_id=match_id)


def _get(row, col):
    v = row.get(col)
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else v


def build_timeline(events, max_chars=20000):
    """One line per event: [mm:ss] Team | Player | Type | detail. Capped by characters (~4 chars per token)."""
    lines = []
    for _, r in events.sort_values(["period", "minute", "second"]).iterrows():
        t = r["type"]
        keep = t in KEEP_TYPES
        if t == "Pass" and (_get(r, "pass_shot_assist") or _get(r, "pass_goal_assist")):
            keep = True
        if not keep:
            continue
        detail = ""
        if t == "Shot":
            detail = f"outcome={_get(r, 'shot_outcome')} xG={_get(r, 'shot_statsbomb_xg')}"
        elif t == "Pass":
            detail = f"key pass to {_get(r, 'pass_recipient')}"
        elif t == "Foul Committed":
            detail = f"card={_get(r, 'foul_committed_card')}"
        elif t == "Substitution":
            detail = f"replaced by {_get(r, 'substitution_replacement')}"
        elif t == "Bad Behaviour":
            detail = f"card={_get(r, 'bad_behaviour_card')}"
        lines.append(
            f"[{int(r['minute']):02d}:{int(r['second']):02d}] {_get(r, 'team')} | "
            f"{_get(r, 'player') or '-'} | {t} | {detail}".rstrip(" |")
        )
    text = "\n".join(lines)
    return text[:max_chars]


def build_prompt(timeline, question):
    """The timeline is the shared prefix; only the question changes between requests."""
    return (
        "You are a soccer match analyst. Use only the event log below.\n\n"
        f"EVENT LOG\n{timeline}\n\n"
        f"QUESTION: {question}\nANSWER:"
    )


if __name__ == "__main__":
    ev = load_events()
    tl = build_timeline(ev)
    print(f"{len(ev)} raw events -> {len(tl.splitlines())} timeline lines, {len(tl)} chars")
    print(tl[:600])

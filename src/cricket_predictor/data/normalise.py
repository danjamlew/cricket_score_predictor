"""Canonical names for teams, venues and seasons (Cricsheet naming drifts over time)."""

TEAM_ALIASES = {
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}

# (lowercase substring, canonical venue). First match wins; order matters.
VENUE_RULES = [
    ("wankhede", "Wankhede Stadium"),
    ("chinnaswamy", "M Chinnaswamy Stadium"),
    ("chidambaram", "MA Chidambaram Stadium"),
    ("arun jaitley", "Arun Jaitley Stadium"),
    ("feroz shah kotla", "Arun Jaitley Stadium"),
    ("rajiv gandhi", "Rajiv Gandhi International Stadium"),
    ("punjab cricket association", "Punjab Cricket Association IS Bindra Stadium"),
    ("maharaja yadavindra singh", "Maharaja Yadavindra Singh International Cricket Stadium"),
    ("sardar patel", "Narendra Modi Stadium"),
    ("narendra modi", "Narendra Modi Stadium"),
    ("zayed", "Sheikh Zayed Stadium"),
    ("subrata roy sahara", "Maharashtra Cricket Association Stadium"),
    ("eden gardens", "Eden Gardens"),
    ("brabourne", "Brabourne Stadium"),
    ("dy patil", "Dr DY Patil Sports Academy"),
    ("dr. y.s. rajasekhara reddy", "ACA-VDCA Cricket Stadium"),
    ("aca-vdca", "ACA-VDCA Cricket Stadium"),
]


def canonical_team(name: str) -> str:
    return TEAM_ALIASES.get(name, name)


def canonical_venue(name: str) -> str:
    """Map a raw venue string to one name per ground."""
    low = name.lower()
    for key, canon in VENUE_RULES:
        if key in low:
            return canon
    # Fallback: drop the trailing ", City" and stray dots.
    return name.split(",")[0].replace(".", "").strip()


def match_year(info: dict) -> int:
    """The `season` field is mixed int/str ("2007/08"), so use the match date."""
    return int(info["dates"][0][:4])

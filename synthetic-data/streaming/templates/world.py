"""Synthetic StreamFlix world data: user queries, catalog, search-augment
scenarios, and user profiles / watch history (for the fetch_user_history step
and the hallucination judge)."""

import random

# Bedrock metadata (for span attributes only; no real API calls).
BEDROCK_REGION = "us-east-1"
BEDROCK_MODEL_ID = "claude-3-7-sonnet-latest"

USER_QUERIES = [
    "something light and funny for Friday night",
    "thriller with a twist ending",
    "documentary about space exploration",
    "feel-good romance for a rainy day",
    "mind-bending sci-fi like Inception",
    "historical drama set in the 1920s",
    "animated family movie everyone will enjoy",
    "true crime documentary series",
]

# Synthetic catalog: id, title, genre, short description
CATALOG = [
    {"id": "mov_001", "title": "Cosmic Horizons", "genre": "documentary", "description": "A journey through the solar system and beyond."},
    {"id": "mov_002", "title": "Friday Night Laughs", "genre": "comedy", "description": "A group of friends navigate a chaotic evening."},
    {"id": "mov_003", "title": "The Twisted Truth", "genre": "thriller", "description": "Nothing is as it seems in this twist-filled mystery."},
    {"id": "mov_004", "title": "Rainy Day Hearts", "genre": "romance", "description": "Two strangers find love during a storm."},
    {"id": "mov_005", "title": "Recursion", "genre": "sci-fi", "description": "Dreams within dreams; reality bends."},
    {"id": "mov_006", "title": "Jazz Age", "genre": "historical", "description": "1920s New York through the eyes of a musician."},
    {"id": "mov_007", "title": "Skyward", "genre": "animated", "description": "A young bird discovers the meaning of family."},
    {"id": "mov_008", "title": "The Heist Files", "genre": "documentary", "description": "The untold story of the century's biggest heist."},
    {"id": "mov_009", "title": "Last Laugh", "genre": "comedy", "description": "A stand-up comic's road to redemption."},
    {"id": "mov_010", "title": "Double Blind", "genre": "thriller", "description": "A detective uncovers a conspiracy with a shocking end."},
]


def _doc_entry(item_id, score):
    item = next((c for c in CATALOG if c["id"] == item_id), None)
    if not item:
        return {"id": item_id, "content": "", "score": score}
    content = f"{item['title']} ({item['genre']}): {item['description']}"
    return {"id": item_id, "content": content, "score": score}


# Per-query: first_pull = retrieval candidates with scores; reranked = reordered
# top picks; response = the canned personalized answer.
SEARCH_AUGMENT_SCENARIOS = [
    # query_idx 0: light and funny
    {
        "query": USER_QUERIES[0],
        "first_pull": [("mov_002", 0.92), ("mov_009", 0.88), ("mov_007", 0.82), ("mov_004", 0.78), ("mov_001", 0.65)],
        "reranked": [("mov_002", 0.96), ("mov_009", 0.94), ("mov_007", 0.90)],
        "response": "For Friday night I'd go with **Friday Night Laughs** — pure chaos and laughs. If you want something the whole family can enjoy, **Skyward** is sweet and funny. **Last Laugh** is another solid pick if you love stand-up.",
    },
    # query_idx 1: thriller with twist
    {
        "query": USER_QUERIES[1],
        "first_pull": [("mov_003", 0.94), ("mov_010", 0.91), ("mov_005", 0.85), ("mov_006", 0.72), ("mov_008", 0.68)],
        "reranked": [("mov_003", 0.98), ("mov_010", 0.95), ("mov_005", 0.88)],
        "response": "You'll love **The Twisted Truth** — it's exactly what you asked for, with a twist that pays off. **Double Blind** is another great thriller with a conspiracy angle. For something more mind-bending, **Recursion** has the twist factor in a sci-fi package.",
    },
    # query_idx 2: documentary space
    {
        "query": USER_QUERIES[2],
        "first_pull": [("mov_001", 0.95), ("mov_008", 0.75), ("mov_005", 0.70), ("mov_003", 0.55), ("mov_007", 0.50)],
        "reranked": [("mov_001", 0.99), ("mov_008", 0.85), ("mov_005", 0.72)],
        "response": "**Cosmic Horizons** is the one — a proper space documentary that takes you through the solar system and beyond. If you're also into true crime, **The Heist Files** is a gripping documentary series.",
    },
    # query_idx 3: feel-good romance
    {
        "query": USER_QUERIES[3],
        "first_pull": [("mov_004", 0.93), ("mov_002", 0.85), ("mov_007", 0.80), ("mov_001", 0.62), ("mov_006", 0.60)],
        "reranked": [("mov_004", 0.97), ("mov_002", 0.91), ("mov_007", 0.86)],
        "response": "**Rainy Day Hearts** is perfect for a rainy day — cozy and romantic. **Friday Night Laughs** has a sweet subplot too. For something the whole family can enjoy, **Skyward** has a lot of heart.",
    },
    # query_idx 4: mind-bending sci-fi
    {
        "query": USER_QUERIES[4],
        "first_pull": [("mov_005", 0.94), ("mov_003", 0.88), ("mov_001", 0.79), ("mov_010", 0.76), ("mov_006", 0.65)],
        "reranked": [("mov_005", 0.98), ("mov_003", 0.92), ("mov_001", 0.85)],
        "response": "**Recursion** is the closest to that Inception vibe — dreams within dreams and reality bending. **The Twisted Truth** has a mind-bend of its own. **Cosmic Horizons** gives you the big-picture sci-fi feel.",
    },
    # query_idx 5: historical 1920s
    {
        "query": USER_QUERIES[5],
        "first_pull": [("mov_006", 0.96), ("mov_002", 0.70), ("mov_004", 0.68), ("mov_008", 0.65), ("mov_010", 0.60)],
        "reranked": [("mov_006", 0.99), ("mov_004", 0.82), ("mov_008", 0.78)],
        "response": "**Jazz Age** is right in the 1920s — New York, music, and that era's energy. **Rainy Day Hearts** has a period feel in places. **The Heist Files** documentary touches on that era too.",
    },
    # query_idx 6: animated family
    {
        "query": USER_QUERIES[6],
        "first_pull": [("mov_007", 0.95), ("mov_002", 0.88), ("mov_009", 0.85), ("mov_004", 0.75), ("mov_001", 0.65)],
        "reranked": [("mov_007", 0.98), ("mov_002", 0.92), ("mov_009", 0.89)],
        "response": "**Skyward** is the one — animated, family-friendly, and full of heart. **Friday Night Laughs** and **Last Laugh** are great if you want more comedy with the family.",
    },
    # query_idx 7: true crime documentary
    {
        "query": USER_QUERIES[7],
        "first_pull": [("mov_008", 0.94), ("mov_001", 0.78), ("mov_003", 0.72), ("mov_010", 0.70), ("mov_006", 0.65)],
        "reranked": [("mov_008", 0.97), ("mov_010", 0.88), ("mov_003", 0.85)],
        "response": "**The Heist Files** is the true crime documentary series you want — the story of the century's biggest heist. **Double Blind** has a documentary-style tension, and **The Twisted Truth** has that investigative angle.",
    },
]

# Synthetic user profiles / watch history (for fetch_user_history step and hallucination eval)
USER_PROFILES = [
    {"user_id": "u1", "recent_watches": "Cosmic Horizons, Friday Night Laughs", "preferences": "documentaries, comedy", "last_active": "2 days ago"},
    {"user_id": "u2", "recent_watches": "The Twisted Truth, Dark Signal", "preferences": "thrillers, mystery", "last_active": "1 day ago"},
    {"user_id": "u3", "recent_watches": "Rainy Day Hearts, Skyward", "preferences": "romance, family", "last_active": "3 hours ago"},
    {"user_id": "u4", "recent_watches": "Recursion, The Twisted Truth", "preferences": "sci-fi, psychological", "last_active": "5 hours ago"},
    {"user_id": "u5", "recent_watches": "The Heist Files, Beyond Earth", "preferences": "documentaries, true crime", "last_active": "1 day ago"},
]


def _format_user_history(profile: dict) -> str:
    return (
        f"User {profile['user_id']}: recent watches — {profile['recent_watches']}; "
        f"preferences — {profile['preferences']}; last active — {profile['last_active']}"
    )


def get_search_augment_data(query_id=None):
    """
    Get synthetic data for one search-augment trace.
    query_id: int index into SEARCH_AUGMENT_SCENARIOS, or None for random.
    Returns: dict with query, retrieved_docs (first-pull), reranked_docs,
    response (personalized text), user_history, user_profile.
    """
    if query_id is None:
        query_id = random.randint(0, len(SEARCH_AUGMENT_SCENARIOS) - 1)
    scenario = SEARCH_AUGMENT_SCENARIOS[query_id]
    query = scenario["query"]
    first_pull = [_doc_entry(uid, s) for uid, s in scenario["first_pull"]]
    reranked = [_doc_entry(uid, s) for uid, s in scenario["reranked"]]
    response = scenario["response"]
    profile = random.choice(USER_PROFILES)
    user_history = _format_user_history(profile)
    return {
        "query_id": query_id,
        "query": query,
        "retrieved_docs": first_pull,
        "reranked_docs": reranked,
        "response": response,
        "user_history": user_history,
        "user_profile": profile,
    }

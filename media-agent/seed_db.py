#!/usr/bin/env python3
"""
Seed the media demo SQLite database with realistic synthetic data.

Creates four tables:
  - authors (20 rows): reporters across beats
  - articles (~200 rows): 2023-2024 content
  - revenue (40 rows): 8 quarters × 5 segments, with realistic trends
  - traffic (~1825 rows): daily 2024 data per source, with weekly seasonality

Trends baked in:
  - digital_ads: growing +8% QoQ
  - print_ads: declining -5% QoQ
  - subscriptions: plateauing +1% QoQ
  - Traffic: weekday peaks, slight upward annual trend
"""

import os
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

_DEFAULT_DB_PATH = Path(__file__).parent / "data" / "media.db"
DB_PATH = Path(os.environ.get("MEDIA_DB_PATH", str(_DEFAULT_DB_PATH)))

SECTIONS = ["Markets", "Technology", "Opinion", "Wealth"]

TITLES = {
    "Markets": [
        "Fed Signals Pause in Rate Hike Cycle",
        "S&P 500 Reaches New All-Time High",
        "Bond Yields Surge on Strong Employment Data",
        "Dollar Strengthens Against Euro on ECB Concerns",
        "Emerging Markets Face Currency Pressure",
        "Goldman Sachs Upgrades Tech Sector Outlook",
        "Oil Prices Fall on OPEC Production Increase",
        "Credit Markets Show Signs of Stress",
        "Investors Flee to Safe Havens Amid Volatility",
        "Hedge Funds Increase Short Positions in Banks",
        "Treasury Yield Curve Inversion Deepens",
        "Private Equity Deals Dry Up on Financing Costs",
    ],
    "Technology": [
        "AI Chip Demand Drives Semiconductor Surge",
        "Cloud Computing Revenue Beats Estimates",
        "Cybersecurity Spending Rises Amid Threat Landscape",
        "Electric Vehicle Startups Face Cash Crunch",
        "Big Tech Faces Antitrust Scrutiny in Europe",
        "Quantum Computing Milestone Reached",
        "Social Media Platforms Struggle with AI Content",
        "Semiconductor Supply Chain Diversification Accelerates",
        "Tech Layoffs Continue Despite Strong Earnings",
        "Open Source AI Models Challenge Big Tech",
        "Data Center Energy Demand Hits Grid Capacity",
        "Startup Valuations Reset After Rate Shock",
    ],
    "Opinion": [
        "Why the Fed Is Getting It Wrong on Rates",
        "The ESG Backlash Is Overblown",
        "Private Equity Must Adapt to Higher Rates",
        "Digital Currencies Will Reshape Central Banking",
        "Why AI Won't Replace Financial Analysts",
        "The Case for Emerging Market Equities",
        "Corporate Debt: A Risk Too Long Ignored",
        "Reshoring Is Real, But The Math Is Hard",
        "The Coming Commercial Real Estate Reckoning",
        "Passive Investing Has Its Limits",
    ],
    "Wealth": [
        "Ultra-High-Net-Worth Individuals Shift to Alternatives",
        "Family Offices Increase Real Asset Allocations",
        "Philanthropy Trends Among Billionaires in 2024",
        "How the Super-Rich Are Hedging Inflation",
        "Art Market Cools After Post-Pandemic Boom",
        "Private Credit Attracts Institutional Capital",
        "Succession Planning Gaps at Family Businesses",
        "Sports Franchises as Alternative Investments",
        "Tax Optimization Strategies for HNW Clients",
        "Yacht Market Stays Resilient Amid Uncertainty",
    ],
}

TITLE_SUFFIXES = [
    "— Analysis", "— Report", "— Feature", "— Deep Dive",
    "— Exclusive", "— Q&A", "— Data", "— Outlook",
]

AUTHOR_DATA = [
    # (name, beat, join_date, productivity)
    ("Sarah Chen",       "Technology", "2018-03-15", 22),
    ("James Whitaker",   "Markets",    "2015-07-22", 25),
    ("Priya Sharma",     "Wealth",     "2020-01-10",  8),
    ("Marcus Thompson",  "Energy",     "2016-11-05", 15),
    ("Emma Rodriguez",   "Technology", "2019-04-18", 12),
    ("David Park",       "Markets",    "2017-09-30", 18),
    ("Aisha Johnson",    "Politics",   "2021-02-14",  6),
    ("Robert Chen",      "Healthcare", "2018-06-20",  9),
    ("Laura Martinez",   "Real Estate","2019-08-12",  7),
    ("Kevin O'Brien",    "ESG",        "2020-11-03", 11),
    ("Diana Walsh",      "Markets",    "2014-05-16", 28),
    ("Thomas Hughes",    "Technology", "2016-03-28", 16),
    ("Nina Patel",       "Wealth",     "2021-07-09",  5),
    ("Carlos Mendez",    "Energy",     "2017-12-01", 13),
    ("Rachel Kim",       "Politics",   "2019-01-22", 10),
    ("Anthony Davis",    "Markets",    "2015-10-14", 20),
    ("Sophie Laurent",   "Technology", "2020-05-06", 14),
    ("Michael Chang",    "Healthcare", "2018-09-19",  8),
    ("Olivia Brooks",    "Wealth",     "2022-03-08",  4),
    ("Nathan Foster",    "ESG",        "2021-06-25",  9),
]

# Which sections each beat maps to
BEAT_TO_SECTIONS = {
    "Markets":     ["Markets", "Opinion"],
    "Technology":  ["Technology", "Opinion"],
    "Wealth":      ["Wealth", "Opinion"],
    "Energy":      ["Markets", "Opinion"],
    "Politics":    ["Opinion"],
    "Healthcare":  ["Opinion", "Technology"],
    "Real Estate": ["Wealth", "Opinion"],
    "ESG":         ["Opinion", "Wealth"],
}

REVENUE_SEGMENTS = ["digital_ads", "print_ads", "subscriptions", "events", "licensing"]
QUARTERS = [
    ("Q1", 2023), ("Q2", 2023), ("Q3", 2023), ("Q4", 2023),
    ("Q1", 2024), ("Q2", 2024), ("Q3", 2024), ("Q4", 2024),
]
BASE_REVENUE = {
    "digital_ads":   1_800_000,
    "print_ads":     2_200_000,
    "subscriptions": 4_500_000,
    "events":          800_000,
    "licensing":       600_000,
}
GROWTH_RATES = {
    "digital_ads":    0.08,   # growing
    "print_ads":     -0.05,   # declining
    "subscriptions":  0.01,   # plateauing
    "events":         0.04,
    "licensing":      0.03,
}

TRAFFIC_SOURCES = ["organic", "social", "direct", "referral", "newsletter"]
SOURCE_BASE_VIEWS = {
    "organic":    45_000,
    "social":     28_000,
    "direct":     35_000,
    "referral":   12_000,
    "newsletter": 18_000,
}
SECTION_WEIGHTS = [0.40, 0.30, 0.15, 0.15]  # Markets, Technology, Opinion, Wealth


def seed():
    random.seed(42)
    DB_PATH.parent.mkdir(exist_ok=True)

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.executescript("""
        DROP TABLE IF EXISTS traffic;
        DROP TABLE IF EXISTS revenue;
        DROP TABLE IF EXISTS articles;
        DROP TABLE IF EXISTS authors;

        CREATE TABLE authors (
            id               INTEGER PRIMARY KEY,
            name             TEXT    NOT NULL,
            beat             TEXT    NOT NULL,
            articles_published INTEGER NOT NULL DEFAULT 0,
            join_date        TEXT    NOT NULL
        );

        CREATE TABLE articles (
            id           INTEGER PRIMARY KEY,
            title        TEXT    NOT NULL,
            category     TEXT    NOT NULL,
            publish_date TEXT    NOT NULL,
            author_id    INTEGER NOT NULL REFERENCES authors(id),
            word_count   INTEGER NOT NULL,
            section      TEXT    NOT NULL
        );

        CREATE TABLE revenue (
            id         INTEGER PRIMARY KEY,
            quarter    TEXT    NOT NULL,
            year       INTEGER NOT NULL,
            segment    TEXT    NOT NULL,
            amount_usd REAL    NOT NULL
        );

        CREATE TABLE traffic (
            id               INTEGER PRIMARY KEY,
            date             TEXT    NOT NULL,
            page_views       INTEGER NOT NULL,
            unique_visitors  INTEGER NOT NULL,
            source           TEXT    NOT NULL,
            section          TEXT    NOT NULL
        );
    """)

    # ── Authors ──────────────────────────────────────────────────────────────
    authors_rows = [
        (i + 1, name, beat, prod, jd)
        for i, (name, beat, jd, prod) in enumerate(AUTHOR_DATA)
    ]
    cur.executemany("INSERT INTO authors VALUES (?,?,?,?,?)", authors_rows)

    # ── Articles ─────────────────────────────────────────────────────────────
    start_date = date(2023, 1, 1)
    date_range_days = (date(2024, 12, 31) - start_date).days

    articles_raw = []
    for i, (name, beat, _, productivity) in enumerate(AUTHOR_DATA):
        author_id = i + 1
        section_pool = BEAT_TO_SECTIONS.get(beat, ["Opinion"])
        for _ in range(productivity):
            section = random.choice(section_pool)
            title_pool = TITLES.get(section, TITLES["Opinion"])
            title = random.choice(title_pool) + " " + random.choice(TITLE_SUFFIXES)
            pub_date = (start_date + timedelta(days=random.randint(0, date_range_days))).isoformat()
            if section == "Opinion":
                word_count = random.randint(600, 1_200)
            elif section == "Markets":
                word_count = random.randint(350, 900)
            else:
                word_count = random.randint(500, 2_000)
            articles_raw.append((title, section.lower(), pub_date, author_id, word_count, section))

    # Shuffle and cap at 200; re-assign sequential IDs
    random.shuffle(articles_raw)
    articles_rows = [
        (i + 1, title, cat, pub_date, auth_id, wc, sec)
        for i, (title, cat, pub_date, auth_id, wc, sec) in enumerate(articles_raw[:200])
    ]
    cur.executemany("INSERT INTO articles VALUES (?,?,?,?,?,?,?)", articles_rows)

    # ── Revenue ───────────────────────────────────────────────────────────────
    revenue_rows = []
    rev_id = 1
    for qi, (quarter, year) in enumerate(QUARTERS):
        for segment in REVENUE_SEGMENTS:
            base = BASE_REVENUE[segment]
            rate = GROWTH_RATES[segment]
            amount = round(base * ((1 + rate) ** qi) * random.uniform(0.95, 1.05), 2)
            revenue_rows.append((rev_id, quarter, year, segment, amount))
            rev_id += 1
    cur.executemany("INSERT INTO revenue VALUES (?,?,?,?,?)", revenue_rows)

    # ── Traffic ───────────────────────────────────────────────────────────────
    traffic_rows = []
    traffic_id = 1
    year_start = date(2024, 1, 1)
    year_end = date(2024, 12, 31)
    total_days = (year_end - year_start).days

    current = year_start
    while current <= year_end:
        dow = current.weekday()  # 0=Mon … 6=Sun
        days_elapsed = (current - year_start).days
        annual_trend = 1 + (days_elapsed / total_days) * 0.15  # up 15% over the year

        if dow < 4:        # Mon–Thu
            day_mult = random.uniform(1.00, 1.20)
        elif dow == 4:     # Fri
            day_mult = random.uniform(0.85, 1.00)
        else:              # Sat–Sun
            day_mult = random.uniform(0.55, 0.75)

        for source in TRAFFIC_SOURCES:
            base = SOURCE_BASE_VIEWS[source]
            views = int(base * day_mult * annual_trend * random.uniform(0.90, 1.10))
            visitors = int(views * random.uniform(0.55, 0.70))
            section = random.choices(SECTIONS, weights=SECTION_WEIGHTS)[0]
            traffic_rows.append((traffic_id, current.isoformat(), views, visitors, source, section))
            traffic_id += 1

        current += timedelta(days=1)

    cur.executemany("INSERT INTO traffic VALUES (?,?,?,?,?,?)", traffic_rows)

    con.commit()
    con.close()

    print(f"Database seeded: {DB_PATH}")
    print(f"  authors:  {len(authors_rows)}")
    print(f"  articles: {len(articles_rows)}")
    print(f"  revenue:  {len(revenue_rows)}")
    print(f"  traffic:  {len(traffic_rows)}")


if __name__ == "__main__":
    seed()

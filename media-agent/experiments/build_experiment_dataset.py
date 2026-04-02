#!/usr/bin/env python3
"""
Build the ~150-question experiment dataset for the Media Agent.

One-time script. Writes to experiments/data/experiment_dataset.json.

Categories:
  simple              (~35)  Single-table lookups, counts, filters
  aggregation         (~35)  GROUP BY, ranking, temporal comparisons
  multi_hop           (~35)  Joins across 2+ tables, sequential reasoning
  constraint_following (~25) Explicit formatting/exclusion/persona instructions
  edge_case           (~20) Out-of-scope, missing data, schema mismatch

Golden SQL and answers are intentionally mixed: some are optimal, others are
correct but less efficient so that experiment scoring can surface better model outputs.

Usage:
    python -m experiments.build_experiment_dataset
    python experiments/build_experiment_dataset.py
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def _examples() -> list[dict]:
    """Return all eval examples as dicts."""
    rows: list[dict] = []
    _id = 0

    def add(
        question: str,
        category: str,
        difficulty: str,
        expected_sql: str,
        expected_answer: str,
        must_include: list[str] | None = None,
        must_not_include: list[str] | None = None,
        notes: str = "",
    ):
        nonlocal _id
        _id += 1
        rows.append(
            {
                "id": f"eval_{_id:03d}",
                "question": question,
                "category": category,
                "difficulty": difficulty,
                "expected_sql": expected_sql,
                "expected_answer": expected_answer,
                "must_include": must_include or [],
                "must_not_include": must_not_include or [],
                "notes": notes,
                "role": "finance",
            }
        )

    # ── SIMPLE (~35) ──────────────────────────────────────────────────────────

    add(
        "How many articles were published in total?",
        "simple", "easy",
        "SELECT COUNT(*) AS article_count FROM articles",
        "There are 200 articles in total.",
        ["200"],
    )
    add(
        "How many authors are in the database?",
        "simple", "easy",
        "SELECT COUNT(*) AS author_count FROM authors",
        "There are 20 authors.",
        ["20"],
    )
    add(
        "List all revenue segments.",
        "simple", "easy",
        "SELECT DISTINCT segment FROM revenue ORDER BY segment",
        "The revenue segments are: digital_ads, events, licensing, print_ads, and subscriptions.",
        ["digital_ads", "events", "licensing", "print_ads", "subscriptions"],
    )
    add(
        "What sections appear in the traffic data?",
        "simple", "easy",
        "SELECT DISTINCT section FROM traffic ORDER BY section",
        "The traffic data covers: Markets, Opinion, Technology, and Wealth.",
        ["Markets", "Technology"],
    )
    add(
        "What is Sarah Chen's beat?",
        "simple", "easy",
        "SELECT beat FROM authors WHERE name = 'Sarah Chen'",
        "Sarah Chen covers the Technology beat.",
        ["Technology"],
    )
    add(
        "How many articles were published in 2024?",
        "simple", "easy",
        "SELECT COUNT(*) AS cnt FROM articles WHERE publish_date >= '2024-01-01' AND publish_date < '2025-01-01'",
        "Approximately 100 articles were published in 2024.",
        notes="Exact count depends on seed; answer should be in the ballpark.",
    )
    add(
        "How many articles were published in 2023?",
        "simple", "easy",
        "SELECT COUNT(*) AS cnt FROM articles WHERE publish_date >= '2023-01-01' AND publish_date < '2024-01-01'",
        "Approximately 100 articles were published in 2023.",
    )
    add(
        "Which author has the most articles published?",
        "simple", "easy",
        # Intentionally less efficient: subquery instead of ORDER BY LIMIT
        "SELECT name, articles_published FROM authors WHERE articles_published = (SELECT MAX(articles_published) FROM authors)",
        "Diana Walsh has the most articles published with 28.",
        ["Diana Walsh", "28"],
        notes="Less efficient golden SQL uses subquery instead of ORDER BY ... LIMIT 1.",
    )
    add(
        "What was total revenue in Q2 2024?",
        "simple", "easy",
        "SELECT SUM(amount_usd) AS total_revenue FROM revenue WHERE quarter = 'Q2' AND year = 2024",
        "Total revenue in Q2 2024 was approximately $10.5 million across all segments.",
        ["Q2 2024"],
    )
    add(
        "What traffic sources are tracked?",
        "simple", "easy",
        "SELECT DISTINCT source FROM traffic ORDER BY source",
        "Traffic sources tracked are: direct, newsletter, organic, referral, and social.",
        ["organic", "social", "direct", "referral", "newsletter"],
    )
    add(
        "What is the average word count across all articles?",
        "simple", "easy",
        "SELECT AVG(word_count) AS avg_word_count FROM articles",
        "The average word count across all articles is approximately 900 words.",
    )
    add(
        "When did James Whitaker join?",
        "simple", "easy",
        "SELECT join_date FROM authors WHERE name = 'James Whitaker'",
        "James Whitaker joined on 2015-07-22.",
        ["2015-07-22"],
    )
    add(
        "List all authors who cover the Markets beat.",
        "simple", "easy",
        "SELECT name FROM authors WHERE beat = 'Markets' ORDER BY name",
        "Markets beat authors: Anthony Davis, David Park, Diana Walsh, and James Whitaker.",
        ["James Whitaker", "Diana Walsh"],
    )
    add(
        "What was the licensing revenue in Q1 2023?",
        "simple", "easy",
        "SELECT amount_usd FROM revenue WHERE segment = 'licensing' AND quarter = 'Q1' AND year = 2023",
        "Licensing revenue in Q1 2023 was approximately $600,000.",
        ["licensing", "Q1 2023"],
    )
    add(
        "How many rows are in the traffic table?",
        "simple", "easy",
        "SELECT COUNT(*) AS row_count FROM traffic",
        "The traffic table contains 1,830 rows (365 days times 5 sources).",
        notes="Exact count depends on seed.",
    )
    add(
        "What is the most recent article publish date?",
        "simple", "easy",
        "SELECT MAX(publish_date) AS latest FROM articles",
        "The most recent article was published in late December 2024.",
    )
    add(
        "What is the earliest article publish date?",
        "simple", "easy",
        "SELECT MIN(publish_date) AS earliest FROM articles",
        "The earliest article was published in early January 2023.",
    )
    add(
        "How many distinct sections are in the articles table?",
        "simple", "easy",
        "SELECT COUNT(DISTINCT section) AS section_count FROM articles",
        "There are 4 distinct sections in the articles table.",
        ["4"],
    )
    add(
        "What is the highest single-quarter revenue amount recorded?",
        "simple", "medium",
        "SELECT MAX(amount_usd) AS max_revenue FROM revenue",
        "The highest single-quarter revenue amount is approximately $5 million, from the subscriptions segment.",
    )
    add(
        "Which author joined most recently?",
        "simple", "easy",
        # Less efficient: subquery
        "SELECT name, join_date FROM authors WHERE join_date = (SELECT MAX(join_date) FROM authors)",
        "Olivia Brooks joined most recently, on 2022-03-08.",
        ["Olivia Brooks"],
        notes="Uses subquery instead of ORDER BY join_date DESC LIMIT 1.",
    )
    add(
        "How many quarters of revenue data are available?",
        "simple", "easy",
        "SELECT COUNT(DISTINCT quarter || '-' || year) AS quarter_count FROM revenue",
        "There are 8 quarters of revenue data, from Q1 2023 through Q4 2024.",
        ["8"],
    )
    add(
        "What is the total number of articles in the Opinion section?",
        "simple", "easy",
        "SELECT COUNT(*) AS cnt FROM articles WHERE section = 'Opinion'",
        "There are approximately 30 articles in the Opinion section.",
    )
    add(
        "Show me the revenue for events in Q3 2024.",
        "simple", "easy",
        "SELECT amount_usd FROM revenue WHERE segment = 'events' AND quarter = 'Q3' AND year = 2024",
        "Events revenue in Q3 2024 was approximately $950,000.",
        ["events", "Q3 2024"],
    )
    add(
        "What beats do the authors cover?",
        "simple", "easy",
        "SELECT DISTINCT beat FROM authors ORDER BY beat",
        "Authors cover: ESG, Energy, Healthcare, Markets, Politics, Real Estate, Technology, and Wealth.",
        ["Technology", "Markets", "ESG"],
    )
    add(
        "How many articles have a word count over 1500?",
        "simple", "medium",
        "SELECT COUNT(*) AS cnt FROM articles WHERE word_count > 1500",
        "There are approximately 20-30 articles with a word count over 1,500.",
    )
    add(
        "What is the minimum word count of any article?",
        "simple", "easy",
        "SELECT MIN(word_count) AS min_wc FROM articles",
        "The minimum word count is approximately 350 words.",
    )
    add(
        "How many authors have published more than 15 articles?",
        "simple", "medium",
        "SELECT COUNT(*) AS cnt FROM authors WHERE articles_published > 15",
        "There are 6 authors who have published more than 15 articles.",
    )
    add(
        "What was the total print_ads revenue in 2024?",
        "simple", "medium",
        "SELECT SUM(amount_usd) AS total FROM revenue WHERE segment = 'print_ads' AND year = 2024",
        "Total print_ads revenue in 2024 was approximately $7 million, reflecting a declining trend.",
        ["print_ads", "2024"],
    )
    add(
        "List all authors sorted by join date, oldest first.",
        "simple", "easy",
        "SELECT name, join_date FROM authors ORDER BY join_date ASC",
        "The earliest-joining author is Diana Walsh (2014-05-16), followed by James Whitaker and Anthony Davis.",
        ["Diana Walsh"],
    )
    add(
        "What is the average page_views from the traffic table?",
        "simple", "easy",
        "SELECT AVG(page_views) AS avg_views FROM traffic",
        "The average daily page views across all sources and sections is approximately 25,000-30,000.",
    )
    add(
        "How many revenue records exist for 2023?",
        "simple", "easy",
        "SELECT COUNT(*) AS cnt FROM revenue WHERE year = 2023",
        "There are 20 revenue records for 2023 (4 quarters times 5 segments).",
        ["20"],
    )
    add(
        "What was the subscriptions revenue in Q4 2024?",
        "simple", "easy",
        "SELECT amount_usd FROM revenue WHERE segment = 'subscriptions' AND quarter = 'Q4' AND year = 2024",
        "Subscriptions revenue in Q4 2024 was approximately $4.9 million, reflecting the plateauing trend.",
        ["subscriptions", "Q4 2024"],
    )
    add(
        "How many articles are in the Technology section?",
        "simple", "easy",
        "SELECT COUNT(*) AS cnt FROM articles WHERE section = 'Technology'",
        "There are approximately 30-40 articles in the Technology section.",
        ["Technology"],
    )
    add(
        "Show me all authors on the Wealth beat.",
        "simple", "easy",
        "SELECT name FROM authors WHERE beat = 'Wealth' ORDER BY name",
        "Wealth beat authors: Nina Patel, Olivia Brooks, and Priya Sharma.",
        ["Priya Sharma"],
    )
    add(
        "What is the total digital_ads revenue across all quarters?",
        "simple", "medium",
        "SELECT SUM(amount_usd) AS total FROM revenue WHERE segment = 'digital_ads'",
        "Total digital_ads revenue across all 8 quarters is approximately $17 million, with a strong growth trend of +8% QoQ.",
        ["digital_ads"],
    )

    # ── AGGREGATION (~35) ─────────────────────────────────────────────────────

    add(
        "What is the total revenue by segment for 2024?",
        "aggregation", "medium",
        "SELECT segment, SUM(amount_usd) AS total_revenue FROM revenue WHERE year = 2024 GROUP BY segment ORDER BY total_revenue DESC",
        "In 2024, subscriptions led with approximately $18.5M, followed by digital_ads at around $9.5M, print_ads near $7M, events around $3.8M, and licensing at roughly $2.6M.",
        ["subscriptions", "digital_ads"],
    )
    add(
        "Which revenue segment had the highest total in 2024?",
        "aggregation", "easy",
        # Less efficient: computes all then filters
        "SELECT segment, SUM(amount_usd) AS total FROM revenue WHERE year = 2024 GROUP BY segment ORDER BY total DESC LIMIT 1",
        "Subscriptions was the highest revenue segment in 2024.",
        ["subscriptions"],
    )
    add(
        "Rank the top 5 authors by number of articles published.",
        "aggregation", "easy",
        "SELECT name, articles_published FROM authors ORDER BY articles_published DESC LIMIT 5",
        "Top 5: Diana Walsh (28), James Whitaker (25), Sarah Chen (22), Anthony Davis (20), David Park (18).",
        ["Diana Walsh", "James Whitaker"],
    )
    add(
        "What is the average revenue per segment across all quarters?",
        "aggregation", "medium",
        "SELECT segment, AVG(amount_usd) AS avg_revenue FROM revenue GROUP BY segment ORDER BY avg_revenue DESC",
        "Subscriptions averages the highest per quarter at roughly $4.6M, while licensing averages the lowest at around $650K.",
        ["subscriptions"],
    )
    add(
        "How many articles were published per section?",
        "aggregation", "easy",
        "SELECT section, COUNT(*) AS article_count FROM articles GROUP BY section ORDER BY article_count DESC",
        "Markets has the most articles, followed by Technology, Opinion, and Wealth.",
        ["Markets"],
    )
    add(
        "What is the average word count by section?",
        "aggregation", "medium",
        "SELECT section, AVG(word_count) AS avg_wc FROM articles GROUP BY section ORDER BY avg_wc DESC",
        "Technology tends to have the longest articles on average, while Markets articles are typically shorter.",
    )
    add(
        "What was the total page views by traffic source for 2024?",
        "aggregation", "medium",
        "SELECT source, SUM(page_views) AS total_views FROM traffic GROUP BY source ORDER BY total_views DESC",
        "Organic traffic leads with the highest total page views, followed by direct, social, newsletter, and referral.",
        ["organic"],
    )
    add(
        "Show quarterly revenue trends for digital_ads from Q1 2023 to Q4 2024.",
        "aggregation", "medium",
        "SELECT quarter, year, amount_usd FROM revenue WHERE segment = 'digital_ads' ORDER BY year, quarter",
        "Digital ads revenue shows consistent growth of approximately 8% quarter-over-quarter, rising from about $1.8M in Q1 2023 to around $3M by Q4 2024.",
        ["digital_ads", "growth"],
    )
    add(
        "Compare total 2023 revenue to total 2024 revenue.",
        "aggregation", "medium",
        # Less efficient: two subqueries instead of conditional aggregation
        "SELECT (SELECT SUM(amount_usd) FROM revenue WHERE year = 2023) AS total_2023, (SELECT SUM(amount_usd) FROM revenue WHERE year = 2024) AS total_2024",
        "Total revenue grew from approximately $39M in 2023 to roughly $42M in 2024, driven by digital ads and subscriptions growth offsetting print ads decline.",
        ["2023", "2024"],
        notes="Less efficient: uses subqueries instead of CASE WHEN with GROUP BY.",
    )
    add(
        "What is the month-by-month article count for 2024?",
        "aggregation", "medium",
        "SELECT strftime('%Y-%m', publish_date) AS month, COUNT(*) AS cnt FROM articles WHERE publish_date >= '2024-01-01' AND publish_date < '2025-01-01' GROUP BY month ORDER BY month",
        "Article output in 2024 was distributed across months, typically 7-12 articles per month.",
    )
    add(
        "Which traffic source has the highest average unique visitors?",
        "aggregation", "easy",
        "SELECT source, AVG(unique_visitors) AS avg_visitors FROM traffic GROUP BY source ORDER BY avg_visitors DESC LIMIT 1",
        "Organic search has the highest average unique visitors per day.",
        ["organic"],
    )
    add(
        "What is the total revenue per quarter across all segments?",
        "aggregation", "medium",
        "SELECT quarter, year, SUM(amount_usd) AS total FROM revenue GROUP BY quarter, year ORDER BY year, quarter",
        "Quarterly total revenue ranges from roughly $9.5M in Q1 2023 to around $11M in Q4 2024, showing an overall upward trend.",
    )
    add(
        "How many articles did each author publish (using the articles table)?",
        "aggregation", "medium",
        "SELECT a.name, COUNT(ar.id) AS article_count FROM authors a LEFT JOIN articles ar ON a.id = ar.author_id GROUP BY a.id, a.name ORDER BY article_count DESC",
        "The most prolific authors in the articles table are led by Diana Walsh and James Whitaker.",
        ["Diana Walsh"],
        notes="Uses LEFT JOIN to include authors with zero articles.",
    )
    add(
        "What is the average daily page views by day of week?",
        "aggregation", "hard",
        "SELECT CASE CAST(strftime('%w', date) AS INTEGER) WHEN 0 THEN 'Sunday' WHEN 1 THEN 'Monday' WHEN 2 THEN 'Tuesday' WHEN 3 THEN 'Wednesday' WHEN 4 THEN 'Thursday' WHEN 5 THEN 'Friday' WHEN 6 THEN 'Saturday' END AS day_name, AVG(page_views) AS avg_views FROM traffic GROUP BY strftime('%w', date) ORDER BY CAST(strftime('%w', date) AS INTEGER)",
        "Weekdays (Monday through Thursday) see the highest average page views, with a noticeable drop on Friday and a significant dip on weekends.",
        ["Monday", "weekend"],
    )
    add(
        "What is the quarter-over-quarter revenue growth rate for print_ads?",
        "aggregation", "hard",
        "SELECT r1.quarter || ' ' || r1.year AS period, r1.amount_usd, ROUND((r1.amount_usd - r2.amount_usd) / r2.amount_usd * 100, 1) AS growth_pct FROM revenue r1 JOIN revenue r2 ON r1.segment = r2.segment AND ((r1.year = r2.year AND CASE r1.quarter WHEN 'Q2' THEN 'Q1' WHEN 'Q3' THEN 'Q2' WHEN 'Q4' THEN 'Q3' END = r2.quarter) OR (r1.quarter = 'Q1' AND r1.year = r2.year + 1 AND r2.quarter = 'Q4')) WHERE r1.segment = 'print_ads' ORDER BY r1.year, r1.quarter",
        "Print ads revenue declines approximately 5% each quarter, falling from about $2.2M in Q1 2023 to under $1.6M by Q4 2024.",
        ["print_ads", "declin"],
        notes="Complex self-join for QoQ growth. Model may simplify.",
    )
    add(
        "Which section gets the most total traffic?",
        "aggregation", "easy",
        "SELECT section, SUM(page_views) AS total_views FROM traffic GROUP BY section ORDER BY total_views DESC LIMIT 1",
        "Markets receives the most total traffic.",
        ["Markets"],
    )
    add(
        "What is the total revenue by year?",
        "aggregation", "easy",
        "SELECT year, SUM(amount_usd) AS total FROM revenue GROUP BY year ORDER BY year",
        "Total revenue was approximately $39M in 2023 and $42M in 2024.",
        ["2023", "2024"],
    )
    add(
        "Show the revenue distribution by segment for Q4 2024.",
        "aggregation", "medium",
        "SELECT segment, amount_usd, ROUND(amount_usd * 100.0 / SUM(amount_usd) OVER (), 1) AS pct FROM revenue WHERE quarter = 'Q4' AND year = 2024 ORDER BY amount_usd DESC",
        "In Q4 2024, subscriptions dominates at roughly 45% of total revenue, followed by digital_ads and print_ads.",
        ["subscriptions", "Q4 2024"],
    )
    add(
        "What is the average word count for articles published in each year?",
        "aggregation", "easy",
        "SELECT strftime('%Y', publish_date) AS year, AVG(word_count) AS avg_wc FROM articles GROUP BY year ORDER BY year",
        "Average word count is similar across both years, around 850-950 words.",
    )
    add(
        "How many articles were published per quarter in 2024?",
        "aggregation", "medium",
        "SELECT CASE WHEN publish_date BETWEEN '2024-01-01' AND '2024-03-31' THEN 'Q1' WHEN publish_date BETWEEN '2024-04-01' AND '2024-06-30' THEN 'Q2' WHEN publish_date BETWEEN '2024-07-01' AND '2024-09-30' THEN 'Q3' ELSE 'Q4' END AS quarter, COUNT(*) AS cnt FROM articles WHERE publish_date >= '2024-01-01' AND publish_date < '2025-01-01' GROUP BY quarter ORDER BY quarter",
        "Article output in 2024 was roughly evenly distributed across quarters, approximately 25 articles per quarter.",
    )
    add(
        "What is the total unique visitors by section?",
        "aggregation", "medium",
        "SELECT section, SUM(unique_visitors) AS total_visitors FROM traffic GROUP BY section ORDER BY total_visitors DESC",
        "Markets leads in total unique visitors, followed by Technology.",
        ["Markets"],
    )
    add(
        "Show the top 3 revenue segments by total 2024 earnings.",
        "aggregation", "easy",
        "SELECT segment, SUM(amount_usd) AS total FROM revenue WHERE year = 2024 GROUP BY segment ORDER BY total DESC LIMIT 3",
        "Top 3 revenue segments in 2024: subscriptions, digital_ads, and print_ads.",
        ["subscriptions", "digital_ads", "print_ads"],
    )
    add(
        "What was the maximum daily page views recorded?",
        "aggregation", "easy",
        "SELECT MAX(page_views) AS max_views FROM traffic",
        "The maximum daily page views recorded was approximately 55,000-60,000.",
    )
    add(
        "Show the number of authors per beat.",
        "aggregation", "easy",
        "SELECT beat, COUNT(*) AS cnt FROM authors GROUP BY beat ORDER BY cnt DESC",
        "Markets and Technology have the most authors (4 each), while Real Estate has the fewest (1).",
        ["Markets", "Technology"],
    )
    add(
        "What was the total newsletter traffic in 2024?",
        "aggregation", "medium",
        "SELECT SUM(page_views) AS total_views FROM traffic WHERE source = 'newsletter'",
        "Total newsletter-driven page views in 2024 were approximately 6.5 million.",
        ["newsletter"],
    )
    add(
        "What percentage of total revenue comes from digital_ads in 2024?",
        "aggregation", "medium",
        # Less efficient: two subqueries
        "SELECT ROUND((SELECT SUM(amount_usd) FROM revenue WHERE segment = 'digital_ads' AND year = 2024) * 100.0 / (SELECT SUM(amount_usd) FROM revenue WHERE year = 2024), 1) AS pct",
        "Digital ads accounts for approximately 22-25% of total 2024 revenue.",
        ["digital_ads"],
        notes="Less efficient golden SQL; model could use a single GROUP BY.",
    )
    add(
        "How has events revenue changed from 2023 to 2024?",
        "aggregation", "medium",
        "SELECT year, SUM(amount_usd) AS total FROM revenue WHERE segment = 'events' GROUP BY year ORDER BY year",
        "Events revenue grew from approximately $3.4M in 2023 to $3.8M in 2024, reflecting steady 4% QoQ growth.",
        ["events", "2023", "2024"],
    )
    add(
        "Which month in 2024 had the highest total traffic?",
        "aggregation", "hard",
        "SELECT strftime('%Y-%m', date) AS month, SUM(page_views) AS total FROM traffic GROUP BY month ORDER BY total DESC LIMIT 1",
        "The month with the highest total traffic was likely in Q4 2024 due to the annual upward trend.",
    )
    add(
        "Show quarterly subscriptions revenue for 2023 and 2024.",
        "aggregation", "medium",
        "SELECT quarter, year, amount_usd FROM revenue WHERE segment = 'subscriptions' ORDER BY year, quarter",
        "Subscriptions revenue remained relatively flat at around $4.5M-$4.9M per quarter, growing only ~1% QoQ.",
        ["subscriptions", "plateau"],
    )
    add(
        "What is the average article word count by author beat?",
        "aggregation", "medium",
        "SELECT au.beat, AVG(ar.word_count) AS avg_wc FROM authors au JOIN articles ar ON au.id = ar.author_id GROUP BY au.beat ORDER BY avg_wc DESC",
        "Authors covering different beats produce articles of varying lengths; Technology beat authors tend to write longer pieces.",
    )
    add(
        "What is the longest article by word count?",
        "aggregation", "easy",
        "SELECT title, word_count, section FROM articles ORDER BY word_count DESC LIMIT 1",
        "The longest article has approximately 2,000 words and is in the Technology section.",
    )
    add(
        "How many distinct dates have traffic data?",
        "aggregation", "easy",
        "SELECT COUNT(DISTINCT date) AS day_count FROM traffic",
        "There are 366 distinct dates with traffic data (all of 2024).",
        ["366"],
    )
    add(
        "What is the median revenue amount across all records?",
        "aggregation", "hard",
        "SELECT amount_usd FROM revenue ORDER BY amount_usd LIMIT 1 OFFSET (SELECT COUNT(*) FROM revenue) / 2",
        "The median revenue amount is approximately $1.5-2M, reflecting the wide range between licensing (~$600K) and subscriptions (~$4.5M+).",
        notes="SQLite doesn't have a native MEDIAN; this approximation is acceptable.",
    )

    # ── MULTI-HOP (~35) ──────────────────────────────────────────────────────

    add(
        "Which authors in the Technology beat wrote articles in the Opinion section?",
        "multi_hop", "medium",
        "SELECT DISTINCT au.name FROM authors au JOIN articles ar ON au.id = ar.author_id WHERE au.beat = 'Technology' AND ar.section = 'Opinion' ORDER BY au.name",
        "Technology beat authors who also wrote Opinion pieces include Sarah Chen, Emma Rodriguez, and others.",
        ["Technology", "Opinion"],
    )
    add(
        "What was the total revenue in quarters where more than 25 articles were published?",
        "multi_hop", "hard",
        "SELECT r.quarter, r.year, SUM(r.amount_usd) AS total_revenue FROM revenue r WHERE r.quarter || '-' || r.year IN (SELECT CASE WHEN publish_date BETWEEN (year || '-01-01') AND (year || '-03-31') THEN 'Q1' WHEN publish_date BETWEEN (year || '-04-01') AND (year || '-06-30') THEN 'Q2' WHEN publish_date BETWEEN (year || '-07-01') AND (year || '-09-30') THEN 'Q3' ELSE 'Q4' END || '-' || strftime('%Y', publish_date) FROM articles GROUP BY 1 HAVING COUNT(*) > 25) GROUP BY r.quarter, r.year ORDER BY r.year, r.quarter",
        "Quarters with more than 25 articles published had total revenues ranging from $9M to $11M.",
        notes="Complex correlation between articles and revenue. SQL may vary significantly.",
    )
    add(
        "Show me the top 3 authors by article count along with their average word count.",
        "multi_hop", "medium",
        "SELECT au.name, COUNT(ar.id) AS article_count, ROUND(AVG(ar.word_count)) AS avg_wc FROM authors au JOIN articles ar ON au.id = ar.author_id GROUP BY au.id, au.name ORDER BY article_count DESC LIMIT 3",
        "The top 3 authors by article count are Diana Walsh, James Whitaker, and Sarah Chen, with average word counts varying by beat.",
        ["Diana Walsh"],
    )
    add(
        "Compare the average daily traffic for Markets vs Technology sections.",
        "multi_hop", "medium",
        "SELECT section, ROUND(AVG(page_views)) AS avg_daily_views FROM traffic WHERE section IN ('Markets', 'Technology') GROUP BY section",
        "Markets sees higher average daily traffic than Technology, reflecting its larger weight in the distribution.",
        ["Markets", "Technology"],
    )
    add(
        "Which section had the most articles and how does its traffic compare to other sections?",
        "multi_hop", "hard",
        "SELECT ar.section, COUNT(DISTINCT ar.id) AS article_count, SUM(t.page_views) AS total_traffic FROM articles ar JOIN traffic t ON ar.section = t.section GROUP BY ar.section ORDER BY article_count DESC",
        "Markets leads in both article count and total traffic. There is a positive correlation between content volume and traffic.",
        ["Markets"],
        notes="Join between articles and traffic on section. Traffic data is daily while articles are individual records, so the join is approximate.",
    )
    add(
        "Show authors who joined before 2019 and their total article count from the articles table.",
        "multi_hop", "medium",
        "SELECT au.name, au.join_date, COUNT(ar.id) AS article_count FROM authors au LEFT JOIN articles ar ON au.id = ar.author_id WHERE au.join_date < '2019-01-01' GROUP BY au.id, au.name ORDER BY article_count DESC",
        "Pre-2019 authors like Diana Walsh, James Whitaker, and Anthony Davis are among the most productive.",
        ["Diana Walsh"],
    )
    add(
        "What is the relationship between article output in the Markets section and organic traffic?",
        "multi_hop", "hard",
        "SELECT strftime('%m', ar.publish_date) AS month, COUNT(ar.id) AS article_count, SUM(t.page_views) AS organic_views FROM articles ar JOIN traffic t ON strftime('%m', ar.publish_date) = strftime('%m', t.date) AND t.source = 'organic' AND t.section = 'Markets' WHERE ar.section = 'Markets' AND strftime('%Y', ar.publish_date) = '2024' AND strftime('%Y', t.date) = '2024' GROUP BY month ORDER BY month",
        "There appears to be a positive relationship between Markets article output and organic traffic, though traffic is also influenced by the annual upward trend.",
        ["Markets", "organic"],
    )
    add(
        "List authors whose articles have an average word count above 1000.",
        "multi_hop", "medium",
        "SELECT au.name, au.beat, ROUND(AVG(ar.word_count)) AS avg_wc FROM authors au JOIN articles ar ON au.id = ar.author_id GROUP BY au.id, au.name, au.beat HAVING AVG(ar.word_count) > 1000 ORDER BY avg_wc DESC",
        "Several authors, particularly those covering Technology, tend to write longer articles averaging over 1,000 words.",
    )
    add(
        "How does digital ad revenue compare to total traffic volume over 2024?",
        "multi_hop", "hard",
        "SELECT r.quarter, r.amount_usd AS digital_ads_revenue, (SELECT SUM(page_views) FROM traffic WHERE date BETWEEN CASE r.quarter WHEN 'Q1' THEN '2024-01-01' WHEN 'Q2' THEN '2024-04-01' WHEN 'Q3' THEN '2024-07-01' WHEN 'Q4' THEN '2024-10-01' END AND CASE r.quarter WHEN 'Q1' THEN '2024-03-31' WHEN 'Q2' THEN '2024-06-30' WHEN 'Q3' THEN '2024-09-30' WHEN 'Q4' THEN '2024-12-31' END) AS quarterly_views FROM revenue r WHERE r.segment = 'digital_ads' AND r.year = 2024 ORDER BY r.quarter",
        "Both digital ad revenue and total traffic grew throughout 2024, with digital ads up 8% QoQ and traffic rising 15% annually. The trends are directionally aligned.",
        ["digital_ads", "traffic", "growth"],
        notes="Complex correlated subquery. Model output may differ structurally.",
    )
    add(
        "Which beat produces the most articles per author on average?",
        "multi_hop", "medium",
        "SELECT au.beat, COUNT(ar.id) * 1.0 / COUNT(DISTINCT au.id) AS avg_articles_per_author FROM authors au JOIN articles ar ON au.id = ar.author_id GROUP BY au.beat ORDER BY avg_articles_per_author DESC",
        "Markets beat authors are among the most productive per author, averaging the highest article count per person.",
    )
    add(
        "Show revenue by segment alongside the number of articles published in the same year.",
        "multi_hop", "hard",
        "SELECT r.year, r.segment, SUM(r.amount_usd) AS total_revenue, (SELECT COUNT(*) FROM articles WHERE strftime('%Y', publish_date) = CAST(r.year AS TEXT)) AS article_count FROM revenue r GROUP BY r.year, r.segment ORDER BY r.year, r.segment",
        "Each year saw roughly 100 articles published. Revenue varies by segment: subscriptions leads, while licensing is the smallest.",
        notes="Less efficient correlated subquery repeated per row.",
    )
    add(
        "Which authors wrote articles longer than 1500 words and in which sections?",
        "multi_hop", "medium",
        "SELECT DISTINCT au.name, au.beat, ar.section, ar.word_count FROM authors au JOIN articles ar ON au.id = ar.author_id WHERE ar.word_count > 1500 ORDER BY ar.word_count DESC",
        "Long-form articles (>1500 words) come primarily from Technology section authors.",
    )
    add(
        "For each traffic source, show the average page views on weekdays vs weekends.",
        "multi_hop", "hard",
        "SELECT source, CASE WHEN CAST(strftime('%w', date) AS INTEGER) IN (0, 6) THEN 'weekend' ELSE 'weekday' END AS day_type, ROUND(AVG(page_views)) AS avg_views FROM traffic GROUP BY source, day_type ORDER BY source, day_type",
        "All traffic sources show significantly higher page views on weekdays compared to weekends. Organic traffic shows the largest weekday/weekend gap.",
        ["weekday", "weekend"],
    )
    add(
        "How many articles did Markets-beat authors publish in the Technology section?",
        "multi_hop", "medium",
        "SELECT COUNT(*) AS cnt FROM articles ar JOIN authors au ON ar.author_id = au.id WHERE au.beat = 'Markets' AND ar.section = 'Technology'",
        "Markets-beat authors typically don't write for the Technology section, so the count should be zero or very low.",
    )
    add(
        "Show the quarterly trend: total revenue vs total traffic page views.",
        "multi_hop", "hard",
        "SELECT r.quarter || ' ' || r.year AS period, SUM(r.amount_usd) AS total_revenue, COALESCE((SELECT SUM(page_views) FROM traffic WHERE date BETWEEN CASE r.quarter WHEN 'Q1' THEN r.year || '-01-01' WHEN 'Q2' THEN r.year || '-04-01' WHEN 'Q3' THEN r.year || '-07-01' WHEN 'Q4' THEN r.year || '-10-01' END AND CASE r.quarter WHEN 'Q1' THEN r.year || '-03-31' WHEN 'Q2' THEN r.year || '-06-30' WHEN 'Q3' THEN r.year || '-09-30' WHEN 'Q4' THEN r.year || '-12-31' END), 0) AS total_views FROM revenue r GROUP BY r.quarter, r.year ORDER BY r.year, r.quarter",
        "Both revenue and traffic trend upward through 2024. Note that traffic data is only available for 2024, so 2023 quarters show zero traffic.",
        notes="Traffic is 2024-only; complex correlated subquery.",
    )
    add(
        "Which author has written in the most different sections?",
        "multi_hop", "medium",
        "SELECT au.name, COUNT(DISTINCT ar.section) AS section_count FROM authors au JOIN articles ar ON au.id = ar.author_id GROUP BY au.id, au.name ORDER BY section_count DESC LIMIT 1",
        "The most versatile author writes across 2-3 different sections.",
    )
    add(
        "What is the average traffic per section on days when articles were published in that section?",
        "multi_hop", "hard",
        "SELECT ar.section, ROUND(AVG(t.page_views)) AS avg_views_on_pub_days FROM articles ar JOIN traffic t ON ar.section = t.section AND ar.publish_date = t.date GROUP BY ar.section ORDER BY avg_views_on_pub_days DESC",
        "Sections tend to see similar traffic on publication days vs non-publication days, since daily traffic is driven more by source and day-of-week patterns.",
    )
    add(
        "Show the top 5 authors by total word count written.",
        "multi_hop", "medium",
        "SELECT au.name, SUM(ar.word_count) AS total_words FROM authors au JOIN articles ar ON au.id = ar.author_id GROUP BY au.id, au.name ORDER BY total_words DESC LIMIT 5",
        "The authors with the highest total word output include Diana Walsh and James Whitaker, who are also the most prolific by article count.",
        ["Diana Walsh"],
    )
    add(
        "What fraction of total 2024 traffic came from social media?",
        "multi_hop", "medium",
        # Less efficient: two separate queries
        "SELECT ROUND(SUM(CASE WHEN source = 'social' THEN page_views ELSE 0 END) * 100.0 / SUM(page_views), 1) AS social_pct FROM traffic",
        "Social media accounts for approximately 20% of total 2024 traffic.",
        ["social"],
    )
    add(
        "For each author, show their beat and the total page views their section receives.",
        "multi_hop", "hard",
        "SELECT au.name, au.beat, ar.section, SUM(t.page_views) AS section_views FROM authors au JOIN articles ar ON au.id = ar.author_id JOIN traffic t ON ar.section = t.section GROUP BY au.name, au.beat, ar.section ORDER BY section_views DESC",
        "Authors writing for the Markets section benefit from the highest section traffic, while Wealth section authors see lower overall traffic.",
        notes="Three-table join. Traffic is not directly attributable to individual authors.",
    )
    add(
        "How many articles were published by authors who joined in 2020 or later?",
        "multi_hop", "medium",
        "SELECT COUNT(ar.id) AS cnt FROM articles ar JOIN authors au ON ar.author_id = au.id WHERE au.join_date >= '2020-01-01'",
        "Authors who joined in 2020 or later have published approximately 50-60 articles collectively.",
    )
    add(
        "Show the quarterly revenue alongside article count for the same quarter.",
        "multi_hop", "hard",
        "SELECT r.quarter, r.year, SUM(r.amount_usd) AS revenue, (SELECT COUNT(*) FROM articles WHERE (CASE WHEN publish_date BETWEEN r.year || '-01-01' AND r.year || '-03-31' THEN 'Q1' WHEN publish_date BETWEEN r.year || '-04-01' AND r.year || '-06-30' THEN 'Q2' WHEN publish_date BETWEEN r.year || '-07-01' AND r.year || '-09-30' THEN 'Q3' ELSE 'Q4' END) = r.quarter AND strftime('%Y', publish_date) = CAST(r.year AS TEXT)) AS article_count FROM revenue r GROUP BY r.quarter, r.year ORDER BY r.year, r.quarter",
        "Each quarter sees roughly 25 articles published and $9-11M in total revenue. The relationship between content volume and revenue is relatively stable.",
    )
    add(
        "Show authors along with their earliest and latest article dates.",
        "multi_hop", "medium",
        "SELECT au.name, MIN(ar.publish_date) AS first_article, MAX(ar.publish_date) AS last_article FROM authors au JOIN articles ar ON au.id = ar.author_id GROUP BY au.id, au.name ORDER BY first_article",
        "Authors' publishing spans range from early 2023 to late 2024, with longer-tenured authors having wider date ranges.",
    )
    add(
        "Which section has the highest revenue-per-pageview ratio?",
        "multi_hop", "hard",
        "SELECT t.section, SUM(r.amount_usd) / SUM(t.page_views) AS revenue_per_view FROM traffic t CROSS JOIN revenue r WHERE r.year = 2024 GROUP BY t.section ORDER BY revenue_per_view DESC LIMIT 1",
        "This comparison is approximate since revenue is not broken down by section. Using a cross join gives a rough ratio, with smaller-traffic sections showing higher per-view ratios.",
        notes="Imperfect join—revenue isn't sectioned. Tests model reasoning about data gaps.",
    )
    add(
        "How many Technology beat authors have published more than 10 articles?",
        "multi_hop", "medium",
        "SELECT COUNT(*) AS cnt FROM authors WHERE beat = 'Technology' AND articles_published > 10",
        "Three Technology beat authors have published more than 10 articles: Sarah Chen (22), Thomas Hughes (16), and Sophie Laurent (14).",
        ["Sarah Chen"],
    )
    add(
        "Show the weekly traffic pattern averaged across all of 2024.",
        "multi_hop", "hard",
        "SELECT CASE CAST(strftime('%w', date) AS INTEGER) WHEN 0 THEN 'Sun' WHEN 1 THEN 'Mon' WHEN 2 THEN 'Tue' WHEN 3 THEN 'Wed' WHEN 4 THEN 'Thu' WHEN 5 THEN 'Fri' WHEN 6 THEN 'Sat' END AS day, ROUND(AVG(page_views)) AS avg_views FROM traffic GROUP BY strftime('%w', date) ORDER BY CAST(strftime('%w', date) AS INTEGER)",
        "Traffic peaks Monday through Thursday, dips on Friday, and drops significantly on weekends (Saturday and Sunday).",
        ["Monday", "weekend"],
    )

    # ── CONSTRAINT-FOLLOWING (~25) ────────────────────────────────────────────

    add(
        "In exactly 2 sentences, explain the relationship between article output and page views by section.",
        "constraint_following", "medium",
        "SELECT ar.section, COUNT(ar.id) AS articles, SUM(t.page_views) AS views FROM articles ar JOIN traffic t ON ar.section = t.section GROUP BY ar.section",
        "Markets leads in both article output and page views, suggesting a positive correlation between content volume and readership. Technology follows as the second most active section, while Opinion and Wealth trail in both metrics.",
        notes="Must be exactly 2 sentences.",
    )
    add(
        "Summarize 2024 revenue trends but exclude any mention of the events segment.",
        "constraint_following", "medium",
        "SELECT segment, SUM(amount_usd) AS total FROM revenue WHERE year = 2024 AND segment != 'events' GROUP BY segment ORDER BY total DESC",
        "In 2024, subscriptions remained the dominant revenue source at roughly $18.5M. Digital ads grew strongly at 8% QoQ to approximately $9.5M. Print ads continued its decline to around $7M. Licensing contributed about $2.6M with modest 3% growth.",
        must_not_include=["events"],
        notes="Must not mention events at all.",
    )
    add(
        "Explain Q3 traffic patterns as if briefing a non-technical CMO — no jargon, no raw numbers over 1 million.",
        "constraint_following", "hard",
        "SELECT source, SUM(page_views) AS total_views, ROUND(AVG(page_views)) AS avg_daily FROM traffic WHERE date BETWEEN '2024-07-01' AND '2024-09-30' GROUP BY source ORDER BY total_views DESC",
        "In the third quarter, search engines drove the most visitors to our site, followed by people typing our URL directly. Social media and email newsletters each brought in a healthy share. Referrals from other websites contributed a smaller but steady stream. Weekdays consistently outperformed weekends.",
        must_not_include=["page_views", "million"],
        notes="No jargon, no numbers over 1M.",
    )
    add(
        "List the top 5 authors by article count, then separately give a 1-sentence summary of overall content production trends. Do not combine these into one paragraph.",
        "constraint_following", "medium",
        "SELECT name, articles_published FROM authors ORDER BY articles_published DESC LIMIT 5",
        "**Top 5 Authors by Article Count:**\n1. Diana Walsh — 28\n2. James Whitaker — 25\n3. Sarah Chen — 22\n4. Anthony Davis — 20\n5. David Park — 18\n\n**Summary:** Content production across 2023-2024 was concentrated among Markets and Technology beat reporters, with the top 5 authors accounting for over half of all published articles.",
        ["Diana Walsh", "James Whitaker"],
        notes="Must be two separate sections, not one paragraph.",
    )
    add(
        "Give me a 3-bullet summary of subscription revenue trends. Each bullet must start with the quarter name.",
        "constraint_following", "medium",
        "SELECT quarter, year, amount_usd FROM revenue WHERE segment = 'subscriptions' ORDER BY year, quarter",
        "- Q1-Q2 2023: Subscriptions started at approximately $4.5M and grew minimally through mid-2023.\n- Q3-Q4 2023: Growth plateaued around the $4.6M mark, with only ~1% quarterly increases.\n- Q1-Q4 2024: The plateau continued through 2024, ending near $4.9M—signaling market saturation.",
        ["Q1", "Q2", "plateau"],
        notes="Must be exactly 3 bullets, each starting with a quarter reference.",
    )
    add(
        "In one paragraph of no more than 50 words, summarize the overall revenue trend from 2023 to 2024.",
        "constraint_following", "medium",
        "SELECT year, SUM(amount_usd) AS total FROM revenue GROUP BY year",
        "Revenue grew from $39M in 2023 to $42M in 2024, driven by digital ads (+8% QoQ) and steady subscriptions. Print ads declined 5% quarterly. Overall growth was modest but positive.",
        notes="Must be one paragraph, max 50 words.",
    )
    add(
        "Describe traffic source distribution using only percentages, no absolute numbers.",
        "constraint_following", "medium",
        "SELECT source, ROUND(SUM(page_views) * 100.0 / (SELECT SUM(page_views) FROM traffic), 1) AS pct FROM traffic GROUP BY source ORDER BY pct DESC",
        "Organic search accounts for roughly 33% of all traffic, followed by direct at 25%, social at 20%, newsletter at 13%, and referral at 9%.",
        must_not_include=["page_views"],
        notes="Only percentages allowed.",
    )
    add(
        "Compare print_ads and digital_ads trends over 2023-2024 in exactly 3 sentences.",
        "constraint_following", "medium",
        "SELECT quarter, year, segment, amount_usd FROM revenue WHERE segment IN ('print_ads', 'digital_ads') ORDER BY year, quarter, segment",
        "Digital ads revenue grew consistently at 8% quarter-over-quarter, rising from $1.8M in Q1 2023 to approximately $3M by Q4 2024. In contrast, print ads declined 5% each quarter, falling from $2.2M to under $1.6M over the same period. By mid-2024, digital ads had overtaken print ads in quarterly revenue.",
        ["digital_ads", "print_ads"],
        notes="Exactly 3 sentences.",
    )
    add(
        "Rank all revenue segments by 2024 total and present as a numbered list with no additional commentary.",
        "constraint_following", "easy",
        "SELECT segment, SUM(amount_usd) AS total FROM revenue WHERE year = 2024 GROUP BY segment ORDER BY total DESC",
        "1. Subscriptions\n2. Digital Ads\n3. Print Ads\n4. Events\n5. Licensing",
        notes="Numbered list, no commentary.",
    )
    add(
        "Explain the traffic weekday vs weekend pattern without using the word 'traffic'.",
        "constraint_following", "hard",
        "SELECT CASE WHEN CAST(strftime('%w', date) AS INTEGER) IN (0, 6) THEN 'weekend' ELSE 'weekday' END AS period, ROUND(AVG(page_views)) AS avg_views FROM traffic GROUP BY period",
        "Readership follows a clear weekly cycle: Monday through Thursday sees the highest engagement, Friday begins tapering off, and Saturday-Sunday visitor counts drop to roughly 60-70% of the weekday average.",
        must_not_include=["traffic"],
    )
    add(
        "Summarize author productivity in a table format with columns: Name, Beat, Articles.",
        "constraint_following", "medium",
        "SELECT name, beat, articles_published FROM authors ORDER BY articles_published DESC",
        "| Name | Beat | Articles |\n|---|---|---|\n| Diana Walsh | Markets | 28 |\n| James Whitaker | Markets | 25 |\n| Sarah Chen | Technology | 22 |\n...",
        ["Diana Walsh", "Markets"],
        notes="Must use table format.",
    )
    add(
        "Without using any numbers, describe the relative size of each revenue segment.",
        "constraint_following", "hard",
        "SELECT segment, SUM(amount_usd) AS total FROM revenue GROUP BY segment ORDER BY total DESC",
        "Subscriptions is by far the largest revenue segment, contributing nearly half of all revenue. Digital ads is the second largest and growing rapidly. Print ads is the third largest but shrinking. Events and licensing are the smallest segments, each contributing a modest slice.",
        must_not_include=["$", "million", "%"],
        notes="No numbers of any kind.",
    )
    add(
        "Give me a one-line answer: what is the total article count?",
        "constraint_following", "easy",
        "SELECT COUNT(*) FROM articles",
        "There are 200 articles in the database.",
        ["200"],
        notes="Must be a single line.",
    )
    add(
        "Summarize 2024 digital ads performance as a haiku.",
        "constraint_following", "hard",
        "SELECT quarter, amount_usd FROM revenue WHERE segment = 'digital_ads' AND year = 2024 ORDER BY quarter",
        "Ads climb every quarter\nEight percent growth fuels the rise\nDigital leads now",
        notes="Haiku format: 5-7-5 syllables. Creative constraint.",
    )
    add(
        "List all beats alphabetically, one per line, no additional text.",
        "constraint_following", "easy",
        "SELECT DISTINCT beat FROM authors ORDER BY beat",
        "ESG\nEnergy\nHealthcare\nMarkets\nPolitics\nReal Estate\nTechnology\nWealth",
        notes="One per line, no commentary.",
    )
    add(
        "Explain the top 3 traffic sources using only metaphors — no data terms.",
        "constraint_following", "hard",
        "SELECT source, SUM(page_views) AS total FROM traffic GROUP BY source ORDER BY total DESC LIMIT 3",
        "Think of our readers arriving three ways: the largest river flows from search engines, a steady well-worn highway brings those who know exactly where they're going, and a bustling town square channels the social media crowd our way.",
        notes="Metaphorical language only.",
    )
    add(
        "Compare Q1 2023 vs Q1 2024 total revenue. Answer with exactly one number and one comparison word.",
        "constraint_following", "medium",
        "SELECT year, SUM(amount_usd) AS total FROM revenue WHERE quarter = 'Q1' GROUP BY year ORDER BY year",
        "Q1 2024 was approximately 8% higher.",
        notes="Minimal answer format.",
    )
    add(
        "Describe the print_ads decline using only questions, not statements.",
        "constraint_following", "hard",
        "SELECT quarter, year, amount_usd FROM revenue WHERE segment = 'print_ads' ORDER BY year, quarter",
        "Did you know print ad revenue started at $2.2M per quarter in 2023? Can you guess how much it dropped by Q4 2024 — to under $1.6M? Isn't it striking that every single quarter showed a decline of about 5%? And doesn't this raise the question of when digital will fully replace print?",
        ["print_ads", "decline"],
        notes="All questions, no declarative statements.",
    )
    add(
        "Summarize author distribution by beat — respond in JSON format only.",
        "constraint_following", "medium",
        "SELECT beat, COUNT(*) AS cnt FROM authors GROUP BY beat ORDER BY beat",
        '{"ESG": 2, "Energy": 2, "Healthcare": 2, "Markets": 4, "Politics": 2, "Real Estate": 1, "Technology": 4, "Wealth": 3}',
        notes="Must be valid JSON, no prose.",
    )
    add(
        "In bullet points, list each revenue segment's 2024 trend direction (growing/declining/flat). No numbers.",
        "constraint_following", "medium",
        "SELECT segment, SUM(CASE WHEN year = 2024 THEN amount_usd ELSE 0 END) - SUM(CASE WHEN year = 2023 THEN amount_usd ELSE 0 END) AS delta FROM revenue GROUP BY segment",
        "- Digital Ads: Growing\n- Print Ads: Declining\n- Subscriptions: Flat\n- Events: Growing\n- Licensing: Growing",
        must_not_include=["$", "%", "million"],
        notes="Bullet points, direction only, no numbers.",
    )
    add(
        "Provide a 2-sentence executive summary of the entire media data landscape.",
        "constraint_following", "medium",
        "SELECT 'articles' AS tbl, COUNT(*) AS cnt FROM articles UNION ALL SELECT 'authors', COUNT(*) FROM authors UNION ALL SELECT 'revenue', COUNT(*) FROM revenue UNION ALL SELECT 'traffic', COUNT(*) FROM traffic",
        "The content operation spans 200 articles across four sections by 20 authors, generating roughly $40M annually across five revenue streams. Traffic data shows strong weekday engagement patterns and a 15% annual growth trend driven primarily by organic search.",
        notes="Exactly 2 sentences.",
    )

    # ── Additional multi_hop ─────────────────────────────────────────────────

    add(
        "What is the total revenue in quarters where organic traffic exceeded 4 million page views?",
        "multi_hop", "hard",
        "SELECT r.quarter, r.year, SUM(r.amount_usd) AS total_revenue FROM revenue r WHERE EXISTS (SELECT 1 FROM traffic t WHERE t.source = 'organic' AND t.date BETWEEN CASE r.quarter WHEN 'Q1' THEN r.year || '-01-01' WHEN 'Q2' THEN r.year || '-04-01' WHEN 'Q3' THEN r.year || '-07-01' WHEN 'Q4' THEN r.year || '-10-01' END AND CASE r.quarter WHEN 'Q1' THEN r.year || '-03-31' WHEN 'Q2' THEN r.year || '-06-30' WHEN 'Q3' THEN r.year || '-09-30' WHEN 'Q4' THEN r.year || '-12-31' END GROUP BY 1 HAVING SUM(page_views) > 4000000) GROUP BY r.quarter, r.year",
        "Quarters with high organic traffic (likely Q3-Q4 2024 given the annual growth trend) correspond to total revenues of approximately $10-11M.",
        notes="Complex EXISTS subquery correlating traffic to revenue quarters.",
    )
    add(
        "Show each author's name, how many articles they wrote, and the most common section they wrote for.",
        "multi_hop", "hard",
        "SELECT au.name, COUNT(ar.id) AS article_count, ar.section FROM authors au JOIN articles ar ON au.id = ar.author_id GROUP BY au.id, au.name, ar.section HAVING COUNT(ar.id) = (SELECT MAX(cnt) FROM (SELECT COUNT(*) AS cnt FROM articles WHERE author_id = au.id GROUP BY section))",
        "Each author primarily writes for sections aligned with their beat, e.g., Markets beat authors mostly publish in Markets.",
        notes="Correlated subquery to find mode section per author.",
    )
    add(
        "Which authors have published articles in every section that exists in the database?",
        "multi_hop", "hard",
        "SELECT au.name FROM authors au WHERE (SELECT COUNT(DISTINCT section) FROM articles WHERE author_id = au.id) = (SELECT COUNT(DISTINCT section) FROM articles)",
        "Most likely no author has published in all 4 sections; authors tend to stay within 1-2 sections aligned with their beat.",
        notes="Tests understanding of universal quantification.",
    )
    add(
        "Compare the total word count output between authors who joined before 2018 and those who joined after.",
        "multi_hop", "medium",
        "SELECT CASE WHEN join_date < '2018-01-01' THEN 'pre-2018' ELSE 'post-2018' END AS cohort, SUM(ar.word_count) AS total_words, COUNT(ar.id) AS articles FROM authors au JOIN articles ar ON au.id = ar.author_id GROUP BY cohort",
        "Pre-2018 authors have produced significantly more total word output, driven by their longer tenure and higher individual productivity.",
    )
    add(
        "For each revenue segment, show the quarter with the highest revenue.",
        "multi_hop", "medium",
        "SELECT r.segment, r.quarter, r.year, r.amount_usd FROM revenue r WHERE r.amount_usd = (SELECT MAX(amount_usd) FROM revenue WHERE segment = r.segment) ORDER BY r.segment",
        "Each segment's peak quarter varies: digital_ads peaks in Q4 2024 (latest quarter, due to growth), while print_ads peaked in Q1 2023 (due to decline).",
        ["digital_ads", "print_ads"],
    )
    add(
        "Show the total article word count per month alongside monthly newsletter traffic.",
        "multi_hop", "hard",
        "SELECT strftime('%Y-%m', ar.publish_date) AS month, SUM(ar.word_count) AS total_words, COALESCE((SELECT SUM(page_views) FROM traffic WHERE source = 'newsletter' AND strftime('%Y-%m', date) = strftime('%Y-%m', ar.publish_date)), 0) AS newsletter_views FROM articles ar WHERE ar.publish_date >= '2024-01-01' GROUP BY month ORDER BY month",
        "Monthly word count output and newsletter traffic both show variability across 2024, with no strong direct correlation between the two.",
        notes="Correlated subquery joining on month; newsletter views from traffic.",
    )

    # ── Additional constraint_following ────────────────────────────────────────

    add(
        "Summarize the difference between the top and bottom revenue segments in 2024. Use exactly one statistic.",
        "constraint_following", "medium",
        "SELECT segment, SUM(amount_usd) AS total FROM revenue WHERE year = 2024 GROUP BY segment ORDER BY total DESC",
        "Subscriptions outearned licensing by approximately 7x in 2024.",
        notes="Exactly one statistic.",
    )
    add(
        "Name every author on the Technology beat and nothing else.",
        "constraint_following", "easy",
        "SELECT name FROM authors WHERE beat = 'Technology' ORDER BY name",
        "Emma Rodriguez, Sarah Chen, Sophie Laurent, Thomas Hughes.",
        notes="Names only, no commentary.",
    )
    add(
        "Describe the revenue trajectory of digital_ads as if it were a stock price — use financial language.",
        "constraint_following", "hard",
        "SELECT quarter, year, amount_usd FROM revenue WHERE segment = 'digital_ads' ORDER BY year, quarter",
        "Digital ads opened at $1.8M in Q1 2023 and rallied steadily, posting consistent 8% quarterly gains. By Q4 2024 the segment closed near $3M, charting an uninterrupted bull run across eight consecutive quarters with no drawdowns.",
        ["rally", "bull"],
        notes="Financial/stock metaphor language.",
    )
    add(
        "In a single word, characterize the print_ads revenue trend.",
        "constraint_following", "easy",
        "SELECT SUM(CASE WHEN year = 2024 THEN amount_usd ELSE 0 END) - SUM(CASE WHEN year = 2023 THEN amount_usd ELSE 0 END) AS delta FROM revenue WHERE segment = 'print_ads'",
        "Declining.",
        notes="Single word answer.",
    )

    # ── EDGE CASES (~20) ─────────────────────────────────────────────────────

    add(
        "How did the company's market share compare to Reuters in Q4 2024?",
        "edge_case", "medium",
        "SELECT 'No competitor market share data available in the database' AS note",
        "The database does not contain competitor market share data. I can only report on the company's own revenue, content, and traffic metrics.",
        notes="Missing data: no competitor info exists.",
    )
    add(
        "Summarize article performance for January 2022.",
        "edge_case", "medium",
        "SELECT COUNT(*) AS cnt FROM articles WHERE publish_date >= '2022-01-01' AND publish_date < '2022-02-01'",
        "No articles exist for January 2022 in this database. The earliest data begins in January 2023.",
        notes="Out-of-range: data starts 2023.",
    )
    add(
        "What is the average word count per article broken down by author seniority level?",
        "edge_case", "medium",
        "SELECT 'No seniority_level column exists in the authors table' AS note",
        "The database does not include an author seniority level field. I can break down word count by beat or by individual author if that would be helpful.",
        notes="Schema mismatch: no seniority column.",
    )
    add(
        "Show me the click-through rate for each traffic source.",
        "edge_case", "medium",
        "SELECT 'No click-through rate data available — traffic table has page_views and unique_visitors only' AS note",
        "Click-through rate is not available in this database. The traffic table tracks page views and unique visitors by source and section, but does not include click or impression data.",
        notes="Schema mismatch: no CTR field.",
    )
    add(
        "What was the advertising CPM trend over 2024?",
        "edge_case", "medium",
        "SELECT 'No CPM or impression data available in the database' AS note",
        "CPM (cost per thousand impressions) data is not available. Revenue is tracked at the segment level (digital_ads, print_ads, etc.) without impression or CPM breakdowns.",
        notes="Missing data: no CPM metrics.",
    )
    add(
        "Compare the company's article quality scores across sections.",
        "edge_case", "medium",
        "SELECT 'No quality_score column exists in the articles table' AS note",
        "The database does not contain article quality scores. Available article metrics include word count, section, publish date, and author.",
        notes="Schema mismatch: no quality score.",
    )
    add(
        "What was the year-over-year traffic growth from 2023 to 2024?",
        "edge_case", "medium",
        "SELECT 'Traffic data only available for 2024' AS note",
        "Year-over-year traffic comparison is not possible because the traffic table only contains 2024 data. Within 2024, there was approximately 15% growth from January to December.",
        notes="Missing data: traffic is 2024-only.",
    )
    add(
        "Show the revenue broken down by geographic region.",
        "edge_case", "medium",
        "SELECT 'No geographic region data in the revenue table' AS note",
        "Revenue data is not available by geographic region. Revenue is broken down by segment (digital_ads, print_ads, subscriptions, events, licensing) and by quarter/year.",
        notes="Schema mismatch: no geography field.",
    )
    add(
        "What is the churn rate for subscriptions?",
        "edge_case", "medium",
        "SELECT 'No churn rate or subscriber count data available' AS note",
        "Subscription churn rate is not tracked in this database. The revenue table only records total subscription revenue per quarter, without subscriber counts or cancellations.",
        notes="Missing data: no churn metrics.",
    )
    add(
        "How many video articles were published?",
        "edge_case", "easy",
        "SELECT 'No article type/format column exists — all articles are text' AS note",
        "The articles table does not distinguish between text and video content. All 200 records appear to be text articles based on the schema (which includes word_count).",
        notes="Schema mismatch: no content type field.",
    )
    add(
        "What was the total cost of content production in 2024?",
        "edge_case", "medium",
        "SELECT 'No cost or expense data available' AS note",
        "Content production costs are not tracked in this database. The data covers revenue, articles, authors, and traffic — but no expense or cost information.",
        notes="Missing data: no cost/expense tables.",
    )
    add(
        "Show me the sentiment analysis of articles by section.",
        "edge_case", "medium",
        "SELECT 'No sentiment data available in the articles table' AS note",
        "Sentiment analysis data is not available. The articles table contains metadata like title, section, publish date, and word count, but no sentiment scores or NLP-derived fields.",
        notes="Schema mismatch: no sentiment column.",
    )
    add(
        "What is the correlation coefficient between traffic and revenue?",
        "edge_case", "hard",
        "SELECT 'SQLite does not support statistical correlation functions natively' AS note",
        "SQLite does not have a built-in correlation function. I can show the raw quarterly traffic and revenue data side-by-side so you can assess the relationship directionally.",
        notes="Technical limitation: SQLite lacks CORR().",
    )
    add(
        "Show me the email open rates for the newsletter source.",
        "edge_case", "medium",
        "SELECT 'No email open rate data — traffic table only has page_views and unique_visitors' AS note",
        "Email open rates are not available. The traffic table tracks page views and unique visitors attributed to the newsletter source, but does not include email-level metrics like open rate or click rate.",
        notes="Schema mismatch: no email metrics.",
    )
    add(
        "Which articles went viral on social media?",
        "edge_case", "medium",
        "SELECT 'No per-article social engagement data available' AS note",
        "The database does not track social media engagement at the individual article level. Traffic from social media is recorded as aggregate daily page views by section, not per article.",
        notes="Missing data: no per-article social metrics.",
    )
    add(
        "What was revenue in Q1 2025?",
        "edge_case", "easy",
        "SELECT COUNT(*) AS cnt FROM revenue WHERE quarter = 'Q1' AND year = 2025",
        "No revenue data exists for Q1 2025. The dataset covers Q1 2023 through Q4 2024.",
        notes="Out-of-range: data ends Q4 2024.",
    )
    add(
        "What is the profit margin by segment?",
        "edge_case", "medium",
        "SELECT 'No cost or margin data available — only revenue amounts' AS note",
        "Profit margin data is not available. The revenue table contains gross revenue by segment and quarter, but no cost or margin information.",
        notes="Missing data: revenue only, no costs.",
    )
    add(
        "Show the bounce rate trends for each section.",
        "edge_case", "medium",
        "SELECT 'No bounce rate data in the traffic table' AS note",
        "Bounce rate is not tracked in this database. The traffic table includes page views and unique visitors by source and section, but no engagement metrics like bounce rate or time-on-page.",
        notes="Schema mismatch: no bounce rate.",
    )
    add(
        "What's the trending topic across all articles?",
        "edge_case", "hard",
        "SELECT section, COUNT(*) AS cnt FROM articles GROUP BY section ORDER BY cnt DESC",
        "The database doesn't contain topic tags or NLP-derived topics. The closest approximation is section distribution: Markets has the most articles, followed by Technology. Individual article titles suggest themes around AI, Fed policy, and market volatility.",
        notes="Partial data: no topic tags, but section and titles give hints.",
    )
    add(
        "How many articles were written by freelancers vs staff?",
        "edge_case", "medium",
        "SELECT 'No employment_type column exists in the authors table' AS note",
        "The database does not distinguish between freelance and staff authors. All 20 authors are listed with name, beat, article count, and join date — no employment type field exists.",
        notes="Schema mismatch: no freelance/staff distinction.",
    )

    return rows


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    examples = _examples()

    out_path = DATA_DIR / "experiment_dataset.json"
    with open(out_path, "w") as f:
        json.dump(examples, f, indent=2)

    cats = {}
    diffs = {}
    for ex in examples:
        cats[ex["category"]] = cats.get(ex["category"], 0) + 1
        diffs[ex["difficulty"]] = diffs.get(ex["difficulty"], 0) + 1

    print(f"Wrote {len(examples)} examples to {out_path}")
    print(f"  Categories: {cats}")
    print(f"  Difficulties: {diffs}")


if __name__ == "__main__":
    main()

"""Single source of truth for Spring 2026 survey analysis v2.

Every parsing rule, vocabulary, bucket map, and the model registry lives here so
that the generated `parsing_decisions.md` and `parsing_review.md` stay 1:1 with
the code that actually runs. If you change a rule, change it here and both the
analysis and the review docs update on the next run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../../"))

INPUT_DIR = os.path.join(
    REPO_ROOT, "local/private_input_data/fall_and_spring_survey_data"
)
SURVEY_CSV = os.path.join(
    INPUT_DIR,
    "Spring 2026 FutureEval Participant Survey (Responses) - Form Responses 1.csv",
)
PRIZE_STATS_CSV = os.path.join(
    INPUT_DIR, "Spring 2026 FutureEval Prize and participation stats.csv"
)
# Human-reviewed adjustment files (see manual_adjustments.py). They live in the
# non-committed private input directory because they reference individual bots;
# keeping them beside the raw survey data preserves them for reruns.
MANUAL_WINNER_OVERRIDES_CSV = os.path.join(INPUT_DIR, "manual_winner_overrides.csv")
MANUAL_ANSWER_ADJUSTMENTS_CSV = os.path.join(INPUT_DIR, "manual_answer_adjustments.csv")
BOT_TOURNAMENT_JSON = os.path.join(
    REPO_ROOT, "local/spring_2026_simulations/2_bot_tournament.json"
)

OUTPUT_DIR = os.path.join(REPO_ROOT, "local/spring_survey_analysis")
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
LEADERBOARD_CACHE_CSV = os.path.join(DATA_DIR, "spring_leaderboard.csv")

REPORT_MD = os.path.join(OUTPUT_DIR, "spring_survey_analysis.md")
PARSING_DECISIONS_MD = os.path.join(OUTPUT_DIR, "parsing_decisions.md")
PARSING_REVIEW_MD = os.path.join(OUTPUT_DIR, "parsing_review.md")

# --------------------------------------------------------------------------- #
# Group definitions (confirmed with user)
# --------------------------------------------------------------------------- #
# winner       = won any AIB prize (prize-stats winner_count > 0 OR aib_prize > 0)
# top_10       = top 10 bots by sum of spot peer score on the FULL leaderboard
# non_winner   = every responding bot that is not a winner
TOP_N_FOR_TOP_GROUP = 10

# Performance metric for the analysis is each bot's AVERAGE spot peer score
# (its summed score divided by the number of questions it forecast), which
# isolates per-question skill from how many questions a bot happened to answer.
# The correlation analysis only includes bots that forecast at least this many
# scored questions, so a bot with a handful of questions cannot swing a result
# with one lucky forecast. This floor applies ONLY to the correlations and their
# score charts; the answer distributions and the winner/top-10 groups still use
# every in-scope participant.
MIN_QUESTIONS_FOR_CORRELATION = 100

# Evidence-summary threshold. "Significant" means the Benjamini-Hochberg q-value
# (false-discovery-rate adjusted p, computed across all measured features) clears
# this bar.
EVIDENCE_SIGNIFICANT_Q = 0.05

# --------------------------------------------------------------------------- #
# Frontier model definition (confirmed with user)
# --------------------------------------------------------------------------- #
# A bot is "frontier" if the model it used for its FINAL prediction is BOTH
# high-powered AND released after the cutoff below (e.g. GPT-5.4 yes, GPT-5.4
# mini no). Release dates were looked up online (Aug 2026); released_after_cutoff
# is computed from the stored date, so correcting a date is enough.
FRONTIER_RELEASE_CUTOFF = date(2025, 11, 1)
FRONTIER_RELEASE_CUTOFF_LABEL = "released after 2025-11-01"


@dataclass(frozen=True)
class ModelInfo:
    """One row of the model registry.

    normalized_key : lowercase, spaces/hyphens stripped form used for matching.
    high_power     : True for full-size flagship models, False for mini / nano /
                     flash / fast / haiku / free-tier / small variants.
    release_date   : looked-up public release date, or None if unknown/unspecified.
    """

    display: str
    normalized_key: str
    high_power: bool
    release_date: date | None
    note: str = ""

    @property
    def released_after_cutoff(self) -> bool:
        return self.release_date is not None and self.release_date > FRONTIER_RELEASE_CUTOFF

    @property
    def is_frontier(self) -> bool:
        return self.high_power and self.released_after_cutoff

    @property
    def release_date_str(self) -> str:
        return self.release_date.isoformat() if self.release_date else "unknown"


def _norm_model(raw: str) -> str:
    """Normalize a model string for matching: lowercase, drop spaces/hyphens/dots-noise."""
    lowered = raw.strip().lower()
    for ch in (" ", "-", "_"):
        lowered = lowered.replace(ch, "")
    return lowered


def _m(
    display: str,
    high_power: bool,
    release_date: date | None,
    note: str = "",
) -> ModelInfo:
    return ModelInfo(
        display=display,
        normalized_key=_norm_model(display),
        high_power=high_power,
        release_date=release_date,
        note=note,
    )


# Registry. Release dates were looked up online in Aug 2026. Order matters only
# for readability; matching sorts by key length. high_power is False whenever the
# name carries a small-tier marker (mini/nano/flash/fast/haiku/free/small/lite).
MODEL_REGISTRY: list[ModelInfo] = [
    # OpenAI GPT-5 family
    _m("GPT-5.5", high_power=True, release_date=date(2026, 4, 23)),
    _m("GPT-5.4 mini", high_power=False, release_date=date(2026, 3, 5)),
    _m("GPT-5.4", high_power=True, release_date=date(2026, 3, 5)),
    _m("GPT-5.2", high_power=True, release_date=date(2025, 12, 11)),
    _m("GPT-5.1", high_power=True, release_date=date(2025, 11, 13)),
    _m("GPT-5 Mini", high_power=False, release_date=date(2025, 8, 7)),
    _m("GPT-5 nano", high_power=False, release_date=date(2025, 8, 7)),
    _m("GPT-5", high_power=True, release_date=date(2025, 8, 7)),
    # OpenAI older
    _m("GPT-4o-mini", high_power=False, release_date=date(2024, 7, 18)),
    _m("gpt-4o-search-preview", high_power=False, release_date=date(2025, 3, 11)),
    _m("GPT-4o", high_power=True, release_date=date(2024, 5, 13)),
    _m("gpt 4.1", high_power=True, release_date=date(2025, 4, 14)),
    _m("o4-mini-deep-research", high_power=False, release_date=date(2025, 6, 26)),
    _m("o4-mini", high_power=False, release_date=date(2025, 4, 16)),
    _m("o3", high_power=True, release_date=date(2025, 4, 16)),
    # Anthropic Claude
    _m("Claude Opus 4.8", high_power=True, release_date=date(2026, 5, 28)),
    _m("Claude Opus 4.7", high_power=True, release_date=date(2026, 4, 16)),
    _m("Claude Opus 4.6", high_power=True, release_date=date(2026, 2, 5)),
    _m("Claude Opus 4.5", high_power=True, release_date=date(2025, 11, 24)),
    _m("Claude Sonnet 4.6", high_power=False, release_date=date(2026, 2, 17),
       note="Sonnet is Anthropic's mid tier below Opus; not counted as highest-power/frontier"),
    _m("Claude Sonnet 4.5", high_power=False, release_date=date(2025, 9, 29),
       note="Sonnet is Anthropic's mid tier below Opus; not counted as highest-power/frontier"),
    _m("Claude Haiku 4.5", high_power=False, release_date=date(2025, 10, 15)),
    _m("claude-3-haiku", high_power=False, release_date=date(2024, 3, 13)),
    # Google Gemini
    _m("Gemini 3.5 Flash", high_power=False, release_date=date(2026, 5, 19)),
    _m("Gemini 3.1 Pro", high_power=True, release_date=date(2026, 2, 19)),
    _m("Gemini 3.1 Flash", high_power=False, release_date=date(2026, 2, 19),
       note="date approximate (shipped with the 3.1 line)"),
    _m("Gemini 3 Pro", high_power=True, release_date=date(2025, 11, 18)),
    _m("Gemini 3 Flash", high_power=False, release_date=date(2025, 12, 17)),
    _m("Gemini 2.5 Pro", high_power=True, release_date=date(2025, 3, 25)),
    _m("Gemini 2.5 Flash", high_power=False, release_date=date(2025, 4, 17)),
    _m("gemini-1.5-flash", high_power=False, release_date=date(2024, 5, 14)),
    # xAI Grok
    _m("Grok 4.3", high_power=True, release_date=date(2026, 4, 17)),
    _m("Grok 4.20", high_power=True, release_date=date(2026, 1, 15),
       note="date approximate (early Jan 2026)"),
    _m("Grok 4.1 Fast", high_power=False, release_date=date(2025, 11, 19)),
    # DeepSeek / others
    _m("Deepseek V4 Pro", high_power=True, release_date=date(2026, 4, 24)),
    _m("Deepseek- v4 flash", high_power=False, release_date=date(2026, 4, 24)),
    _m("Perplexity sonar-reasoning-pro", high_power=False, release_date=date(2025, 2, 1),
       note="specialized search/reasoning model, not a frontier flagship; date approximate"),
    _m("Perplexity Sonar Pro", high_power=False, release_date=date(2025, 2, 1),
       note="specialized search model; date approximate"),
    _m("Kimi", high_power=True, release_date=None,
       note="generic 'Kimi', version unspecified: K2 was Jul 2025 (pre-cutoff), "
            "K2.5 Jan 2026 / K2.6 Apr 2026 (post-cutoff). Left unknown, so not counted as frontier."),
    _m("Cohere", high_power=False, release_date=None),
    # Generic / unspecified buckets (never counted as frontier; flagged instead)
    _m("Finetuned Proprietary Model", high_power=False, release_date=None,
       note="unspecified; excluded from frontier"),
    _m("Open Source Model", high_power=False, release_date=None,
       note="unspecified; excluded from frontier"),
]

# Raw tokens that are parsing artifacts or too vague to map to a model. They are
# recorded in the review doc and never treated as frontier.
MODEL_TOKENS_IGNORED: set[str] = {
    _norm_model(x)
    for x in (
        "etc)",
        "Mutiple Models",
        "openrouter/free",
        "openrouter/openrouter/free",
        "OpenRouter free-tier auto-router",
        "Chinese SOTAs",
        "Chinese SOTAs (GLM",
    )
}

# --------------------------------------------------------------------------- #
# Column slugs -> exact (trimmed) survey header text
# --------------------------------------------------------------------------- #
COLUMNS: dict[str, str] = {
    "timestamp": "Timestamp",
    "bot_name": "What is your bot's name as listed in the Spring 2026 Leaderboard?",
    "confirm": "I confirm that I will answer questions about my bot as it was in the Spring 2026 season",
    "final_model": "Which LLM model(s) did you use to make your final prediction/answer?",
    "support_model": "Which LLM model(s) did you use in supporting roles (i.e. not final predictions)?",
    "iterations": "How many iterations of your primary bot did you make that ended up forecasting tournament questions live?",
    "research": "How did your bot research questions?",
    "strategies": "Did your bot use any of the below forecasting strategies?",
    "development": "What went into the development of your bot?",
    "abandoned": "Tell us about any approaches you tried and abandoned, and why",
    "verification_env": "Did you give an LLM a verification environment (backtest harness, eval set, scoring loop) and let it self-experiment to produce part of your system?",
    "aggregate": "How did you aggregate?",
    "combine": "How did you combine ensemble outputs into the final forecast?",
    "role": "What best describes you?",
    "team_size": "How many people are on your team?",
    "hours": "What is your best estimate for how many total active hours (between all team members) have been put into developing your bot?",
    "llm_calls": "Your best estimate of the number of LLM calls per question?",
    "cost_per_q": "What is your best estimate of cost per Question? (USD)",
    "research_vs_reasoning": "When building, have you optimized more for research (external information retrieval) or reasoning (processing information given to the LLM)?",
    "changed_since_fall": "Did you change how your bot predicted questions in Spring compared to Fall?",
    "code_link": "If you are a prize winner, provide a way to review your code or a general overview of your bot",
    "share_code_consent": "If you included a link to your code or a description, can we share it publicly?",
    "share_response_consent": "Can we share your individual survey response in association with your bot outside our aggregated results?",
    "minibench": "Should we continue running MiniBench?",
    "writeup_rating": "How would you rate the quality and usefulness of Metaculus's research write-ups?",
    "lessons": "In summary, what should other bot makers learn from your experience?",
    "anything_else": "Anything else you want to share? Anything that might be missing from this survey?",
}

# --------------------------------------------------------------------------- #
# Multi-select option vocabularies. Parsing matches these full strings as
# substrings of the raw cell (longest first) so embedded commas never break it.
# Anything left over is captured as an "Other" write-in and logged.
# --------------------------------------------------------------------------- #
MULTISELECT_VOCAB: dict[str, list[str]] = {
    "research": [
        "AskNews DeepNews",
        "Other AskNews",
        "Exa",
        "Tavily",
        "Perplexity",
        "Gemini web search",
        "OpenAI web search",
        "Anthropic web search",
        "XAI web search",
        "Google/Bing/DuckDuckGo Search API (or equivalent like Serp API)",
        "Computer Use Model + Web Browser",
        "Static web scraping (Only HTML, possibly converted to markdown)",
        "Interactive web scraping w/o computer use (Rendering, screenshots, playwright MCP, etc)",
        "A Deep Research Tool",
        "Dedicated tools, such as metaculus/manifold specific search",
    ],
    "strategies": [
        "Capping predictions at a max/min",
        "Mathmatically calibrating/adjusting predictions based on past forecast data",
        "Mathmatically extremizing predictions via code",
        "Taking the median/mean/aggregate of multiple forecasts in probability space",
        "Taking the median/mean/aggregate of multiple forecasts in logit space",
        "Other mathematical adjustments",
        "Check for forecasts on similar Metaculus questions or prediction markets",
        "Explicitly calculate/estimate base rates in a rigorous way",
        "Explicitly run Fermi estimates in a rigorous way",
        "Explicitly consider consider/categorize future scenarios in a rigorous way",
        "Have the LLM do self critiquing or red teaming",
        "Generate and then research subquestions",
        "Generate and then explicity forecast subquestions",
        "Simulating multiple personalities or experts",
        "Run generated code",
        "Collect and analyze pre-existing datasets",
        "Use a library for ochestrating agentic loops (or build your own implementation)",
        "Use skills",
        "Use tool calling",
        "Use subagents",
        "Use MCP servers",
        "Used a specialized fine-tuned LLM for research",
        "Used a specialized fine-tuned LLM for prediction",
        "Custom smoothing math for numeric predictions",
    ],
    "development": [
        "LLM finetuning",
        "Testing via pastcasting (questions that have already resolved)",
        "Testing against community prediction on prediction platforms",
        "Testing via generating >50 of your own questions, forecasting them, and resolving them",
        "Significant testing via manual review of bot outputs (more than sanity checks)",
        "Running multiple bots in parallel to compare results",
        "Custom Evals (e.g. testing base rate finding against known answers)",
        "Running my bot in MiniBench and using results to inform design decisions",
        "Gave an LLM a verification environment, and let it self experiment to produce a part of your system",
    ],
    "verification_env": [
        "No",
        "Yes, for prompt iteration",
        "Yes, for model/ensemble selection",
        "Yes, for retrieval/research strategy",
        "Yes, for code generation of the bot itself",
    ],
    "aggregate": [
        "Same prompt run multiple times, same model (self-consistency)",
        "Same prompt, varied models",
        "Varied prompts, same model",
        "Varied prompts AND varied models",
        "Varied research/retrieval pipelines feeding the same predictor",
        "Varied tooling (e.g. code execution vs. no code execution)",
        "Single model, no ensemble",
    ],
    "combine": [
        "Single model, no ensemble",
        "Simple mean/median",
        "Trimmed mean or outlier-removed mean",
        "Weighted average, weights based on model identity/hardcoded",
        "Weighted average, weights based on self-reported confidence",
        "Weighted average, weights based on research quality/source count",
        "LLM-as-aggregator (a model reads the others' outputs and decides)",
    ],
    # Parsed (so reviewed write-in adjustments still resolve) but excluded from
    # the report; see EXCLUDED_COLUMNS.
    "minibench": [
        "Yes, it is a valuable tool for iterating with my bot",
        "Yes, I wouldn't have joined otherwise, it was useful to get started",
        "Yes, its another chance to earn prize money",
        "Yes, other reasons",
        "No, I would not be disappointed if it went away",
        "Only if you add more diversity of questions (e.g. LLM generated questions)",
    ],
}

# --------------------------------------------------------------------------- #
# Single-select canonical option sets. A raw value that matches one of these
# (exactly, case-insensitively) is kept; anything else is bucketed as
# "Other (excluded)" and logged.
# --------------------------------------------------------------------------- #
SINGLE_SELECT_VOCAB: dict[str, list[str]] = {
    "iterations": [
        "0 (I made it and let it loose)",
        "1-2",
        "3-5",
        "6-10",
        "11-20",
        "21-50",
        "51+",
    ],
    "role": [
        "Hobbyist(s) with professional software experience",
        "Hobbyist(s) without professional software experience",
        "Student(s)",
        "Academic Researcher(s)",
        "Commercial Entity",
    ],
    # Exactly the buckets the Spring form offered: it jumps from 41-80hrs to the
    # full-time-month options (no 81-160hrs / 161-320hrs, which were Fall buckets).
    "hours": [
        "0-8hrs",
        "9-15hrs",
        "16-40hrs",
        "41-80hrs",
        "2 full time weeks - 1 full time month",
        "1 full time month - 4 full time months",
        "4 full time months +",
    ],
    "llm_calls": ["1", "2-5", "5-10", "10-20", "20-50", "50-100", "100+"],
    "cost_per_q": [
        "$0-0.09",
        "$0.1-0.99",
        "$1-2.99",
        "$3-4.99",
        "$5-9.99",
        "$10-19.99",
        "$20+",
    ],
    "research_vs_reasoning": [
        "Strong research lean",
        "Slight research lean",
        "About the same for both",
        "Slight reasoning lean",
        "Strong reasoning lean",
    ],
    "changed_since_fall": [
        "Yes",
        "No",
        "I didn't participate in the Fall tournament",
    ],
    # Parsed (so reviewed write-in adjustments still resolve) but excluded from
    # the report; see EXCLUDED_COLUMNS.
    "writeup_rating": [
        "Neglible: They did not impact my performance",
        "Useful: I would have probably done worse without them",
        "Very Useful: I would have done notably worse with out them",
    ],
}

# Ordinal ordering (low -> high) for single-selects that are ranked. Used for
# Spearman correlation and for ordering chart categories. Must stay monotone
# with the midpoint maps below (tested in test_config_integrity).
ORDINAL_ORDER: dict[str, list[str]] = {
    "iterations": SINGLE_SELECT_VOCAB["iterations"],
    "hours": SINGLE_SELECT_VOCAB["hours"],
    "llm_calls": SINGLE_SELECT_VOCAB["llm_calls"],
    "cost_per_q": SINGLE_SELECT_VOCAB["cost_per_q"],
    "research_vs_reasoning": SINGLE_SELECT_VOCAB["research_vs_reasoning"],
}

# --------------------------------------------------------------------------- #
# Bucket -> numeric midpoint maps (for correlation against peer score).
# Mixed-unit hours are converted to approximate total hours.
# --------------------------------------------------------------------------- #
# The full-time buckets are converted with 1 full-time week = 40 hours (so a
# full-time month = 160 hours): "2 full time weeks - 1 full time month" = 80-160h
# (mid 120), "1-4 full time months" = 160-640h (mid 400), "4 full time months +"
# = 640h+ (open-ended, 700).
HOURS_MIDPOINT: dict[str, float] = {
    "0-8hrs": 4,
    "9-15hrs": 12,
    "16-40hrs": 28,
    "41-80hrs": 60,
    "2 full time weeks - 1 full time month": 120,
    "1 full time month - 4 full time months": 400,
    "4 full time months +": 700,
}
LLM_CALLS_MIDPOINT: dict[str, float] = {
    "1": 1,
    "2-5": 3.5,
    "5-10": 7.5,
    "10-20": 15,
    "20-50": 35,
    "50-100": 75,
    "100+": 150,
}
COST_MIDPOINT: dict[str, float] = {
    "$0-0.09": 0.045,
    "$0.1-0.99": 0.5,
    "$1-2.99": 2.0,
    "$3-4.99": 4.0,
    "$5-9.99": 7.5,
    "$10-19.99": 15.0,
    "$20+": 25.0,
}
ITERATIONS_MIDPOINT: dict[str, float] = {
    "0 (I made it and let it loose)": 0,
    "1-2": 1.5,
    "3-5": 4,
    "6-10": 8,
    "11-20": 15,
    "21-50": 35,
    "51+": 75,
}

# Research tools counted toward the Fall "breadth of research" headline metric.
RESEARCH_SOURCE_OPTIONS: list[str] = MULTISELECT_VOCAB["research"]

# --------------------------------------------------------------------------- #
# Derived boolean features (Fall headline habits). Each maps to a multi-select
# column and a substring that marks the habit as present.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BooleanFeature:
    key: str
    label: str
    column_slug: str
    match_substring: str
    # `definition` states plainly what a "yes" means, so a reader of the yes/no
    # chart knows which survey answer triggers it. It should complete the sentence
    # "Yes = ...". The exact matched option text is the corresponding row of the
    # research/strategy/etc. vocabulary in this file.
    definition: str = ""


BOOLEAN_FEATURES: list[BooleanFeature] = [
    BooleanFeature("uses_asknews", "Uses AskNews", "research", "asknews",
                   "the bot's research used AskNews (AskNews DeepNews or Other AskNews)"),
    BooleanFeature("uses_exa", "Uses Exa", "research", "exa",
                   "the bot's research used Exa"),
    BooleanFeature("uses_perplexity", "Uses Perplexity", "research", "perplexity",
                   "the bot's research used Perplexity"),
    BooleanFeature("uses_openai_search", "Uses OpenAI web search", "research", "openai web search",
                   "the bot's research used OpenAI web search"),
    BooleanFeature("uses_scraping", "Uses web scraping", "research", "scraping",
                   "the bot's research used static or interactive web scraping"),
    BooleanFeature("uses_aggregation", "Aggregates multiple forecasts", "strategies", "median/mean/aggregate",
                   "the bot took the median, mean, or aggregate of multiple forecasts"),
    BooleanFeature("uses_base_rates", "Uses explicit base rates", "strategies", "base rates",
                   "the bot explicitly estimated base rates in a rigorous way"),
    BooleanFeature("uses_scenarios", "Uses scenario analysis", "strategies", "future scenarios",
                   "the bot explicitly considered or categorized future scenarios"),
    BooleanFeature("uses_similar_qs", "Checks similar questions/markets", "strategies", "similar Metaculus questions",
                   "the bot checked similar Metaculus questions or prediction markets"),
    BooleanFeature("uses_capping", "Caps predictions", "strategies", "capping predictions",
                   "the bot capped predictions at a max/min"),
    BooleanFeature("uses_extremizing", "Extremizes predictions", "strategies", "extremizing",
                   "the bot mathematically extremized predictions via code"),
    BooleanFeature("uses_self_critique", "Uses LLM self-critique / red team", "strategies", "self critiquing",
                   "the bot had the LLM self-critique or red-team its forecasts"),
    BooleanFeature("uses_subquestions", "Researches subquestions", "strategies", "research subquestions",
                   "the bot generated and researched subquestions"),
    BooleanFeature("manual_review", "Does manual review of outputs", "development", "manual review",
                   "the maker did significant manual review of bot outputs, beyond sanity checks"),
    BooleanFeature("uses_pastcasting", "Tests via pastcasting", "development", "pastcasting",
                   "the bot was tested via pastcasting (questions that already resolved)"),
    BooleanFeature("vs_community", "Tests vs community prediction", "development", "community prediction",
                   "the bot was tested against community predictions on prediction platforms"),
    BooleanFeature("uses_minibench", "Uses MiniBench for design", "development", "minibench",
                   "the maker ran the bot in MiniBench and used the results to inform design"),
    BooleanFeature("used_verification_env", "Gave LLM a verification env", "verification_env", "yes,",
                   "the maker answered 'Yes' (for any purpose) to giving an LLM a verification environment"),
    BooleanFeature("varied_models", "Ensemble uses multiple models", "aggregate", "varied models",
                   "the ensemble combined more than one model (answered 'Same prompt, varied models' or "
                   "'Varied prompts AND varied models')"),
]

# --------------------------------------------------------------------------- #
# Question specs that drive the report. One entry per structured question we
# chart. `chart` selects the plot style; `correlations` lists the analyses run.
# Free-text / consent / identifier columns are intentionally omitted (see
# EXCLUDED_COLUMNS) and reported as such in the review doc.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QuestionSpec:
    slug: str
    title: str
    kind: str  # "multiselect" | "single_select" | "ordinal" | "numeric" | "model"
    correlations: list[str] = field(default_factory=list)


QUESTION_SPECS: list[QuestionSpec] = [
    QuestionSpec(
        "final_model",
        "Final-prediction models",
        "model",
        ["frontier", "gpt_5_4", "gpt_5x_high", "opus", "opus_4_6", "final_model_release"],
    ),
    QuestionSpec("support_model", "Supporting-role models", "model", ["frontier_support"]),
    QuestionSpec("research", "How bots researched questions", "multiselect",
                 ["n_research_sources", "uses_asknews", "uses_exa", "uses_perplexity",
                  "uses_openai_search", "uses_scraping"]),
    QuestionSpec("strategies", "Forecasting strategies used", "multiselect",
                 ["uses_aggregation", "uses_base_rates", "uses_similar_qs", "uses_subquestions",
                  "uses_scenarios", "uses_self_critique", "uses_capping", "uses_extremizing"]),
    QuestionSpec("development", "What went into development", "multiselect",
                 ["manual_review", "uses_minibench", "uses_pastcasting", "vs_community"]),
    QuestionSpec("verification_env", "LLM self-experimentation (verification env)", "multiselect",
                 ["used_verification_env"]),
    QuestionSpec("aggregate", "Ensemble aggregation approach", "multiselect",
                 ["varied_models"]),
    QuestionSpec("combine", "Combining ensemble outputs", "multiselect", []),
    QuestionSpec("iterations", "Iterations that went live", "ordinal",
                 ["iterations_mid"]),
    QuestionSpec("role", "Who the makers are", "single_select", []),
    QuestionSpec("team_size", "Team size", "numeric", ["team_size"]),
    QuestionSpec("hours", "Total active hours on the bot", "ordinal", ["hours_mid"]),
    QuestionSpec("llm_calls", "LLM calls per question", "ordinal", ["llm_calls_mid"]),
    QuestionSpec("cost_per_q", "Cost per question (USD)", "ordinal", ["cost_mid"]),
    QuestionSpec("research_vs_reasoning", "Research vs reasoning optimization", "ordinal",
                 ["research_vs_reasoning_ord"]),
    QuestionSpec("changed_since_fall", "Changed approach since Fall", "single_select", []),
]

# Columns deliberately not charted, with the reason surfaced in the review doc.
EXCLUDED_COLUMNS: dict[str, str] = {
    "timestamp": "Form metadata, not an analysis variable.",
    "bot_name": "Respondent identifier / join key, not a survey answer.",
    "confirm": "Consent checkbox with free-text variants; no analytic value.",
    "abandoned": "Open free text; not chartable, no fixed options.",
    "code_link": "Free-text code link; identifier, not a variable.",
    "share_code_consent": "Consent field; gates sharing, not analyzed.",
    "share_response_consent": "Consent field; gates sharing, not analyzed.",
    "lessons": "Open free text; not chartable.",
    "anything_else": "Open free text; not chartable.",
    "minibench": "Opinion poll on whether to keep MiniBench; not a bot behavior, removed from the analysis.",
    "writeup_rating": "Rating of Metaculus write-ups; removed from the analysis.",
}

# Group display config for split charts.
# GROUP_ORDER is the set of analytic groups a respondent can belong to.
# CHART_GROUP_ORDER adds an "Everyone" baseline series shown on distribution
# charts (all in-scope participants, regardless of group).
GROUP_ORDER: list[str] = ["non_winner", "winner", "top_10"]
CHART_GROUP_ORDER: list[str] = ["everyone", "non_winner", "winner", "top_10"]
GROUP_LABELS: dict[str, str] = {
    "everyone": "Everyone",
    "non_winner": "Non-winners",
    "winner": "Winners",
    "top_10": "Top 10 (peer score)",
}
GROUP_COLORS: dict[str, str] = {
    "everyone": "#2d3748",
    "non_winner": "#9aa5b1",
    "winner": "#2f80ed",
    "top_10": "#f2994a",
}

"""Season configs. Add a new one here to analyse a new survey."""

from __future__ import annotations

from pathlib import Path

from aib_analysis.survey_analysis.config import (
    BinaryFeature,
    CategoricalFeature,
    CountFeature,
    ModelTiering,
    OrdinalFeature,
    SeasonConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Shared bucket -> midpoint maps
# ---------------------------------------------------------------------------

HOURS_MIDPOINTS = {
    "0-8hrs": 4,
    "9-15hrs": 12,
    "1-15hrs": 8,
    "<16hrs": 8,
    "16-40hrs": 28,
    "41-80hrs": 60,
    "80-160hrs": 120,
    "81-160hrs": 120,
    "160-320hrs": 240,
    "161-320hrs": 240,
    "320+hrs": 500,
    "321+hrs": 500,
    ">320hrs": 500,
    # Full-time phrasings. One full-time week is treated as 40 hours.
    "2 full time weeks - 1 full time month": 120,
    "1 full time month - 4 full time months": 400,
    "4 full time months +": 700,
}

LLM_CALL_MIDPOINTS = {
    "1": 1,
    "<5": 3,
    "1-5": 3,
    "2-5": 3.5,
    "5-10": 7.5,
    "10-20": 15,
    "20-50": 35,
    "50-100": 75,
    "100+": 150,
}

COST_MIDPOINTS = {
    "$0-0.09": 0.045,
    "$0-$0.09": 0.045,
    "<$0.1": 0.05,
    "$0.1-0.99": 0.5,
    "$0.1-$0.99": 0.5,
    "$1-2.99": 2.0,
    "$1-$2.99": 2.0,
    "$1-4.99": 2.5,
    "$1-$4.99": 2.5,
    "$3-4.99": 4.0,
    "$3-$4.99": 4.0,
    "$5-9.99": 7.5,
    "$5-$9.99": 7.5,
    "$10-19.99": 15.0,
    "$10-24.99": 17.0,
    "$25-49.99": 37.0,
    "$50+": 75.0,
}

ITERATION_MIDPOINTS = {
    "0 (i made it and let it loose)": 0,
    "1-2": 1.5,
    "3-5": 4,
    "6-10": 8,
    "11-20": 15,
    "21-50": 35,
    "51+": 60,
}

# ---------------------------------------------------------------------------
# Multi-select option catalogs (verbatim prefixes from the Google Form)
# ---------------------------------------------------------------------------

RESEARCH_SOURCES = (
    "AskNews DeepNews",
    "Other AskNews",
    "Exa",
    "Tavily",
    "Perplexity",
    "OpenAI web search",
    "Anthropic web search",
    "Gemini web search",
    "XAI web search",
    "Google/Bing/DuckDuckGo Search API",
    "Computer Use Model + Web Browser",
    "Static web scraping",
    "Interactive web scraping",
)

SPRING_2026 = SeasonConfig(
    season="Spring 2026",
    survey_csv=REPO_ROOT
    / "local/private_input_data/fall_and_spring_survey_data"
    / "Spring 2026 FutureEval Participant Survey (Responses) - Form Responses 1.csv",
    participation_csv=REPO_ROOT
    / "local/private_input_data/fall_and_spring_survey_data"
    / "Spring 2026 FutureEval Prize and participation stats.csv",
    leaderboard_csv=REPO_ROOT
    / "temp/spring_survey_analysis/data/spring_2026_leaderboard.csv",
    output_dir=REPO_ROOT / "temp/spring_survey_analysis",
    # 'MWG Bot' in the survey is registered as bot 'MWG' under owner MWGHuman.
    name_aliases={"mwgbot": "mwg"},
    # Preseen-Chestnut placed 5th on the raw leaderboard but was disqualified
    # and paid no prize.
    disqualified_bots=("Preseen-Chestnut",),
    binary_features=(
        # --- Forecasting strategies -------------------------------------
        BinaryFeature(
            "uses_aggregation",
            "forecasting_strategies",
            ("median/mean/aggregate",),
            "Aggregate multiple forecasts",
            "strategy",
        ),
        BinaryFeature(
            "uses_logit_space",
            "forecasting_strategies",
            ("aggregate of multiple forecasts in logit space",),
            "Aggregate in logit space",
            "strategy",
        ),
        BinaryFeature(
            "uses_capping",
            "forecasting_strategies",
            ("capping predictions at a max/min",),
            "Cap predictions at max/min",
            "strategy",
        ),
        BinaryFeature(
            "uses_extremizing",
            "forecasting_strategies",
            ("extremizing predictions via code",),
            "Extremize via code",
            "strategy",
        ),
        BinaryFeature(
            "uses_calibration",
            "forecasting_strategies",
            ("calibrating/adjusting predictions based on past",),
            "Calibrate on past forecast data",
            "strategy",
        ),
        BinaryFeature(
            "uses_base_rates",
            "forecasting_strategies",
            ("base rate",),
            "Calculate base rates",
            "strategy",
        ),
        BinaryFeature(
            "uses_scenarios",
            "forecasting_strategies",
            ("categorize future scenarios",),
            "Categorize future scenarios",
            "strategy",
        ),
        BinaryFeature(
            "uses_similar_qs",
            "forecasting_strategies",
            ("forecasts on similar metaculus questions",),
            "Check similar questions/markets",
            "strategy",
        ),
        BinaryFeature(
            "uses_self_critique",
            "forecasting_strategies",
            ("self critiquing or red teaming",),
            "LLM self-critique / red team",
            "strategy",
        ),
        BinaryFeature(
            "uses_subquestions",
            "forecasting_strategies",
            ("generate and then research subquestions",),
            "Research subquestions",
            "strategy",
        ),
        BinaryFeature(
            "uses_personas",
            "forecasting_strategies",
            ("multiple personalities or experts",),
            "Simulate multiple experts",
            "strategy",
        ),
        BinaryFeature(
            "uses_tool_calling",
            "forecasting_strategies",
            ("use tool calling",),
            "Tool calling",
            "strategy",
        ),
        BinaryFeature(
            "uses_agentic_loop",
            "forecasting_strategies",
            ("ochestrating agentic loops", "orchestrating agentic loops"),
            "Agentic loop library",
            "strategy",
        ),
        BinaryFeature(
            "uses_code_execution",
            "forecasting_strategies",
            ("run generated code",),
            "Run generated code",
            "strategy",
        ),
        BinaryFeature(
            "uses_smoothing",
            "forecasting_strategies",
            ("custom smoothing math",),
            "Custom numeric smoothing",
            "strategy",
        ),
        BinaryFeature(
            "uses_fermi",
            "forecasting_strategies",
            ("fermi estimates",),
            "Fermi estimates",
            "strategy",
        ),
        # --- Development practices --------------------------------------
        BinaryFeature(
            "manual_review",
            "development",
            ("manual review of bot outputs",),
            "Manual review of outputs",
            "development",
        ),
        BinaryFeature(
            "pastcasting",
            "development",
            ("pastcasting",),
            "Pastcasting on resolved questions",
            "development",
        ),
        BinaryFeature(
            "vs_community",
            "development",
            ("against community prediction",),
            "Test vs community prediction",
            "development",
        ),
        BinaryFeature(
            "uses_minibench",
            "development",
            ("minibench",),
            "Iterate using MiniBench",
            "development",
        ),
        BinaryFeature(
            "parallel_bots",
            "development",
            ("multiple bots in parallel",),
            "Run bots in parallel to compare",
            "development",
        ),
        BinaryFeature(
            "custom_evals",
            "development",
            ("custom evals",),
            "Custom evals",
            "development",
        ),
        BinaryFeature(
            "finetuning",
            "development",
            ("llm finetuning",),
            "LLM finetuning",
            "development",
        ),
        BinaryFeature(
            "self_generated_questions",
            "development",
            ("generating >50 of your own questions",),
            "Generate own question set",
            "development",
        ),
        # --- Research sources -------------------------------------------
        BinaryFeature(
            "uses_asknews", "research", ("asknews",), "AskNews (any)", "research"
        ),
        BinaryFeature(
            "uses_exa", "research", ("exa",), "Exa", "research"
        ),
        BinaryFeature(
            "uses_tavily", "research", ("tavily",), "Tavily", "research"
        ),
        BinaryFeature(
            "uses_perplexity", "research", ("perplexity",), "Perplexity", "research"
        ),
        BinaryFeature(
            "uses_openai_search",
            "research",
            ("openai web search",),
            "OpenAI web search",
            "research",
        ),
        BinaryFeature(
            "uses_search_api",
            "research",
            ("google/bing/duckduckgo",),
            "Google/Bing/DDG search API",
            "research",
        ),
        BinaryFeature(
            "uses_scraping",
            "research",
            ("web scraping",),
            "Web scraping (any)",
            "research",
        ),
        BinaryFeature(
            "uses_computer_use",
            "research",
            ("computer use model",),
            "Computer-use browsing",
            "research",
        ),
        # --- Spring-specific additions ----------------------------------
        BinaryFeature(
            "used_verification_env",
            "verification_env",
            ("yes,",),
            "LLM self-experiment harness",
            "development",
        ),
        BinaryFeature(
            "ensemble_varied_models",
            "aggregation_method",
            ("varied models",),
            "Ensemble across varied models",
            "ensemble",
        ),
        BinaryFeature(
            "ensemble_varied_prompts",
            "aggregation_method",
            ("varied prompts",),
            "Ensemble across varied prompts",
            "ensemble",
        ),
        BinaryFeature(
            "ensemble_varied_research",
            "aggregation_method",
            ("varied research/retrieval pipelines",),
            "Ensemble across varied research",
            "ensemble",
        ),
        BinaryFeature(
            "ensemble_self_consistency",
            "aggregation_method",
            ("self-consistency",),
            "Self-consistency resampling",
            "ensemble",
        ),
        BinaryFeature(
            "combine_llm_aggregator",
            "ensemble_combination",
            ("llm-as-aggregator",),
            "LLM-as-aggregator",
            "ensemble",
        ),
        BinaryFeature(
            "combine_trimmed",
            "ensemble_combination",
            ("trimmed mean",),
            "Trimmed / outlier-removed mean",
            "ensemble",
        ),
        BinaryFeature(
            "combine_weighted",
            "ensemble_combination",
            ("weighted average",),
            "Weighted average",
            "ensemble",
        ),
        BinaryFeature(
            "combine_simple_mean",
            "ensemble_combination",
            ("simple mean/median",),
            "Simple mean/median",
            "ensemble",
        ),
        BinaryFeature(
            "no_ensemble",
            "ensemble_combination",
            ("single model, no ensemble",),
            "No ensemble at all",
            "ensemble",
        ),
        # Returning participants had a prior season to iterate against. Tested
        # as a feature in its own right so it can be checked as a confounder
        # for anything else that separates the cohorts.
        BinaryFeature(
            "returning_participant",
            "changed_since_last",
            ("didn't participate", "did not participate"),
            "Competed in the previous season",
            "experience",
            negate=True,
        ),
    ),
    ordinal_features=(
        OrdinalFeature(
            "hours_mid",
            "hours",
            HOURS_MIDPOINTS,
            "Dev hours",
            "hrs",
            log_scale=True,
        ),
        OrdinalFeature(
            "llm_calls_mid",
            "llm_calls",
            LLM_CALL_MIDPOINTS,
            "LLM calls per question",
            "calls",
            fallback_patterns=((r"probably\s*5-10", 7.5),),
            log_scale=True,
        ),
        OrdinalFeature(
            "cost_mid",
            "cost_per_q",
            COST_MIDPOINTS,
            "Cost per question",
            "USD",
            fallback_patterns=(
                (r"free tier", 0.0),
                (r"between\s*%?\$?0\.70\s*and\s*\$?1\.30", 1.0),
                (r"more like \$?1-5", 3.0),
            ),
            log_scale=True,
        ),
        OrdinalFeature(
            "iterations_mid",
            "iterations",
            ITERATION_MIDPOINTS,
            "Live bot iterations",
            "revisions",
            fallback_patterns=(
                (r"~?10\b", 10),
                (r"continuous development", 20),
            ),
        ),
    ),
    count_features=(
        CountFeature(
            "n_research_sources",
            "research",
            RESEARCH_SOURCES,
            "Distinct research sources",
        ),
    ),
    categorical_features=(
        CategoricalFeature(
            "research_vs_reasoning",
            "research_vs_reasoning",
            "Research vs reasoning lean",
            order=(
                "Strong research lean",
                "Slight research lean",
                "About the same for both",
                "Slight reasoning lean",
                "Strong reasoning lean",
            ),
            ordinal_scores={
                "Strong research lean": 2,
                "Slight research lean": 1,
                "About the same for both": 0,
                "Slight reasoning lean": -1,
                "Strong reasoning lean": -2,
            },
        ),
        CategoricalFeature(
            "respondent_type",
            "respondent_type",
            "Respondent type",
            order=(
                "Hobbyist(s) with professional software experience",
                "Hobbyist(s) without professional software experience",
                "Student(s)",
                "Commercial Entity",
                "Academic Researcher(s)",
            ),
        ),
        CategoricalFeature(
            "writeup_rating",
            "writeup_rating",
            "Rating of Metaculus research write-ups",
            order=(
                "Very Useful: I would have done notably worse with out them",
                "Useful: I would have probably done worse without them",
                "Neglible: They did not impact my performance",
            ),
            ordinal_scores={
                "Very Useful: I would have done notably worse with out them": 2,
                "Useful: I would have probably done worse without them": 1,
                "Neglible: They did not impact my performance": 0,
            },
        ),
        CategoricalFeature(
            "ensemble_combination",
            "ensemble_combination",
            "Ensemble combination method",
        ),
    ),
    model_tiering=ModelTiering(
        frontier=(
            r"\bgpt[- ]?5(\.\d)?\b",
            r"\bclaude[- ]?(opus|sonnet)[- ]?4\.[5-9]\b",
            r"\b(opus|sonnet)[- ]?4\.[5-9]\b",
            r"\bgemini[- ]?3(\.\d)?[- ]?pro\b",
            r"\bgrok[- ]?4\.\d\b",
            r"\bo[34](-pro|-high)?\b",
            r"\bdeepseek[- ]?(r1|v3\.2|v4)\b",
            r"\bqwen3[- ]?max\b",
            r"finetuned",
        ),
        mid=(
            r"\bgemini[- ]?3[- ]?flash\b",
            r"\bgrok[- ]?4\b",
            r"\bclaude[- ]?(opus|sonnet)[- ]?4(\.[0-4])?\b",
            r"\bgpt[- ]?4\.1\b",
            r"\bgemini[- ]?2\.5[- ]?pro\b",
            r"\bo4[- ]?mini\b",
            # Strong open-weight models named without a version by respondents.
            r"\bglm[- ]?\d?\b",
            r"\bkimi\b",
            r"chinese sota",
        ),
        legacy=(
            r"\bgpt[- ]?4o\b",
            r"\b4o\b",
            r"\bgpt[- ]?4\b",
            r"\bclaude[- ]?3\.\d\b",
            r"\bsonnet[- ]?3\.\d\b",
            r"\bgemini[- ]?2\.\d[- ]?flash\b",
            r"openrouter/free",
            r"free tier",
        ),
    ),
)


SEASONS: dict[str, SeasonConfig] = {
    "spring-2026": SPRING_2026,
}

"""Alternative capability-metric robustness check.

The 19-predictor OLS regression is sensitive to the choice of model
capability metric. The default specification uses the Artificial Analysis
Intelligence Index, which is a composite of single-turn benchmarks and may
conflate models with different agentic behaviour. This script re-runs the
regression with two alternative capability metrics and reports whether the
core findings are robust to the choice of metric.

Metric A — Agentic Capability Index (ACI):
    Each model's mean single-agent performance across all six benchmarks.
    Directly measures agentic ability rather than static benchmark composites.

Metric B — Per-dataset Single-Agent Baseline (PSB):
    Each model's single-agent performance on the same dataset as the
    regression row. This is already present in the formula as
    ``single_agent_baseline`` but is here used in place of (not in addition
    to) the Intelligence Index, replacing ``intelligence_centered`` and
    ``intelligence_sq_centered``.

For each metric the script reports R²_train, R²_CV (5-fold), a full
coefficient table, and a key-findings check. A final comparison summarises
which findings are robust to the choice of capability metric.

Run:
    python scripts/alternative_capability_metric.py
"""

import os
import sys
import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')

# ==============================================================================
# EMBEDDED DATA — ORIGINAL 180 CONFIGURATIONS (4 benchmarks)
# Source: extended_mixed_effects_6benchmarks.py  (DO NOT MODIFY)
# ==============================================================================
ORIGINAL_DATA = [
    # BrowseComp-Plus
    ['browsecomp_plus_sampled_100', 'multi-agent-centralized',   'anthropic', 'claude-3-7-sonnet-20250219', 0.3434343434343434],
    ['browsecomp_plus_sampled_100', 'multi-agent-centralized',   'anthropic', 'claude-sonnet-4-20250514',   0.32323232323232326],
    ['browsecomp_plus_sampled_100', 'multi-agent-centralized',   'anthropic', 'claude-sonnet-4-5',          0.42857142857142855],
    ['browsecomp_plus_sampled_100', 'multi-agent-centralized',   'gemini',    'gemini-2.0-flash',           0.22],
    ['browsecomp_plus_sampled_100', 'multi-agent-centralized',   'gemini',    'gemini-2.5-flash',           0.31],
    ['browsecomp_plus_sampled_100', 'multi-agent-centralized',   'gemini',    'gemini-2.5-pro',             0.37],
    ['browsecomp_plus_sampled_100', 'multi-agent-centralized',   'openai',    'gpt-5',                      0.34],
    ['browsecomp_plus_sampled_100', 'multi-agent-centralized',   'openai',    'gpt-5-mini',                 0.26],
    ['browsecomp_plus_sampled_100', 'multi-agent-centralized',   'openai',    'gpt-5-nano',                 0.27],
    ['browsecomp_plus_sampled_100', 'multi-agent-decentralized', 'anthropic', 'claude-3-7-sonnet-20250219', 0.29292929292929293],
    ['browsecomp_plus_sampled_100', 'multi-agent-decentralized', 'anthropic', 'claude-sonnet-4-20250514',   0.37373737373737376],
    ['browsecomp_plus_sampled_100', 'multi-agent-decentralized', 'anthropic', 'claude-sonnet-4-5',          0.43434343434343436],
    ['browsecomp_plus_sampled_100', 'multi-agent-decentralized', 'gemini',    'gemini-2.0-flash',           0.18],
    ['browsecomp_plus_sampled_100', 'multi-agent-decentralized', 'gemini',    'gemini-2.5-flash',           0.26],
    ['browsecomp_plus_sampled_100', 'multi-agent-decentralized', 'gemini',    'gemini-2.5-pro',             0.43],
    ['browsecomp_plus_sampled_100', 'multi-agent-decentralized', 'openai',    'gpt-5',                      0.5],
    ['browsecomp_plus_sampled_100', 'multi-agent-decentralized', 'openai',    'gpt-5-mini',                 0.33],
    ['browsecomp_plus_sampled_100', 'multi-agent-decentralized', 'openai',    'gpt-5-nano',                 0.32],
    ['browsecomp_plus_sampled_100', 'multi-agent-hybrid',        'anthropic', 'claude-3-7-sonnet-20250219', 0.3333333333333333],
    ['browsecomp_plus_sampled_100', 'multi-agent-hybrid',        'anthropic', 'claude-sonnet-4-20250514',   0.41414141414141414],
    ['browsecomp_plus_sampled_100', 'multi-agent-hybrid',        'anthropic', 'claude-sonnet-4-5',          0.40404040404040403],
    ['browsecomp_plus_sampled_100', 'multi-agent-hybrid',        'gemini',    'gemini-2.0-flash',           0.2],
    ['browsecomp_plus_sampled_100', 'multi-agent-hybrid',        'gemini',    'gemini-2.5-flash',           0.32],
    ['browsecomp_plus_sampled_100', 'multi-agent-hybrid',        'gemini',    'gemini-2.5-pro',             0.4],
    ['browsecomp_plus_sampled_100', 'multi-agent-hybrid',        'openai',    'gpt-5',                      0.38],
    ['browsecomp_plus_sampled_100', 'multi-agent-hybrid',        'openai',    'gpt-5-mini',                 0.24],
    ['browsecomp_plus_sampled_100', 'multi-agent-hybrid',        'openai',    'gpt-5-nano',                 0.33],
    ['browsecomp_plus_sampled_100', 'multi-agent-independent',   'anthropic', 'claude-3-7-sonnet-20250219', 0.18181818181818182],
    ['browsecomp_plus_sampled_100', 'multi-agent-independent',   'anthropic', 'claude-sonnet-4-20250514',   0.1111111111111111],
    ['browsecomp_plus_sampled_100', 'multi-agent-independent',   'anthropic', 'claude-sonnet-4-5',          0.1414141414141414],
    ['browsecomp_plus_sampled_100', 'multi-agent-independent',   'gemini',    'gemini-2.0-flash',           0.1],
    ['browsecomp_plus_sampled_100', 'multi-agent-independent',   'gemini',    'gemini-2.5-flash',           0.21],
    ['browsecomp_plus_sampled_100', 'multi-agent-independent',   'gemini',    'gemini-2.5-pro',             0.24],
    ['browsecomp_plus_sampled_100', 'multi-agent-independent',   'openai',    'gpt-5',                      0.44],
    ['browsecomp_plus_sampled_100', 'multi-agent-independent',   'openai',    'gpt-5-mini',                 0.25],
    ['browsecomp_plus_sampled_100', 'multi-agent-independent',   'openai',    'gpt-5-nano',                 0.18],
    ['browsecomp_plus_sampled_100', 'single-agent',              'anthropic', 'claude-3-7-sonnet-20250219', 0.26262626262626265],
    ['browsecomp_plus_sampled_100', 'single-agent',              'anthropic', 'claude-sonnet-4-20250514',   0.29292929292929293],
    ['browsecomp_plus_sampled_100', 'single-agent',              'anthropic', 'claude-sonnet-4-5',          0.3434343434343434],
    ['browsecomp_plus_sampled_100', 'single-agent',              'gemini',    'gemini-2.0-flash',           0.17],
    ['browsecomp_plus_sampled_100', 'single-agent',              'gemini',    'gemini-2.5-flash',           0.28],
    ['browsecomp_plus_sampled_100', 'single-agent',              'gemini',    'gemini-2.5-pro',             0.36],
    ['browsecomp_plus_sampled_100', 'single-agent',              'openai',    'gpt-5',                      0.37],
    ['browsecomp_plus_sampled_100', 'single-agent',              'openai',    'gpt-5-mini',                 0.41],
    ['browsecomp_plus_sampled_100', 'single-agent',              'openai',    'gpt-5-nano',                 0.37],
    # Finance-Agent
    ['finance-agent', 'multi-agent-centralized',   'anthropic', 'claude-3-7-sonnet-20250219', 0.3],
    ['finance-agent', 'multi-agent-centralized',   'anthropic', 'claude-sonnet-4-20250514',   0.42],
    ['finance-agent', 'multi-agent-centralized',   'anthropic', 'claude-sonnet-4-5',          0.46],
    ['finance-agent', 'multi-agent-centralized',   'gemini',    'gemini-2.0-flash',           0.7],
    ['finance-agent', 'multi-agent-centralized',   'gemini',    'gemini-2.5-flash',           0.74],
    ['finance-agent', 'multi-agent-centralized',   'gemini',    'gemini-2.5-pro',             0.78],
    ['finance-agent', 'multi-agent-centralized',   'openai',    'gpt-5',                      0.8],
    ['finance-agent', 'multi-agent-centralized',   'openai',    'gpt-5-mini',                 0.72],
    ['finance-agent', 'multi-agent-centralized',   'openai',    'gpt-5-nano',                 0.76],
    ['finance-agent', 'multi-agent-decentralized', 'anthropic', 'claude-3-7-sonnet-20250219', 0.2],
    ['finance-agent', 'multi-agent-decentralized', 'anthropic', 'claude-sonnet-4-20250514',   0.32],
    ['finance-agent', 'multi-agent-decentralized', 'anthropic', 'claude-sonnet-4-5',          0.44],
    ['finance-agent', 'multi-agent-decentralized', 'gemini',    'gemini-2.0-flash',           0.72],
    ['finance-agent', 'multi-agent-decentralized', 'gemini',    'gemini-2.5-flash',           0.74],
    ['finance-agent', 'multi-agent-decentralized', 'gemini',    'gemini-2.5-pro',             0.76],
    ['finance-agent', 'multi-agent-decentralized', 'openai',    'gpt-5',                      0.78],
    ['finance-agent', 'multi-agent-decentralized', 'openai',    'gpt-5-mini',                 0.76],
    ['finance-agent', 'multi-agent-decentralized', 'openai',    'gpt-5-nano',                 0.76],
    ['finance-agent', 'multi-agent-hybrid',        'anthropic', 'claude-3-7-sonnet-20250219', 0.24],
    ['finance-agent', 'multi-agent-hybrid',        'anthropic', 'claude-sonnet-4-20250514',   0.38],
    ['finance-agent', 'multi-agent-hybrid',        'anthropic', 'claude-sonnet-4-5',          0.46],
    ['finance-agent', 'multi-agent-hybrid',        'gemini',    'gemini-2.0-flash',           0.68],
    ['finance-agent', 'multi-agent-hybrid',        'gemini',    'gemini-2.5-flash',           0.74],
    ['finance-agent', 'multi-agent-hybrid',        'gemini',    'gemini-2.5-pro',             0.76],
    ['finance-agent', 'multi-agent-hybrid',        'openai',    'gpt-5',                      0.78],
    ['finance-agent', 'multi-agent-hybrid',        'openai',    'gpt-5-mini',                 0.66],
    ['finance-agent', 'multi-agent-hybrid',        'openai',    'gpt-5-nano',                 0.74],
    ['finance-agent', 'multi-agent-independent',   'anthropic', 'claude-3-7-sonnet-20250219', 0.12],
    ['finance-agent', 'multi-agent-independent',   'anthropic', 'claude-sonnet-4-20250514',   0.16],
    ['finance-agent', 'multi-agent-independent',   'anthropic', 'claude-sonnet-4-5',          0.28],
    ['finance-agent', 'multi-agent-independent',   'gemini',    'gemini-2.0-flash',           0.62],
    ['finance-agent', 'multi-agent-independent',   'gemini',    'gemini-2.5-flash',           0.68],
    ['finance-agent', 'multi-agent-independent',   'gemini',    'gemini-2.5-pro',             0.76],
    ['finance-agent', 'multi-agent-independent',   'openai',    'gpt-5',                      0.76],
    ['finance-agent', 'multi-agent-independent',   'openai',    'gpt-5-mini',                 0.78],
    ['finance-agent', 'multi-agent-independent',   'openai',    'gpt-5-nano',                 0.76],
    ['finance-agent', 'single-agent',              'anthropic', 'claude-3-7-sonnet-20250219', 0.3],
    ['finance-agent', 'single-agent',              'anthropic', 'claude-sonnet-4-20250514',   0.32],
    ['finance-agent', 'single-agent',              'anthropic', 'claude-sonnet-4-5',          0.28],
    ['finance-agent', 'single-agent',              'gemini',    'gemini-2.0-flash',           0.1],
    ['finance-agent', 'single-agent',              'gemini',    'gemini-2.5-flash',           0.16],
    ['finance-agent', 'single-agent',              'gemini',    'gemini-2.5-pro',             0.58],
    ['finance-agent', 'single-agent',              'openai',    'gpt-5',                      0.62],
    ['finance-agent', 'single-agent',              'openai',    'gpt-5-mini',                 0.54],
    ['finance-agent', 'single-agent',              'openai',    'gpt-5-nano',                 0.24],
    # PlanCraft
    ['plancraft-test', 'multi-agent-centralized',   'anthropic', 'claude-3-7-sonnet-20250219', 0.1919191919191919],
    ['plancraft-test', 'multi-agent-centralized',   'anthropic', 'claude-sonnet-4-20250514',   0.1717171717171717],
    ['plancraft-test', 'multi-agent-centralized',   'anthropic', 'claude-sonnet-4-5',          0.1919191919191919],
    ['plancraft-test', 'multi-agent-centralized',   'gemini',    'gemini-2.0-flash',           0.3],
    ['plancraft-test', 'multi-agent-centralized',   'gemini',    'gemini-2.5-flash',           0.38],
    ['plancraft-test', 'multi-agent-centralized',   'gemini',    'gemini-2.5-pro',             0.34],
    ['plancraft-test', 'multi-agent-centralized',   'openai',    'gpt-5',                      0.32],
    ['plancraft-test', 'multi-agent-centralized',   'openai',    'gpt-5-mini',                 0.35],
    ['plancraft-test', 'multi-agent-centralized',   'openai',    'gpt-5-nano',                 0.29],
    ['plancraft-test', 'multi-agent-decentralized', 'anthropic', 'claude-3-7-sonnet-20250219', 0.1111111111111111],
    ['plancraft-test', 'multi-agent-decentralized', 'anthropic', 'claude-sonnet-4-20250514',   0.20202020202020202],
    ['plancraft-test', 'multi-agent-decentralized', 'anthropic', 'claude-sonnet-4-5',          0.16161616161616163],
    ['plancraft-test', 'multi-agent-decentralized', 'gemini',    'gemini-2.0-flash',           0.44],
    ['plancraft-test', 'multi-agent-decentralized', 'gemini',    'gemini-2.5-flash',           0.41],
    ['plancraft-test', 'multi-agent-decentralized', 'gemini',    'gemini-2.5-pro',             0.38],
    ['plancraft-test', 'multi-agent-decentralized', 'openai',    'gpt-5',                      0.46],
    ['plancraft-test', 'multi-agent-decentralized', 'openai',    'gpt-5-mini',                 0.45],
    ['plancraft-test', 'multi-agent-decentralized', 'openai',    'gpt-5-nano',                 0.38],
    ['plancraft-test', 'multi-agent-hybrid',        'anthropic', 'claude-3-7-sonnet-20250219', 0.30303030303030304],
    ['plancraft-test', 'multi-agent-hybrid',        'anthropic', 'claude-sonnet-4-20250514',   0.2828282828282828],
    ['plancraft-test', 'multi-agent-hybrid',        'anthropic', 'claude-sonnet-4-5',          0.3434343434343434],
    ['plancraft-test', 'multi-agent-hybrid',        'gemini',    'gemini-2.0-flash',           0.32],
    ['plancraft-test', 'multi-agent-hybrid',        'gemini',    'gemini-2.5-flash',           0.41],
    ['plancraft-test', 'multi-agent-hybrid',        'gemini',    'gemini-2.5-pro',             0.42],
    ['plancraft-test', 'multi-agent-hybrid',        'openai',    'gpt-5',                      0.336],
    ['plancraft-test', 'multi-agent-hybrid',        'openai',    'gpt-5-mini',                 0.35],
    ['plancraft-test', 'multi-agent-hybrid',        'openai',    'gpt-5-nano',                 0.35],
    ['plancraft-test', 'multi-agent-independent',   'anthropic', 'claude-3-7-sonnet-20250219', 0.09090909090909091],
    ['plancraft-test', 'multi-agent-independent',   'anthropic', 'claude-sonnet-4-20250514',   0.09090909090909091],
    ['plancraft-test', 'multi-agent-independent',   'anthropic', 'claude-sonnet-4-5',          0.0707070707070707],
    ['plancraft-test', 'multi-agent-independent',   'gemini',    'gemini-2.0-flash',           0.15],
    ['plancraft-test', 'multi-agent-independent',   'gemini',    'gemini-2.5-flash',           0.19],
    ['plancraft-test', 'multi-agent-independent',   'gemini',    'gemini-2.5-pro',             0.14],
    ['plancraft-test', 'multi-agent-independent',   'openai',    'gpt-5',                      0.28],
    ['plancraft-test', 'multi-agent-independent',   'openai',    'gpt-5-mini',                 0.28],
    ['plancraft-test', 'multi-agent-independent',   'openai',    'gpt-5-nano',                 0.24],
    ['plancraft-test', 'single-agent',              'anthropic', 'claude-3-7-sonnet-20250219', 0.5959595959595959],
    ['plancraft-test', 'single-agent',              'anthropic', 'claude-sonnet-4-20250514',   0.6767676767676768],
    ['plancraft-test', 'single-agent',              'anthropic', 'claude-sonnet-4-5',          0.7676767676767676],
    ['plancraft-test', 'single-agent',              'gemini',    'gemini-2.0-flash',           0.52],
    ['plancraft-test', 'single-agent',              'gemini',    'gemini-2.5-flash',           0.51],
    ['plancraft-test', 'single-agent',              'gemini',    'gemini-2.5-pro',             0.51],
    ['plancraft-test', 'single-agent',              'openai',    'gpt-5',                      0.61],
    ['plancraft-test', 'single-agent',              'openai',    'gpt-5-mini',                 0.54],
    ['plancraft-test', 'single-agent',              'openai',    'gpt-5-nano',                 0.38],
    # Workbench
    ['workbench', 'multi-agent-centralized',   'anthropic', 'claude-3-7-sonnet-20250219', 0.63],
    ['workbench', 'multi-agent-centralized',   'anthropic', 'claude-sonnet-4-20250514',   0.68],
    ['workbench', 'multi-agent-centralized',   'anthropic', 'claude-sonnet-4-5',          0.72],
    ['workbench', 'multi-agent-centralized',   'gemini',    'gemini-2.0-flash',           0.52],
    ['workbench', 'multi-agent-centralized',   'gemini',    'gemini-2.5-flash',           0.58],
    ['workbench', 'multi-agent-centralized',   'gemini',    'gemini-2.5-pro',             0.66],
    ['workbench', 'multi-agent-centralized',   'openai',    'gpt-5',                      0.64],
    ['workbench', 'multi-agent-centralized',   'openai',    'gpt-5-mini',                 0.6],
    ['workbench', 'multi-agent-centralized',   'openai',    'gpt-5-nano',                 0.56],
    ['workbench', 'multi-agent-decentralized', 'anthropic', 'claude-3-7-sonnet-20250219', 0.67],
    ['workbench', 'multi-agent-decentralized', 'anthropic', 'claude-sonnet-4-20250514',   0.72],
    ['workbench', 'multi-agent-decentralized', 'anthropic', 'claude-sonnet-4-5',          0.81],
    ['workbench', 'multi-agent-decentralized', 'gemini',    'gemini-2.0-flash',           0.52],
    ['workbench', 'multi-agent-decentralized', 'gemini',    'gemini-2.5-flash',           0.58],
    ['workbench', 'multi-agent-decentralized', 'gemini',    'gemini-2.5-pro',             0.69],
    ['workbench', 'multi-agent-decentralized', 'openai',    'gpt-5',                      0.76],
    ['workbench', 'multi-agent-decentralized', 'openai',    'gpt-5-mini',                 0.62],
    ['workbench', 'multi-agent-decentralized', 'openai',    'gpt-5-nano',                 0.61],
    ['workbench', 'multi-agent-hybrid',        'anthropic', 'claude-3-7-sonnet-20250219', 0.66],
    ['workbench', 'multi-agent-hybrid',        'anthropic', 'claude-sonnet-4-20250514',   0.71],
    ['workbench', 'multi-agent-hybrid',        'anthropic', 'claude-sonnet-4-5',          0.74],
    ['workbench', 'multi-agent-hybrid',        'gemini',    'gemini-2.0-flash',           0.55],
    ['workbench', 'multi-agent-hybrid',        'gemini',    'gemini-2.5-flash',           0.63],
    ['workbench', 'multi-agent-hybrid',        'gemini',    'gemini-2.5-pro',             0.66],
    ['workbench', 'multi-agent-hybrid',        'openai',    'gpt-5',                      0.6],
    ['workbench', 'multi-agent-hybrid',        'openai',    'gpt-5-mini',                 0.56],
    ['workbench', 'multi-agent-hybrid',        'openai',    'gpt-5-nano',                 0.48],
    ['workbench', 'multi-agent-independent',   'anthropic', 'claude-3-7-sonnet-20250219', 0.55],
    ['workbench', 'multi-agent-independent',   'anthropic', 'claude-sonnet-4-20250514',   0.65],
    ['workbench', 'multi-agent-independent',   'anthropic', 'claude-sonnet-4-5',          0.7],
    ['workbench', 'multi-agent-independent',   'gemini',    'gemini-2.0-flash',           0.53],
    ['workbench', 'multi-agent-independent',   'gemini',    'gemini-2.5-flash',           0.54],
    ['workbench', 'multi-agent-independent',   'gemini',    'gemini-2.5-pro',             0.56],
    ['workbench', 'multi-agent-independent',   'openai',    'gpt-5',                      0.59],
    ['workbench', 'multi-agent-independent',   'openai',    'gpt-5-mini',                 0.45],
    ['workbench', 'multi-agent-independent',   'openai',    'gpt-5-nano',                 0.44],
    ['workbench', 'single-agent',              'anthropic', 'claude-3-7-sonnet-20250219', 0.53],
    ['workbench', 'single-agent',              'anthropic', 'claude-sonnet-4-20250514',   0.64],
    ['workbench', 'single-agent',              'anthropic', 'claude-sonnet-4-5',          0.65],
    ['workbench', 'single-agent',              'gemini',    'gemini-2.0-flash',           0.55],
    ['workbench', 'single-agent',              'gemini',    'gemini-2.5-flash',           0.63],
    ['workbench', 'single-agent',              'gemini',    'gemini-2.5-pro',             0.64],
    ['workbench', 'single-agent',              'openai',    'gpt-5',                      0.7],
    ['workbench', 'single-agent',              'openai',    'gpt-5-mini',                 0.7],
    ['workbench', 'single-agent',              'openai',    'gpt-5-nano',                 0.62],
]

# ==============================================================================
# COORDINATION METRICS (Table 5, paper) — per-architecture, dataset-independent
# ==============================================================================
COORDINATION_METRICS = {
    'single-agent': {
        'overhead_pct': 0.0,
        'message_density': 0.00,
        'redundancy': 0.00,
        'efficiency': 0.466,
        'error_amplification': 1.0,
        'success_per_1k_tokens': 67.7,
    },
    'multi-agent-independent': {
        'overhead_pct': 58.0,
        'message_density': 0.00,
        'redundancy': 0.48,
        'efficiency': 0.234,
        'error_amplification': 17.2,
        'success_per_1k_tokens': 42.4,
    },
    'multi-agent-decentralized': {
        'overhead_pct': 263.0,
        'message_density': 0.41,
        'redundancy': 0.50,
        'efficiency': 0.132,
        'error_amplification': 7.8,
        'success_per_1k_tokens': 23.9,
    },
    'multi-agent-centralized': {
        'overhead_pct': 285.0,
        'message_density': 0.39,
        'redundancy': 0.41,
        'efficiency': 0.120,
        'error_amplification': 4.4,
        'success_per_1k_tokens': 21.5,
    },
    'multi-agent-hybrid': {
        'overhead_pct': 515.0,
        'message_density': 0.24,
        'redundancy': 0.46,
        'efficiency': 0.074,
        'error_amplification': 5.1,
        'success_per_1k_tokens': 13.6,
    },
}

# ==============================================================================
# INTELLIGENCE INDEX SCORES (Table 6, paper)
# Used only for the ORIGINAL baseline model in the comparison table.
# ==============================================================================
INTELLIGENCE_SCORES = {
    "claude-3-7-sonnet": 42,
    "claude-sonnet-4": 47,
    "claude-sonnet-4-5": 55,
    "gemini-2.0-flash": 47,
    "gemini-2.5-flash": 58,
    "gemini-2.5-pro": 65,
    "gpt-5-nano": 59,
    "gpt-5-mini": 68,
    "gpt-5": 71,
    "gemini-3-flash-preview": 71,
}

# ==============================================================================
# TOOL COUNTS PER DATASET
# ==============================================================================
TOOL_COUNTS = {
    'browsecomp_plus_sampled_100': 3,
    'workbench': 16,
    'plancraft-test': 4,
    'finance-agent': 5,
    'swebench-verified': 7,
    'terminalbench': 2,
}

# ==============================================================================
# AGENT COUNTS PER ARCHITECTURE
# ==============================================================================
AGENT_COUNTS = {
    'single-agent': 1,
    'multi-agent-independent': 3,
    'multi-agent-centralized': 4,
    'multi-agent-decentralized': 3,
    'multi-agent-hybrid': 4,
}

# ==============================================================================
# FORMULA TEMPLATES
# Identical structure to paper; only the capability terms differ across metrics.
# ==============================================================================

# Original formula — uses Intelligence Index (centered + squared)
FORMULA_ORIGINAL = """performance ~ intelligence_centered + intelligence_sq_centered + log_tools + log_agents +
             log_overhead + message_density + redundancy + efficiency + log_error_amp +
             single_agent_baseline +
             intel_x_efficiency + error_x_baseline + overhead_x_tools +
             redundancy_x_agents + msg_density_x_intel + efficiency_x_tools +
             baseline_x_agents + intel_x_tools + error_x_tools"""

# Metric A — uses Agentic Capability Index (centered + squared) instead of Intelligence Index
FORMULA_METRIC_A = """performance ~ aci_centered + aci_sq_centered + log_tools + log_agents +
             log_overhead + message_density + redundancy + efficiency + log_error_amp +
             single_agent_baseline +
             aci_x_efficiency + error_x_baseline + overhead_x_tools +
             redundancy_x_agents + msg_density_x_aci + efficiency_x_tools +
             baseline_x_agents + aci_x_tools + error_x_tools"""

# Metric B — uses per-dataset Single-Agent Baseline (centered + squared) as capability proxy
# single_agent_baseline itself becomes the primary capability term;
# the interaction single_agent_baseline*agents (baseline_x_agents) is retained as a
# distinct interaction effect alongside the quadratic capability terms.
FORMULA_METRIC_B = """performance ~ psb_centered + psb_sq_centered + log_tools + log_agents +
             log_overhead + message_density + redundancy + efficiency + log_error_amp +
             single_agent_baseline +
             psb_x_efficiency + error_x_baseline + overhead_x_tools +
             redundancy_x_agents + msg_density_x_psb + efficiency_x_tools +
             baseline_x_agents + psb_x_tools + error_x_tools"""

# Feature columns for each model (for StandardScaler)
FEATURE_COLS_ORIGINAL = [
    'intelligence_centered', 'intelligence_sq_centered',
    'n_tools', 'log_tools', 'n_agents', 'log_agents',
    'overhead_pct', 'log_overhead',
    'message_density', 'redundancy', 'efficiency',
    'error_amplification', 'log_error_amp', 'success_per_1k',
    'single_agent_baseline',
    'intel_x_efficiency', 'error_x_baseline', 'overhead_x_tools',
    'redundancy_x_agents', 'msg_density_x_intel', 'efficiency_x_tools',
    'baseline_x_agents', 'intel_x_tools', 'error_x_tools',
]

FEATURE_COLS_METRIC_A = [
    'aci_centered', 'aci_sq_centered',
    'n_tools', 'log_tools', 'n_agents', 'log_agents',
    'overhead_pct', 'log_overhead',
    'message_density', 'redundancy', 'efficiency',
    'error_amplification', 'log_error_amp', 'success_per_1k',
    'single_agent_baseline',
    'aci_x_efficiency', 'error_x_baseline', 'overhead_x_tools',
    'redundancy_x_agents', 'msg_density_x_aci', 'efficiency_x_tools',
    'baseline_x_agents', 'aci_x_tools', 'error_x_tools',
]

FEATURE_COLS_METRIC_B = [
    'psb_centered', 'psb_sq_centered',
    'n_tools', 'log_tools', 'n_agents', 'log_agents',
    'overhead_pct', 'log_overhead',
    'message_density', 'redundancy', 'efficiency',
    'error_amplification', 'log_error_amp', 'success_per_1k',
    'single_agent_baseline',
    'psb_x_efficiency', 'error_x_baseline', 'overhead_x_tools',
    'redundancy_x_agents', 'msg_density_x_psb', 'efficiency_x_tools',
    'baseline_x_agents', 'psb_x_tools', 'error_x_tools',
]


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_new_data(csv_path: str) -> list:
    """
    Load swebench-verified and terminalbench data from CSV.
    Aggregates per-instance results to resolution_rate per (dataset, architecture, model).
    Returns list in same format as ORIGINAL_DATA.
    """
    df = pd.read_csv(csv_path)
    required = {'dataset', 'agent_type', 'provider', 'model', 'resolved'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    grp = df.groupby(['dataset', 'agent_type', 'provider', 'model']).agg(
        resolution_rate=('resolved', 'mean'),
    ).reset_index()

    rows = []
    for _, row in grp.iterrows():
        rows.append([
            row['dataset'],
            row['agent_type'],
            row['provider'],
            row['model'],
            row['resolution_rate'],
        ])
    return rows


def normalize_model_name(model_name: str) -> str:
    """Strip date suffixes; mirrors extended_mixed_effects_6benchmarks.py."""
    short = re.sub(r'-\d{8}$', '', model_name)
    mapping = {
        'claude-3-7-sonnet': 'claude-3-7-sonnet',
        'claude-sonnet-4': 'claude-sonnet-4',
        'claude-sonnet-4-5': 'claude-sonnet-4-5',
        'gemini-2.0-flash': 'gemini-2.0-flash',
        'gemini-2.5-flash': 'gemini-2.5-flash',
        'gemini-2.5-pro': 'gemini-2.5-pro',
        'gemini-3-flash-preview': 'gemini-3-flash-preview',
        'gpt-5-nano': 'gpt-5-nano',
        'gpt-5-mini': 'gpt-5-mini',
        'gpt-5': 'gpt-5',
    }
    if short in mapping:
        return mapping[short]
    short2 = re.sub(r'-\d{8}$', '', short)
    if short2 in mapping:
        return mapping[short2]
    return short


# ==============================================================================
# BASE DATAFRAME BUILDER (shared structure for all three models)
# ==============================================================================

def build_base_dataframe(data_rows: list) -> pd.DataFrame:
    """
    Build the shared feature columns that are identical across all three model
    variants: coordination metrics, tool/agent counts, log transforms,
    single_agent_baseline, and error/overhead interactions.

    Does NOT include capability-metric-specific columns (those are added by
    the variant-specific builders below).
    """
    df = pd.DataFrame(data_rows, columns=['dataset', 'architecture', 'vendor', 'model', 'performance'])
    df['model_short'] = df['model'].apply(normalize_model_name)

    # Task properties
    df['n_tools'] = df['dataset'].map(TOOL_COUNTS)
    unmapped_ds = df[df['n_tools'].isna()]['dataset'].unique()
    if len(unmapped_ds) > 0:
        raise ValueError(f"[ERROR] Datasets with no tool count: {unmapped_ds.tolist()}")

    df['n_agents'] = df['architecture'].map(AGENT_COUNTS)

    # Coordination metrics (per-architecture)
    for key in ['overhead_pct', 'message_density', 'redundancy', 'efficiency',
                'error_amplification', 'success_per_1k_tokens']:
        col = key.replace('_tokens', '')
        df[col] = df['architecture'].map({k: v[key] for k, v in COORDINATION_METRICS.items()})

    # Single-agent baseline: same model, same dataset, single-agent architecture
    sa_perf = df[df['architecture'] == 'single-agent'][['dataset', 'model_short', 'performance']].copy()
    sa_perf.columns = ['dataset', 'model_short', 'single_agent_baseline']
    df = df.merge(sa_perf, on=['dataset', 'model_short'], how='left')

    # For single-agent rows: use dataset-level mean of single-agent performance (matches Cell 6)
    dataset_sa_mean = df[df['architecture'] == 'single-agent'].groupby('dataset')['performance'].mean()
    mask_sa = df['architecture'] == 'single-agent'
    df.loc[mask_sa, 'single_agent_baseline'] = df.loc[mask_sa, 'dataset'].map(dataset_sa_mean)

    # Log transforms
    df['log_tools'] = np.log1p(df['n_tools'])
    df['log_agents'] = np.log1p(df['n_agents'])
    df['log_overhead'] = np.log1p(df['overhead_pct'])
    df['log_error_amp'] = np.log1p(df['error_amplification'])

    # Interactions NOT involving capability metric
    df['error_x_baseline']    = df['error_amplification']  * df['single_agent_baseline']
    df['overhead_x_tools']    = df['overhead_pct']         * df['n_tools']
    df['redundancy_x_agents'] = df['redundancy']           * df['n_agents']
    df['efficiency_x_tools']  = df['efficiency']           * df['n_tools']
    df['baseline_x_agents']   = df['single_agent_baseline'] * np.log1p(df['n_agents'])
    df['error_x_tools']       = df['error_amplification']  * df['n_tools']

    return df


# ==============================================================================
# VARIANT DATAFRAME BUILDERS
# ==============================================================================

def build_original_df(data_rows: list):
    """Original model: capability = Intelligence Index (centered)."""
    df = build_base_dataframe(data_rows)

    df['intelligence'] = df['model_short'].map(INTELLIGENCE_SCORES)
    unmapped = df[df['intelligence'].isna()]['model_short'].unique()
    if len(unmapped) > 0:
        raise ValueError(f"[ERROR] Models with no intelligence score: {unmapped.tolist()}")

    intel_mean = df['intelligence'].mean()
    df['intelligence_centered']    = df['intelligence'] - intel_mean
    df['intelligence_sq_centered'] = df['intelligence_centered'] ** 2

    # Capability interactions
    df['intel_x_efficiency']  = df['intelligence_centered'] * df['efficiency']
    df['msg_density_x_intel'] = df['message_density']       * df['intelligence_centered']
    df['intel_x_tools']       = df['intelligence_centered'] * np.log1p(df['n_tools'])

    return df, intel_mean


def build_metric_a_df(data_rows: list):
    """
    Metric A: capability = Agentic Capability Index (ACI).

    ACI for a model = mean single-agent performance across ALL 6 benchmarks.
    Computed from the 270-row dataset itself (single-agent rows only).
    This directly measures agentic ability on real tasks, distinguishing models
    that share the same static Intelligence Index but behave differently in
    agentic settings (e.g., GPT-5.2 vs Gemini-3.0-Pro, both at index 75).
    """
    df = build_base_dataframe(data_rows)

    # ACI: mean single-agent performance across all datasets for each model
    sa_rows = df[df['architecture'] == 'single-agent']
    aci_map = sa_rows.groupby('model_short')['performance'].mean()
    df['aci_raw'] = df['model_short'].map(aci_map)

    unmapped = df[df['aci_raw'].isna()]['model_short'].unique()
    if len(unmapped) > 0:
        raise ValueError(f"[ERROR] Models with no ACI: {unmapped.tolist()}")

    aci_mean = df['aci_raw'].mean()
    df['aci_centered']    = df['aci_raw'] - aci_mean
    df['aci_sq_centered'] = df['aci_centered'] ** 2

    # Capability interactions (replace intel_ with aci_)
    df['aci_x_efficiency']  = df['aci_centered'] * df['efficiency']
    df['msg_density_x_aci'] = df['message_density'] * df['aci_centered']
    df['aci_x_tools']       = df['aci_centered'] * np.log1p(df['n_tools'])

    return df, aci_mean


def build_metric_b_df(data_rows: list):
    """
    Metric B: capability = Per-dataset Single-Agent Baseline (PSB).

    PSB for a (model, dataset) pair = that model's single-agent performance on
    exactly that dataset. This is already used as a control variable
    (single_agent_baseline) in the original formula. Here we use it INSTEAD OF
    the Intelligence Index as the primary capability proxy, centering and
    squaring it to mirror the Intelligence Index treatment.

    The original single_agent_baseline column is retained in the formula so
    its interaction with n_agents (baseline_x_agents) preserves the baseline
    paradox finding.
    """
    df = build_base_dataframe(data_rows)

    # PSB = single_agent_baseline (already populated by build_base_dataframe)
    psb_mean = df['single_agent_baseline'].mean()
    df['psb_centered']    = df['single_agent_baseline'] - psb_mean
    df['psb_sq_centered'] = df['psb_centered'] ** 2

    # Capability interactions (replace intel_ with psb_)
    df['psb_x_efficiency']  = df['psb_centered'] * df['efficiency']
    df['msg_density_x_psb'] = df['message_density'] * df['psb_centered']
    df['psb_x_tools']       = df['psb_centered'] * np.log1p(df['n_tools'])

    return df, psb_mean


# ==============================================================================
# STANDARDIZATION & CROSS-VALIDATION
# ==============================================================================

def standardize(df: pd.DataFrame, feature_cols: list):
    """Standardize feature columns with StandardScaler. Returns (df_scaled, scaler)."""
    df_scaled = df.copy()
    scaler = StandardScaler()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df_scaled, scaler


def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot


def run_5fold_cv(df_scaled: pd.DataFrame, formula: str, feature_cols: list) -> np.ndarray:
    """5-fold CV matching original methodology (random_state=42)."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = []
    for train_idx, test_idx in kf.split(df_scaled):
        train_data = df_scaled.iloc[train_idx]
        test_data  = df_scaled.iloc[test_idx]
        cv_model   = smf.ols(formula, data=train_data).fit()
        y_pred     = cv_model.predict(test_data)
        cv_r2.append(compute_r2(test_data['performance'].values, y_pred))
    return np.array(cv_r2)


# ==============================================================================
# OUTPUT HELPERS
# ==============================================================================

def print_section(title: str, width: int = 90):
    print()
    print("=" * width)
    print(title)
    print("=" * width)


def print_subsection(title: str, width: int = 90):
    print()
    print("-" * width)
    print(title)
    print("-" * width)


def sig_stars(p: float) -> str:
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.10:  return '†'
    return ''


# ==============================================================================
# SINGLE MODEL RUN
# ==============================================================================

def run_model(label: str, df_scaled: pd.DataFrame, formula: str,
              feature_cols: list, cap_col_linear: str, cap_col_sq: str):
    """
    Fit OLS, run 5-fold CV, print results.
    Returns (model, coef_df, cv_r2_arr).

    cap_col_linear / cap_col_sq: names of the linear and quadratic capability
    terms so we can look them up in the coefficient table for key-findings check.
    """
    model = smf.ols(formula, data=df_scaled).fit()
    cv_r2 = run_5fold_cv(df_scaled, formula, feature_cols)
    cv_mean, cv_std = np.mean(cv_r2), np.std(cv_r2)

    print_section(f"MODEL: {label}  (N={len(df_scaled)})")
    print(f"\n  Training R²      = {model.rsquared:.4f}")
    print(f"  Adjusted R²      = {model.rsquared_adj:.4f}")
    print(f"  R²_CV (5-fold)   = {cv_mean:.4f} (±{cv_std:.4f})")
    print(f"  AIC              = {model.aic:.2f}")

    coef_df = pd.DataFrame({
        'Coefficient': model.params,
        'Std_Error':   model.bse,
        't_value':     model.tvalues,
        'p_value':     model.pvalues,
    })
    coef_df['CI_lower'] = coef_df['Coefficient'] - 1.96 * coef_df['Std_Error']
    coef_df['CI_upper'] = coef_df['Coefficient'] + 1.96 * coef_df['Std_Error']
    coef_df['Sig'] = coef_df['p_value'].apply(sig_stars)

    print_subsection("ALL COEFFICIENTS (standardized)")
    fmt = "  {:<32s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>8s}  {:>6s}"
    print(fmt.format("Variable", "Beta", "SE", "CI_lower", "CI_upper", "p-value", "Sig"))
    print("  " + "-" * 90)
    for var, row in coef_df.iterrows():
        print("  {:<32s}  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>8.4f}  {:>6s}".format(
            str(var), row['Coefficient'], row['Std_Error'],
            row['CI_lower'], row['CI_upper'], row['p_value'], row['Sig']))

    print_subsection("SIGNIFICANT PREDICTORS (p < 0.05), sorted by |beta|")
    sig = coef_df[(coef_df['p_value'] < 0.05) & (coef_df.index != 'Intercept')].copy()
    sig['abs_beta'] = sig['Coefficient'].abs()
    sig = sig.sort_values('abs_beta', ascending=False)
    for var, row in sig.iterrows():
        print("  {:<32s}  beta={:>7.4f}  95%CI [{:>7.4f}, {:>7.4f}]  p={:.4f} {}".format(
            str(var), row['Coefficient'], row['CI_lower'], row['CI_upper'],
            row['p_value'], row['Sig']))

    return model, coef_df, cv_r2


# ==============================================================================
# KEY-FINDINGS CHECK (adapted from original, generalized across metric names)
# ==============================================================================

def check_key_findings_single(label: str, coef_df: pd.DataFrame,
                               cap_linear_col: str, cap_sq_col: str,
                               cap_x_eff_col: str):
    """
    Check whether the paper's three primary findings hold for this model.

    The check is IDENTICAL in logic to extended_mixed_effects_6benchmarks.py
    but uses the generic capability-column names passed in.
    """
    checks = [
        # (predictor_in_coef_df, expected_sign, description)
        ('efficiency_x_tools',  -1, "efficiency×tools: dominant negative (tool-heavy + inefficient)"),
        ('baseline_x_agents',   -1, "baseline×agents: baseline paradox (high SA perf -> diminishing MAS)"),
        ('redundancy_x_agents', +1, "redundancy×agents: modest error-correction benefit"),
        (cap_linear_col,        +1, f"capability (linear={cap_linear_col}): higher -> better performance"),
    ]

    print(f"\n  Key findings check for: {label}")
    fmt = "  {:<30s}  {:>8s}  {:>8s}  {:>8s}  {:>10s}"
    print(fmt.format("Predictor", "beta", "p", "sign_ok", "p<0.05_ok"))
    print("  " + "-" * 70)

    results = {}
    for pred, expected_sign, desc in checks:
        if pred in coef_df.index:
            b = coef_df.loc[pred, 'Coefficient']
            p = coef_df.loc[pred, 'p_value']
            sign_ok = "YES" if np.sign(b) == expected_sign else "NO"
            sig_ok  = "YES" if p < 0.05 else "NO"
        else:
            b, p, sign_ok, sig_ok = np.nan, np.nan, "N/A", "N/A"
        status = "HOLDS" if sign_ok == "YES" and sig_ok == "YES" else (
                 "WEAKENED" if sign_ok == "YES" else "REVERSED/MISSING")
        b_s = f"{b:.4f}" if not np.isnan(b) else "  N/A"
        p_s = f"{p:.4f}" if not np.isnan(p) else "  N/A"
        print(f"  {pred:<30s}  {b_s:>8s}  {p_s:>8s}  {sign_ok:>8s}  {sig_ok:>10s}   {status}")
        results[pred] = {'beta': b, 'p': p, 'sign_ok': sign_ok, 'sig_ok': sig_ok, 'status': status}

    return results


# ==============================================================================
# MASTER COMPARISON TABLE
# ==============================================================================

def print_comparison_table(results_dict: dict):
    """
    Print a side-by-side comparison of all three capability metrics.

    results_dict: {label: {'r2_train': float, 'r2_cv': float,
                            'cv_std': float, 'aic': float,
                            'findings': {pred: {beta, p, sign_ok, sig_ok}}}}
    """
    print_section("ROBUSTNESS CHECK: COMPARISON ACROSS CAPABILITY METRICS")
    print("""
  MOTIVATION :
    Both reviewers questioned whether the Artificial Analysis Intelligence Index
    (a composite of static single-turn benchmarks) is appropriate for predicting
    agentic task performance.  This table shows whether the core findings are
    stable across three operationalizations of "model capability":

    Original  — Intelligence Index (static composite, Table 6)
    Metric A  — Agentic Capability Index (mean single-agent performance, all 6 tasks)
    Metric B  — Per-dataset Single-Agent Baseline (same dataset, same model, SA arch)
    """)

    labels  = list(results_dict.keys())
    metrics = ['r2_train', 'r2_cv', 'cv_std', 'aic']
    metric_names = {
        'r2_train': 'R²_train',
        'r2_cv':    'R²_CV (5-fold mean)',
        'cv_std':   'R²_CV (5-fold std)',
        'aic':      'AIC',
    }

    # Header
    header_fmt = "  {:<28s}" + "  {:>20s}" * len(labels)
    print(header_fmt.format("Metric", *labels))
    print("  " + "-" * (28 + 22 * len(labels)))

    for m in metrics:
        vals = [results_dict[l][m] for l in labels]
        if m == 'aic':
            row_vals = [f"{v:>20.2f}" for v in vals]
        else:
            row_vals = [f"{v:>20.4f}" for v in vals]
        print(f"  {metric_names[m]:<28s}" + "".join(row_vals))

    # Key findings across metrics
    print_subsection("KEY FINDINGS STATUS ACROSS METRICS")
    key_preds = ['efficiency_x_tools', 'baseline_x_agents', 'redundancy_x_agents']
    # Map cap_linear col for each label
    cap_linear_cols = {
        'Original (Intelligence Index)':       'intelligence_centered',
        'Metric A (Agentic Capability Index)': 'aci_centered',
        'Metric B (Per-dataset SA Baseline)':  'psb_centered',
    }

    all_preds = key_preds + list(cap_linear_cols.values())

    pred_labels = {
        'efficiency_x_tools':  'efficiency×tools (negative)',
        'baseline_x_agents':   'baseline×agents  (negative)',
        'redundancy_x_agents': 'redundancy×agents (positive)',
        'intelligence_centered': 'capability linear (positive)',
        'aci_centered':          'capability linear (positive)',
        'psb_centered':          'capability linear (positive)',
    }

    hdr = "  {:<30s}" + "  {:>26s}" * len(labels)
    print(hdr.format("Finding", *[l[:26] for l in labels]))
    print("  " + "-" * (30 + 28 * len(labels)))

    # Shared findings
    for pred in key_preds:
        row = f"  {pred_labels.get(pred, pred):<30s}"
        for lbl in labels:
            fd = results_dict[lbl]['findings']
            if pred in fd:
                b   = fd[pred]['beta']
                p   = fd[pred]['p']
                st  = fd[pred]['status']
                b_s = f"{b:.3f}" if not np.isnan(b) else "N/A"
                p_s = f"p={p:.3f}" if not np.isnan(p) else ""
                cell = f"{st} (b={b_s},{p_s})"
            else:
                cell = "N/A"
            row += f"  {cell:>26s}"
        print(row)

    # Capability linear term (differs per model)
    print()
    print("  Capability (linear) term:")
    row = "  {:<30s}".format("capability_centered")
    for lbl in labels:
        fd = results_dict[lbl]['findings']
        cap_col = cap_linear_cols.get(lbl, '')
        if cap_col in fd:
            b   = fd[cap_col]['beta']
            p   = fd[cap_col]['p']
            st  = fd[cap_col]['status']
            b_s = f"{b:.3f}" if not np.isnan(b) else "N/A"
            p_s = f"p={p:.3f}" if not np.isnan(p) else ""
            cell = f"{st} (b={b_s},{p_s})"
        else:
            cell = "N/A"
        row += f"  {cell:>26s}"
    print(row)

    # Conclusion
    print_subsection("CONCLUSION")
    all_statuses = []
    for lbl in labels:
        fd = results_dict[lbl]['findings']
        all_statuses.extend([v['status'] for v in fd.values()])

    n_holds    = sum(1 for s in all_statuses if s == 'HOLDS')
    n_weakened = sum(1 for s in all_statuses if s == 'WEAKENED')
    n_reversed = sum(1 for s in all_statuses if 'REVERSED' in s or 'MISSING' in s)

    print(f"""
  Findings across {len(labels)} capability metrics ({len(all_statuses)} finding×metric pairs):
    HOLDS    (correct sign AND p<0.05): {n_holds}
    WEAKENED (correct sign, p>=0.05) : {n_weakened}
    REVERSED/MISSING                 : {n_reversed}

  R² range:  {min(results_dict[l]['r2_train'] for l in labels):.4f} – {max(results_dict[l]['r2_train'] for l in labels):.4f} (train)
             {min(results_dict[l]['r2_cv'] for l in labels):.4f} – {max(results_dict[l]['r2_cv'] for l in labels):.4f} (5-fold CV)

  INTERPRETATION:
    Metric A (Agentic Capability Index) directly addresses the capability-metric concern:
    it uses task performance on the six actual benchmarks rather than a static
    benchmark composite.  If findings hold under Metric A, the core results
    do not depend on the choice of Intelligence Index.

    Metric B (Per-dataset Single-Agent Baseline) is the most conservative
    robustness check: the capability proxy is exactly the single-agent
    performance on the same task, already included as a control variable
    in the original formula.

    If the key findings (efficiency×tools negative, baseline×agents negative,
    redundancy×agents positive) hold with correct sign across all three metrics,
    the conclusions are robust to the operationalization of model capability.
    If any finding reverses sign or loses significance, this is reported
    honestly above — the script does not suppress negative results.
    """)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print_section("ALTERNATIVE CAPABILITY METRIC ROBUSTNESS CHECK")
    print("  Capability-metric robustness check")
    print("  Capability-metric robustness analysis")
    print("  Question: Are the findings robust to the choice of capability metric?")
    print("  Methodology: identical 19-predictor OLS to extended_mixed_effects_6benchmarks.py")

    # ------------------------------------------------------------------
    # Locate CSV
    # ------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'per_instance_results_swe_tb.csv')
    if not os.path.exists(csv_path):
        sys.exit(f"[ERROR] Cannot find per_instance_results_swe_tb.csv at:\n  {csv_path}")

    print(f"\n  CSV: {csv_path}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    new_rows  = load_new_data(csv_path)
    all_rows  = ORIGINAL_DATA + new_rows
    print(f"  Original rows : {len(ORIGINAL_DATA)}")
    print(f"  New rows (CSV): {len(new_rows)}")
    print(f"  Total rows    : {len(all_rows)}")
    assert len(all_rows) == 270, f"Expected 270 rows, got {len(all_rows)}"

    # ------------------------------------------------------------------
    # Build & verify ACI (agentic capability) rankings vs Intelligence Index
    # ------------------------------------------------------------------
    print_section("AGENTIC CAPABILITY INDEX vs INTELLIGENCE INDEX")
    print("""
  ACI = mean single-agent performance across all 6 benchmarks.
  This directly measures what each model achieves on agentic tasks.
  Correlation with Intelligence Index indicates whether the static proxy
  tracks real agentic ability .
    """)

    df_base = build_base_dataframe(all_rows)
    df_base['intelligence'] = df_base['model_short'].map(INTELLIGENCE_SCORES)
    sa_rows = df_base[df_base['architecture'] == 'single-agent']
    aci_map = sa_rows.groupby('model_short')['performance'].mean()
    intel_map = df_base.groupby('model_short')['intelligence'].first()

    aci_df = pd.DataFrame({'ACI': aci_map, 'Intelligence': intel_map}).dropna().sort_values('ACI', ascending=False)
    print(f"  {'Model':<30s}  {'ACI':>8s}  {'Intel':>8s}  {'ACI rank':>10s}  {'Intel rank':>10s}")
    print("  " + "-" * 72)
    aci_df['aci_rank']   = aci_df['ACI'].rank(ascending=False).astype(int)
    aci_df['intel_rank'] = aci_df['Intelligence'].rank(ascending=False, method='min').astype(int)
    for model, row in aci_df.iterrows():
        print(f"  {model:<30s}  {row['ACI']:>8.4f}  {row['Intelligence']:>8.0f}"
              f"  {int(row['aci_rank']):>10d}  {int(row['intel_rank']):>10d}")

    corr = aci_df['ACI'].corr(aci_df['Intelligence'])
    print(f"\n  Pearson r (ACI vs Intelligence Index) = {corr:.4f}")
    if abs(corr) < 0.7:
        print("  => Low correlation: ACI and Intelligence Index diverge substantially.")
        print("     Metric A provides genuinely different information about model capability.")
    elif abs(corr) < 0.9:
        print("  => Moderate correlation: some divergence between metrics.")
        print("     Metric A captures ordering differences not visible in Intelligence Index.")
    else:
        print("  => High correlation: metrics largely agree on model ordering.")
        print("     Metric A provides a task-grounded confirmation of the static index.")

    # ------------------------------------------------------------------
    # Build dataframes for all three variants
    # ------------------------------------------------------------------
    df_orig_raw, intel_mean   = build_original_df(all_rows)
    df_metA_raw, aci_mean     = build_metric_a_df(all_rows)
    df_metB_raw, psb_mean     = build_metric_b_df(all_rows)

    df_orig_scaled, _ = standardize(df_orig_raw, FEATURE_COLS_ORIGINAL)
    df_metA_scaled, _ = standardize(df_metA_raw, FEATURE_COLS_METRIC_A)
    df_metB_scaled, _ = standardize(df_metB_raw, FEATURE_COLS_METRIC_B)

    print_section("CAPABILITY METRIC DESCRIPTIVES")
    print(f"  Intelligence Index:          mean={intel_mean:.4f}  "
          f"range=[{df_orig_raw['intelligence'].min():.0f}, {df_orig_raw['intelligence'].max():.0f}]")
    print(f"  Agentic Capability Index:    mean={aci_mean:.4f}  "
          f"range=[{df_metA_raw['aci_raw'].min():.4f}, {df_metA_raw['aci_raw'].max():.4f}]")
    print(f"  Per-dataset SA Baseline:     mean={psb_mean:.4f}  "
          f"range=[{df_metB_raw['single_agent_baseline'].min():.4f}, "
          f"{df_metB_raw['single_agent_baseline'].max():.4f}]")

    # ------------------------------------------------------------------
    # Run all three models
    # ------------------------------------------------------------------
    model_orig, coef_orig, cv_orig = run_model(
        "Original — Intelligence Index (N=270)",
        df_orig_scaled, FORMULA_ORIGINAL, FEATURE_COLS_ORIGINAL,
        'intelligence_centered', 'intelligence_sq_centered',
    )

    model_metA, coef_metA, cv_metA = run_model(
        "Metric A — Agentic Capability Index (N=270)",
        df_metA_scaled, FORMULA_METRIC_A, FEATURE_COLS_METRIC_A,
        'aci_centered', 'aci_sq_centered',
    )

    model_metB, coef_metB, cv_metB = run_model(
        "Metric B — Per-dataset Single-Agent Baseline (N=270)",
        df_metB_scaled, FORMULA_METRIC_B, FEATURE_COLS_METRIC_B,
        'psb_centered', 'psb_sq_centered',
    )

    # ------------------------------------------------------------------
    # Key findings check for each model
    # ------------------------------------------------------------------
    print_section("KEY FINDINGS CHECK ACROSS METRICS")

    findings_orig = check_key_findings_single(
        "Original (Intelligence Index)",
        coef_orig, 'intelligence_centered', 'intelligence_sq_centered', 'intel_x_efficiency',
    )
    findings_metA = check_key_findings_single(
        "Metric A (Agentic Capability Index)",
        coef_metA, 'aci_centered', 'aci_sq_centered', 'aci_x_efficiency',
    )
    findings_metB = check_key_findings_single(
        "Metric B (Per-dataset SA Baseline)",
        coef_metB, 'psb_centered', 'psb_sq_centered', 'psb_x_efficiency',
    )

    # ------------------------------------------------------------------
    # Master comparison table
    # ------------------------------------------------------------------
    results_dict = {
        'Original (Intelligence Index)': {
            'r2_train': model_orig.rsquared,
            'r2_cv':    np.mean(cv_orig),
            'cv_std':   np.std(cv_orig),
            'aic':      model_orig.aic,
            'findings': findings_orig,
        },
        'Metric A (Agentic Capability Index)': {
            'r2_train': model_metA.rsquared,
            'r2_cv':    np.mean(cv_metA),
            'cv_std':   np.std(cv_metA),
            'aic':      model_metA.aic,
            'findings': findings_metA,
        },
        'Metric B (Per-dataset SA Baseline)': {
            'r2_train': model_metB.rsquared,
            'r2_cv':    np.mean(cv_metB),
            'cv_std':   np.std(cv_metB),
            'aic':      model_metB.aic,
            'findings': findings_metB,
        },
    }

    print_comparison_table(results_dict)

    print_section("REPRODUCIBILITY NOTES")
    print(f"""
  Script    : scripts/alternative_capability_metric.py
  Data      : 180 original rows (ORIGINAL_DATA embedded) +
              {len(new_rows)} rows from scripts/per_instance_results_swe_tb.csv
  Total N   : 270 (6 benchmarks × 9 models × 5 architectures)
  Formula   : 19-predictor OLS, identical structure to extended_mixed_effects_6benchmarks.py
  CV        : 5-fold KFold(shuffle=True, random_state=42)
  Scaling   : StandardScaler on full dataset (same as original)
  Cap. term : centered before squaring to reduce VIF (same procedure as original)

  Metric A rationale:
    ACI = mean single-agent performance across all 6 benchmarks.
    Computed from the same 270-row dataset (single-agent rows only).
    No external source; fully reproducible from the paper's own data.
    distinguishes models with identical static indices
    but different actual agentic performance.

  Metric B rationale:
    PSB = single_agent_baseline (already in original formula as control).
    Using it as the PRIMARY capability proxy is the most conservative check:
    it tests whether the coordination findings survive when the capability
    metric is the exact single-task performance rather than a cross-task or
    static composite.
    """)


if __name__ == '__main__':
    main()

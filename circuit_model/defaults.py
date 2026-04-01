"""
Shared default paths for CLI commands.

This module defines the default file paths used across all CLI commands
(circuit_model.cli and circuit_model.ring.cli) to ensure consistency.
"""

from pathlib import Path

# Default circuit parameters (fitted to firing rates)
DEFAULT_WT_PARAMS_PATH = Path("params/new/ring_firing_rate/WT_1mo_article_ko.json")
DEFAULT_APP_PARAMS_PATH = Path("params/new/ring_firing_rate/WT_APP_1mo_article_ko.json")

# Default ring parameters (topology and inter-node weights)
DEFAULT_WT_RING_PARAMS_PATH = Path("params/new/ring_firing_rate/WT_1mo_article_ring_ko.json")
DEFAULT_APP_RING_PARAMS_PATH = Path("params/new/ring_firing_rate/WT_APP_1mo_article_ring_ko.json")

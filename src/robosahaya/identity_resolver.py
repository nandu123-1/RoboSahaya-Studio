"""Robot Identity Resolver for RoboSahaya Studio.

Decouples natural language user instructions and robot hints from simulator
compatibility, resolving queries to canonical robot models or signaling ambiguity.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from robosahaya.robot_registry import RobotModel, RobotRegistry, get_robot_registry, _normalize_alias


@dataclass
class IdentityResolutionResult:
    """Result of resolving a robot identity from text or user input."""
    resolved_id: Optional[str] = None
    confidence: float = 0.0
    match_type: str = "none"  # "canonical", "exact_alias", "normalized_alias", "fuzzy", "ambiguous", "none"
    matched_alias: Optional[str] = None
    candidates: List[Tuple[str, float]] = field(default_factory=list)
    requires_clarification: bool = False
    clarification_message: Optional[str] = None
    notes: str = ""


# Common category trigger words that imply ambiguity if no specific model is mentioned
CATEGORY_KEYWORDS = {
    "arm": ["so_arm101", "franka_panda", "ur5e", "so_arm100"],
    "manipulator": ["so_arm101", "franka_panda", "ur5e", "so_arm100"],
}


class IdentityResolver:
    """Resolves natural-language queries and aliases to canonical robot identities."""

    def __init__(self, registry: Optional[RobotRegistry] = None):
        self.registry = registry or get_robot_registry()

    def resolve(self, text: str) -> IdentityResolutionResult:
        """Resolve a user prompt, robot name, or alias to a canonical RobotModel."""
        if not text or not text.strip():
            return IdentityResolutionResult(
                resolved_id=None,
                confidence=0.0,
                match_type="none",
                notes="Empty query text provided.",
            )

        raw_query = text.strip()
        norm_query = _normalize_alias(raw_query)

        # 1. Direct canonical ID lookup
        direct_model = self.registry.get_model(raw_query)
        if direct_model and direct_model.id == raw_query:
            return IdentityResolutionResult(
                resolved_id=direct_model.id,
                confidence=1.0,
                match_type="canonical",
                matched_alias=direct_model.id,
                candidates=[(direct_model.id, 1.0)],
                notes=f"Exact canonical ID match for '{direct_model.id}'.",
            )

        # 2. Exact or normalized alias lookup for the full string
        if direct_model:
            return IdentityResolutionResult(
                resolved_id=direct_model.id,
                confidence=1.0,
                match_type="normalized_alias",
                matched_alias=norm_query,
                candidates=[(direct_model.id, 1.0)],
                notes=f"Direct alias match for '{direct_model.id}'.",
            )

        # 3. Scan for model aliases contained as subphrases/tokens in the user prompt
        # We search from longest alias to shortest to match most specific phrases first (e.g. 'franka panda' over 'panda')
        models = self.registry.list_models()
        alias_matches: List[Tuple[str, str, int]] = []  # (robot_id, alias, length)

        query_tokens = set(norm_query.split())

        for model in models:
            # Check canonical ID
            c_norm = _normalize_alias(model.id)
            if self._is_subphrase_match(c_norm, norm_query, query_tokens):
                alias_matches.append((model.id, model.id, len(c_norm)))

            # Check display name
            d_norm = _normalize_alias(model.display_name)
            if self._is_subphrase_match(d_norm, norm_query, query_tokens):
                alias_matches.append((model.id, model.display_name, len(d_norm)))

            # Check aliases
            for alias in model.aliases:
                a_norm = _normalize_alias(alias)
                if self._is_subphrase_match(a_norm, norm_query, query_tokens):
                    alias_matches.append((model.id, alias, len(a_norm)))

        if alias_matches:
            # Sort by longest matched alias length descending
            alias_matches.sort(key=lambda x: x[2], reverse=True)
            best_id, best_alias, _ = alias_matches[0]
            
            # Check if there are conflicting matches of similar specificity from different robots
            unique_matched_ids = list(dict.fromkeys(m[0] for m in alias_matches))
            if len(unique_matched_ids) > 1 and alias_matches[0][2] == alias_matches[1][2]:
                cand_list = [(mid, 0.75) for mid in unique_matched_ids]
                return IdentityResolutionResult(
                    resolved_id=None,
                    confidence=0.5,
                    match_type="ambiguous",
                    candidates=cand_list,
                    requires_clarification=True,
                    clarification_message=f"Prompt matched multiple robots: {', '.join(unique_matched_ids)}. Please specify which one.",
                    notes="Multiple conflicting alias matches found with equal length.",
                )

            return IdentityResolutionResult(
                resolved_id=best_id,
                confidence=0.95,
                match_type="exact_alias",
                matched_alias=best_alias,
                candidates=[(best_id, 0.95)],
                notes=f"Matched alias phrase '{best_alias}' in query.",
            )

        # 4. Check for ambiguous category words (e.g., 'build an arm', 'manipulator')
        is_specific_arm = any(w in norm_query for w in ["6 dof", "6dof", "robotic arm", "6 degree"])
        if not is_specific_arm:
            for cat_key, candidate_ids in CATEGORY_KEYWORDS.items():
                cat_norm = _normalize_alias(cat_key)
                if self._is_subphrase_match(cat_norm, norm_query, query_tokens):
                    # If multiple robots match this category
                    valid_candidates = [cid for cid in candidate_ids if self.registry.get_model(cid)]
                    if len(valid_candidates) > 1:
                        display_names = [
                            self.registry.get_model(cid).display_name for cid in valid_candidates
                        ]
                        return IdentityResolutionResult(
                            resolved_id=None,
                            confidence=0.4,
                            match_type="ambiguous",
                            candidates=[(cid, 0.4) for cid in valid_candidates],
                            requires_clarification=True,
                            clarification_message=(
                                f"Multiple {cat_key} models are available: {', '.join(display_names)}. "
                                f"Please clarify which specific model you wish to build."
                            ),
                            notes=f"Generic category term '{cat_key}' matched multiple candidates without a specific qualifier.",
                        )
                    elif len(valid_candidates) == 1:
                        cid = valid_candidates[0]
                        return IdentityResolutionResult(
                            resolved_id=cid,
                            confidence=0.8,
                            match_type="category_single",
                            candidates=[(cid, 0.8)],
                            notes=f"Category '{cat_key}' uniquely resolved to {cid}.",
                        )

        # 5. Bounded Fuzzy Matching across all aliases and IDs
        fuzzy_candidates: List[Tuple[str, str, float]] = []  # (robot_id, alias, score)
        # Compare individual query words or ngrams against alias words
        for model in models:
            all_names = [model.id, model.display_name] + list(model.aliases)
            for name in all_names:
                n_norm = _normalize_alias(name)
                # Compute sequence similarity
                ratio = difflib.SequenceMatcher(None, norm_query, n_norm).ratio()
                if ratio >= 0.70:
                    fuzzy_candidates.append((model.id, name, ratio))

                # Also compare token-level similarity
                for token in norm_query.split():
                    if len(token) >= 3:
                        t_ratio = difflib.SequenceMatcher(None, token, n_norm).ratio()
                        if t_ratio >= 0.85:
                            fuzzy_candidates.append((model.id, name, t_ratio))

        if fuzzy_candidates:
            fuzzy_candidates.sort(key=lambda x: x[2], reverse=True)
            top_id, top_alias, top_score = fuzzy_candidates[0]
            if top_score >= 0.82:
                return IdentityResolutionResult(
                    resolved_id=top_id,
                    confidence=round(top_score, 2),
                    match_type="fuzzy",
                    matched_alias=top_alias,
                    candidates=[(top_id, round(top_score, 2))],
                    notes=f"Fuzzy matched '{top_alias}' with score {top_score:.2f}.",
                )

        # 6. Safe failure: no confident match. DO NOT guess or default silently.
        return IdentityResolutionResult(
            resolved_id=None,
            confidence=0.0,
            match_type="none",
            candidates=[],
            requires_clarification=False,
            notes=f"Could not resolve any registered robot model for query '{raw_query}'.",
        )

    def _is_subphrase_match(self, needle: str, haystack: str, haystack_tokens: Set[str]) -> bool:
        """Check if needle appears as a whole phrase or word in haystack."""
        if not needle:
            return False
        # If single word, must match a token exactly
        if " " not in needle:
            return needle in haystack_tokens
        # If multi-word, search with word boundaries
        pattern = r"\b" + re.escape(needle) + r"\b"
        return bool(re.search(pattern, haystack))

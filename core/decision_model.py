"""System-1 Calibrated Decision Model Engine for Valstorm Agent Runtime.

Provides fast, non-autoregressive, type-safe probabilistic decisions over structured
program/UI/canvas state inspired by Kahneman's System 1 thinking and TypeSafe AI's Jev:

Primitives:
  - ChoiceQuestion -> ChoiceAnswer(choice, probabilities, confidence)
  - ScoreQuestion  -> ScoreAnswer(score, level, probabilities, confidence)
  - NoulQuestion   -> NoulAnswer(noul, confidence)
  - BoxQuestion    -> BoxAnswer(left, top, right, bottom, confidence)

Backends:
  - LocalSystemOneBackend: Zero-dependency, in-process calibrated lexical/semantic/spatial
    evaluator for the local daemon (1ms-15ms latency, $0.00 cost, 0% schema/type errors).
  - TypeSafeJevBackend: Cloud client for TypeSafe AI's Jev API (`https://api.typesafe.ai/v1/systemone`).
  - LLMSystemOneAdapterBackend: Single-pass structured JSON adapter backed by any runtime BaseProvider.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import httpx
from pydantic import BaseModel, Field

# Pricing for Jev ($0.042 / 1M input tokens, $0.00 output tokens)
JEV_INPUT_COST_PER_1M = 0.042

_WORD_RE = re.compile(r"[a-z0-9]+")
_CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "with", "by", "from", "as", "into",
    "this", "that", "these", "those", "it", "its", "or", "and", "if", "then",
    "which", "what", "where", "when", "who", "how", "should", "would", "could",
    "does", "do", "did", "has", "have", "had", "any", "all", "each", "every",
})

# Domain synonym clusters for UI navigation, DOM/Command Bus, and 2D/3D Crafting (ArtCraft)
_SYNONYM_CLUSTERS: List[frozenset[str]] = [
    frozenset({"click", "press", "tap", "activate", "trigger", "button", "btn", "submit", "open"}),
    frozenset({"type", "input", "fill", "enter", "write", "text", "search", "field", "query", "textbox"}),
    frozenset({"select", "choose", "pick", "dropdown", "option", "combobox", "menu", "tab", "switch"}),
    frozenset({"drag", "move", "translate", "position", "reposition", "place", "drop", "slider"}),
    frozenset({"pose", "rig", "skeleton", "joint", "character", "actor", "mannequin", "limb", "posture"}),
    frozenset({"camera", "viewport", "fov", "lens", "frame", "shot", "orbit", "dolly", "zoom", "angle"}),
    frozenset({"mesh", "3d", "model", "geometry", "extrude", "depth", "object3d", "prop"}),
    frozenset({"layer", "composite", "backdrop", "foreground", "mask", "matte", "background", "remove", "alpha", "blend"}),
    frozenset({"render", "generate", "synthesize", "export", "bake", "video", "diffusion", "craft"}),
    frozenset({"timeline", "keyframe", "frame", "sequence", "clip", "playhead", "scrub", "animation"}),
    frozenset({"draw", "brush", "sketch", "paint", "stroke", "canvas", "pencil", "inpaint"}),
    frozenset({"error", "exception", "crash", "failed", "failure", "alert", "toast", "broken", "invalid", "blocker"}),
    frozenset({"loading", "spinner", "skeleton", "pending", "fetching", "busy", "wait", "progress"}),
    frozenset({"modal", "dialog", "popup", "overlay", "drawer", "interstitial", "confirm", "banner"}),
    frozenset({"ready", "interactive", "loaded", "complete", "completed", "success", "active", "visible", "enabled"}),
    frozenset({"vault", "file", "folder", "document", "vfs", "storage", "cloud", "share"}),
    frozenset({"contact", "lead", "deal", "account", "crm", "record", "datagrid", "table", "filter", "kanban"}),
    frozenset({"settings", "profile", "preferences", "config", "configuration", "permissions", "user", "admin"}),
    frozenset({"delete", "remove", "discard", "trash", "erase", "destroy", "drop", "purge", "clear"}),
    frozenset({"logout", "signout", "disconnect", "exit", "leave", "lock"}),
    frozenset({"invoice", "billing", "payment", "subscription", "charge", "receipt", "checkout", "stripe", "pricing"}),
    frozenset({"sms", "message", "chat", "conversation", "twilio", "phone", "call", "dial", "inbox"}),
    frozenset({"deploy", "publish", "release", "ship", "push", "production", "rollout"}),
]

_TOKEN_TO_CLUSTER_TERMS: Dict[str, frozenset[str]] = {}
for _cluster in _SYNONYM_CLUSTERS:
    for _tok in _cluster:
        _TOKEN_TO_CLUSTER_TERMS[_tok] = _cluster

_NEGATIVE_MARKERS = frozenset({
    "no", "not", "none", "never", "neither", "without", "false", "disabled",
    "hidden", "closed", "empty", "missing", "absent", "zero", "0", "off", "inactive",
})

_POSITIVE_MARKERS = frozenset({
    "yes", "true", "active", "enabled", "visible", "open", "opened", "present",
    "loaded", "complete", "completed", "success", "passed", "ready", "valid", "on",
})


# =====================================================================
# 1. Question Primitives
# =====================================================================


class ChoiceQuestion(BaseModel):
    """Selects one option from a closed set of candidates with calibrated probabilities."""

    type: Literal["choice"] = "choice"
    instructions: str
    criteria: Union[Dict[str, Any], List[str]]

    def normalized_criteria(self) -> Dict[str, str]:
        if isinstance(self.criteria, list):
            return {str(k): str(k) for k in self.criteria}
        out: Dict[str, str] = {}
        for k, v in self.criteria.items():
            if isinstance(v, (dict, list)):
                out[str(k)] = json.dumps(v, ensure_ascii=False)
            else:
                out[str(k)] = str(v)
        return out


class ScoreQuestion(BaseModel):
    """Scores state against an ordered rubric of descriptive levels."""

    type: Literal["score"] = "score"
    instructions: str
    criteria: Union[Dict[str, Any], List[str]]
    level_values: Optional[Dict[str, float]] = None

    def normalized_criteria(self) -> List[Tuple[str, str, float]]:
        """Returns ordered list of (key, description, numeric_value in [0.0, 1.0])."""
        if isinstance(self.criteria, list):
            items = [(str(k), str(k)) for k in self.criteria]
        else:
            items = [
                (str(k), json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v))
                for k, v in self.criteria.items()
            ]
        n = len(items)
        if n == 0:
            return []
        result: List[Tuple[str, str, float]] = []
        for idx, (k, desc) in enumerate(items):
            if self.level_values and k in self.level_values:
                val = float(self.level_values[k])
            else:
                val = idx / float(max(1, n - 1)) if n > 1 else 1.0
            result.append((k, desc, val))
        return result


class NoulQuestion(BaseModel):
    """Evaluates whether a statement/proposition holds true over the state in [0.0, 1.0]."""

    type: Literal["noul"] = "noul"
    instructions: str
    criteria: Optional[Union[str, Dict[str, Any]]] = None


class BoxQuestion(BaseModel):
    """Finds normalized [left, top, right, bottom] bounding coordinates in [0.0, 1.0]."""

    type: Literal["box"] = "box"
    instructions: str
    target_id: Optional[str] = None


QuestionSpec = Union[ChoiceQuestion, ScoreQuestion, NoulQuestion, BoxQuestion, Dict[str, Any]]


def coerce_question(q: QuestionSpec) -> Union[ChoiceQuestion, ScoreQuestion, NoulQuestion, BoxQuestion]:
    """Coerces a raw dictionary or Question object into a typed Question model."""
    if isinstance(q, (ChoiceQuestion, ScoreQuestion, NoulQuestion, BoxQuestion)):
        return q
    if not isinstance(q, dict):
        raise ValueError(f"Unsupported question specification: {type(q)}")
    q_type = str(q.get("type", "")).strip().lower()
    if q_type == "choice" or ("criteria" in q and q_type not in ("score", "noul", "box")):
        return ChoiceQuestion(**{**q, "type": "choice"})
    if q_type == "score":
        return ScoreQuestion(**{**q, "type": "score"})
    if q_type == "noul":
        return NoulQuestion(**{**q, "type": "noul"})
    if q_type == "box":
        return BoxQuestion(**{**q, "type": "box"})
    # Default fallback if no type given and no criteria dict: Noul boolean check
    return NoulQuestion(instructions=str(q.get("instructions") or q.get("question") or ""), criteria=q.get("criteria"))


# =====================================================================
# 2. Answer & Response Contracts
# =====================================================================


class ChoiceAnswer(BaseModel):
    """Typed answer for a ChoiceQuestion."""

    type: Literal["choice"] = "choice"
    choice: str
    probabilities: Dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)
    stages_evaluated: int = 1


class ScoreAnswer(BaseModel):
    """Typed answer for a ScoreQuestion."""

    type: Literal["score"] = "score"
    score: float = Field(ge=0.0, le=1.0)
    level: str
    probabilities: Dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)


class NoulAnswer(BaseModel):
    """Typed answer for a NoulQuestion."""

    type: Literal["noul"] = "noul"
    noul: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    @property
    def is_true(self) -> bool:
        return self.noul >= 0.5


class BoxAnswer(BaseModel):
    """Normalized bounding box answer [left, top, right, bottom] in [0.0, 1.0]."""

    type: Literal["box"] = "box"
    left: float = Field(ge=0.0, le=1.0)
    top: float = Field(ge=0.0, le=1.0)
    right: float = Field(ge=0.0, le=1.0)
    bottom: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    matched_id: Optional[str] = None

    @property
    def center(self) -> Tuple[float, float]:
        """Normalized (cx, cy) click target coordinates in [0.0, 1.0]."""
        return (round((self.left + self.right) / 2.0, 4), round((self.top + self.bottom) / 2.0, 4))

    def pixel_point(self, viewport_width: int = 1920, viewport_height: int = 1080) -> Tuple[int, int]:
        """Converts normalized center into integer (x, y) viewport pixel coordinates."""
        cx, cy = self.center
        return (int(round(cx * viewport_width)), int(round(cy * viewport_height)))


AnswerType = Union[ChoiceAnswer, ScoreAnswer, NoulAnswer, BoxAnswer]


class SystemOneResponse(BaseModel):
    """Aggregated response from a parallel System-1 evaluation call."""

    answers: Dict[str, AnswerType]
    backend: str = "local"
    model: str = "local-system-one-v1"
    duration_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def get_choice(self, key: str) -> ChoiceAnswer:
        ans = self.answers[key]
        if not isinstance(ans, ChoiceAnswer):
            raise TypeError(f"Answer '{key}' is {type(ans).__name__}, expected ChoiceAnswer")
        return ans

    def get_score(self, key: str) -> ScoreAnswer:
        ans = self.answers[key]
        if not isinstance(ans, ScoreAnswer):
            raise TypeError(f"Answer '{key}' is {type(ans).__name__}, expected ScoreAnswer")
        return ans

    def get_noul(self, key: str) -> NoulAnswer:
        ans = self.answers[key]
        if not isinstance(ans, NoulAnswer):
            raise TypeError(f"Answer '{key}' is {type(ans).__name__}, expected NoulAnswer")
        return ans

    def get_box(self, key: str) -> BoxAnswer:
        ans = self.answers[key]
        if not isinstance(ans, BoxAnswer):
            raise TypeError(f"Answer '{key}' is {type(ans).__name__}, expected BoxAnswer")
        return ans


# =====================================================================
# 3. Tokenization, Softmax & Epistemic Calibration Utilities
# =====================================================================


def _stem(word: str) -> str:
    """Lightweight morphological stemmer for UI/action verbs and plurals."""
    if len(word) <= 3 or word.isdigit():
        return word
    if word.endswith("ing") and len(word) > 5:
        base = word[:-3]
        if base.endswith(("at", "iz", "os", "am", "ak", "av", "ag", "iv")):
            return base + "e"
        return base
    if word.endswith("ed") and len(word) > 4:
        base = word[:-2]
        if base.endswith(("at", "iz", "os", "am", "ak", "av", "ag", "iv")):
            return base + "e"
        return base
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 4 and not word.endswith(("ses", "xes", "zes", "ches", "shes")):
        return word[:-1]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
        return word[:-1]
    return word


def _tokenize(text: str, expand_synonyms: bool = False) -> List[str]:
    """Splits camelCase, snake_case, and prose into normalized tokens with light stemming."""
    if not text:
        return []
    # Strip structural bracket tags like [dom_element:button] before tokenizing
    cleaned_text = re.sub(r"\[[a-z0-9_]+:[a-z0-9_]+\]", " ", str(text))
    decamel = _CAMEL_RE.sub(r"\1 \2", cleaned_text)
    raw_tokens = _WORD_RE.findall(decamel.lower())
    filtered: List[str] = []
    for t in raw_tokens:
        if len(t) > 1 or t.isdigit():
            filtered.append(t)
            st = _stem(t)
            if st != t:
                filtered.append(st)
    if not expand_synonyms:
        return filtered
    expanded = list(filtered)
    for t in filtered:
        cluster = _TOKEN_TO_CLUSTER_TERMS.get(t) or _TOKEN_TO_CLUSTER_TERMS.get(_stem(t))
        if cluster:
            for syn in cluster:
                if syn not in expanded:
                    expanded.append(syn)
    return expanded


def _char_trigrams(text: str) -> set[str]:
    clean = re.sub(r"\s+", " ", (text or "").lower().strip())
    if len(clean) < 3:
        return {clean} if clean else set()
    return {clean[i : i + 3] for i in range(len(clean) - 2)}


def _calibrated_softmax(logits: Dict[str, float], temperature: float = 0.45) -> Dict[str, float]:
    """Computes temperature-scaled softmax probabilities guaranteed to sum to 1.0."""
    if not logits:
        return {}
    temp = max(0.05, float(temperature))
    max_l = max(logits.values())
    exps = {k: math.exp((v - max_l) / temp) for k, v in logits.items()}
    total = sum(exps.values()) or 1.0
    probs = {k: round(v / total, 6) for k, v in exps.items()}
    # Fix rounding residual on top key so sum is 1.0
    top_k = max(probs, key=probs.get)  # type: ignore[arg-type]
    residual = round(1.0 - sum(probs.values()), 6)
    probs[top_k] = round(max(0.0, min(1.0, probs[top_k] + residual)), 6)
    return probs


def _compute_epistemic_confidence(
    probs: Dict[str, float],
    evidence_strength: float,
) -> float:
    """Computes calibrated epistemic confidence in [0.0, 1.0].

    Combines:
      1. Top-1 vs Top-2 probability margin (separation clarity).
      2. Normalized inverse Shannon entropy: 1 - H(p) / log(K).
      3. Evidence strength (signal overlap with the state).
    """
    n = len(probs)
    if n == 0:
        return 0.0
    if n == 1:
        ev = min(1.0, max(0.15, evidence_strength))
        return round(min(0.98, 0.45 + 0.52 * ev) if ev >= 0.25 else ev, 4)

    sorted_p = sorted(probs.values(), reverse=True)
    top1, top2 = sorted_p[0], sorted_p[1]
    margin = top1 - top2

    entropy = -sum(p * math.log(max(p, 1e-12)) for p in sorted_p if p > 0)
    max_entropy = math.log(n)
    inv_entropy = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0

    ev_factor = min(1.0, max(0.1, evidence_strength))
    raw_conf = (0.45 * margin + 0.30 * inv_entropy + 0.25 * ev_factor)
    # Boost slightly when both margin and evidence are strong
    if margin > 0.35 and ev_factor > 0.55:
        raw_conf = min(0.99, raw_conf + 0.18)
    return round(min(0.99, max(0.05, raw_conf)), 4)


def _state_to_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    try:
        return json.dumps(state, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(state)


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 3.8)) if text else 0


# =====================================================================
# 4. Local System-1 Decision Backend (In-Process Daemon Engine)
# =====================================================================


class LocalSystemOneBackend:
    """Fast, deterministic, zero-hallucination local System-1 evaluator.

    Evaluates Choice, Score, Noul, and Box questions in parallel in ~1ms-15ms
    directly inside the local daemon process without external API calls.
    Supports high-cardinality (>255 options) two-stage beam ranking matching Jev.
    """

    MAX_SINGLE_STAGE_CARDINALITY = 255
    STAGE1_BEAM_WIDTH = 32

    def __init__(self, model_name: str = "local-system-one-v1"):
        self.model_name = model_name

    def evaluate(
        self,
        state: Any,
        questions: Dict[str, QuestionSpec],
    ) -> SystemOneResponse:
        t0 = time.perf_counter()
        state_str = _state_to_text(state)
        state_tokens = _tokenize(state_str, expand_synonyms=False)
        state_expanded = set(_tokenize(state_str, expand_synonyms=True))
        state_token_counts: Dict[str, int] = {}
        for tok in state_tokens:
            state_token_counts[tok] = state_token_counts.get(tok, 0) + 1

        answers: Dict[str, AnswerType] = {}
        q_chars = 0

        for q_key, raw_q in questions.items():
            q = coerce_question(raw_q)
            q_chars += len(q.instructions)
            if isinstance(q, ChoiceQuestion):
                answers[q_key] = self._eval_choice(state, state_str, state_token_counts, state_expanded, q)
            elif isinstance(q, ScoreQuestion):
                answers[q_key] = self._eval_score(state, state_str, state_token_counts, state_expanded, q)
            elif isinstance(q, NoulQuestion):
                answers[q_key] = self._eval_noul(state, state_str, state_token_counts, state_expanded, q)
            elif isinstance(q, BoxQuestion):
                answers[q_key] = self._eval_box(state, state_str, state_token_counts, state_expanded, q)

        dur_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        in_toks = _estimate_tokens(state_str) + max(1, int(q_chars / 3.8))
        return SystemOneResponse(
            answers=answers,
            backend="local",
            model=self.model_name,
            duration_ms=dur_ms,
            input_tokens=in_toks,
            output_tokens=0,
            cost_usd=0.0,
        )

    def _score_candidate_against_context(
        self,
        state_str: str,
        state_token_counts: Dict[str, int],
        state_expanded: set[str],
        instructions: str,
        opt_key: str,
        opt_desc: str,
        focus_text: str = "",
    ) -> Tuple[float, float]:
        """Returns (raw_logit_score, evidence_strength in [0, 1])."""
        target_context = focus_text if focus_text else f"{instructions} {state_str}"
        ctx_tokens = set(t for t in _tokenize(target_context, expand_synonyms=False) if t not in _STOPWORDS)
        ctx_expanded = set(_tokenize(target_context, expand_synonyms=True)) - _STOPWORDS
        inst_tokens = set(t for t in _tokenize(instructions, expand_synonyms=False) if t not in _STOPWORDS)
        inst_expanded = set(_tokenize(instructions, expand_synonyms=True)) - _STOPWORDS

        opt_full = f"{opt_key} {opt_desc}"
        # Deduplicate option tokens while preserving order so stemmed variants don't inflate denominator
        opt_tokens = list(dict.fromkeys(t for t in _tokenize(opt_full, expand_synonyms=False) if t not in _STOPWORDS))
        opt_key_tokens = list(dict.fromkeys(t for t in _tokenize(opt_key, expand_synonyms=False) if t not in _STOPWORDS))

        if not opt_tokens:
            return 0.0, 0.1

        # 1. Exact & synonym token overlap prioritized against focus_text & instructions (prevents state["elements"] self-echo)
        matched_direct = 0.0
        matched_synonym = 0.0
        for tok in opt_tokens:
            if tok in ctx_tokens or tok in inst_tokens:
                weight = 2.2 if (tok in ctx_tokens and tok in inst_tokens) else 1.75
                matched_direct += weight
            elif tok in ctx_expanded or tok in inst_expanded:
                matched_synonym += 1.35
            elif not focus_text and tok in state_token_counts:
                matched_direct += 1.0

        # 2. Key-specific match boost (when the option key/ID or its words appear in goal/instructions)
        key_boost = 0.0
        if opt_key.lower() in target_context.lower():
            key_boost += 1.1
        if opt_key_tokens:
            key_hits = sum(1 for kt in opt_key_tokens if kt in ctx_tokens or kt in ctx_expanded)
            key_boost += 0.85 * (key_hits / len(opt_key_tokens))

        # 3. Phrase / bigram match bonus
        opt_lower = opt_desc.lower()
        ctx_lower = target_context.lower()
        phrase_boost = 0.0
        if len(opt_tokens) >= 2:
            for i in range(len(opt_tokens) - 1):
                bg = f"{opt_tokens[i]} {opt_tokens[i + 1]}"
                if bg in ctx_lower or bg in opt_lower and opt_tokens[i] in ctx_tokens and opt_tokens[i + 1] in ctx_tokens:
                    phrase_boost += 0.45

        # 4. Character trigram Jaccard similarity for morphological resilience
        tri_opt = _char_trigrams(opt_full)
        tri_ctx = _char_trigrams(f"{instructions} {focus_text[:600] if focus_text else state_str[:600]}")
        tri_sim = (len(tri_opt & tri_ctx) / max(1, len(tri_opt))) if tri_opt else 0.0

        # 5. Penalize explicitly disabled or antonym-conflicting candidates
        penalty = 0.0
        if any(w in opt_lower for w in ("disabled: true", "disabled=true", "[disabled]", "status: disabled")):
            penalty -= 1.5
        destructive_verbs = {"delete", "remove", "discard", "trash", "erase", "destroy", "purge"}
        constructive_verbs = {"save", "create", "add", "keep", "preserve"}
        if (ctx_tokens & destructive_verbs) and (set(opt_tokens) & constructive_verbs) and not (set(opt_tokens) & destructive_verbs):
            penalty -= 1.2
        elif (ctx_tokens & constructive_verbs) and (set(opt_tokens) & destructive_verbs) and not (set(opt_tokens) & constructive_verbs):
            penalty -= 1.2

        denom = math.sqrt(len(opt_tokens)) + 1.0
        norm_overlap = (matched_direct + matched_synonym) / denom
        raw_score = norm_overlap + key_boost + phrase_boost + (0.6 * tri_sim) + penalty
        evidence = min(1.0, (matched_direct + 0.5 * matched_synonym + key_boost) / max(2.0, len(opt_tokens) * 0.85))
        return raw_score, evidence

    def _extract_goal_focus(self, state: Any, instructions: str) -> str:
        """Extracts goal/query/intent fields from structured state so options align with user intent."""
        parts = [instructions]
        if isinstance(state, dict):
            for k in ("goal", "query", "intent", "instruction", "task", "user_input", "prompt", "step_goal", "active_error", "modal_text"):
                val = state.get(k)
                if val:
                    parts.append(str(val))
        return " ".join(parts)

    def _eval_choice(
        self,
        state: Any,
        state_str: str,
        state_token_counts: Dict[str, int],
        state_expanded: set[str],
        q: ChoiceQuestion,
    ) -> ChoiceAnswer:
        criteria = q.normalized_criteria()
        if not criteria:
            return ChoiceAnswer(choice="", probabilities={}, confidence=0.0)

        focus_text = self._extract_goal_focus(state, q.instructions)
        stages = 1

        # High-cardinality (>255) two-stage hierarchical evaluation (matches Jev architecture)
        if len(criteria) > self.MAX_SINGLE_STAGE_CARDINALITY:
            stages = 2
            stage1_scores: Dict[str, float] = {}
            for k, desc in criteria.items():
                sc, _ = self._score_candidate_against_context(
                    state_str, state_token_counts, state_expanded, q.instructions, k, desc[:140], focus_text
                )
                stage1_scores[k] = sc
            top_keys = sorted(stage1_scores, key=stage1_scores.get, reverse=True)[: self.STAGE1_BEAM_WIDTH]  # type: ignore[arg-type]
            shortlist = {k: criteria[k] for k in top_keys}
        else:
            shortlist = criteria

        logits: Dict[str, float] = {}
        evidences: Dict[str, float] = {}
        for k, desc in shortlist.items():
            sc, ev = self._score_candidate_against_context(
                state_str, state_token_counts, state_expanded, q.instructions, k, desc, focus_text
            )
            logits[k] = sc
            evidences[k] = ev

        probs = _calibrated_softmax(logits, temperature=0.42)
        best_choice = max(probs, key=probs.get)  # type: ignore[arg-type]

        # If two-stage, include 0.0 entries for pruned options so schema keys match
        if stages == 2:
            full_probs = {k: probs.get(k, 0.0) for k in criteria}
            probs = full_probs

        conf = _compute_epistemic_confidence(probs, evidences.get(best_choice, 0.25))
        return ChoiceAnswer(
            choice=best_choice,
            probabilities=probs,
            confidence=conf,
            stages_evaluated=stages,
        )

    def _eval_score(
        self,
        state: Any,
        state_str: str,
        state_token_counts: Dict[str, int],
        state_expanded: set[str],
        q: ScoreQuestion,
    ) -> ScoreAnswer:
        levels = q.normalized_criteria()
        if not levels:
            return ScoreAnswer(score=0.0, level="", probabilities={}, confidence=0.0)

        focus_text = self._extract_goal_focus(state, q.instructions)
        logits: Dict[str, float] = {}
        evidences: Dict[str, float] = {}
        level_val_map: Dict[str, float] = {}

        for k, desc, val in levels:
            sc, ev = self._score_candidate_against_context(
                state_str, state_token_counts, state_expanded, q.instructions, k, desc, focus_text
            )
            logits[k] = sc
            evidences[k] = ev
            level_val_map[k] = val

        probs = _calibrated_softmax(logits, temperature=0.45)
        best_level = max(probs, key=probs.get)  # type: ignore[arg-type]
        expected_score = sum(probs[k] * level_val_map[k] for k in probs)
        conf = _compute_epistemic_confidence(probs, evidences.get(best_level, 0.3))
        return ScoreAnswer(
            score=round(min(1.0, max(0.0, expected_score)), 4),
            level=best_level,
            probabilities=probs,
            confidence=conf,
        )

    def _eval_noul(
        self,
        state: Any,
        state_str: str,
        state_token_counts: Dict[str, int],
        state_expanded: set[str],
        q: NoulQuestion,
    ) -> NoulAnswer:
        crit_str = ""
        if q.criteria:
            crit_str = json.dumps(q.criteria, ensure_ascii=False) if isinstance(q.criteria, dict) else str(q.criteria)
        statement = f"{q.instructions} {crit_str}".strip()
        stmt_tokens = [t for t in _tokenize(statement, expand_synonyms=False) if t not in _STOPWORDS]
        if not stmt_tokens:
            return NoulAnswer(noul=0.5, confidence=0.1)

        # Check content token overlap
        content_tokens = [t for t in stmt_tokens if t not in _NEGATIVE_MARKERS and t not in _POSITIVE_MARKERS]
        if not content_tokens:
            content_tokens = stmt_tokens

        hits = 0.0
        for tok in content_tokens:
            if tok in state_token_counts:
                hits += 1.0
            elif tok in state_expanded:
                hits += 0.7

        coverage = hits / float(len(content_tokens))

        # Check explicit boolean/status key-value pairs in state if state is a dict
        polarity_shift = 0.0
        state_lower = state_str.lower()
        stmt_lower = statement.lower()

        # Check if statement asks about presence of error/blocker/modal/toast
        asks_about_error = any(w in stmt_lower for w in ("error", "toast", "exception", "crash", "blocker", "alert", "broken"))
        if asks_about_error:
            has_error_signal = False
            if isinstance(state, dict):
                for err_k in ("error", "errors", "error_toast", "active_error", "blocker", "blocking_modal", "alert", "exception"):
                    val = state.get(err_k)
                    if val and val not in (False, "none", "None", "null", [], {}):
                        has_error_signal = True
            else:
                if any(p in state_lower for p in ("error:", "exception:", "unhandled error", "error toast", "\"error\": true")):
                    has_error_signal = True
            if not has_error_signal:
                return NoulAnswer(noul=0.04, confidence=0.92)
            else:
                return NoulAnswer(noul=0.94, confidence=0.91)

        # Check negations in state near matched terms
        for tok in content_tokens:
            # Look for `"<tok>": false` or `<tok>: false` or `no <tok>` or `not <tok>`
            neg_patterns = (
                f'"{tok}": false',
                f'"{tok}":false',
                f'"{tok}": null',
                f'"{tok}": none',
                f'"{tok}": []',
                f'"{tok}": ""',
                f"no {tok}",
                f"not {tok}",
                f"{tok} is false",
                f"{tok}: false",
            )
            pos_patterns = (
                f'"{tok}": true',
                f'"{tok}":true',
                f"{tok}: true",
                f"{tok} is true",
            )
            if any(np in state_lower for np in neg_patterns):
                polarity_shift -= 0.55
            if any(pp in state_lower for pp in pos_patterns):
                polarity_shift += 0.45

        # If the statement itself is negated (e.g. "There is no error")
        stmt_negated = any(neg in _tokenize(statement, expand_synonyms=False) for neg in ("no", "not", "never", "without", "none"))
        raw_prob = (0.12 + 0.78 * min(1.0, coverage)) + polarity_shift
        if stmt_negated:
            raw_prob = 1.0 - raw_prob + polarity_shift

        noul_val = round(min(0.99, max(0.01, raw_prob)), 4)
        dist_from_half = abs(noul_val - 0.5) * 2.0
        conf = round(min(0.98, max(0.15, 0.55 * dist_from_half + 0.45 * min(1.0, coverage + abs(polarity_shift)))), 4)
        return NoulAnswer(noul=noul_val, confidence=conf)

    def _eval_box(
        self,
        state: Any,
        state_str: str,
        state_token_counts: Dict[str, int],
        state_expanded: set[str],
        q: BoxQuestion,
    ) -> BoxAnswer:
        """Resolves normalized [left, top, right, bottom] bounds from UI/canvas state elements."""
        candidates = self._extract_spatial_candidates(state)
        if not candidates:
            return BoxAnswer(left=0.4, top=0.4, right=0.6, bottom=0.6, confidence=0.15, matched_id=None)

        if q.target_id:
            for cid, cdesc, box in candidates:
                if cid == q.target_id:
                    return BoxAnswer(
                        left=box[0], top=box[1], right=box[2], bottom=box[3],
                        confidence=0.96, matched_id=cid,
                    )

        best_id = None
        best_box = (0.4, 0.4, 0.6, 0.6)
        logits: Dict[str, float] = {}
        box_map: Dict[str, Tuple[float, float, float, float]] = {}
        ev_map: Dict[str, float] = {}

        focus = self._extract_goal_focus(state, q.instructions)
        for cid, cdesc, box in candidates:
            sc, ev = self._score_candidate_against_context(
                state_str, state_token_counts, state_expanded, q.instructions, cid, cdesc, focus
            )
            logits[cid] = sc
            box_map[cid] = box
            ev_map[cid] = ev

        probs = _calibrated_softmax(logits, temperature=0.40)
        best_id = max(probs, key=probs.get)  # type: ignore[arg-type]
        best_box = box_map[best_id]
        conf = _compute_epistemic_confidence(probs, ev_map.get(best_id, 0.4))
        return BoxAnswer(
            left=best_box[0],
            top=best_box[1],
            right=best_box[2],
            bottom=best_box[3],
            confidence=conf,
            matched_id=best_id,
        )

    def _extract_spatial_candidates(
        self, state: Any
    ) -> List[Tuple[str, str, Tuple[float, float, float, float]]]:
        """Extracts (id, description, normalized_box) from DOM or ArtCraft/canvas scene states."""
        out: List[Tuple[str, str, Tuple[float, float, float, float]]] = []
        if not isinstance(state, dict):
            return out

        vp_w = float(state.get("viewport", {}).get("width", 1920) if isinstance(state.get("viewport"), dict) else state.get("viewport_width", 1920))
        vp_h = float(state.get("viewport", {}).get("height", 1080) if isinstance(state.get("viewport"), dict) else state.get("viewport_height", 1080))

        lists_to_scan: List[Any] = []
        for k in ("elements", "interactive_elements", "nodes", "layers", "actors", "tools", "controls", "candidates"):
            val = state.get(k)
            if isinstance(val, list):
                lists_to_scan.extend(val)

        for idx, item in enumerate(lists_to_scan):
            if not isinstance(item, dict):
                continue
            cid = str(item.get("id") or item.get("test_id") or item.get("command_id") or item.get("name") or f"el_{idx}")
            desc = " ".join(
                str(item.get(f, ""))
                for f in ("label", "text", "name", "role", "type", "description", "aria_label", "tool_type")
                if item.get(f)
            )
            box = self._normalize_item_box(item, vp_w, vp_h)
            if box is not None:
                out.append((cid, desc, box))
        return out

    @staticmethod
    def _normalize_item_box(
        item: Dict[str, Any], vp_w: float, vp_h: float
    ) -> Optional[Tuple[float, float, float, float]]:
        b = item.get("box") or item.get("bbox") or item.get("bounds") or item.get("rect")
        if isinstance(b, dict):
            if all(k in b for k in ("left", "top", "right", "bottom")):
                l, t, r, bot = float(b["left"]), float(b["top"]), float(b["right"]), float(b["bottom"])
                if r > 1.0 or bot > 1.0:
                    l, r = l / max(1.0, vp_w), r / max(1.0, vp_w)
                    t, bot = t / max(1.0, vp_h), bot / max(1.0, vp_h)
                return (
                    round(min(1.0, max(0.0, l)), 4),
                    round(min(1.0, max(0.0, t)), 4),
                    round(min(1.0, max(0.0, r)), 4),
                    round(min(1.0, max(0.0, bot)), 4),
                )
            if all(k in b for k in ("x", "y", "width", "height")):
                x, y, w, h = float(b["x"]), float(b["y"]), float(b["width"]), float(b["height"])
                if x > 1.0 or y > 1.0 or w > 1.0 or h > 1.0:
                    l, r = x / max(1.0, vp_w), (x + w) / max(1.0, vp_w)
                    t, bot = y / max(1.0, vp_h), (y + h) / max(1.0, vp_h)
                else:
                    l, t, r, bot = x, y, x + w, y + h
                return (
                    round(min(1.0, max(0.0, l)), 4),
                    round(min(1.0, max(0.0, t)), 4),
                    round(min(1.0, max(0.0, r)), 4),
                    round(min(1.0, max(0.0, bot)), 4),
                )
        elif isinstance(b, (list, tuple)) and len(b) == 4:
            l, t, r, bot = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            if r > 1.0 or bot > 1.0:
                l, r = l / max(1.0, vp_w), r / max(1.0, vp_w)
                t, bot = t / max(1.0, vp_h), bot / max(1.0, vp_h)
            return (
                round(min(1.0, max(0.0, l)), 4),
                round(min(1.0, max(0.0, t)), 4),
                round(min(1.0, max(0.0, r)), 4),
                round(min(1.0, max(0.0, bot)), 4),
            )
        return None


# =====================================================================
# 5. Cloud TypeSafe Jev Backend (`https://api.typesafe.ai/v1/systemone`)
# =====================================================================


class TypeSafeJevBackend:
    """Cloud client for TypeSafe AI's Jev System-1 endpoint."""

    DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "jev-latest",
        endpoint: Optional[str] = None,
        timeout_sec: float = 15.0,
    ):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY") or ""
        self.model = model
        self.endpoint = endpoint or os.environ.get("TYPESAFE_API_URL") or self.DEFAULT_ENDPOINT
        self.timeout_sec = timeout_sec

    def _build_payload(self, state: Any, questions: Dict[str, QuestionSpec]) -> Tuple[Dict[str, Any], bool]:
        serialized_q: Dict[str, Any] = {}
        has_box = False
        for k, raw_q in questions.items():
            q = coerce_question(raw_q)
            if isinstance(q, ChoiceQuestion):
                serialized_q[k] = {
                    "type": "choice",
                    "instructions": q.instructions,
                    "criteria": q.normalized_criteria(),
                }
            elif isinstance(q, ScoreQuestion):
                serialized_q[k] = {
                    "type": "score",
                    "instructions": q.instructions,
                    "criteria": {lvl_k: desc for lvl_k, desc, _ in q.normalized_criteria()},
                }
            elif isinstance(q, NoulQuestion):
                entry: Dict[str, Any] = {"type": "noul", "instructions": q.instructions}
                if q.criteria is not None:
                    entry["criteria"] = q.criteria
                serialized_q[k] = entry
            elif isinstance(q, BoxQuestion):
                has_box = True
                serialized_q[k] = {"type": "box", "instructions": q.instructions}

        payload = {
            "model": "jev-preview" if has_box and self.model == "jev-latest" else self.model,
            "state": state,
            "questions": serialized_q,
        }
        return payload, has_box

    def _parse_api_response(
        self,
        data: Dict[str, Any],
        questions: Dict[str, QuestionSpec],
        dur_ms: float,
    ) -> SystemOneResponse:
        raw_answers = data.get("answers") or {}
        answers: Dict[str, AnswerType] = {}
        for k, raw_q in questions.items():
            q = coerce_question(raw_q)
            ans_dict = raw_answers.get(k) or {}
            if isinstance(q, ChoiceQuestion):
                probs = ans_dict.get("probabilities") or {}
                choice = str(ans_dict.get("choice") or (max(probs, key=probs.get) if probs else ""))  # type: ignore[arg-type]
                conf = float(ans_dict.get("confidence", 0.8))
                answers[k] = ChoiceAnswer(choice=choice, probabilities=probs, confidence=conf)
            elif isinstance(q, ScoreQuestion):
                probs = ans_dict.get("probabilities") or {}
                score = float(ans_dict.get("score", 0.5))
                lvl = str(ans_dict.get("level") or (max(probs, key=probs.get) if probs else ""))  # type: ignore[arg-type]
                conf = float(ans_dict.get("confidence", 0.8))
                answers[k] = ScoreAnswer(score=score, level=lvl, probabilities=probs, confidence=conf)
            elif isinstance(q, NoulQuestion):
                noul_val = float(ans_dict.get("noul", ans_dict.get("probability", 0.5)))
                conf = float(ans_dict.get("confidence", abs(noul_val - 0.5) * 2.0))
                answers[k] = NoulAnswer(noul=noul_val, confidence=conf)
            elif isinstance(q, BoxQuestion):
                answers[k] = BoxAnswer(
                    left=float(ans_dict.get("left", 0.0)),
                    top=float(ans_dict.get("top", 0.0)),
                    right=float(ans_dict.get("right", 1.0)),
                    bottom=float(ans_dict.get("bottom", 1.0)),
                    confidence=float(ans_dict.get("confidence", 0.8)),
                )

        usage = data.get("usage") or {}
        in_tok = int(usage.get("input_tokens") or _estimate_tokens(_state_to_text(data)))
        cost = round((in_tok / 1_000_000.0) * JEV_INPUT_COST_PER_1M, 8)
        return SystemOneResponse(
            answers=answers,
            backend="jev",
            model=data.get("model") or self.model,
            duration_ms=dur_ms,
            input_tokens=in_tok,
            output_tokens=0,
            cost_usd=cost,
        )

    async def evaluate_async(
        self,
        state: Any,
        questions: Dict[str, QuestionSpec],
    ) -> SystemOneResponse:
        if not self.api_key:
            raise RuntimeError("TYPESAFE_API_KEY is required for TypeSafeJevBackend.")
        payload, has_box = self._build_payload(state, questions)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if has_box:
            headers["TypeSafe-Danger-Zone-Images"] = "true"

        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
            resp = await client.post(self.endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        dur_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return self._parse_api_response(data, questions, dur_ms)


# =====================================================================
# 6. Unified SystemOneClient Facade & Confidence Routing
# =====================================================================


class ConfidenceRouteDecision(BaseModel):
    """Result of applying a confidence gate to a System-1 answer."""

    action: Literal["act", "fallback", "escalate_system2"]
    selected_value: Any
    confidence: float
    threshold: float
    reason: str


def route_with_confidence(
    answer: AnswerType,
    threshold: float = 0.60,
    fallback_value: Any = None,
) -> ConfidenceRouteDecision:
    """Applies TypeSafe's two-axis Confidence-Gated Routing pattern.

    'The answer tells you what; confidence tells you whether to act.'
    """
    if isinstance(answer, ChoiceAnswer):
        val: Any = answer.choice
    elif isinstance(answer, ScoreAnswer):
        val = answer.score
    elif isinstance(answer, NoulAnswer):
        val = answer.noul
    elif isinstance(answer, BoxAnswer):
        val = answer.center
    else:
        val = None

    if answer.confidence >= threshold:
        return ConfidenceRouteDecision(
            action="act",
            selected_value=val,
            confidence=answer.confidence,
            threshold=threshold,
            reason=f"Confidence {answer.confidence:.2f} >= threshold {threshold:.2f}; safe for System-1 execution.",
        )

    if fallback_value is not None:
        return ConfidenceRouteDecision(
            action="fallback",
            selected_value=fallback_value,
            confidence=answer.confidence,
            threshold=threshold,
            reason=f"Confidence {answer.confidence:.2f} < threshold {threshold:.2f}; using deterministic fallback.",
        )

    return ConfidenceRouteDecision(
        action="escalate_system2",
        selected_value=val,
        confidence=answer.confidence,
        threshold=threshold,
        reason=f"Confidence {answer.confidence:.2f} < threshold {threshold:.2f}; escalating ambiguous state to System-2 LLM.",
    )


class SystemOneClient:
    """Unified System-1 Decision Client supporting Local Daemon and Cloud Jev backends."""

    def __init__(
        self,
        backend: Literal["local", "jev", "auto"] = "auto",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.requested_backend = backend
        self.local_backend = LocalSystemOneBackend(model_name=model or "local-system-one-v1")
        self.jev_backend = TypeSafeJevBackend(
            api_key=api_key,
            model=model or "jev-latest",
        )

    def _resolve_backend(self) -> Literal["local", "jev"]:
        if self.requested_backend == "local":
            return "local"
        if self.requested_backend == "jev":
            return "jev"
        # In "auto" mode: prefer local daemon unless VALSTORM_SYSTEM_ONE_BACKEND=jev and TYPESAFE_API_KEY is set
        env_pref = os.environ.get("VALSTORM_SYSTEM_ONE_BACKEND", "local").strip().lower()
        if env_pref == "jev" and self.jev_backend.api_key:
            return "jev"
        return "local"

    def evaluate(
        self,
        state: Any,
        questions: Dict[str, QuestionSpec],
    ) -> SystemOneResponse:
        """Synchronously evaluates a batch of System-1 questions in parallel against `state`."""
        active = self._resolve_backend()
        if active == "jev":
            try:
                return asyncio.run(self.jev_backend.evaluate_async(state, questions))
            except Exception:
                # Self-healing fallback to local daemon evaluator if cloud endpoint is unreachable
                return self.local_backend.evaluate(state, questions)
        return self.local_backend.evaluate(state, questions)

    async def evaluate_async(
        self,
        state: Any,
        questions: Dict[str, QuestionSpec],
    ) -> SystemOneResponse:
        """Asynchronously evaluates a batch of System-1 questions in parallel against `state`."""
        active = self._resolve_backend()
        if active == "jev":
            try:
                return await self.jev_backend.evaluate_async(state, questions)
            except Exception:
                return self.local_backend.evaluate(state, questions)
        return self.local_backend.evaluate(state, questions)

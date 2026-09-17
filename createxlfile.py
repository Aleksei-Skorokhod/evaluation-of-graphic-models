"""
Reliable image-generation model collector v8.

Goals:
- high-recall Hugging Face discovery without pretending discovery is exhaustive;
- conservative capability classification: missing evidence => Not found;
- evidence-level tracking for every capability;
- Model Card metadata + Model Card text;
- explicit separation of models from model components/variants;
- multi-source evidence graph with dynamic source-provider matrix;
- official-model enrichment with optional HF repository IDs/aliases;
- 30-day and all-time HF download counters when available;
- Arena ELO only through explicit/verified aliases;
- auditable Excel output.

Requirements:
    pip install -U huggingface_hub pandas openpyxl requests
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
import re
import time
from urllib.parse import quote_plus, urlparse
from html.parser import HTMLParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from huggingface_hub import HfApi, ModelCard


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_PIPELINES = (
    "text-to-image",
    "image-to-image",
    "unconditional-image-generation",
    "depth-to-image",
)

# Search terms are a recall-oriented supplement to pipeline-tag discovery.
# They deliberately produce candidates; later stages remain conservative.
DISCOVERY_SEARCH_TERMS = (
    "text-to-image",
    "image-to-image",
    "inpainting",
    "image editing",
    "text-to-video image",
    "diffusion",
    "stable diffusion",
    "flux",
    "sdxl",
)

VARIANT_PATTERNS = {
    "LoRA": [r"\blora\b", r"-lora\b", r"_lora\b"],
    "ControlNet": [r"\bcontrolnet\b", r"\bcontrol-net\b"],
    "Adapter": [r"\badapter\b", r"\bip-adapter\b", r"\bt2i-adapter\b"],
    "Embedding": [r"\bembedding\b", r"\btextual[-_ ]?inversion\b"],
    "Quantized": [
        r"\bquant\b", r"\bquantized\b", r"\bgguf\b", r"\bgptq\b",
        r"\bawq\b", r"\bint[248]\b", r"\bfp8\b",
    ],
    "Upscaler": [r"\bupscal", r"\bsuper[-_ ]?resolution\b"],
    "Inpainting Component": [r"\binpainting[-_ ]?(?:model|adapter|controlnet)\b"],
}

EXCLUDED_COMPONENT_TYPES = {
    "LoRA", "ControlNet", "Adapter", "Embedding",
    "Upscaler", "Inpainting Component", "Quantized",
}

DISCOVERY_BATCH_SIZE = 500

VARIANT_TAGS = {
    "lora": "LoRA",
    "controlnet": "ControlNet",
    "adapter": "Adapter",
    "ip-adapter": "Adapter",
    "t2i-adapter": "Adapter",
    "textual-inversion": "Embedding",
    "embedding": "Embedding",
    "upscaler": "Upscaler",
    "quantized": "Quantized",
    "gguf": "Quantized",
    "gptq": "Quantized",
    "awq": "Quantized",
}

CAPABILITIES = (
    "Text-to-Image",
    "Image-to-Image",
    "Inpainting",
    "Image Editing",
    "Control / Conditioning",
)

# Only fixed, commonly used image-generation benchmarks are kept in the main table.
# Values are populated only when the exact benchmark is identified in structured
# evaluation metadata; arbitrary model-specific evaluations are ignored.
COMMON_BENCHMARKS = (
    "GenEval",
    "DPG-Bench",
    "ImageReward",
    "PickScore",
    "HPS v2",
)
NOT_FOUND = "Not found"
ELO_NOT_FOUND = NOT_FOUND
YES = "Yes"
NO = "No"

CAPABILITY_PATTERNS = {
    "Text-to-Image": (
        r"\btext[- ]to[- ]image\b",
        r"\btext2image\b",
        r"\bt2i\b",
        r"\btext[- ]conditioned image generation\b",
    ),
    "Image-to-Image": (
        r"\bimage[- ]to[- ]image\b",
        r"\bimg2img\b",
        r"\bimage2image\b",
        r"\bimg[- ]to[- ]img\b",
    ),
    "Inpainting": (
        r"\binpainting\b",
        r"\bin[- ]paint(?:ing)?\b",
    ),
    "Image Editing": (
        r"\bimage editing\b",
        r"\bedit(?:ing)? images?\b",
        r"\bimage[- ]edit\b",
        r"\bediting[- ]model\b",
    ),
    "Control / Conditioning": (
        r"\bcontrolnet\b",
        r"\bcontrol[- ]net\b",
        r"\bip[- ]adapter\b",
        r"\bt2i[- ]adapter\b",
        r"\bimage prompt(?:ing)?\b",
        r"\bconditioning\b",
        r"\bconditional image generation\b",
    ),
}

# Phrases that strongly indicate a capability is the intended function of
# the repository/model, rather than merely mentioning it in a limitation.
POSITIVE_CONTEXT_PATTERNS = (
    r"\bsupport(?:s|ed)?\b",
    r"\bcan\b",
    r"\bcapable\b",
    r"\bprovides?\b",
    r"\benables?\b",
    r"\bused for\b",
    r"\bdesigned for\b",
    r"\btrained for\b",
    r"\btask\b",
    r"\busage\b",
    r"\bexample\b",
)

MODEL_COLUMNS = [
    "Model",
    "Developer / HF Owner",
    "Model Family",
    "Model Variant",
    "Artifact Type",
    "Availability",
    "Release Date",
    "HF Repository Created",
    "Architecture",
    "Parameters",
    "Repository Size (GB)",
    "Weights Size (GB)",
    "License",
    "Base Model",
    "Text-to-Image",
    "Image-to-Image",
    "Inpainting",
    "Image Editing",
    "Control / Conditioning",
    "HF Downloads (30d)",
    "HF Downloads (All Time)",
    "HF Likes",
    "HF Trending Score",
    "Arena ELO",
    "GenEval",
    "DPG-Bench",
    "ImageReward",
    "PickScore",
    "HPS v2",
    "Model URL",
    "Official URL",
]

# Source/provenance information is exported separately so the main model table
# stays compact and readable.
SOURCE_SHEET_NAME = "Sources"
SOURCE_LIST_SHEET_NAME = "All Sources"

CACHE_VERSION = 8

# Persistent collector checkpoint.
# Unlike the Model Card cache, this stores the already processed rows and
# auxiliary records. It is written after every successfully completed model.
CHECKPOINT_VERSION = 7
DEFAULT_CHECKPOINT = "cache/collector_checkpoint_v9.json"

# Performance controls. External web discovery is deliberately bounded because
# the previous version could execute up to 8 search queries + 3 page fetches
# per model, turning a 100-model test into a multi-hour run.
EXTERNAL_SEARCH_QUERIES = 2
EXTERNAL_MAX_SOURCES = 6
EXTERNAL_FACT_SOURCES = 2
EXTERNAL_SEARCH_TIMEOUT = 6
EXTERNAL_PAGE_TIMEOUT = 8
MAX_SMALL_HF_CONFIGS = 4
CHECKPOINT_SAVE_EVERY = 5

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    value = str(value).strip()
    return value if value else None


def normalize_key(value: str) -> str:
    """Normalize names for conservative cross-source matching."""
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def normalize_arena_key(value: Any) -> str:
    """Normalize Artificial Analysis names to Hugging Face-style model names.

    Artificial Analysis currently appends labels such as ``Open Weights`` to
    model names and uses bracketed variants, e.g. ``FLUX.1 [dev]Open Weights``
    versus the HF repository ``FLUX.1-dev``.  Both should resolve to the same
    identity without using fuzzy matching.
    """
    text = clean(value) or ""
    text = html_lib.unescape(text)
    text = re.sub(r"\s*open\s*weights\s*$", "", text, flags=re.I)
    text = re.sub(r"\s*\[[^\]]+\]", lambda m: " " + m.group(0)[1:-1] + " ", text)
    text = text.replace("/", " ")
    return normalize_key(text)


def model_identity_candidates(value: Any) -> List[str]:
    """Return deterministic normalized aliases for a model/repository name."""
    text = clean(value)
    if not text:
        return []

    values = [text]
    basename = text.rsplit("/", 1)[-1]
    if basename != text:
        values.append(basename)

    # HF repositories commonly use '-'/'_' where leaderboard names use spaces
    # or bracketed variant labels.  Generate only deterministic variants.
    for item in list(values):
        values.extend([
            item.replace("_", " "),
            item.replace("-", " "),
            re.sub(r"\[([^]]+)\]", r" \1 ", item),
        ])

    result = []
    for item in values:
        key = normalize_arena_key(item)
        if key and key not in result:
            result.append(key)
    return result


def normalize_repo_id(repo_id: str) -> str:
    return repo_id.strip().lower()


def iso_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    text = str(value)
    return text[:10] if re.match(r"^\d{4}-\d{2}-\d{2}", text) else text


def first_value(*values: Any) -> Optional[Any]:
    for value in values:
        if value is not None and clean(value) is not None:
            return value
    return None


def hf_url(repo_id: str) -> str:
    return f"https://huggingface.co/{repo_id}"


def official_url(value: Any) -> Optional[str]:
    value = clean(value)
    if not value:
        return None
    return value if value.startswith(("http://", "https://")) else None


def source_display_name(url: str, label: Optional[str] = None) -> str:
    """Return a stable human-readable source name for the source columns."""
    label_text = (clean(label) or "").strip()
    low_label = label_text.lower()
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")

    if "huggingface.co" in host:
        return "Hugging Face"
    if "artificialanalysis.ai" in host:
        return "Artificial Analysis"
    if "github.com" in host:
        return "GitHub"
    if "gitlab.com" in host:
        return "GitLab"
    if "paperswithcode.com" in host:
        return "Papers with Code"
    if "arxiv.org" in host:
        return "arXiv"
    if "civitai.com" in host:
        return "Civitai"
    if "replicate.com" in host:
        return "Replicate"
    if "modelscope.cn" in host:
        return "ModelScope"
    if "openreview.net" in host:
        return "OpenReview"
    if label_text:
        return label_text
    return host or "External source"


def split_multivalue(value: Any) -> List[str]:
    text = clean(value) or ""
    if not text or text == NOT_FOUND:
        return []
    return [item.strip() for item in text.split("\n") if item.strip()]


def add_row_source(row: Dict[str, Any], name: str, url: Any) -> None:
    """Append one deduplicated source name + URL to the row-level audit columns."""
    url_text = official_url(url)
    name_text = clean(name) or (source_display_name(url_text) if url_text else None)
    if name_text:
        names = split_multivalue(row.get("Sources"))
        if name_text not in names:
            names.append(name_text)
        row["Sources"] = "\n".join(names)
    if url_text:
        urls = split_multivalue(row.get("Source URLs"))
        if url_text not in urls:
            urls.append(url_text)
        row["Source URLs"] = "\n".join(urls)


def add_row_source_with_field(row: Dict[str, Any], field: str, name: str, url: Any) -> None:
    """Record source provenance internally and retain the compact row source list for compatibility."""
    source_name = source_display_name(official_url(url) or "", name)
    add_row_source(row, source_name, url)
    provenance = row.setdefault("_field_sources", {})
    entries = provenance.setdefault(field, [])
    url_text = official_url(url)
    if url_text and url_text not in entries:
        entries.append(url_text)


def _source_provider(url: Any, label: Any = None) -> str:
    """Stable source-provider name used for the dynamic source columns."""
    url_text = official_url(url)
    return source_display_name(url_text, clean(label)) if url_text else (clean(label) or "External source")


def _record_field_source(row: Dict[str, Any], field: str, source: Any, source_type: Any = None) -> None:
    """Attach one URL to one exported field's provenance map."""
    url_text = official_url(source)
    if not url_text:
        return
    provenance = row.setdefault("_field_sources", {})
    urls = provenance.setdefault(field, [])
    if url_text not in urls:
        urls.append(url_text)
    providers = row.setdefault("_providers", set())
    providers.add(_source_provider(url_text, source_type))


def cache_path(cache_dir: Path, repo_id: str) -> Path:
    return cache_dir / f"{repo_id.replace('/', '__')}.json"


def atomic_json_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def add_source(
    sources: List[Dict[str, Any]],
    model: str,
    field: str,
    value: Any,
    source: str,
    source_type: str,
    confidence: str,
) -> None:
    value = clean(value)
    if value is None:
        return
    sources.append({
        "Model": model,
        "Field": field,
        "Value": value,
        "Source": source,
        "Source Type": source_type,
        "Confidence": confidence,
        "Checked At": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    })


def add_capability_evidence(
    evidence: List[Dict[str, Any]],
    model: str,
    capability: str,
    value: str,
    confidence: str,
    source: str,
    source_type: str,
    evidence_text: Optional[str],
    evidence_kind: str,
) -> None:
    evidence.append({
        "Model": model,
        "Capability": capability,
        "Value": value,
        "Confidence": confidence,
        "Source": source,
        "Source Type": source_type,
        "Evidence": evidence_text,
        "Evidence Kind": evidence_kind,
        "Checked At": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    })



def load_checkpoint(path: Path) -> Dict[str, Any]:
    """Load the persistent collector state. A missing/broken checkpoint is ignored."""
    if not path.exists():
        return {
            "checkpoint_version": CHECKPOINT_VERSION,
            "processed_repo_ids": [],
            "models": {},
            "sources": [],
            "evaluations": [],
            "capabilities": [],
            "errors": [],
        }

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("checkpoint root is not an object")
        if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError("unsupported checkpoint version")

        return {
            "checkpoint_version": CHECKPOINT_VERSION,
            "processed_repo_ids": list(payload.get("processed_repo_ids") or []),
            "models": dict(payload.get("models") or {}),
            "sources": list(payload.get("sources") or []),
            "evaluations": list(payload.get("evaluations") or []),
            "capabilities": list(payload.get("capabilities") or []),
            "errors": list(payload.get("errors") or []),
        }
    except Exception as exc:
        print(f"[Checkpoint] Cannot load {path}: {exc}")
        print("[Checkpoint] Starting with an empty state; the old file is not modified.")
        return {
            "checkpoint_version": CHECKPOINT_VERSION,
            "processed_repo_ids": [],
            "models": {},
            "sources": [],
            "evaluations": [],
            "capabilities": [],
            "errors": [],
        }


def save_checkpoint(
    path: Path,
    models: Dict[str, Dict[str, Any]],
    sources: List[Dict[str, Any]],
    evaluations: List[Dict[str, Any]],
    capabilities: List[Dict[str, Any]],
    errors: List[Dict[str, Any]],
    processed_repo_ids: Iterable[str],
) -> None:
    """Atomically persist all collector progress."""
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "processed_repo_ids": sorted(
            set(normalize_repo_id(x) for x in processed_repo_ids)
        ),
        "models": models,
        "sources": sources,
        "evaluations": evaluations,
        "capabilities": capabilities,
        "errors": errors,
    }
    atomic_json_write(path, payload)


# ---------------------------------------------------------------------------
# Hugging Face Model Card handling
# ---------------------------------------------------------------------------

def metadata_dict(model_info: Any) -> Dict[str, Any]:
    card_data = getattr(model_info, "cardData", None)
    if card_data is None:
        return {}
    if isinstance(card_data, dict):
        return card_data
    if hasattr(card_data, "to_dict"):
        try:
            result = card_data.to_dict()
            return result if isinstance(result, dict) else {}
        except Exception:
            pass
    try:
        return dict(card_data)
    except Exception:
        return {}


def load_cached_card(
    repo_id: str,
    cache_dir: Path,
    max_age_days: Optional[int],
) -> Optional[Dict[str, Any]]:
    path = cache_path(cache_dir, repo_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("cache_version") != CACHE_VERSION:
            return None
        fetched_at = payload.get("fetched_at")
        if max_age_days is not None and fetched_at:
            age = datetime.now(timezone.utc) - datetime.fromisoformat(fetched_at)
            if age.total_seconds() < 0 or age.days > max_age_days:
                return None
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def fetch_model_card(
    repo_id: str,
    cache_dir: Path,
    sleep_seconds: float = 0.0,
    max_age_days: Optional[int] = None,
) -> Tuple[Dict[str, Any], str]:
    cached = load_cached_card(repo_id, cache_dir, max_age_days)
    if cached is not None:
        return cached, "cache"

    try:
        card = ModelCard.load(repo_id)
        data = getattr(card, "data", None)
        metadata = {}
        if data is not None and hasattr(data, "to_dict"):
            result = data.to_dict()
            if isinstance(result, dict):
                metadata = result

        text = clean(getattr(card, "text", None)) or ""
        content = clean(getattr(card, "content", None)) or ""

        payload = {
            "cache_version": CACHE_VERSION,
            "repo_id": repo_id,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata,
            "text": text,
            "content": content,
        }
        atomic_json_write(cache_path(cache_dir, repo_id), payload)

        if sleep_seconds:
            time.sleep(sleep_seconds)

        return payload, "network"
    except Exception as exc:
        print(f"[HF Card] {repo_id}: {exc}")
        return {}, "failed"


def get_license(model_info: Any, card_meta: Dict[str, Any]) -> Optional[str]:
    return clean(first_value(
        card_meta.get("license"),
        getattr(model_info, "license", None),
    ))


def get_pipeline(model_info: Any, card_meta: Dict[str, Any]) -> Optional[str]:
    return clean(first_value(
        card_meta.get("pipeline_tag"),
        getattr(model_info, "pipeline_tag", None),
    ))


def get_tags(model_info: Any, card_meta: Dict[str, Any]) -> List[str]:
    tags = list(getattr(model_info, "tags", None) or [])
    card_tags = card_meta.get("tags", []) or []
    if isinstance(card_tags, str):
        card_tags = [card_tags]
    tags.extend(card_tags)
    return list(dict.fromkeys(str(tag) for tag in tags))


def get_base_model(card_meta: Dict[str, Any], model_info: Any = None) -> Optional[str]:
    base = first_value(
        card_meta.get("base_model"),
        card_meta.get("base_models"),
        getattr(model_info, "base_models", None) if model_info else None,
    )
    if isinstance(base, list):
        return ", ".join(map(str, base))
    return clean(base)


def get_official_urls(card_meta: Dict[str, Any], card_text: str = "") -> List[Tuple[str, str]]:
    """Extract all reasonably strong project/source links from metadata and Model Card text."""
    found: List[Tuple[str, str]] = []
    seen: set[str] = set()

    def add(label: str, value: Any) -> None:
        url = official_url(value)
        if not url:
            return
        normalized = url.rstrip(".,)")
        if "huggingface.co" in normalized.lower():
            return
        if normalized.lower() in seen:
            return
        seen.add(normalized.lower())
        found.append((label, normalized))

    structured_keys = (
        ("Official project", "homepage"),
        ("Official project", "website"),
        ("Official project", "project_url"),
        ("Official project", "project"),
        ("Official project", "homepage_url"),
        ("Source repository", "github"),
        ("Source repository", "github_url"),
        ("Source repository", "repository"),
        ("Source repository", "repo"),
    )
    for label, key in structured_keys:
        add(label, card_meta.get(key))

    text = card_text or ""
    if not text:
        text = "\n".join(
            str(card_meta.get(key) or "")
            for key in ("text", "content", "README", "readme")
        )

    if text:
        md = re.compile(
            r"\[([^\]]{2,120})\]\((https?://[^)\s]+)\)", re.I
        )
        strong_words = (
            "official", "project", "homepage", "website", "source code",
            "github", "gitlab", "repository", "repo", "code",
        )
        excluded_domains = (
            "huggingface.co", "arxiv.org", "paperswithcode.com",
            "wandb.ai", "tensorboard.dev",
        )
        for label, url in md.findall(text):
            label_l = label.lower()
            url_l = url.lower().rstrip(".,)")
            if any(domain in url_l for domain in excluded_domains):
                continue
            if any(word in label_l for word in strong_words):
                add("Official/source link", url)

        # Also retain explicit bare GitHub/GitLab links when they appear in a
        # sentence containing a source/repository cue.
        bare = re.findall(r"https?://(?:github\.com|gitlab\.com)/[^\s<>)]+", text, re.I)
        for url in bare:
            add("Source repository", url)

    return found


def get_official_url(card_meta: Dict[str, Any], card_text: str = "") -> Optional[str]:
    urls = get_official_urls(card_meta, card_text)
    return urls[0][1] if urls else None


def get_referenced_source_links(card_text: str) -> List[Tuple[str, str]]:
    """Collect recognized ecosystem links explicitly referenced by a Model Card."""
    if not card_text:
        return []
    found: List[Tuple[str, str]] = []
    seen: set[str] = set()
    pattern = re.compile(r"(?:\[[^\]]{1,200}\]\()?((?:https?://)[^\s<>)]+)", re.I)
    for match in pattern.finditer(card_text):
        url = match.group(1).rstrip(".,);]}")
        if not _allowed_external_source(url):
            continue
        key = url.lower()
        if key in seen:
            continue
        seen.add(key)
        found.append((source_display_name(url), url))
    return found


def get_author(repo_id: str, model_info: Any, card_meta: Dict[str, Any]) -> str:
    author = first_value(
        card_meta.get("author"),
        card_meta.get("developers"),
        card_meta.get("developer"),
        getattr(model_info, "author", None),
    )
    if author:
        return str(author)
    return repo_id.split("/", 1)[0] if "/" in repo_id else NOT_FOUND


def get_numeric_bytes(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value > 0:
        return int(value)
    return None


def get_weights_size(model_info: Any) -> Optional[float]:
    safetensors = getattr(model_info, "safetensors", None)
    total = None
    if isinstance(safetensors, dict):
        total = get_numeric_bytes(safetensors.get("total"))
    else:
        total = get_numeric_bytes(getattr(safetensors, "total", None))
    return round(total / (1024 ** 3), 2) if total else None


def get_repository_size(model_info: Any) -> Optional[float]:
    used_storage = get_numeric_bytes(getattr(model_info, "used_storage", None))
    if used_storage:
        return round(used_storage / (1024 ** 3), 2)

    total_bytes = sum(
        get_numeric_bytes(getattr(file_info, "size", None)) or 0
        for file_info in (getattr(model_info, "siblings", None) or [])
    )
    return round(total_bytes / (1024 ** 3), 2) if total_bytes else None


def _config(model_info: Any) -> Dict[str, Any]:
    value = getattr(model_info, "config", None)
    if isinstance(value, dict):
        return value
    try:
        value = value.to_dict()
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _walk_dicts(value: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _first_config_value(config: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    for item in _walk_dicts(config):
        for key in keys:
            value = item.get(key)
            if value is not None and clean(value) is not None:
                return value
    return None


def get_architecture(
    model_info: Any,
    card_meta: Dict[str, Any],
    extra_configs: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Resolve architecture from HF metadata, model-index and nested Diffusers configs."""
    config_candidates = [_config(model_info)]
    config_candidates.extend(extra_configs or [])

    transformers_info = getattr(model_info, "transformers_info", None)
    values: List[Any] = [
        card_meta.get("architecture"),
        card_meta.get("architectures"),
        getattr(transformers_info, "auto_model", None) if transformers_info else None,
        getattr(transformers_info, "custom_class", None) if transformers_info else None,
    ]
    for config in config_candidates:
        if not config:
            continue
        values.extend([
            config.get("architectures"),
            config.get("architecture"),
            config.get("model_type"),
            config.get("_class_name"),
            config.get("library_name"),
        ])
        nested = _first_config_value(
            config,
            ("architectures", "architecture", "model_type", "_class_name"),
        )
        if nested is not None:
            values.append(nested)

    cleaned_values: List[str] = []
    for value in values:
        if isinstance(value, (list, tuple, set)):
            cleaned_values.extend(
                x for x in (clean(item) for item in value) if x
            )
        else:
            value_text = clean(value)
            if value_text:
                cleaned_values.append(value_text)

    # Prefer architectures/classes over generic *Pipeline wrappers.
    for value in cleaned_values:
        if not re.search(r"pipeline$", value, flags=re.I):
            return value
    return cleaned_values[0] if cleaned_values else None


def get_model_family(repo_id: str, model_info: Any, card_meta: Dict[str, Any]) -> Optional[str]:
    config = _config(model_info)
    value = first_value(
        card_meta.get("model_name"),
        card_meta.get("model_family"),
        config.get("model_type"),
        config.get("_name_or_path"),
    )
    return clean(value) or repo_id.rsplit("/", 1)[-1]


def _numeric_sum(value: Any) -> Optional[int]:
    if isinstance(value, dict):
        nums = [
            int(v) for v in value.values()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
        ]
        return sum(nums) if nums else None
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return int(value)
    return None


def _format_parameter_count(value: Any) -> Optional[str]:
    total = _numeric_sum(value)
    if total is None:
        return None
    if total >= 1_000_000_000:
        return f"{total / 1_000_000_000:.2f}B"
    if total >= 1_000_000:
        return f"{total / 1_000_000:.2f}M"
    return str(total)


def _extract_parameter_mentions(text: str) -> Optional[str]:
    if not text:
        return None
    patterns = (
        r"(?<![\w.])(\d+(?:\.\d+)?)\s*(billion|bn|b)\s*(?:trainable\s*)?(?:parameters?|params?)",
        r"(?<![\w.])(\d+(?:\.\d+)?)\s*(million|mn|m)\s*(?:trainable\s*)?(?:parameters?|params?)",
        r"(?:parameters?|params?)\s*[:=]\s*(\d+(?:\.\d+)?)\s*(billion|bn|b|million|mn|m)",
    )
    multipliers = {"b": 1_000_000_000, "bn": 1_000_000_000, "billion": 1_000_000_000,
                   "m": 1_000_000, "mn": 1_000_000, "million": 1_000_000}
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if not match:
            continue
        groups = match.groups()
        number = groups[0] if groups[0] else groups[1]
        unit = groups[1] if groups[0] else groups[2]
        if number and unit:
            try:
                total = float(number) * multipliers[unit.lower()]
                return _format_parameter_count(int(total))
            except (TypeError, ValueError, KeyError):
                continue
    return None


def get_parameters(
    model_info: Any,
    card_meta: Dict[str, Any],
    extra_configs: Optional[List[Dict[str, Any]]] = None,
    extra_text: str = "",
) -> Optional[str]:
    """Resolve parameter count from HF safetensors metadata, model config and text."""
    safetensors = getattr(model_info, "safetensors", None)
    if safetensors is not None:
        parameters = getattr(safetensors, "parameters", None)
        if parameters is None and isinstance(safetensors, dict):
            parameters = safetensors.get("parameters")
        formatted = _format_parameter_count(parameters)
        if formatted:
            return formatted

    config = _config(model_info)
    configs = [config] + (extra_configs or [])
    for item in configs:
        if not item:
            continue
        value = _first_config_value(
            item,
            ("num_parameters", "parameter_count", "n_parameters", "params", "parameters"),
        )
        formatted = _format_parameter_count(value)
        if formatted:
            return formatted
        # Some configs store a scalar string such as "1.3B".
        text_value = clean(value)
        if text_value and re.fullmatch(r"\d+(?:\.\d+)?\s*[bBmMkK]", text_value):
            return text_value.upper().replace(" ", "")

    for value in (
        card_meta.get("parameters"),
        card_meta.get("num_parameters"),
        card_meta.get("parameter_count"),
    ):
        formatted = _format_parameter_count(value)
        if formatted:
            return formatted
        text_value = clean(value)
        if text_value:
            mention = _extract_parameter_mentions(text_value)
            if mention:
                return mention

    for text in (extra_text,):
        mention = _extract_parameter_mentions(text)
        if mention:
            return mention

    return None


def load_small_hf_configs(
    repo_id: str,
    model_info: Any,
    timeout: int = 20,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Fetch only small JSON configuration files that can reveal architecture/parameters.

    The list of repository files comes from HF metadata, so no large model weights
    are downloaded. This is particularly important for Diffusers repositories whose
    architecture lives under unet/, transformer/ or model_index.json.
    """
    siblings = getattr(model_info, "siblings", None) or []
    filenames = []
    for sibling in siblings:
        name = clean(getattr(sibling, "rfilename", None) or getattr(sibling, "path", None))
        if not name:
            continue
        lower = name.lower()
        if lower.endswith(".json") and (
            lower in {
                "config.json", "model_index.json", "params.json",
                "configuration.json", "config/config.json",
            }
            or lower.endswith("/config.json")
        ):
            filenames.append(name)

    priority = {
        "model_index.json": 0,
        "config.json": 1,
        "transformer/config.json": 2,
        "unet/config.json": 3,
    }
    filenames = sorted(
        dict.fromkeys(filenames),
        key=lambda name: (priority.get(name.lower(), 10), len(name)),
    )[:MAX_SMALL_HF_CONFIGS]

    configs: List[Dict[str, Any]] = []
    paths: List[str] = []
    for filename in filenames:
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        try:
            response = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0 image-model-collector/9.0"},
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                configs.append(payload)
                paths.append(url)
        except Exception:
            continue
    return configs, paths


def fetch_hf_model_api_details(api: HfApi, repo_id: str, timeout: float = 12.0) -> Optional[Any]:
    """Fetch optional expanded HF fields through huggingface_hub itself.

    The old version manually called /api/models/{repo_id} with an ``expand`` query.
    That produced HTTP 400 responses for every repository in the user's run.
    The official client handles repository IDs and the current API contract.
    This call is only made when the ordinary ModelInfo object is missing fields
    that are actually needed.
    """
    try:
        return api.model_info(
            repo_id,
            timeout=timeout,
            # Request only fields that the normal files_metadata response may omit.
            # Smaller responses reduce both latency and server-side work.
            expand=[
                "downloadsAllTime",
                "trendingScore",
                "evalResults",
            ],
        )
    except Exception as exc:
        print(f"[HF API] Optional expanded fields unavailable for {repo_id}: {exc}")
        return None


def _extract_explicit_date(text: str) -> Optional[str]:
    """Extract a date only when it is explicitly tied to release/publication language."""
    if not text:
        return None

    normalized = re.sub(r"\s+", " ", text)

    patterns = (
        # ISO date after an explicit release/publication label.
        r"\b(?:release date|released|release|published|publication date|introduced|announced)"
        r"\s*(?:on|:|-)?\s*(\d{4}-\d{2}-\d{2})\b",
        # US-style date.
        r"\b(?:release date|released|release|published|publication date|introduced|announced)"
        r"\s*(?:on|:|-)?\s*(\d{1,2}/\d{1,2}/\d{4})\b",
        # Month + year/day.
        r"\b(?:release date|released|release|published|publication date|introduced|announced)"
        r"\s*(?:on|in|:|-)?\s*"
        r"((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"(?:\s+\d{1,2}(?:st|nd|rd|th)?)?,?\s+\d{4})\b",
        # Year-only release statements; this is intentionally less precise.
        r"\b(?:released|release|introduced|announced|published)\s+(?:in\s+)?(20\d{2})\b",
    )

    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.I)
        if match:
            return match.group(1).strip()

    return None


def get_release_date(
    model_info: Any,
    card_meta: Dict[str, Any],
    card_text: str = "",
) -> Optional[str]:
    """Resolve an explicit release/publication date without conflating it with HF creation."""
    metadata_value = first_value(
        card_meta.get("release_date"),
        card_meta.get("releaseDate"),
        card_meta.get("publication_date"),
        card_meta.get("publicationDate"),
    )
    explicit_metadata = iso_date(metadata_value) if metadata_value is not None else None
    if explicit_metadata:
        return explicit_metadata

    return _extract_explicit_date(card_text)


def classify_repo(repo_id: str, tags: List[str], card_meta: Dict[str, Any], card_text: str) -> str:
    # Classification is intentionally based first on structured signals and repo
    # name. Searching arbitrary Model Card prose for words like "checkpoint"
    # caused normal models to be misclassified in the previous version.
    repo_and_tags = (repo_id + " " + " ".join(tags)).lower()
    tags_lower = {str(t).lower() for t in tags}

    for tag, category in VARIANT_TAGS.items():
        if tag in tags_lower:
            return category

    for category, patterns in VARIANT_PATTERNS.items():
        if any(re.search(pattern, repo_and_tags) for pattern in patterns):
            return category

    base_model = get_base_model(card_meta)
    if base_model:
        if re.search(r"\b(?:fine[- ]?tune|finetuned|checkpoint)\b", repo_and_tags):
            return "Fine-tune / Checkpoint"
        return "Derived Model"

    if re.search(r"\b(?:fine[- ]?tune|finetuned|checkpoint)\b", repo_and_tags):
        return "Fine-tune / Checkpoint"

    # A small amount of card-text evidence is used only when it explicitly
    # describes the artifact type, not when the word merely occurs somewhere.
    head = (card_text or "")[:12000].lower()
    if re.search(r"(?:this|the) (?:model|repository) (?:is|contains) (?:a )?fine[- ]?tune", head):
        return "Fine-tune / Checkpoint"
    return "Base / Standalone"


def is_usable_image_model(
    repo_id: str,
    info: Any,
    tags: List[str],
    card_meta: Dict[str, Any],
    card_text: str,
    discovery_reasons: List[str],
) -> Tuple[bool, str]:
    artifact_type = classify_repo(repo_id, tags, card_meta, card_text)
    if artifact_type in EXCLUDED_COMPONENT_TYPES:
        return False, artifact_type

    pipeline = get_pipeline(info, card_meta)
    tags_lower = {t.lower() for t in tags}
    image_signal = (
        pipeline in IMAGE_PIPELINES
        or any(t in tags_lower for t in IMAGE_PIPELINES)
        or any(r.startswith("pipeline:") for r in discovery_reasons)
        or any(r.startswith("search:") for r in discovery_reasons)
    )
    return image_signal, artifact_type


# ---------------------------------------------------------------------------
# Capability evidence
# ---------------------------------------------------------------------------

def _positive_match(text: str, capability: str) -> Optional[str]:
    if not text:
        return None

    for pattern in CAPABILITY_PATTERNS[capability]:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue

        start = max(0, match.start() - 180)
        end = min(len(text), match.end() + 240)
        snippet = re.sub(r"\s+", " ", text[start:end]).strip()

        context = snippet.lower()
        if any(re.search(p, context) for p in POSITIVE_CONTEXT_PATTERNS):
            return snippet

        # A clean capability heading/list item is also useful evidence.
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        if line_end == -1:
            line_end = min(len(text), match.end() + 200)
        line = re.sub(r"\s+", " ", text[line_start:line_end]).strip()
        if len(line) <= 300:
            return line

    return None


def capability_evidence(
    repo_id: str,
    tags: List[str],
    pipeline: Optional[str],
    card_text: str,
    card_source: str,
    model_url: str,
    model_type: str,
    evidence_rows: List[Dict[str, Any]],
) -> Dict[str, str]:
    tags_lower = {str(t).lower() for t in tags}
    result: Dict[str, str] = {}

    for capability in CAPABILITIES:
        value = NOT_FOUND
        confidence = "Low"
        source = model_url
        source_type = "HF metadata/text"
        evidence_text = None
        evidence_kind = "No explicit evidence found"

        # Pipeline/tag evidence is stronger than generic prose.
        if capability == "Text-to-Image" and (
            pipeline == "text-to-image" or "text-to-image" in tags_lower
        ):
            value, confidence = "Yes", "High"
            source_type = "Hugging Face pipeline/tag"
            evidence_text = f"pipeline={pipeline}" if pipeline else "tag=text-to-image"
            evidence_kind = "Structured metadata"

        elif capability == "Image-to-Image" and (
            pipeline == "image-to-image" or "image-to-image" in tags_lower
        ):
            value, confidence = "Yes", "High"
            source_type = "Hugging Face pipeline/tag"
            evidence_text = f"pipeline={pipeline}" if pipeline else "tag=image-to-image"
            evidence_kind = "Structured metadata"

        elif capability == "Inpainting" and "inpainting" in tags_lower:
            value, confidence = "Yes", "High"
            source_type = "Hugging Face tag"
            evidence_text = "tag=inpainting"
            evidence_kind = "Structured metadata"

        elif capability == "Image Editing" and any(
            x in tags_lower for x in ("image-editing", "image-edit", "editing")
        ):
            value, confidence = "Yes", "High"
            source_type = "Hugging Face tag"
            evidence_text = "editing-related tag"
            evidence_kind = "Structured metadata"

        elif capability == "Control / Conditioning" and any(
            x in tags_lower for x in (
                "controlnet", "adapter", "ip-adapter",
                "t2i-adapter", "conditioning"
            )
        ):
            value, confidence = "Yes", "High"
            source_type = "Hugging Face tag"
            evidence_text = "control/conditioning-related tag"
            evidence_kind = "Structured metadata"

        else:
            snippet = _positive_match(card_text, capability)
            if snippet:
                value = "Yes"
                confidence = "Medium"
                source_type = "HF Model Card text"
                evidence_text = snippet
                evidence_kind = "Explicit Model Card statement"

        # Components should not automatically be presented as standalone
        # generators. Their capability is still recorded, but confidence is
        # downgraded when the artifact is clearly a component.
        if value == "Yes" and model_type in {
            "LoRA", "ControlNet", "Adapter", "Embedding", "Upscaler",
            "Quantized", "Inpainting Component",
        }:
            confidence = "Medium"

        result[capability] = value
        add_capability_evidence(
            evidence_rows,
            repo_id,
            capability,
            value,
            confidence,
            source,
            source_type,
            evidence_text,
            evidence_kind,
        )

    return result


# ---------------------------------------------------------------------------
# Evaluations
# ---------------------------------------------------------------------------

def _normalized_benchmark_name(value: Any) -> str:
    text = normalize_key(clean(value) or "")
    aliases = {
        "geneval": "GenEval",
        "gen eval": "GenEval",
        "geneval benchmark": "GenEval",
        "dpg bench": "DPG-Bench",
        "dpgbench": "DPG-Bench",
        "dpg bench benchmark": "DPG-Bench",
        "imagereward": "ImageReward",
        "image reward": "ImageReward",
        "pickscore": "PickScore",
        "pick score": "PickScore",
        "hps v2": "HPS v2",
        "hps2": "HPS v2",
        "hps 2": "HPS v2",
        "human preference score v2": "HPS v2",
    }
    if text in aliases:
        return aliases[text]
    compact = text.replace(" ", "")
    for key, name in (
        ("geneval", "GenEval"),
        ("dpgbench", "DPG-Bench"),
        ("imagereward", "ImageReward"),
        ("pickscore", "PickScore"),
        ("hpsv2", "HPS v2"),
        ("hps2", "HPS v2"),
    ):
        if key in compact:
            return name
    return ""


def _value_after_benchmark_label(text: str, benchmark: str) -> Optional[Any]:
    """Extract a score explicitly associated with a known benchmark label."""
    aliases = {
        "GenEval": r"GenEval|Gen\s*Eval",
        "DPG-Bench": r"DPG[\s-]*Bench|DPGBench",
        "ImageReward": r"ImageReward|Image\s*Reward",
        "PickScore": r"PickScore|Pick\s*Score",
        "HPS v2": r"HPS\s*v?2|Human\s*Preference\s*Score\s*v?2",
    }
    pattern = aliases[benchmark]

    for match in re.finditer(pattern, text, flags=re.I):
        segment = text[match.end():match.end() + 240]
        segment = re.sub(r"\s+", " ", segment)

        # Strongest form: label followed by ":" or "=" and then a number.
        explicit = re.search(
            r"^\s*(?:[:=|\-]\s*)?(?:score|value|metric)?\s*[:=]?\s*"
            r"(-?\d+(?:\.\d+)?%?)",
            segment,
            flags=re.I,
        )
        if explicit:
            token = explicit.group(1)
        else:
            # Common markdown/table form: "GenEval | 0.55" / "GenEval 0.55".
            nearby = segment[:100]
            number_match = re.search(
                r"(?<![\w.])(-?(?:0\.\d+|1(?:\.0+)?|\d+(?:\.\d+)?%))(?![\w.])",
                nearby,
            )
            if not number_match:
                continue
            token = number_match.group(1)

        try:
            if token.endswith("%"):
                return token
            value = float(token)
        except ValueError:
            continue

        # Reject obvious years, IDs and ranking counts.
        if 1900 <= value <= 2100:
            continue
        return value

    return None


def _text_benchmark_values(
    card_text: str,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    values = {name: None for name in COMMON_BENCHMARKS}
    if not card_text:
        return values

    search_text = card_text
    if model_name:
        model_variants = [
            clean(model_name) or "",
            (clean(model_name) or "").rsplit("/", 1)[-1],
        ]
        windows: List[str] = []
        lowered = card_text.lower()
        for variant in model_variants:
            if not variant:
                continue
            start = 0
            variant_lower = variant.lower()
            while True:
                pos = lowered.find(variant_lower, start)
                if pos == -1:
                    break
                windows.append(card_text[max(0, pos - 500): min(len(card_text), pos + len(variant) + 800)])
                start = pos + len(variant_lower)
        if not windows:
            return values
        search_text = "\n".join(windows)

    for benchmark in COMMON_BENCHMARKS:
        values[benchmark] = _value_after_benchmark_label(search_text, benchmark)
    return values


def extract_common_benchmarks(
    card_meta: Dict[str, Any],
    card_text: str = "",
    model_info: Any = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract benchmark scores from structured HF API data first, then Model Card text."""
    values: Dict[str, Any] = {name: None for name in COMMON_BENCHMARKS}

    raw_entries: List[Any] = []

    # Newer Hugging Face Hub API exposes parsed evaluation results directly.
    api_eval_results = getattr(model_info, "eval_results", None) if model_info else None
    if api_eval_results:
        raw_entries.extend(api_eval_results if isinstance(api_eval_results, list) else [api_eval_results])

    # ModelInfo may also expose model_index directly.
    model_index_api = getattr(model_info, "model_index", None) if model_info else None
    if model_index_api:
        raw_entries.extend(model_index_api if isinstance(model_index_api, list) else [model_index_api])

    model_index = card_meta.get("model-index", [])
    if isinstance(model_index, dict):
        model_index = [model_index]
    if isinstance(model_index, list):
        raw_entries.extend(model_index)

    def entry_to_mapping(entry: Any) -> Dict[str, Any]:
        if isinstance(entry, dict):
            return entry
        try:
            data = entry.to_dict()
            return data if isinstance(data, dict) else {}
        except Exception:
            return {
                "task_type": getattr(entry, "task_type", None),
                "dataset_type": getattr(entry, "dataset_type", None),
                "dataset_name": getattr(entry, "dataset_name", None),
                "metric_type": getattr(entry, "metric_type", None),
                "metric_value": getattr(entry, "metric_value", None),
            }

    def consume_result(result: Dict[str, Any]) -> None:
        if not isinstance(result, dict):
            return

        dataset = result.get("dataset") or {}
        task = result.get("task") or {}
        metric_candidates = result.get("metrics") or []
        if isinstance(metric_candidates, dict):
            metric_candidates = [metric_candidates]
        if not isinstance(metric_candidates, list):
            metric_candidates = []

        benchmark = next(
            (
                _normalized_benchmark_name(value)
                for value in (
                    result.get("dataset_name"),
                    result.get("dataset_type"),
                    dataset.get("name") if isinstance(dataset, dict) else None,
                    dataset.get("type") if isinstance(dataset, dict) else None,
                    task.get("name") if isinstance(task, dict) else None,
                    task.get("type") if isinstance(task, dict) else None,
                    result.get("name"),
                )
                if value
            ),
            "",
        )
        if not benchmark or values[benchmark] is not None:
            return

        # Direct API EvalResult shape.
        direct_value = result.get("metric_value")
        if direct_value is not None:
            if isinstance(direct_value, (int, float)) and not isinstance(direct_value, bool):
                values[benchmark] = direct_value
                return
            text_value = clean(direct_value)
            if text_value:
                values[benchmark] = text_value
                return

        # Standard model-index metrics shape.
        for metric in metric_candidates:
            if not isinstance(metric, dict):
                try:
                    metric = metric.to_dict()
                except Exception:
                    continue
            value = metric.get("value") if isinstance(metric, dict) else None
            if value is None and isinstance(metric, dict):
                value = metric.get("metric_value")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values[benchmark] = value
                return
            text_value = clean(value)
            if text_value:
                values[benchmark] = text_value
                return

    for raw in raw_entries:
        entry = entry_to_mapping(raw)
        if not entry:
            continue

        nested_results = entry.get("results")
        if nested_results is not None:
            if isinstance(nested_results, dict):
                nested_results = [nested_results]
            if isinstance(nested_results, list):
                for result in nested_results:
                    consume_result(entry_to_mapping(result))
        else:
            consume_result(entry)

    text_values = _text_benchmark_values(card_text, model_name=model_name)
    for benchmark in COMMON_BENCHMARKS:
        if values[benchmark] is None and text_values[benchmark] is not None:
            values[benchmark] = text_values[benchmark]
    return values


def extract_eval_results(
    card_meta: Dict[str, Any],
    model_info: Any = None,
) -> List[Dict[str, Any]]:
    """Extract all structured evaluation results for checkpoint auditing."""
    results: List[Dict[str, Any]] = []

    def mapping(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        try:
            converted = value.to_dict()
            return converted if isinstance(converted, dict) else {}
        except Exception:
            return {
                "task_type": getattr(value, "task_type", None),
                "dataset_type": getattr(value, "dataset_type", None),
                "dataset_name": getattr(value, "dataset_name", None),
                "metric_type": getattr(value, "metric_type", None),
                "metric_value": getattr(value, "metric_value", None),
            }

    api_results = getattr(model_info, "eval_results", None) if model_info else None
    for raw in (api_results or []):
        item = mapping(raw)
        if item:
            results.append({
                "Model Name": clean(item.get("model_name")),
                "Task": clean(item.get("task_name") or item.get("task_type")),
                "Dataset": clean(item.get("dataset_name") or item.get("dataset_type")),
                "Metrics": [{
                    "type": clean(item.get("metric_type")),
                    "value": item.get("metric_value"),
                }],
                "Source": clean(item.get("source")),
            })

    model_index = card_meta.get("model-index", [])
    if isinstance(model_index, dict):
        model_index = [model_index]
    if not isinstance(model_index, list):
        return results

    for model_entry in model_index:
        if not isinstance(model_entry, dict):
            continue
        model_name = clean(model_entry.get("name"))
        model_results = model_entry.get("results", []) or []
        if isinstance(model_results, dict):
            model_results = [model_results]
        if not isinstance(model_results, list):
            continue

        for raw_result in model_results:
            result = mapping(raw_result)
            if not result:
                continue
            dataset = result.get("dataset") or {}
            task = result.get("task") or {}
            metrics = result.get("metrics") or []
            if isinstance(metrics, dict):
                metrics = [metrics]
            if not isinstance(metrics, list):
                metrics = []

            results.append({
                "Model Name": model_name,
                "Task": clean(
                    result.get("task_name")
                    or result.get("task_type")
                    or (task.get("name") if isinstance(task, dict) else None)
                    or (task.get("type") if isinstance(task, dict) else None)
                ),
                "Dataset": clean(
                    result.get("dataset_name")
                    or result.get("dataset_type")
                    or (dataset.get("name") if isinstance(dataset, dict) else None)
                    or (dataset.get("type") if isinstance(dataset, dict) else None)
                ),
                "Metrics": metrics,
                "Source": clean(result.get("source")),
            })

    return results


# ---------------------------------------------------------------------------
# External source discovery
# ---------------------------------------------------------------------------

FACT_SOURCE_DOMAINS = {
    "huggingface.co", "github.com", "gitlab.com", "paperswithcode.com",
    "arxiv.org", "openreview.net", "civitai.com", "replicate.com",
    "modelscope.cn", "artificialanalysis.ai",
}

SOURCE_DOMAIN_LABELS = {
    "github.com": "GitHub",
    "gitlab.com": "GitLab",
    "paperswithcode.com": "Papers with Code",
    "arxiv.org": "arXiv",
    "openreview.net": "OpenReview",
    "civitai.com": "Civitai",
    "replicate.com": "Replicate",
    "modelscope.cn": "ModelScope",
    "artificialanalysis.ai": "Artificial Analysis",
    "huggingface.co": "Hugging Face",
}

def _source_domain(url: str) -> str:
    return (urlparse(url).hostname or "").lower().removeprefix("www.")


def _fact_source_allowed(url: str) -> bool:
    host = _source_domain(url)
    return any(host == domain or host.endswith("." + domain) for domain in FACT_SOURCE_DOMAINS)


def _allowed_external_source(url: str) -> bool:
    host = _source_domain(url)
    blocked = {
        "google.com", "bing.com", "duckduckgo.com", "yahoo.com",
        "facebook.com", "instagram.com", "x.com", "twitter.com",
        "linkedin.com",
    }
    if not host or host in blocked or any(host.endswith("." + d) for d in blocked):
        return False
    return True


class _SearchResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: List[Tuple[str, str, str]] = []
        self._in_result = False
        self._href: Optional[str] = None
        self._title_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag.lower() != "a":
            return
        attrs_dict = dict(attrs)
        classes = (attrs_dict.get("class") or "").split()
        if "result__a" in classes:
            self._in_result = True
            self._href = attrs_dict.get("href")
            self._title_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_result:
            self._title_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or not self._in_result:
            return
        title = re.sub(r"\s+", " ", "".join(self._title_parts)).strip()
        if self._href and title:
            self.results.append((title, self._href, title))
        self._in_result = False
        self._href = None
        self._title_parts = []


def _extract_search_results(html: str) -> List[Tuple[str, str, str]]:
    parser = _SearchResultParser()
    try:
        parser.feed(html)
    except Exception:
        return []
    return parser.results


def _extract_bing_results(html: str) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []
    # Bing search results are typically in <li class="b_algo"><h2><a href="...">.
    pattern = re.compile(
        r'<li[^>]+class=["\'][^"\']*b_algo[^"\']*["\'][^>]*>.*?'
        r'<h2[^>]*>\s*<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
        flags=re.I | re.S,
    )
    for match in pattern.finditer(html):
        url = html_lib.unescape(match.group(1))
        title = _strip_html(match.group(2))
        if url.startswith("http") and title:
            results.append((title, url, title))
    return results


def web_search_sources(query: str, timeout: int = 15, limit: int = 10) -> List[Tuple[str, str]]:
    """High-recall source discovery. Search results are sources only, not facts."""
    result_candidates: List[Tuple[str, str, str]] = []
    try:
        response = requests.get(
            "https://html.duckduckgo.com/html/?q=" + quote_plus(query),
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 image-model-collector/9.0"},
        )
        response.raise_for_status()
        result_candidates = _extract_search_results(response.text)
    except Exception:
        pass

    # Fallback for environments where DuckDuckGo blocks automated HTML queries.
    if not result_candidates:
        try:
            response = requests.get(
                "https://www.bing.com/search?q=" + quote_plus(query),
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0 image-model-collector/9.0"},
            )
            response.raise_for_status()
            result_candidates = _extract_bing_results(response.text)
        except Exception:
            result_candidates = []

    found: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for title, url, _ in result_candidates:
        if not _allowed_external_source(url):
            continue
        normalized = url.rstrip(".,)")
        if normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        found.append((title, normalized))
        if len(found) >= limit:
            break
    return found


def discover_civitai_source(
    model_name: str,
    timeout: int = 15,
) -> List[Tuple[str, str]]:
    """Query Civitai's public model API and keep only close name matches."""
    query = clean(model_name)
    if not query:
        return []
    try:
        response = requests.get(
            "https://civitai.com/api/v1/models",
            params={"query": query, "limit": 10},
            timeout=timeout,
            headers={"User-Agent": "image-model-collector/7.0"},
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    requested_keys = set(model_identity_candidates(query))
    results: List[Tuple[str, str]] = []
    for item in payload.get("items", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        name = clean(item.get("name"))
        if not name:
            continue
        candidate_keys = set(model_identity_candidates(name))
        if requested_keys.isdisjoint(candidate_keys):
            # Also accept exact basename word overlap for names such as
            # "Stable Diffusion XL Base 1.0".
            rq = normalize_key(query).split()
            nk = normalize_key(name)
            if not rq or not all(word in nk for word in rq if len(word) > 1):
                continue
        model_id = item.get("id")
        if model_id:
            results.append((f"Civitai [{name}]", f"https://civitai.com/models/{model_id}"))
    return results[:3]


def discover_external_sources(
    repo_id: str,
    model_family: Optional[str],
    card_text: str,
    timeout: int = 15,
    max_sources: int = 8,
) -> List[Tuple[str, str]]:
    """Discover likely secondary sources for each model.

    Source discovery is deliberately broad but relevance-filtered. The search
    engine finds links; a link is not treated as a factual assertion until its
    page is fetched and parsed.
    """
    basename = repo_id.rsplit("/", 1)[-1]
    identity_tokens = {
        token for token in normalize_key(repo_id + " " + basename).split()
        if len(token) >= 3
    }
    if model_family:
        identity_tokens.update(
            token for token in normalize_key(model_family).split()
            if len(token) >= 3
        )

    # Two broad queries replace the previous ten per model. The second query
    # explicitly asks for benchmark evidence while still allowing repositories
    # and official/project pages to appear. This preserves useful recall without
    # making the collector wait through multiple 15-second search timeouts.
    benchmark_terms = " OR ".join(COMMON_BENCHMARKS)
    queries = [
        f'"{repo_id}"',
        f'"{basename}" ({benchmark_terms})',
    ]

    found: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for query in queries[:EXTERNAL_SEARCH_QUERIES]:
        for title, url in web_search_sources(query, timeout=timeout, limit=max_sources * 2):
            haystack = normalize_key(title + " " + url)
            token_hits = sum(
                1 for token in identity_tokens
                if token in haystack
            )
            # One distinctive token is enough for a model-specific URL; generic
            # benchmark landing pages need two.
            host = _source_domain(url)
            benchmark_page = any(
                word in haystack
                for word in ("geneval", "dpg", "imagereward", "pickscore", "hps")
            )
            if token_hits < 1 and not benchmark_page:
                continue
            if benchmark_page and token_hits < 1 and host not in {
                "paperswithcode.com", "github.com", "arxiv.org", "openreview.net"
            }:
                continue

            key = url.lower()
            if key in seen:
                continue
            seen.add(key)
            found.append((source_display_name(url), url))
            if len(found) >= max_sources:
                return found

    for name, url in discover_civitai_source(basename, timeout=timeout):
        if url.lower() not in seen and len(found) < max_sources:
            seen.add(url.lower())
            found.append((name, url))
    return found


def fetch_text_page(url: str, timeout: int = 15, max_chars: int = 200_000) -> str:
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 image-model-collector/9.0"},
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "text" not in content_type and "json" not in content_type:
            return ""
        return _strip_html(response.text)[:max_chars]
    except Exception:
        return ""


def extract_external_facts(
    source_url: str,
    source_name: str,
    model_name: Optional[str] = None,
    timeout: int = 15,
) -> Dict[str, Any]:
    """Extract only explicit, source-local facts from lightweight public pages."""
    text = fetch_text_page(source_url, timeout=timeout)
    if not text:
        return {}

    facts: Dict[str, Any] = {}
    release_date = _extract_explicit_date(text)
    if release_date:
        model_base = (clean(model_name) or "").rsplit("/", 1)[-1].lower()
        if not model_base or model_base in text.lower():
            facts["Release Date"] = release_date

    parameters = _extract_parameter_mentions(text)
    if parameters:
        facts["Parameters"] = parameters

    # Common architecture/class names in image-generation stacks.
    arch_patterns = (
        r"\b(?:Flux|FLUX)\b",
        r"\bDiT\b",
        r"\bMMDiT\b",
        r"\bU[- ]?Net\b",
        r"\bUNet\b",
        r"\bTransformer\b",
        r"\bLatent Diffusion\b",
        r"\bStable Diffusion\b",
        r"\bSDXL\b",
    )
    for pattern in arch_patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            facts["Architecture"] = match.group(0)
            break

    facts.update({k: v for k, v in _text_benchmark_values(text, model_name=model_name).items() if v is not None})
    return facts

# ---------------------------------------------------------------------------
# Hugging Face discovery / collection
# ---------------------------------------------------------------------------

def discover_huggingface(
    api: HfApi,
    target_new_models: Optional[int],
    processed_repo_ids: set[str],
    include_variants: bool,
) -> Dict[str, Any]:
    """Discover enough candidates to obtain target_new_models usable models.

    The old implementation stopped after N repositories and only then filtered
    variants. This implementation keeps searching until N *accepted* models
    are found (or the available discovery sources are exhausted).
    """
    discovered: Dict[str, Any] = {}
    accepted = 0

    def add(info: Any, reason: str) -> None:
        nonlocal accepted
        repo_id = getattr(info, "id", None)
        if not repo_id:
            return
        key = normalize_repo_id(repo_id)
        if key in processed_repo_ids:
            return
        if key not in discovered:
            discovered[key] = {"info": info, "discovery_reasons": [reason]}
        elif reason not in discovered[key]["discovery_reasons"]:
            discovered[key]["discovery_reasons"].append(reason)

    def scan(iterator: Any, reason: str) -> bool:
        nonlocal accepted
        for info in iterator:
            repo_id = getattr(info, "id", None)
            if not repo_id:
                continue
            key = normalize_repo_id(repo_id)
            if key in processed_repo_ids or key in discovered:
                continue

            meta = metadata_dict(info)
            tags = get_tags(info, meta)
            artifact = classify_repo(repo_id, tags, meta, "")
            if not include_variants and artifact in EXCLUDED_COMPONENT_TYPES:
                continue

            pipeline = get_pipeline(info, meta)
            tags_lower = {t.lower() for t in tags}
            image_signal = (
                pipeline in IMAGE_PIPELINES
                or any(t in tags_lower for t in IMAGE_PIPELINES)
                or reason.startswith("pipeline:")
                or reason.startswith("search:")
            )
            if not image_signal:
                continue

            discovered[key] = {
                "info": info,
                "discovery_reasons": [reason],
            }
            accepted += 1
            print(f"[HF] Accepted {accepted}/{target_new_models}: {repo_id} ({artifact})")
            if target_new_models is not None and accepted >= target_new_models:
                return True
        return False


    for pipeline in IMAGE_PIPELINES:
        if target_new_models is not None and accepted >= target_new_models:
            break
        print(f"[HF] Discovering pipeline: {pipeline}")
        try:
            iterator = api.list_models(
                filter=pipeline,
                full=True,
                cardData=True,
                limit=DISCOVERY_BATCH_SIZE,
                sort="downloads",
            )
            if scan(iterator, f"pipeline:{pipeline}"):
                break
        except Exception as exc:
            print(f"[HF] Pipeline discovery failed ({pipeline}): {exc}")

    if target_new_models is None or accepted < target_new_models:
        for term in DISCOVERY_SEARCH_TERMS:
            if target_new_models is not None and accepted >= target_new_models:
                break
            print(f"[HF] Discovering search: {term}")
            try:
                iterator = api.list_models(
                    search=term,
                    full=True,
                    cardData=True,
                    limit=DISCOVERY_BATCH_SIZE,
                    sort="downloads",
                )
                if scan(iterator, f"search:{term}"):
                    break
            except Exception as exc:
                print(f"[HF] Search discovery failed ({term}): {exc}")

    return discovered



def collect_huggingface(
    mode: str,
    limit: Optional[int],
    include_variants: bool,
    cache_dir: Path,
    cache_max_age_days: Optional[int],
    sleep_seconds: float,
    checkpoint_path: Path,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    api = HfApi()

    # Restore persistent state BEFORE discovery. --limit means the number of
    # NEW usable models wanted in this run, not the number of raw candidates.
    checkpoint = load_checkpoint(checkpoint_path)
    models: Dict[str, Dict[str, Any]] = checkpoint["models"]
    sources: List[Dict[str, Any]] = checkpoint["sources"]
    evaluations: List[Dict[str, Any]] = checkpoint["evaluations"]
    capability_rows: List[Dict[str, Any]] = checkpoint["capabilities"]
    errors: List[Dict[str, Any]] = checkpoint["errors"]
    processed_repo_ids = set(
        normalize_repo_id(x) for x in checkpoint["processed_repo_ids"]
    )

    discovered = discover_huggingface(api, limit, processed_repo_ids, include_variants)

    print(f"[HF] Candidates discovered for this run: {len(discovered)}")
    print(f"[Checkpoint] Previously processed: {len(processed_repo_ids)}")

    try:
        for index, (key, item) in enumerate(discovered.items(), start=1):
            repo_id = getattr(item["info"], "id", None)
            if not repo_id:
                continue

            normalized_repo_id = normalize_repo_id(repo_id)

            if normalized_repo_id in processed_repo_ids:
                print(f"[HF] SKIP already checkpointed: {repo_id}")
                continue

            info = item["info"]

            try:
                card_payload: Dict[str, Any] = {}
                card_source = None

                if mode == "full":
                    card_payload, card_source = fetch_model_card(
                        repo_id,
                        cache_dir,
                        sleep_seconds,
                        cache_max_age_days,
                    )

                card_meta = metadata_dict(info)
                if card_payload.get("metadata"):
                    card_meta = {
                        **card_meta,
                        **card_payload["metadata"],
                    }

                # list_models intentionally returns only a subset of ModelInfo.
                # Fetch the per-repository detail object to obtain fields that
                # were empty in v3/v4-test: downloadsAllTime, usedStorage,
                # file sizes, config/architecture and safetensors information.
                try:
                    detailed_info = api.model_info(repo_id, files_metadata=True, timeout=15)
                    info = detailed_info
                    detailed_meta = metadata_dict(info)
                    if detailed_meta:
                        card_meta = {**card_meta, **detailed_meta}
                except Exception as exc:
                    print(f"[HF] Detail metadata unavailable for {repo_id}: {exc}")

                # Diffusers repositories often keep architecture in nested configs
                # such as transformer/config.json or unet/config.json. These are
                # tiny JSON files; large weight files are never downloaded.

                card_text = (
                    card_payload.get("text")
                    or card_payload.get("content")
                    or ""
                )

                extra_configs: List[Dict[str, Any]] = []
                extra_config_urls: List[str] = []
                # Only download small repository config files when the normal
                # ModelInfo response did not already provide both core fields.
                # The old version fetched up to 12 JSON files for every model.
                initial_architecture = get_architecture(info, card_meta)
                initial_parameters = get_parameters(info, card_meta, extra_text=card_text)
                if initial_architecture is None or initial_parameters is None:
                    try:
                        extra_configs, extra_config_urls = load_small_hf_configs(repo_id, info)
                    except Exception as exc:
                        print(f"[HF] Extra config discovery unavailable for {repo_id}: {exc}")
                tags = get_tags(info, card_meta)
                model_type = classify_repo(repo_id, tags, card_meta, card_text)

                # Discovery should already filter these, but keep a second
                # validation here so a malformed/changed API response cannot
                # put a component into the 100-model dataset.
                if not include_variants and model_type in EXCLUDED_COMPONENT_TYPES:
                    print(f"[HF] REJECT component: {repo_id} ({model_type})")
                    continue

                pipeline = get_pipeline(info, card_meta)
                pipeline_display = pipeline or ", ".join(
                    r.split(":", 1)[1]
                    for r in item["discovery_reasons"]
                    if r.startswith("pipeline:")
                )

                license_value = get_license(info, card_meta)
                base_model = get_base_model(card_meta, info)
                repo_size = get_repository_size(info)
                weights_size = get_weights_size(info)
                parameters = get_parameters(
                    info, card_meta, extra_configs=extra_configs, extra_text=card_text
                )
                architecture = get_architecture(
                    info, card_meta, extra_configs=extra_configs
                )
                model_family = get_model_family(repo_id, info, card_meta)
                release_date = get_release_date(info, card_meta, card_text)

                # Most values are already present on the detailed ModelInfo.
                # Ask HF for expanded fields only when something we need is missing.
                downloads_30d = getattr(info, "downloads", None)
                downloads_all_time = getattr(info, "downloads_all_time", None)
                likes = getattr(info, "likes", None)
                trending = getattr(info, "trending_score", None)
                needs_expanded = (
                    downloads_all_time is None
                    or trending is None
                    or not getattr(info, "eval_results", None)
                    or any(row_value in (None, NOT_FOUND) for row_value in (
                        extract_common_benchmarks(card_meta, card_text, info, model_name=repo_id).values()
                    ))
                )
                expanded_info = None
                if needs_expanded:
                    expanded_info = fetch_hf_model_api_details(api, repo_id)
                    if expanded_info is not None:
                        downloads_30d = first_value(downloads_30d, getattr(expanded_info, "downloads", None))
                        downloads_all_time = first_value(downloads_all_time, getattr(expanded_info, "downloads_all_time", None))
                        likes = first_value(likes, getattr(expanded_info, "likes", None))
                        trending = first_value(trending, getattr(expanded_info, "trending_score", None))
                        expanded_meta = metadata_dict(expanded_info)
                        if expanded_meta:
                            card_meta = {**card_meta, **expanded_meta}

                model_url = hf_url(repo_id)

                capabilities = capability_evidence(
                    repo_id,
                    tags,
                    pipeline,
                    card_text,
                    card_source or model_url,
                    model_url,
                    model_type,
                    capability_rows,
                )

                official_page = get_official_url(card_meta, card_text)

                benchmark_info = expanded_info if expanded_info is not None else info
                benchmark_values = extract_common_benchmarks(
                    card_meta, card_text, benchmark_info, model_name=repo_id
                )

                row: Dict[str, Any] = {
                    "Model": repo_id,
                    "Developer / HF Owner": get_author(repo_id, info, card_meta),
                    "Model Family": model_family,
                    "Model Variant": clean(card_meta.get("variant")) or model_type,
                    "Artifact Type": model_type,
                    "Availability": "Open Weights",
                    "Release Date": release_date,
                    "HF Repository Created": iso_date(getattr(info, "created_at", None)),
                    "Architecture": architecture,
                    "Parameters": parameters,
                    "Repository Size (GB)": repo_size,
                    "Weights Size (GB)": weights_size,
                    "License": license_value,
                    "Base Model": base_model,
                    **capabilities,
                    "HF Downloads (30d)": downloads_30d,
                    "HF Downloads (All Time)": downloads_all_time,
                    "HF Likes": likes,
                    "HF Trending Score": trending,
                    "Arena ELO": ELO_NOT_FOUND,
                    **benchmark_values,
                    "Model URL": model_url,
                    "Official URL": official_page,
                    "Sources": "Hugging Face",
                    "Source URLs": model_url,
                }

                # Record the HF model page as provenance for every characteristic
                # whose value came from HF metadata/model-card text.
                hf_sourced_fields = (
                    "Model Family",
                    "Model Variant",
                    "Artifact Type",
                    "Release Date",
                    "Architecture",
                    "Parameters",
                    "Base Model",
                    *CAPABILITIES,
                    *COMMON_BENCHMARKS,
                )
                for field in hf_sourced_fields:
                    value = row.get(field)
                    if value not in (None, "", NOT_FOUND):
                        add_source(
                            sources,
                            repo_id,
                            field,
                            value,
                            model_url,
                            "Hugging Face Model Card / metadata",
                            "High" if field not in COMMON_BENCHMARKS else "Medium",
                        )

                # Keep every explicitly discovered official/project link in the
                # row-level audit trail, while Official URL remains the primary one.
                for official_name, official_link in get_official_urls(card_meta, card_text):
                    add_row_source(row, official_name, official_link)
                for ref_name, ref_link in get_referenced_source_links(card_text):
                    add_row_source(row, ref_name, ref_link)
                for config_url in extra_config_urls:
                    add_row_source(row, "Hugging Face repository config", config_url)

                add_row_source(row, "Hugging Face", model_url)
                for official_name, official_link in get_official_urls(card_meta, card_text):
                    add_row_source(row, official_name, official_link)

                for field in (
                    "Model Family", "Model Variant", "Artifact Type",
                    "Release Date", "Architecture", "Parameters", "Base Model",
                ):
                    value = row[field]
                    if value is not None:
                        add_source(
                            sources,
                            repo_id,
                            field,
                            value,
                            model_url,
                            "Hugging Face Model Card / metadata",
                            "High",
                        )

                for field, value, label in (
                    ("License", license_value, "Hugging Face metadata"),
                    ("Repository Size (GB)", repo_size, "Hugging Face repository storage"),
                    ("Weights Size (GB)", weights_size, "Hugging Face safetensors metadata"),
                    ("HF Downloads (30d)", downloads_30d, "Hugging Face API"),
                    ("HF Downloads (All Time)", downloads_all_time, "Hugging Face API"),
                    ("HF Likes", likes, "Hugging Face API"),
                    ("HF Trending Score", trending, "Hugging Face API"),
                ):
                    if value is not None:
                        add_source(
                            sources, repo_id, field, value, model_url, label, "High"
                        )

                # Search for secondary sources. These links are added to the audit
                # list; only source-local explicit facts are allowed to enrich fields.
                try:
                    needs_external = (
                        not official_page
                        or row.get("Architecture") in (None, NOT_FOUND)
                        or row.get("Parameters") in (None, NOT_FOUND)
                        or any(row.get(b) in (None, NOT_FOUND) for b in COMMON_BENCHMARKS)
                    )
                    external_sources = []
                    if needs_external:
                        external_sources = discover_external_sources(
                            repo_id,
                            model_family,
                            card_text,
                            timeout=EXTERNAL_SEARCH_TIMEOUT,
                            max_sources=EXTERNAL_MAX_SOURCES,
                        )
                    # Only a small number of source pages are fetched for facts.
                    # Search discovery itself is cheap compared with page fetches.
                    fact_sources = [
                        item for item in external_sources
                        if _fact_source_allowed(item[1])
                    ][:EXTERNAL_FACT_SOURCES]

                    # Fetch the small number of selected evidence pages concurrently.
                    # This removes serial 8-second waits when both pages are slow.
                    fetched_facts: List[Tuple[str, str, Dict[str, Any]]] = []
                    if fact_sources:
                        with ThreadPoolExecutor(max_workers=min(2, len(fact_sources))) as executor:
                            futures = {
                                executor.submit(
                                    extract_external_facts,
                                    ext_url,
                                    ext_name,
                                    repo_id,
                                    EXTERNAL_PAGE_TIMEOUT,
                                ): (ext_name, ext_url)
                                for ext_name, ext_url in fact_sources
                            }
                            for future in as_completed(futures):
                                ext_name, ext_url = futures[future]
                                try:
                                    fetched_facts.append((ext_name, ext_url, future.result()))
                                except Exception as exc:
                                    print(f"[External source] {repo_id}: {ext_url}: {exc}")

                    for ext_name, ext_url, ext_facts in fetched_facts:
                        # Keep every independently observed fact in the detailed
                        # evidence store so conflicting sources can be audited.
                        for fact_field in ("Release Date", "Architecture", "Parameters", *COMMON_BENCHMARKS):
                            fact_value = ext_facts.get(fact_field)
                            if fact_value is None:
                                continue
                            add_source(
                                sources, repo_id, fact_field, fact_value,
                                ext_url, ext_name,
                                "Medium" if fact_field in {"Release Date", "Architecture", "Parameters"} else "Low"
                            )
                            add_row_source(row, ext_name, ext_url)

                            # Fill only if the primary HF/structured value is missing.
                            if row.get(fact_field) in (None, NOT_FOUND):
                                row[fact_field] = fact_value
                except Exception as exc:
                    print(f"[External sources] {repo_id}: {exc}")

                # Re-run these fallbacks because an external source may have filled
                # one of the previously missing core values.
                for field in ("Architecture", "Parameters"):
                    if row.get(field) is None:
                        row[field] = NOT_FOUND

                for benchmark in COMMON_BENCHMARKS:
                    if row.get(benchmark) is None:
                        row[benchmark] = NOT_FOUND

                for ev in extract_eval_results(card_meta, benchmark_info):
                    ev["Repository"] = repo_id
                    evaluations.append(ev)

                # COMMIT POINT:
                # The model is added to the checkpoint only after its complete
                # row, sources, evaluations and capability evidence exist.
                models[key] = row
                processed_repo_ids.add(normalized_repo_id)

                if len(processed_repo_ids) % CHECKPOINT_SAVE_EVERY == 0:
                    save_checkpoint(
                        checkpoint_path,
                        models,
                        sources,
                        evaluations,
                        capability_rows,
                        errors,
                        processed_repo_ids,
                    )
                    print(
                        f"[HF] Checkpoint saved: {len(processed_repo_ids)} processed | "
                        f"{repo_id}"
                    )
                else:
                    print(f"[HF] Processed: {len(processed_repo_ids)} | {repo_id}")

            except Exception as exc:
                errors.append({
                    "Stage": "HF model processing",
                    "Model": repo_id,
                    "Error": repr(exc),
                })
                print(f"[HF] Failed {repo_id}: {exc}")

                # Failed repositories are NOT marked as processed, so a later
                # run can retry them.
                save_checkpoint(
                    checkpoint_path,
                    models,
                    sources,
                    evaluations,
                    capability_rows,
                    errors,
                    processed_repo_ids,
                )

            if index % 25 == 0:
                print(f"[HF] Processed discovery entries {index}/{len(discovered)}")

    except KeyboardInterrupt:
        # The last completed model is already on disk. The model being
        # processed at the exact moment of Ctrl+C is not committed unless its
        # complete row had already been assembled.
        print("\n[HF] Ctrl+C received. Saving the last completed checkpoint...")
        save_checkpoint(
            checkpoint_path,
            models,
            sources,
            evaluations,
            capability_rows,
            errors,
            processed_repo_ids,
        )
        print("[HF] Checkpoint saved safely. Current run can be resumed.")

    return (
        list(models.values()),
        sources,
        evaluations,
        capability_rows,
        errors,
    )




# ---------------------------------------------------------------------------
# Official models
# ---------------------------------------------------------------------------

def load_official_models(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        print(f"[Official] File not found: {path} — no standalone official-model catalog loaded.")
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[Official] Cannot read {path}: {exc}")
        return []

    if isinstance(payload, dict):
        payload = payload.get("models", [])

    rows = (
        [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, list)
        else []
    )
    print(f"[Official] Loaded {len(rows)} catalog models from {path}")
    return rows


def official_aliases(item: Dict[str, Any]) -> List[str]:
    aliases = item.get("Aliases") or item.get("aliases") or []
    if isinstance(aliases, str):
        aliases = [aliases]

    values = []
    for value in aliases:
        value = clean(value)
        if value:
            values.append(value)

    model = clean(item.get("Model") or item.get("model"))
    if model:
        values.append(model)

    return list(dict.fromkeys(values))


def collect_official_models(
    path: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, List[str]]]:
    rows: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    official_alias_map: Dict[str, List[str]] = {}

    for item in load_official_models(path):
        model = clean(item.get("Model") or item.get("model"))
        if not model:
            continue

        hf_repo = clean(
            item.get("HF Repository")
            or item.get("HF Repo")
            or item.get("Hugging Face Repository")
        )

        row = {
            "Model": model,
            "Developer / HF Owner": clean(item.get("Developer")),
            "Model Family": clean(item.get("Model Family")),
            "Model Variant": clean(item.get("Model Variant")),
            "Artifact Type": clean(item.get("Artifact Type") or item.get("Model Type"))
                or "Base / Standalone",
            "Availability": clean(item.get("Availability")) or "Closed / Proprietary",
            "Release Date": clean(item.get("Release Date")),
            "HF Repository Created": None,
            "Architecture": clean(item.get("Architecture")),
            "Parameters": clean(item.get("Parameters")),
            "Repository Size (GB)": None,
            "Weights Size (GB)": None,
            "License": clean(item.get("License")),
            "Base Model": clean(item.get("Base Model")),
            "Text-to-Image": clean(item.get("Text-to-Image")) or NOT_FOUND,
            "Image-to-Image": clean(item.get("Image-to-Image")) or NOT_FOUND,
            "Inpainting": clean(item.get("Inpainting")) or NOT_FOUND,
            "Image Editing": clean(item.get("Image Editing")) or NOT_FOUND,
            "Control / Conditioning": clean(item.get("Control / Conditioning")) or NOT_FOUND,
            "HF Downloads (30d)": None,
            "HF Downloads (All Time)": None,
            "HF Likes": None,
            "HF Trending Score": None,
            "Arena ELO": ELO_NOT_FOUND,
            **{name: None for name in COMMON_BENCHMARKS},
            "Model URL": official_url(item.get("Model URL")) or (
                hf_url(hf_repo) if hf_repo else None
            ),
            "Official URL": official_url(item.get("Official URL")),
            "Sources": "Official project",
            "Source URLs": (
                row.get("Official URL")
                or row.get("Model URL")
                or NOT_FOUND
            ),
        }

        # Keep every official catalog link visible in the row-level source list.
        if row.get("Official URL"):
            add_row_source(row, "Official project", row["Official URL"])
        elif row.get("Model URL"):
            add_row_source(row, "Official model page", row["Model URL"])
        if hf_repo:
            add_row_source(row, "Hugging Face", hf_url(hf_repo))

        for alias in official_aliases(item):
            official_alias_map[normalize_key(alias)] = [model, hf_repo or ""]

        for field in MODEL_COLUMNS:
            value = row.get(field)
            if value is None:
                continue
            source = row.get("Official URL") or row.get("Model URL")
            if source:
                add_source(
                    sources,
                    model,
                    field,
                    value,
                    source,
                    "Official documentation/model page",
                    "High",
                )

        rows.append(row)

    return rows, sources, official_alias_map


# ---------------------------------------------------------------------------
# Arena
# ---------------------------------------------------------------------------

def _strip_html(text: str) -> str:
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html_lib.unescape(text)).strip()


def _parse_artificial_analysis_payload(text: str) -> Dict[str, float]:
    """Parse Artificial Analysis JSON/HTML, including current client-rendered payloads."""
    elo_map: Dict[str, float] = {}

    # 1) Direct JSON document.
    try:
        payload = json.loads(text)
        containers: List[Any] = [payload]
        while containers:
            current = containers.pop()
            if isinstance(current, dict):
                name = clean(current.get("name") or current.get("model_name") or current.get("Model"))
                elo = current.get("elo")
                if name is not None and elo is not None:
                    try:
                        elo_map[name] = float(elo)
                    except (TypeError, ValueError):
                        pass
                containers.extend(current.values())
            elif isinstance(current, list):
                containers.extend(current)
        if elo_map:
            return elo_map
    except Exception:
        pass

    # 2) Embedded JSON objects inside script / Next.js payloads.
    # We intentionally require a model-like name plus an explicit Elo field.
    for match in re.finditer(
        r'"name"\s*:\s*"([^"]{2,160})".{0,500}?"elo"\s*:\s*(-?\d+(?:\.\d+)?)',
        text,
        flags=re.I | re.S,
    ):
        name = html_lib.unescape(match.group(1)).strip()
        try:
            elo_map[name] = float(match.group(2))
        except ValueError:
            pass

    for match in re.finditer(
        r'"model_name"\s*:\s*"([^"]{2,160})".{0,400}?"(?:elo|score)"\s*:\s*(-?\d+(?:\.\d+)?)',
        text,
        flags=re.I | re.S,
    ):
        name = html_lib.unescape(match.group(1)).strip()
        try:
            elo_map[name] = float(match.group(2))
        except ValueError:
            pass

    if elo_map:
        return elo_map

    # 3) Server-rendered HTML table. Current public page exposes:
    # Range | Creator | Model | Elo | 95% CI | ...
    for row_html in re.findall(r"<tr\b[^>]*>(.*?)</tr>", text, flags=re.I | re.S):
        cells = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", row_html, flags=re.I | re.S)
        if len(cells) < 4:
            continue
        cell_text = [_strip_html(cell) for cell in cells]
        lower = [x.lower() for x in cell_text]
        if "model" in lower and "elo" in lower:
            continue

        # Find a plausible model/Elo pair rather than assuming fixed positions
        # if the page adds an extra cell.
        elo_idx = None
        for i, value in enumerate(cell_text):
            if re.fullmatch(r"\d{3,5}(?:\.\d+)?", value.replace(",", ".")):
                elo_idx = i
                break
        if elo_idx is None or elo_idx == 0:
            continue

        name_candidates = [
            cell_text[elo_idx - 1],
            cell_text[2] if len(cell_text) > 2 else "",
        ]
        name = next((x for x in name_candidates if x and not re.fullmatch(r"\d+(?:-\d+)?", x)), "")
        if not name:
            continue
        try:
            elo_map[name] = float(cell_text[elo_idx].replace(",", "."))
        except ValueError:
            continue
    return elo_map


def collect_arena(
    url: str,
    timeout: int = 30,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], List[Dict[str, Any]]]:
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 image-model-collector/9.0"},
    )
    response.raise_for_status()
    elo_map = _parse_artificial_analysis_payload(response.text)

    sources: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    if not elo_map:
        errors.append({
            "Stage": "Artificial Analysis parsing",
            "Model": None,
            "Error": "No leaderboard rows with model name + ELO were found.",
        })
    else:
        print(f"[Artificial Analysis] Parsed {len(elo_map)} leaderboard models")
    return elo_map, sources, errors


def load_arena_aliases(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        print(f"[Arena] Cannot read aliases: {exc}")
        return {}


def _arena_token_key(value: Any) -> Tuple[str, ...]:
    text = normalize_arena_key(value)
    tokens = [t for t in text.split() if t]
    # Remove labels that do not identify a model.
    stop = {"open", "weights", "api", "model", "image", "generation"}
    return tuple(t for t in tokens if t not in stop)


def _safe_arena_match(
    row: Dict[str, Any],
    normalized_arena: Dict[str, Tuple[float, str]],
) -> Optional[Tuple[float, str, str]]:
    values = [
        row.get("Model"),
        row.get("Model Variant"),
        row.get("Model Family"),
    ]
    candidate_keys: List[str] = []
    for value in values:
        candidate_keys.extend(model_identity_candidates(value))

    # First: exact deterministic normalized match.
    seen = set()
    for candidate in candidate_keys:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        hit = normalized_arena.get(candidate)
        if hit is not None:
            return hit[0], hit[1], "deterministic normalized match"

    # Second: conservative token containment. Requires at least two distinctive
    # tokens and only one candidate from the leaderboard to avoid accidental matches.
    row_token_sets = []
    for value in values:
        tokens = set(_arena_token_key(value))
        if len(tokens) >= 2:
            row_token_sets.append(tokens)

    possible: List[Tuple[float, str]] = []
    for arena_key, hit in normalized_arena.items():
        arena_tokens = set(_arena_token_key(arena_key))
        for row_tokens in row_token_sets:
            common = row_tokens.intersection(arena_tokens)
            distinctive = [t for t in common if len(t) >= 2 and not t.isdigit()]
            if len(distinctive) >= 2 and (
                row_tokens.issubset(arena_tokens) or arena_tokens.issubset(row_tokens)
            ):
                possible.append(hit)
                break

    unique = []
    seen_names = set()
    for hit in possible:
        if hit[1] not in seen_names:
            unique.append(hit)
            seen_names.add(hit[1])
    if len(unique) == 1:
        return unique[0][0], unique[0][1], "conservative token match"
    return None


def apply_arena_aliases(
    rows: List[Dict[str, Any]],
    arena_map: Dict[str, float],
    aliases: Dict[str, Any],
    sources: List[Dict[str, Any]],
    arena_url: str,
) -> None:
    """Attach Artificial Analysis ELO using verified aliases + conservative matching."""
    exact = aliases.get("exact", {}) if isinstance(aliases, dict) else {}
    verified = aliases.get("verified_normalized", {}) if isinstance(aliases, dict) else {}
    if not isinstance(exact, dict):
        exact = {}
    if not isinstance(verified, dict):
        verified = {}

    normalized_arena: Dict[str, Tuple[float, str]] = {}
    for name, value in arena_map.items():
        try:
            elo = float(value)
        except (TypeError, ValueError):
            continue
        for key in model_identity_candidates(name):
            normalized_arena.setdefault(key, (elo, name))

    matches = 0
    for row in rows:
        repo_id = clean(row.get("Model")) or ""
        explicit = None

        # Explicit aliases remain highest priority.
        for mapping in (exact, verified):
            alias_value = mapping.get(repo_id)
            if alias_value:
                alias_values = alias_value if isinstance(alias_value, list) else [alias_value]
                for alias_name in alias_values:
                    for key in model_identity_candidates(alias_name):
                        hit = normalized_arena.get(key)
                        if hit is not None:
                            explicit = (hit[0], hit[1], "explicit alias")
                            break
                    if explicit:
                        break
            if explicit:
                break

        matched = explicit or _safe_arena_match(row, normalized_arena)
        if matched is None:
            row["Arena ELO"] = NOT_FOUND
            continue

        elo, arena_name, match_type = matched
        row["Arena ELO"] = elo
        add_row_source_with_field(row, "Arena ELO", "Artificial Analysis", arena_url)
        add_source(
            sources, repo_id, "Arena ELO", elo, arena_url,
            f"Artificial Analysis leaderboard: {arena_name} ({match_type})", "High"
        )
        matches += 1
        print(f"[Artificial Analysis] Matched {repo_id} -> {arena_name}: {elo:g} ({match_type})")

    print(f"[Artificial Analysis] Matches: {matches}/{len(rows)}")


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def _merge_source_lists(existing: Dict[str, Any], incoming: Dict[str, Any]) -> None:
    # Legacy compatibility for checkpoints created by earlier collectors.
    for field in ("Sources", "Source URLs"):
        values = split_multivalue(existing.get(field))
        for value in split_multivalue(incoming.get(field)):
            if value not in values:
                values.append(value)
        existing[field] = "\n".join(values) if values else NOT_FOUND


def merge_rows(
    hf_rows: List[Dict[str, Any]],
    official_rows: List[Dict[str, Any]],
    official_alias_map: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for row in hf_rows:
        merged[normalize_key(row["Model"])] = row

    alias_to_hf_key: Dict[str, str] = {}
    for alias, payload in (official_alias_map or {}).items():
        if not isinstance(payload, list) or len(payload) < 2:
            continue
        official_model, hf_repo = payload[0], payload[1]
        target = normalize_key(hf_repo or official_model)
        if hf_repo:
            for key in model_identity_candidates(hf_repo):
                alias_to_hf_key[key] = target
        for key in model_identity_candidates(official_model):
            alias_to_hf_key[key] = target

    for row in official_rows:
        model = clean(row.get("Model")) or ""
        key_candidates = model_identity_candidates(model)
        model_url = row.get("Model URL") or ""
        if "huggingface.co/" in model_url:
            repo_id = model_url.split("huggingface.co/", 1)[1].strip("/")
            key_candidates.extend(model_identity_candidates(repo_id))
        key_candidates.extend(alias_to_hf_key.get(k, "") for k in model_identity_candidates(model))
        key_candidates = [k for k in key_candidates if k]

        target_key = None
        for candidate in key_candidates:
            if candidate in merged:
                target_key = candidate
                break
        if target_key is None:
            for merged_key, existing in merged.items():
                if set(model_identity_candidates(existing.get("Model"))).intersection(key_candidates):
                    target_key = merged_key
                    break

        if target_key is None:
            merged[normalize_key(model)] = row
            continue

        existing = merged[target_key]
        for field, value in row.items():
            if field.startswith("_"):
                continue
            if value is not None and clean(value) is not None and existing.get(field) in (None, "", NOT_FOUND):
                existing[field] = value
    return list(merged.values())


def _normalize_export_row(row: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for field in MODEL_COLUMNS:
        value = row.get(field)
        if value is None or clean(value) is None:
            cleaned[field] = NOT_FOUND
        elif isinstance(value, str) and value.strip().lower() in {"unknown", "n/a", "na", "none", "null", ""}:
            cleaned[field] = NOT_FOUND
        else:
            cleaned[field] = value
    return cleaned


def _resolve_source_model_key(model: Any) -> str:
    return normalize_key(clean(model) or "")


def build_source_matrix(
    rows: List[Dict[str, Any]],
    sources: List[Dict[str, Any]],
) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, str]]]:
    """Build the field-level provenance sheet for the Models workbook.

    Sheet 2 keeps the exact same field names as Models, but every cell contains
    the URL(s) used to populate that field.
    """
    model_keys: Dict[str, int] = {}
    aliases: Dict[str, int] = {}

    for idx, row in enumerate(rows):
        model = clean(row.get("Model")) or ""
        model_keys[normalize_key(model)] = idx
        for alias in model_identity_candidates(model):
            aliases.setdefault(alias, idx)

    provenance: Dict[int, Dict[str, List[str]]] = {i: {} for i in range(len(rows))}

    def attach(model_name: Any, field: Any, url: Any) -> None:
        model_text = clean(model_name) or ""
        idx = model_keys.get(normalize_key(model_text))
        if idx is None:
            for alias in model_identity_candidates(model_text):
                idx = aliases.get(alias)
                if idx is not None:
                    break

        url_text = official_url(url)
        field_name = clean(field)

        if idx is None or not url_text or field_name not in MODEL_COLUMNS:
            return

        values = provenance[idx].setdefault(field_name, [])
        if url_text not in values:
            values.append(url_text)

    # Structured field-level evidence.
    for item in sources:
        attach(item.get("Model"), item.get("Field"), item.get("Source"))

    # Persisted row-level field provenance from current/new checkpoints.
    for idx, row in enumerate(rows):
        field_sources = row.get("_field_sources") or {}
        if isinstance(field_sources, dict):
            for field_name, urls in field_sources.items():
                for url in urls or []:
                    attach(row.get("Model"), field_name, url)

        # Model URL and Official URL are themselves documented fields.
        attach(row.get("Model"), "Model URL", row.get("Model URL"))
        attach(row.get("Model"), "Official URL", row.get("Official URL"))

    output: List[Dict[str, Any]] = []
    normalized_sources: List[Dict[str, str]] = []
    seen_source_pairs: set[Tuple[str, str]] = set()

    for idx, row in enumerate(rows):
        out: Dict[str, Any] = {"Model": clean(row.get("Model")) or NOT_FOUND}

        for field in MODEL_COLUMNS[1:]:
            urls = provenance[idx].get(field, [])
            out[field] = "\n".join(urls) if urls else NOT_FOUND

        # One row per unique source used by this model.
        for urls in provenance[idx].values():
            for url in urls:
                model_name = clean(row.get("Model")) or NOT_FOUND
                pair = (model_name, url)
                if pair in seen_source_pairs:
                    continue
                seen_source_pairs.add(pair)
                normalized_sources.append({
                    "Model": model_name,
                    "Source": source_display_name(url),
                    "Link": url,
                })

        output.append(out)

    return [], output, normalized_sources


def _style_source_url_cell(cell) -> None:
    from copy import copy
    if not isinstance(cell.value, str) or not cell.value.startswith(("http://", "https://")):
        return
    cell.hyperlink = cell.value
    cell.style = "Hyperlink"
    alignment = copy(cell.alignment)
    alignment.wrap_text = False
    alignment.vertical = "top"
    cell.alignment = alignment


def format_excel(writer: pd.ExcelWriter) -> None:
    for ws in writer.sheets.values():
        ws.freeze_panes = "A2"
        if ws.max_row and ws.max_column:
            ws.auto_filter.ref = ws.dimensions
        for column_cells in ws.columns:
            header = str(column_cells[0].value or "")
            max_length = max((min(len(str(c.value or "")), 70) for c in column_cells[:3000]), default=10)
            if ws.title in {SOURCE_SHEET_NAME, SOURCE_LIST_SHEET_NAME}:
                width = 28 if header != "Model" else 32
            else:
                width = min(max(max_length + 2, 10), 45)
            ws.column_dimensions[column_cells[0].column_letter].width = width
            if ws.title in {SOURCE_SHEET_NAME, SOURCE_LIST_SHEET_NAME}:
                for cell in column_cells[1:]:
                    _style_source_url_cell(cell)
                for row_idx in range(2, ws.max_row + 1):
                    ws.row_dimensions[row_idx].height = 18


def export_excel(
    rows: List[Dict[str, Any]],
    sources: List[Dict[str, Any]],
    filename: str,
) -> None:
    export_rows = [_normalize_export_row(r) for r in rows]
    models_df = pd.DataFrame(export_rows, columns=MODEL_COLUMNS).fillna(NOT_FOUND)

    # Preserve the existing model ordering logic.
    elo_numeric = pd.to_numeric(models_df["Arena ELO"], errors="coerce")
    downloads_numeric = pd.to_numeric(models_df["HF Downloads (30d)"], errors="coerce")
    models_df["_elo_sort"] = elo_numeric.fillna(-float("inf"))
    models_df["_downloads_sort"] = downloads_numeric.fillna(-1)
    models_df.sort_values(
        by=["_elo_sort", "_downloads_sort", "Model"],
        ascending=[False, False, True],
        inplace=True,
    )
    models_df.drop(columns=["_elo_sort", "_downloads_sort"], inplace=True)

    # Keep provenance rows aligned with the sorted model table.
    row_by_model = {clean(r.get("Model")): r for r in rows}
    sorted_rows = [row_by_model.get(model, {}) for model in models_df["Model"].tolist()]

    _, source_matrix, source_list = build_source_matrix(sorted_rows, sources)
    sources_df = pd.DataFrame(source_matrix, columns=MODEL_COLUMNS).fillna(NOT_FOUND)
    source_list_df = pd.DataFrame(source_list, columns=["Model", "Source", "Link"]).fillna(NOT_FOUND)

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        # Sheet 1: all discovered characteristics.
        models_df.to_excel(writer, sheet_name="Models", index=False)

        # Sheet 2: same columns, but values are source URLs for each field.
        sources_df.to_excel(writer, sheet_name=SOURCE_SHEET_NAME, index=False)

        # Sheet 3: one row for every unique Model + Source pair.
        # The model is repeated intentionally so each source can be traced back
        # to the exact model without relying on row positions from another sheet.
        source_list_df.to_excel(
            writer,
            sheet_name=SOURCE_LIST_SHEET_NAME,
            index=False,
        )

        format_excel(writer)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a conservative, source-auditable image model database "
            "with persistent per-model checkpoints."
        )
    )
    parser.add_argument(
        "--output",
        default="image_models_v10.xlsx",
        help="Excel output filename.",
    )
    parser.add_argument("--mode", choices=("fast", "full"), default="full")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help=(
            "Number of NEW usable image models to collect in this run. "
            "The collector searches past rejected components until this many "
            "usable models are found. Default: 100."
        ),
    )
    parser.add_argument("--include-variants", action="store_true")
    parser.add_argument(
        "--arena-url",
        default="https://artificialanalysis.ai/image/leaderboard/text-to-image",
        help="Artificial Analysis public text-to-image leaderboard URL. Use empty string to disable.",
    )
    parser.add_argument(
        "--arena-aliases",
        default="config/arena_aliases.json",
    )
    parser.add_argument(
        "--official-models",
        default="config/official_models.json",
    )
    parser.add_argument(
        "--cache-dir",
        default="cache/hf_cards",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Persistent per-model checkpoint JSON.",
    )
    parser.add_argument(
        "--cache-max-age-days",
        type=int,
        default=7,
    )
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Delete only the current-version checkpoint before starting. Model Card cache is untouched.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    official_path = Path(args.official_models)
    aliases_path = Path(args.arena_aliases)
    checkpoint_path = Path(args.checkpoint)

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer or omitted")

    if args.reset_checkpoint:
        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                print(f"[Checkpoint] Reset: {checkpoint_path}")
            else:
                print(f"[Checkpoint] Nothing to reset: {checkpoint_path}")
        except Exception as exc:
            raise SystemExit(f"Cannot reset checkpoint {checkpoint_path}: {exc}")

    print(f"[Mode] {args.mode}")
    print(f"[Limit] {args.limit} new usable image models")
    print(f"[Checkpoint] {checkpoint_path}")
    print()

    try:
        (
            hf_rows,
            sources,
            evaluations,
            capabilities,
            errors,
        ) = collect_huggingface(
            mode=args.mode,
            limit=args.limit,
            include_variants=args.include_variants,
            cache_dir=cache_dir,
            cache_max_age_days=args.cache_max_age_days,
            sleep_seconds=args.sleep,
            checkpoint_path=checkpoint_path,
        )

        official_rows, official_sources, official_alias_map = collect_official_models(
            official_path
        )
        sources.extend(official_sources)
        print(f"[Official] Standalone catalog rows: {len(official_rows)}")

        rows = merge_rows(
            hf_rows,
            official_rows,
            official_alias_map,
        )

        if args.arena_url:
            try:
                arena_map, arena_sources, arena_errors = collect_arena(args.arena_url)
                sources.extend(arena_sources)
                errors.extend(arena_errors)
                aliases = load_arena_aliases(aliases_path)
                apply_arena_aliases(rows, arena_map, aliases, sources, args.arena_url)
            except Exception as exc:
                errors.append({"Stage": "Artificial Analysis", "Model": None, "Error": repr(exc)})
                print(f"[Artificial Analysis] Failed: {exc}")

        # Every model gets an explicit status when no leaderboard match exists.
        for row in rows:
            if not clean(row.get("Arena ELO")):
                row["Arena ELO"] = ELO_NOT_FOUND

        export_excel(rows, sources, args.output)

        print()
        print(f"Done: {args.output}")
        print(f"Models: {len(rows)}")
        print(f"HF models: {len(hf_rows)}")
        print(f"Official catalog models: {len(official_rows)}")
        print(f"Rows with Official URL: {sum(1 for r in rows if clean(r.get('Official URL')) not in (None, NOT_FOUND))}")
        print(f"Artificial Analysis matches: {sum(1 for r in rows if clean(r.get('Arena ELO')) not in (None, NOT_FOUND))}")
        print(f"Models with 2+ field sources: {sum(1 for r in rows if len({u for urls in (r.get('_field_sources') or {}).values() for u in (urls or [])}) >= 2)}")        
        for benchmark in COMMON_BENCHMARKS:
            print(f"{benchmark} values: {sum(1 for r in rows if clean(r.get(benchmark)) not in (None, NOT_FOUND))}")
        print(f"Errors: {len(errors)}")
        print(f"Checkpoint: {checkpoint_path}")

    except KeyboardInterrupt:
        # Safety net for Ctrl+C during a stage outside the per-model loop.
        print("\n[Main] Ctrl+C received. Existing checkpoint is preserved.")
        print(f"[Main] Last safe checkpoint: {checkpoint_path}")
        return
    except Exception as exc:
        # Never delete the checkpoint because a later stage (e.g. Excel export)
        # failed. Already collected model data remains resumable.
        print(f"\n[Main] Fatal stage error: {exc!r}")
        print(f"[Main] Persistent checkpoint preserved at: {checkpoint_path}")
        raise


if __name__ == "__main__":
    main()

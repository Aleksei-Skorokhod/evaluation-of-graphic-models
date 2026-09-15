"""
Reliable image-generation model collector v5.

Goals:
- high-recall Hugging Face discovery without pretending discovery is exhaustive;
- conservative capability classification: missing evidence => Unknown;
- evidence-level tracking for every capability;
- Model Card metadata + Model Card text;
- explicit separation of models from model components/variants;
- official-model enrichment with optional HF repository IDs/aliases;
- 30-day and all-time HF download counters when available;
- Arena ELO only through explicit/verified aliases;
- auditable Excel output.

Requirements:
    pip install -U huggingface_hub pandas openpyxl requests
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
    "Source Type",
    "Pipeline",
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
    "Model URL",
    "Official URL",
    "Data Sources",
]

CACHE_VERSION = 3

# Persistent collector checkpoint.
# Unlike the Model Card cache, this stores the already processed rows and
# auxiliary records. It is written after every successfully completed model.
CHECKPOINT_VERSION = 2
DEFAULT_CHECKPOINT = "cache/collector_checkpoint_v5.json"


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
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


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


def get_author(repo_id: str, model_info: Any, card_meta: Dict[str, Any]) -> str:
    author = first_value(
        card_meta.get("author"),
        card_meta.get("developers"),
        card_meta.get("developer"),
        getattr(model_info, "author", None),
    )
    if author:
        return str(author)
    return repo_id.split("/", 1)[0] if "/" in repo_id else "Unknown"


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
    return value if isinstance(value, dict) else {}


def get_architecture(model_info: Any, card_meta: Dict[str, Any]) -> Optional[str]:
    config = _config(model_info)
    architecture = first_value(
        card_meta.get("architecture"),
        config.get("architectures"),
        config.get("architecture"),
        config.get("model_type"),
    )
    if isinstance(architecture, list):
        architecture = ", ".join(map(str, architecture))
    return clean(architecture)


def get_model_family(repo_id: str, model_info: Any, card_meta: Dict[str, Any]) -> Optional[str]:
    config = _config(model_info)
    value = first_value(
        card_meta.get("model_name"),
        card_meta.get("model_family"),
        config.get("model_type"),
        config.get("_name_or_path"),
    )
    return clean(value) or repo_id.rsplit("/", 1)[-1]


def get_parameters(model_info: Any, card_meta: Dict[str, Any]) -> Optional[str]:
    config = _config(model_info)
    value = first_value(
        card_meta.get("parameters"),
        card_meta.get("num_parameters"),
        config.get("num_parameters"),
        config.get("parameter_count"),
        getattr(model_info, "num_parameters", None),
    )
    if isinstance(value, dict):
        total = sum(v for v in value.values() if isinstance(v, (int, float)))
        if total:
            value = total
    if isinstance(value, (int, float)):
        return str(int(value))
    return clean(value)


def get_release_date(model_info: Any, card_meta: Dict[str, Any]) -> Optional[str]:
    # Repository creation is deliberately NOT substituted for release date.
    # They are different concepts. We only report an explicit release/date field.
    return iso_date(first_value(
        card_meta.get("release_date"),
        card_meta.get("releaseDate"),
        card_meta.get("date"),
    ))


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
        value = "Unknown"
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

def extract_eval_results(card_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    model_index = card_meta.get("model-index", [])

    if isinstance(model_index, dict):
        model_index = [model_index]
    if not isinstance(model_index, list):
        return results

    for model_entry in model_index:
        if not isinstance(model_entry, dict):
            continue
        model_name = model_entry.get("name")

        for result in model_entry.get("results", []) or []:
            if not isinstance(result, dict):
                continue

            task = result.get("task") or {}
            dataset = result.get("dataset") or {}
            metrics = result.get("metrics") or []
            if isinstance(metrics, dict):
                metrics = [metrics]

            for metric in metrics:
                if not isinstance(metric, dict):
                    continue

                results.append({
                    "Model": model_name,
                    "Task": task.get("type") or task.get("name"),
                    "Dataset": dataset.get("name") or dataset.get("type"),
                    "Metric": metric.get("name") or metric.get("type"),
                    "Value": metric.get("value"),
                    "Verified": metric.get("verified"),
                    "Source": metric.get("source"),
                })

    return results


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
                    detailed_info = api.model_info(repo_id, files_metadata=True)
                    info = detailed_info
                    detailed_meta = metadata_dict(info)
                    if detailed_meta:
                        card_meta = {**card_meta, **detailed_meta}
                except Exception as exc:
                    print(f"[HF] Detail metadata unavailable for {repo_id}: {exc}")

                card_text = (
                    card_payload.get("text")
                    or card_payload.get("content")
                    or ""
                )

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
                parameters = get_parameters(info, card_meta)
                architecture = get_architecture(info, card_meta)
                model_family = get_model_family(repo_id, info, card_meta)
                release_date = get_release_date(info, card_meta)

                downloads_30d = getattr(info, "downloads", None)
                downloads_all_time = getattr(info, "downloads_all_time", None)
                likes = getattr(info, "likes", None)
                trending = getattr(info, "trending_score", None)

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

                row: Dict[str, Any] = {
                    "Model": repo_id,
                    "Developer / HF Owner": get_author(repo_id, info, card_meta),
                    "Model Family": model_family,
                    "Model Variant": clean(card_meta.get("variant")) or model_type,
                    "Artifact Type": model_type,
                    "Availability": "Open Weights",
                    "Source Type": "Hugging Face",
                    "Pipeline": pipeline_display,
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
                    "Arena ELO": None,
                    "Model URL": model_url,
                    "Official URL": None,
                    "Data Sources": "Hugging Face",
                }

                if card_source:
                    row["Data Sources"] += f", Model Card ({card_source})"

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

                for ev in extract_eval_results(card_meta):
                    ev["Repository"] = repo_id
                    evaluations.append(ev)

                # COMMIT POINT:
                # The model is added to the checkpoint only after its complete
                # row, sources, evaluations and capability evidence exist.
                models[key] = row
                processed_repo_ids.add(normalized_repo_id)

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
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[Official] Cannot read {path}: {exc}")
        return []

    if isinstance(payload, dict):
        payload = payload.get("models", [])

    return (
        [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, list)
        else []
    )


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
            "Source Type": "Official",
            "Pipeline": clean(item.get("Pipeline")),
            "Release Date": clean(item.get("Release Date")),
            "HF Repository Created": None,
            "Architecture": clean(item.get("Architecture")),
            "Parameters": clean(item.get("Parameters")),
            "Repository Size (GB)": None,
            "Weights Size (GB)": None,
            "License": clean(item.get("License")),
            "Base Model": clean(item.get("Base Model")),
            "Text-to-Image": clean(item.get("Text-to-Image")) or "Unknown",
            "Image-to-Image": clean(item.get("Image-to-Image")) or "Unknown",
            "Inpainting": clean(item.get("Inpainting")) or "Unknown",
            "Image Editing": clean(item.get("Image Editing")) or "Unknown",
            "Control / Conditioning": clean(item.get("Control / Conditioning")) or "Unknown",
            "HF Downloads (30d)": None,
            "HF Downloads (All Time)": None,
            "HF Likes": None,
            "HF Trending Score": None,
            "Arena ELO": None,
            "Model URL": official_url(item.get("Model URL")) or (
                hf_url(hf_repo) if hf_repo else None
            ),
            "Official URL": official_url(item.get("Official URL")),
            "Data Sources": "Official source",
        }

        if hf_repo:
            row["Data Sources"] += ", HF repository declared in official config"

        for alias in official_aliases(item):
            official_alias_map[normalize_key(alias)] = [model, hf_repo or ""]

        for field in MODEL_COLUMNS:
            value = row.get(field)
            if value is None or field == "Data Sources":
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

def collect_arena(
    url: str,
    timeout: int = 30,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], List[Dict[str, Any]]]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    elo_map: Dict[str, float] = {}
    sources: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    rows = payload.get("rows", payload if isinstance(payload, list) else [])
    if not isinstance(rows, list):
        return elo_map, sources, errors

    for item in rows:
        row = item.get("row", {}) if isinstance(item, dict) else {}
        if not isinstance(row, dict):
            continue

        name = clean(row.get("model_name") or row.get("Model") or row.get("model"))
        elo = row.get("elo")
        if elo is None:
            elo = row.get("Arena ELO")
        if elo is None:
            elo = row.get("score")

        if not name:
            continue

        try:
            elo_value = float(elo)
        except (TypeError, ValueError):
            errors.append({
                "Stage": "Arena parsing",
                "Model": name,
                "Error": f"Invalid ELO: {elo!r}",
            })
            continue

        elo_map[name.lower()] = elo_value
        sources.append({
            "Model": name,
            "Field": "Arena ELO",
            "Value": str(elo_value),
            "Source": url,
            "Source Type": "Arena leaderboard",
            "Confidence": "High",
            "Checked At": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        })

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


def apply_arena_aliases(
    rows: List[Dict[str, Any]],
    arena_map: Dict[str, float],
    aliases: Dict[str, Any],
    sources: List[Dict[str, Any]],
    arena_url: str,
) -> None:
    exact = aliases.get("exact", {})
    verified = aliases.get("verified_normalized", {})

    if not isinstance(exact, dict):
        exact = {}
    if not isinstance(verified, dict):
        verified = {}

    normalized_arena = {
        normalize_key(name): value for name, value in arena_map.items()
    }

    for row in rows:
        repo_id = row["Model"]
        arena_name = exact.get(repo_id)
        match_type = "exact alias"

        if not arena_name:
            arena_name = verified.get(repo_id)
            match_type = "verified normalized alias"

        if not arena_name:
            continue

        elo = arena_map.get(str(arena_name).lower())
        if elo is None:
            elo = normalized_arena.get(normalize_key(str(arena_name)))
        if elo is None:
            continue

        row["Arena ELO"] = elo
        row["Data Sources"] += ", Arena"

        add_source(
            sources,
            repo_id,
            "Arena ELO",
            elo,
            arena_url,
            f"Arena leaderboard ({match_type})",
            "High",
        )


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_rows(
    hf_rows: List[Dict[str, Any]],
    official_rows: List[Dict[str, Any]],
    official_alias_map: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for row in hf_rows:
        merged[normalize_key(row["Model"])] = row

    for row in official_rows:
        key_candidates = [
            normalize_key(row["Model"]),
        ]

        # If an official row explicitly declares an HF repository, prefer
        # merging into that repository instead of relying on name similarity.
        model_url = row.get("Model URL") or ""
        if "huggingface.co/" in model_url:
            repo_id = model_url.split("huggingface.co/", 1)[1].strip("/")
            key_candidates.insert(0, normalize_key(repo_id))

        target_key = next((k for k in key_candidates if k in merged), None)

        if target_key is None:
            merged[normalize_key(row["Model"])] = row
            continue

        existing = merged[target_key]

        for field, value in row.items():
            if value is not None and existing.get(field) in (None, "", "Unknown"):
                existing[field] = value

        existing["Data Sources"] = ", ".join(
            dict.fromkeys(
                filter(
                    None,
                    str(existing.get("Data Sources", "")).split(", "),
                )
            )
        )

    return list(merged.values())


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------

def _safe_sort_value(series: pd.Series, default: float) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def format_excel(writer: pd.ExcelWriter) -> None:
    for ws in writer.sheets.values():
        ws.freeze_panes = "A2"
        if ws.max_row and ws.max_column:
            ws.auto_filter.ref = ws.dimensions

        for column_cells in ws.columns:
            max_length = 0
            col_letter = column_cells[0].column_letter
            for cell in list(column_cells)[:3000]:
                value = "" if cell.value is None else str(cell.value)
                max_length = max(max_length, len(value))
            ws.column_dimensions[col_letter].width = min(
                max(max_length + 2, 10), 55
            )


def export_excel(
    rows: List[Dict[str, Any]],
    sources: List[Dict[str, Any]],
    evaluations: List[Dict[str, Any]],
    capabilities: List[Dict[str, Any]],
    errors: List[Dict[str, Any]],
    filename: str,
    mode: str,
) -> None:
    models_df = pd.DataFrame(rows)

    for col in MODEL_COLUMNS:
        if col not in models_df.columns:
            models_df[col] = None

    models_df = models_df[MODEL_COLUMNS]

    models_df["_elo_sort"] = _safe_sort_value(
        models_df["Arena ELO"], -float("inf")
    )
    models_df["_downloads_sort"] = _safe_sort_value(
        models_df["HF Downloads (30d)"], -1
    )

    models_df.sort_values(
        by=["_elo_sort", "_downloads_sort", "Model"],
        ascending=[False, False, True],
        inplace=True,
    )
    models_df.drop(
        columns=["_elo_sort", "_downloads_sort"],
        inplace=True,
    )

    methodology = pd.DataFrame([
        ["Version", "5"],
        ["Principle", "Unknown means insufficient evidence; it is never treated as No."],
        ["Discovery", "Discovery continues until the requested number of usable image models is reached; components are rejected during discovery."],
        ["Discovery limitation", "Discovery is high-recall but cannot guarantee all image models on HF."],
        ["Model Cards", "Full mode reads structured metadata and Model Card text."],
        ["Capabilities", "Structured tags/pipeline evidence is preferred; explicit Model Card statements are secondary evidence."],
        ["Capability evidence", "Every capability is recorded separately in the Capabilities sheet with evidence and confidence."],
        ["Components", "LoRA, ControlNet, Adapter, Embedding, Upscaler and Quantized artifacts are distinguished from standalone models."],
        ["Official models", "Official entries are trusted as supplied and may optionally declare an HF repository explicitly."],
        ["Arena", "ELO is applied only through exact or verified aliases; no fuzzy matching."],
        ["Downloads", "30-day and all-time counters are stored separately when the API exposes them."],
        ["Repository size", "Repository storage and safetensors weight size are stored separately."],
        ["Evaluations", "Model Card evaluation results remain separate because metrics are not directly comparable."],
        ["Sources", "Source records preserve provenance and confidence for populated fields."],
        ["Limit", "--limit is the number of NEW usable image models requested for this run, not the number of raw HF candidates."],
        ["Checkpoint", "A persistent atomic checkpoint is written after every completed model; failed models remain retryable. v5 uses a new checkpoint version."],
        ["Resume", "On restart, repositories already present in the checkpoint are skipped."],
        ["Ctrl+C", "Ctrl+C is handled after saving the last completed checkpoint; the currently incomplete model is not committed."],
        ["Freshness", "The run date is stored in source/evidence records; cached Model Cards can be controlled with --cache-max-age-days."],
    ], columns=["Rule", "Implementation"])

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        models_df.to_excel(writer, sheet_name="Models", index=False)
        pd.DataFrame(sources).to_excel(writer, sheet_name="Sources", index=False)
        pd.DataFrame(evaluations).to_excel(writer, sheet_name="Evaluations", index=False)
        pd.DataFrame(capabilities).to_excel(writer, sheet_name="Capabilities", index=False)
        pd.DataFrame(errors).to_excel(writer, sheet_name="Errors", index=False)
        methodology.to_excel(writer, sheet_name="Methodology", index=False)

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
        default="image_models_v5_test_100.xlsx",
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
    parser.add_argument("--arena-url", default=None)
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
        help="Delete only the v4 checkpoint before starting. Model Card cache is untouched.",
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

        rows = merge_rows(
            hf_rows,
            official_rows,
            official_alias_map,
        )

        if args.arena_url:
            try:
                arena_map, arena_sources, arena_errors = collect_arena(
                    args.arena_url
                )
                sources.extend(arena_sources)
                errors.extend(arena_errors)

                aliases = load_arena_aliases(aliases_path)
                apply_arena_aliases(
                    rows,
                    arena_map,
                    aliases,
                    sources,
                    args.arena_url,
                )
            except Exception as exc:
                errors.append({
                    "Stage": "Arena",
                    "Model": None,
                    "Error": repr(exc),
                })
                print(f"[Arena] Failed: {exc}")

        export_excel(
            rows,
            sources,
            evaluations,
            capabilities,
            errors,
            args.output,
            args.mode,
        )

        print()
        print(f"Done: {args.output}")
        print(f"Models: {len(rows)}")
        print(f"HF models: {len(hf_rows)}")
        print(f"Official models: {len(official_rows)}")
        print(f"Source records: {len(sources)}")
        print(f"Capability records: {len(capabilities)}")
        print(f"Evaluation records: {len(evaluations)}")
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

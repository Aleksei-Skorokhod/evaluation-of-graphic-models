"""
Evaluate saved images without generating them again.

Input:  benchmark_results/generation_results.xlsx
Output: benchmark_results/generation_results_with_scores.xlsx

Metrics: CLIPScore, ImageReward, PickScore, HPS v2.1.
GenEval is not included because its standard benchmark requires a dedicated
prompt set and evaluation protocol rather than arbitrary prompts.
"""
import gc
import os

import pandas as pd
import torch
from PIL import Image

# --------------------------- SETTINGS ---------------------------

INPUT_XLSX = os.path.join("benchmark_results", "generation_results.xlsx")
OUTPUT_XLSX = os.path.join("benchmark_results", "generation_results_with_scores.xlsx")
SUMMARY_CSV = os.path.join("benchmark_results", "benchmark_summary_with_scores.csv")

RUN_CLIP_SCORE = True
RUN_IMAGE_REWARD = True
RUN_PICKSCORE = True
RUN_HPS_V2 = True

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
PICKSCORE_PROCESSOR_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
PICKSCORE_MODEL_ID = "yuvalkirstain/PickScore_v1"
HPS_VERSION = "v2.1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
SCORE_COLUMNS = ["CLIPScore", "ImageReward", "PickScore", "HPS_v2_1"]


# --------------------------- HELPERS ---------------------------

def resolve_image_path(path):
    if not isinstance(path, str) or not path.strip():
        return None
    path = os.path.expanduser(path.strip())
    candidates = [
        path,
        os.path.abspath(path),
        os.path.join(os.path.dirname(INPUT_XLSX), path),
        os.path.join(os.path.dirname(os.path.dirname(INPUT_XLSX)), path),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return None


def load_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")



def score_to_float(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().reshape(-1)[0].item()

    elif isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Metric returned an empty result.")
        return score_to_float(value[0])

    elif hasattr(value, "size") and hasattr(value, "item"):
        if value.size != 1:
            raise ValueError(
                f"Expected one score, got array with size {value.size}"
            )
        value = value.item()

    return float(value)


def save_workbook(df):
    os.makedirs(os.path.dirname(OUTPUT_XLSX) or ".", exist_ok=True)
    valid = df.copy()
    if "status" in valid.columns:
        valid = valid[valid["status"] == "success"]

    numeric_columns = [
        c for c in [
            "generation_time_s", "peak_vram_allocated_mb",
            "peak_vram_reserved_mb", *SCORE_COLUMNS,
        ] if c in valid.columns
    ]
    if "model" in valid.columns and numeric_columns and not valid.empty:
        summary = valid.groupby("model")[numeric_columns].mean(numeric_only=True)
        if "image_path" in valid.columns:
            summary.insert(0, "evaluated_images", valid.groupby("model")["image_path"].count())
        summary = summary.reset_index()
    else:
        summary = pd.DataFrame()

    notes = pd.DataFrame([
        ["CLIPScore", "Image-text semantic similarity; higher is generally better.",
         "Does not reliably verify exact object counts or fine spatial relations."],
        ["ImageReward", "Learned text-to-image human-preference score; higher is generally better.",
         "Raw values are model-specific, not probabilities."],
        ["PickScore", "Learned human-preference score for a prompt-image pair; higher is generally better.",
         "Raw values are not probabilities; compare using the same model/version."],
        ["HPS_v2_1", "Human Preference Score v2.1 for a prompt-image pair; higher is generally better.",
         "Do not mix v2.0 and v2.1 scores."],
        ["GenEval", "Not run by this script.",
         "Its standard benchmark requires a specialized prompt set and evaluation protocol."],
    ], columns=["metric", "meaning", "caution"])

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Generation Results", index=False)
        summary.to_excel(writer, sheet_name="Model Summary", index=False)
        notes.to_excel(writer, sheet_name="Metric Notes", index=False)

    if not summary.empty:
        summary.to_csv(SUMMARY_CSV, index=False)


def should_score(row, path):
    return (
        row.get("status") == "success"
        and path is not None
        and isinstance(row.get("prompt"), str)
    )


def cleanup_model():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------- CLIPSCORE ---------------------------

def evaluate_clipscore(df):
    from transformers import CLIPModel, CLIPProcessor

    print("\\n[CLIPScore] Loading model...")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model = CLIPModel.from_pretrained(
        CLIP_MODEL_ID, torch_dtype=MODEL_DTYPE
    ).to(DEVICE).eval()

    for index, row in df.iterrows():
        path = resolve_image_path(row.get("image_path"))
        if not should_score(row, path):
            continue
        try:
            image = load_image(path)
            image_inputs = processor(images=image, return_tensors="pt")
            text_inputs = processor(
                text=[row["prompt"]], padding=True, truncation=True,
                return_tensors="pt",
            )
            image_inputs = {k: v.to(DEVICE) for k, v in image_inputs.items()}
            text_inputs = {k: v.to(DEVICE) for k, v in text_inputs.items()}
            with torch.inference_mode():
                image_features = torch.nn.functional.normalize(
                    model.get_image_features(**image_inputs), dim=-1
                )
                text_features = torch.nn.functional.normalize(
                    model.get_text_features(**text_inputs), dim=-1
                )
                score = (100 * (image_features * text_features).sum(dim=-1)).clamp(min=0)
            df.at[index, "CLIPScore"] = round(score.item(), 4)
            print(f"  row {index}: {df.at[index, 'CLIPScore']}")
        except Exception as exc:
            print(f"  CLIPScore failed for row {index}: {exc}")

    del model, processor
    cleanup_model()


# --------------------------- IMAGEREWARD ---------------------------

def evaluate_image_reward(df):
    import ImageReward as RM

    print("\\n[ImageReward] Loading model...")
    model = RM.load("ImageReward-v1.0", device=DEVICE)
    for index, row in df.iterrows():
        path = resolve_image_path(row.get("image_path"))
        if not should_score(row, path):
            continue
        try:
            with torch.inference_mode():
                score = model.score(row["prompt"], path)
            df.at[index, "ImageReward"] = round(score_to_float(score), 6)
            print(f"  row {index}: {df.at[index, 'ImageReward']}")
        except Exception as exc:
            print(f"  ImageReward failed for row {index}: {exc}")

    del model
    cleanup_model()


# --------------------------- PICKSCORE ---------------------------

def evaluate_pickscore(df):
    from transformers import AutoModel, AutoProcessor

    print("\\n[PickScore] Loading model...")
    processor = AutoProcessor.from_pretrained(PICKSCORE_PROCESSOR_ID)
    model = AutoModel.from_pretrained(
        PICKSCORE_MODEL_ID, torch_dtype=MODEL_DTYPE
    ).to(DEVICE).eval()

    for index, row in df.iterrows():
        path = resolve_image_path(row.get("image_path"))
        if not should_score(row, path):
            continue
        try:
            image = load_image(path)
            image_inputs = processor(
                images=[image], padding=True, truncation=True, max_length=77,
                return_tensors="pt",
            ).to(DEVICE)
            text_inputs = processor(
                text=[row["prompt"]], padding=True, truncation=True, max_length=77,
                return_tensors="pt",
            ).to(DEVICE)
            with torch.inference_mode():
                image_features = torch.nn.functional.normalize(
                    model.get_image_features(**image_inputs), dim=-1
                )
                text_features = torch.nn.functional.normalize(
                    model.get_text_features(**text_inputs), dim=-1
                )
                score = model.logit_scale.exp() * (text_features @ image_features.T)[0, 0]
            df.at[index, "PickScore"] = round(score.item(), 6)
            print(f"  row {index}: {df.at[index, 'PickScore']}")
        except Exception as exc:
            print(f"  PickScore failed for row {index}: {exc}")

    del model, processor
    cleanup_model()


# --------------------------- HPS V2.1 ---------------------------


# --------------------------- HPS V2.1 ---------------------------

def evaluate_hps(df):
    import hpsv2

    print(f"\n[HPS {HPS_VERSION}] Loading model on first use...")
    print(f"  HPS package: {hpsv2.__file__}")

    scored = 0
    skipped = 0
    failed = 0

    for index, row in df.iterrows():
        path = resolve_image_path(row.get("image_path"))

        if not should_score(row, path):
            skipped += 1
            print(
                f"  row {index}: SKIPPED "
                f"(status={row.get('status')}, "
                f"image_exists={path is not None}, "
                f"prompt_valid={isinstance(row.get('prompt'), str)})"
            )
            continue

        try:
            with torch.inference_mode():
                # Official HPS API: a single image path as a string
                result = hpsv2.score(
                    path,
                    str(row["prompt"]),
                    hps_version="v2.1"
                )

            score = score_to_float(result)

            if not torch.isfinite(torch.tensor(score)):
                raise ValueError(f"Invalid HPS score: {score}")

            df.at[index, "HPS_v2_1"] = round(score, 6)
            scored += 1

            print(
                f"  row {index}: HPS_v2_1={df.at[index, 'HPS_v2_1']}"
            )

        except Exception as exc:
            failed += 1
            print(
                f"  HPS v2.1 FAILED for row {index}: "
                f"{type(exc).__name__}: {exc}"
            )

    print(
        f"\n[HPS v2.1] Finished: "
        f"scored={scored}, skipped={skipped}, failed={failed}"
    )

    cleanup_model()


# --------------------------- MAIN ---------------------------

def main():
    if not os.path.isfile(INPUT_XLSX):
        raise FileNotFoundError(
            f"Cannot find {INPUT_XLSX!r}. Run this script from the project folder "
            "or edit INPUT_XLSX at the top of the file."
        )

    print("=" * 72)
    print("IMAGE QUALITY EVALUATION")
    print("=" * 72)
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    df = pd.read_excel(INPUT_XLSX)
    required = {"prompt", "image_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if "status" not in df.columns:
        df["status"] = "success"

    for column in SCORE_COLUMNS:
        if column not in df.columns:
            df[column] = float("nan")

    accessible = sum(resolve_image_path(p) is not None for p in df["image_path"])
    print(f"Rows in workbook: {len(df)}")
    print(f"Rows with accessible images: {accessible}")

    # Save a valid output workbook immediately, even if a metric dependency is missing.
    save_workbook(df)

    metrics = [
        ("CLIPScore", RUN_CLIP_SCORE, evaluate_clipscore),
        ("ImageReward", RUN_IMAGE_REWARD, evaluate_image_reward),
        ("PickScore", RUN_PICKSCORE, evaluate_pickscore),
        ("HPS v2.1", RUN_HPS_V2, evaluate_hps),
    ]
    for name, enabled, function in metrics:
        if not enabled:
            print(f"\\n[{name}] Skipped (disabled in settings).")
            continue
        try:
            function(df)
        except Exception as exc:
            print(f"\\n[{name}] Could not run: {exc}")
        finally:
            # Checkpoint after each metric so completed scores are not lost.
            save_workbook(df)

    save_workbook(df)
    print("\\nEvaluation finished.")
    print(f"Excel: {os.path.abspath(OUTPUT_XLSX)}")
    if os.path.isfile(SUMMARY_CSV):
        print(f"Summary CSV: {os.path.abspath(SUMMARY_CSV)}")
    print("Blank scores mean a metric was disabled, unavailable, or failed for that image.")


if __name__ == "__main__":
    main()

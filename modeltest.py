import os
import json
import time
import gc

import torch
import pandas as pd
from diffusers import DiffusionPipeline

# ============================================================
# MODELS
# ============================================================

MODEL_IDS = [
    "segmind/tiny-sd",
    "optimum-intel-internal-testing/tiny-stable-diffusion-torch-custom-variant",
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "stabilityai/sd-turbo",
    "stabilityai/stable-diffusion-xl-base-1.0",
]

# ============================================================
# PROMPTS
# ============================================================

BENCHMARK_PROMPTS = [
    "A neon shop sign that reads 'OPEN SOURCE', rainy night, reflections on wet pavement.",
    "A macro photograph of exactly three red apples and two green pears on a rustic wooden table.",
    "A cyberpunk street market in Neo-Tokyo, cinematic lighting, detailed architecture, realistic materials.",
    "A small red cube to the left of a large blue sphere, both objects fully visible against a white background.",
    "A photorealistic golden retriever sitting in a library, soft natural light, sharp focus.",
]

# ============================================================
# SETTINGS
# ============================================================

OUTPUT_DIR = "benchmark_results"
SEED = 12345
WARMUP_RUNS = 1

# Every model in MODEL_IDS must have an entry here.
MODEL_CONFIG = {
    "segmind/tiny-sd": {
        "width": 512, "height": 512, "steps": 4,
        "guidance": 7.5, "cpu_offload": False,
    },
    "optimum-intel-internal-testing/tiny-stable-diffusion-torch-custom-variant": {
        "width": 64, "height": 64, "steps": 4,
        "guidance": 7.5, "cpu_offload": False,
    },
    "stable-diffusion-v1-5/stable-diffusion-v1-5": {
        "width": 512, "height": 512, "steps": 20,
        "guidance": 7.5, "cpu_offload": False,
    },
    "stabilityai/sd-turbo": {
        "width": 512, "height": 512, "steps": 4,
        "guidance": 0.0, "cpu_offload": False,
    },
    "stabilityai/stable-diffusion-xl-base-1.0": {
        "width": 1024, "height": 1024, "steps": 25,
        "guidance": 5.0, "cpu_offload": True,
    },
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_dtype():
    if DEVICE == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


DTYPE = get_dtype()


# ============================================================
# UTILITIES
# ============================================================

def synchronize():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def cleanup():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def reset_peak_memory():
    if DEVICE == "cuda":
        synchronize()
        torch.cuda.reset_peak_memory_stats()


def get_peak_memory():
    if DEVICE != "cuda":
        return {
            "peak_vram_allocated_mb": None,
            "peak_vram_reserved_mb": None,
        }

    synchronize()
    return {
        "peak_vram_allocated_mb": round(
            torch.cuda.max_memory_allocated() / (1024 ** 2), 2
        ),
        "peak_vram_reserved_mb": round(
            torch.cuda.max_memory_reserved() / (1024 ** 2), 2
        ),
    }


def save_results(records):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, "raw_results.json")
    temp_path = json_path + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, ensure_ascii=False)

    os.replace(temp_path, json_path)


def save_excel(records):
    excel_path = os.path.join(OUTPUT_DIR, "generation_results.xlsx")
    pd.DataFrame(records).to_excel(excel_path, index=False)


# ============================================================
# PIPELINE
# ============================================================

def load_pipeline(model_id):
    config = MODEL_CONFIG[model_id]

    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=DTYPE,
    )

    if DEVICE == "cuda" and config["cpu_offload"]:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(DEVICE)

    return pipe


def generate_image(pipe, prompt, config, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)

    result = pipe(
        prompt=prompt,
        width=config["width"],
        height=config["height"],
        num_inference_steps=config["steps"],
        guidance_scale=config["guidance"],
        generator=generator,
        num_images_per_prompt=1,
    )

    return result.images[0]


# ============================================================
# BENCHMARK
# ============================================================

def run_benchmark():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("IMAGE GENERATION BENCHMARK")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Data type: {DTYPE}")

    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_memory_gb = (
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        )
        print(f"Total GPU memory: {total_memory_gb:.2f} GiB")
    else:
        print("CUDA is unavailable. Running on CPU.")

    # Fail early with a clear message if a model has no settings.
    missing_configs = [model_id for model_id in MODEL_IDS
                       if model_id not in MODEL_CONFIG]
    if missing_configs:
        raise ValueError(f"Missing MODEL_CONFIG entries: {missing_configs}")

    records = []

    for model_id in MODEL_IDS:
        print("\n" + "=" * 70)
        print(f"Testing model: {model_id}")
        print("=" * 70)

        pipe = None
        load_start = None
        config = MODEL_CONFIG[model_id]
        model_dir = os.path.join(OUTPUT_DIR, model_id.replace("/", "_"))
        os.makedirs(model_dir, exist_ok=True)

        try:
            # Load model
            load_start = time.perf_counter()
            pipe = load_pipeline(model_id)
            synchronize()
            model_load_time = time.perf_counter() - load_start

            print(f"Model loading time: {model_load_time:.2f}s")
            print(f"Resolution: {config['width']}x{config['height']}")
            print(f"Steps: {config['steps']}")
            print(f"Guidance: {config['guidance']}")
            print(f"CPU offloading: {config['cpu_offload']}")

            # Warmup (not included in measured results)
            print("\nWarming up model...")
            for warmup_idx in range(WARMUP_RUNS):
                generate_image(
                    pipe,
                    BENCHMARK_PROMPTS[0],
                    config,
                    SEED + 10000 + warmup_idx,
                )
                synchronize()

            # Measured generations
            for prompt_idx, prompt in enumerate(BENCHMARK_PROMPTS):
                seed = SEED + prompt_idx
                image = None

                row = {
                    "model": model_id,
                    "prompt_id": prompt_idx,
                    "prompt": prompt,
                    "seed": seed,
                    "steps": config["steps"],
                    "guidance_scale": config["guidance"],
                    "width": config["width"],
                    "height": config["height"],
                    "dtype": str(DTYPE),
                    "device": DEVICE,
                    "cpu_offload": config["cpu_offload"],
                    "model_load_time_s": round(model_load_time, 4),
                    "generation_time_s": None,
                    "peak_vram_allocated_mb": None,
                    "peak_vram_reserved_mb": None,
                    "image_path": None,
                    "status": "failed",
                    "error": None,
                }

                try:
                    cleanup()
                    reset_peak_memory()
                    synchronize()
                    start = time.perf_counter()

                    image = generate_image(pipe, prompt, config, seed)

                    synchronize()
                    elapsed = time.perf_counter() - start
                    row.update(get_peak_memory())

                    image_path = os.path.join(
                        model_dir, f"prompt_{prompt_idx:04d}.png"
                    )
                    image.save(image_path)

                    row["generation_time_s"] = round(elapsed, 4)
                    row["image_path"] = image_path
                    row["status"] = "success"

                    print(
                        f"Prompt {prompt_idx + 1}/{len(BENCHMARK_PROMPTS)}: "
                        f"{elapsed:.2f}s"
                    )

                    if DEVICE == "cuda":
                        print(
                            f"  Peak allocated VRAM: "
                            f"{row['peak_vram_allocated_mb']} MB"
                        )
                        print(
                            f"  Peak reserved VRAM: "
                            f"{row['peak_vram_reserved_mb']} MB"
                        )

                except Exception as exc:
                    row["error"] = repr(exc)
                    print(f"Generation failed for prompt {prompt_idx}: {exc}")

                finally:
                    records.append(row)
                    save_results(records)
                    if image is not None:
                        del image
                    cleanup()

        except Exception as exc:
            print(f"Model failed: {model_id}: {exc}")
            records.append({
                "model": model_id,
                "status": "model_failed",
                "error": repr(exc),
                "model_load_time_s": (
                    round(time.perf_counter() - load_start, 4)
                    if load_start is not None else None
                ),
            })
            save_results(records)

        finally:
            if pipe is not None:
                del pipe
            cleanup()

    # Final exports and summary
    save_results(records)
    save_excel(records)

    df = pd.DataFrame(records)
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    if df.empty or "status" not in df.columns:
        print("No results were recorded.")
        return

    successful = df[df["status"] == "success"].copy()
    if successful.empty:
        print("No successful generations.")
        return

    summary = successful.groupby("model").agg(
        successful_generations=("status", "count"),
        average_generation_time_s=("generation_time_s", "mean"),
        median_generation_time_s=("generation_time_s", "median"),
        min_generation_time_s=("generation_time_s", "min"),
        max_generation_time_s=("generation_time_s", "max"),
        average_peak_vram_allocated_mb=("peak_vram_allocated_mb", "mean"),
        max_peak_vram_allocated_mb=("peak_vram_allocated_mb", "max"),
    )

    print(summary.to_string())
    summary.to_csv(
        os.path.join(OUTPUT_DIR, "benchmark_summary.csv")
    )
    print("\nResults saved to:")
    print(os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    run_benchmark()

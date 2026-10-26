import sys, subprocess, os, gc, importlib

# This file intentionally contains no user image or private data.
# It expects the Colab notebook to have already defined SRC, PROMPT and optionally NEGATIVE.

pkgs = [
    "diffusers>=0.36.0",
    "transformers>=4.52",
    "accelerate",
    "gguf",
    "sentencepiece",
    "safetensors",
    "imageio[ffmpeg]",
    "huggingface_hub",
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", *pkgs])
importlib.invalidate_caches()

import torch
from PIL import Image
from transformers import T5EncoderModel
from diffusers import LTXImageToVideoPipeline, LTXVideoTransformer3DModel, GGUFQuantizationConfig
from diffusers.utils import export_to_video
from IPython.display import Video, display

assert torch.cuda.is_available(), "CUDA GPU is required"
assert "SRC" in globals() and os.path.exists(SRC), "SRC must point to the existing source image"
assert "PROMPT" in globals(), "PROMPT must already be defined"
if "NEGATIVE" not in globals():
    NEGATIVE = "worst quality, inconsistent motion, blurry, jittery, distorted"

print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))

dtype = torch.float16
TRANSFORMER_URL = "https://huggingface.co/city96/LTX-Video-0.9.6-distilled-gguf/resolve/main/ltxv-2b-0.9.6-distilled-04-25-Q3_K_S.gguf"

print("Loading LTX 2B Distilled Q3_K_S GGUF transformer...")
transformer = LTXVideoTransformer3DModel.from_single_file(
    TRANSFORMER_URL,
    config="Lightricks/LTX-Video",
    subfolder="transformer",
    quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
    dtype=dtype,
)

print("Loading Q4 T5-XXL GGUF text encoder...")
text_encoder = T5EncoderModel.from_pretrained(
    "calcuis/ltxv-gguf",
    gguf_file="t5xxl_fp16-q4_0.gguf",
    dtype=dtype,
    low_cpu_mem_usage=True,
)

print("Building LTX image-to-video pipeline...")
pipe = LTXImageToVideoPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    transformer=transformer,
    text_encoder=text_encoder,
    dtype=dtype,
    low_cpu_mem_usage=True,
)
pipe.to("cuda")
if hasattr(pipe.vae, "enable_tiling"):
    pipe.vae.enable_tiling()
if hasattr(pipe.vae, "enable_slicing"):
    pipe.vae.enable_slicing()
gc.collect()
torch.cuda.empty_cache()
print("PIPELINE_READY")
print("allocated GB:", round(torch.cuda.memory_allocated()/1024**3, 2))
print("reserved GB:", round(torch.cuda.memory_reserved()/1024**3, 2))

source = Image.open(SRC).convert("RGB")
OUT = "/content/ltx_v7_i2v.mp4"

def center_crop_resize(im, w, h):
    sw, sh = im.size
    target_ar = w / h
    source_ar = sw / sh
    if source_ar > target_ar:
        nw = int(sh * target_ar)
        left = (sw - nw) // 2
        im = im.crop((left, 0, left + nw, sh))
    else:
        nh = int(sw / target_ar)
        top = (sh - nh) // 2
        im = im.crop((0, top, sw, top + nh))
    return im.resize((w, h), Image.Resampling.LANCZOS)

# Keep the first proof tiny. If this succeeds, quality/resolution can be increased afterward.
attempts = [(384, 160, 25), (320, 128, 25), (256, 128, 17)]
last = None
for w, h, frames in attempts:
    try:
        gc.collect()
        torch.cuda.empty_cache()
        cond = center_crop_resize(source, w, h)
        print(f"TRY {w}x{h} {frames}f 8steps")
        out = pipe(
            image=cond,
            prompt=PROMPT,
            negative_prompt=NEGATIVE,
            width=w,
            height=h,
            num_frames=frames,
            num_inference_steps=8,
            guidance_scale=1.0,
            generator=torch.Generator(device="cuda").manual_seed(43021),
        ).frames[0]
        export_to_video(out, OUT, fps=24)
        SUCCESS = (w, h, frames)
        print("SUCCESS", SUCCESS, OUT, os.path.getsize(OUT))
        break
    except RuntimeError as exc:
        last = exc
        if "out of memory" not in str(exc).lower():
            raise
        print("OOM", (w, h, frames), str(exc)[:500])
        gc.collect()
        torch.cuda.empty_cache()
else:
    raise RuntimeError(f"All low-memory GGUF attempts failed: {last}")

display(Video(OUT, embed=True))
print("OUTPUT_PATH=" + OUT)
print("OUTPUT_SIZE=" + str(os.path.getsize(OUT)))

try:
    from google.colab import files
    files.download(OUT)
    print("DOWNLOAD_TRIGGERED")
except Exception as exc:
    print("DOWNLOAD_TRIGGER_FAILED", repr(exc))

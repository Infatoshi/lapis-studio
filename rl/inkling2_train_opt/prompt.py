"""CUA prompts rendered exactly like the server does (sglang inkling_renderer, spec inkling2)."""

import base64

import torch

IMAGE_PLACEHOLDER = -101  # sglang.srt.parser.inkling_tokenizer.IMAGE_TOKEN_ID

SYSTEM_PROMPT_1280 = (
    "You are a computer-use agent operating a 1280x800 Linux desktop with Chromium open.\n"
    "Respond ONLY with a single JSON object (no markdown fences, no extra text):\n"
    '  {"action": "scroll", "x": 640, "y": 400, "direction": "down", "amount": 5}\n'
    '  {"action": "click", "x": <0..1279>, "y": <0..799>}\n'
    '  {"action": "done", "answer": "p012"}\n'
    "Scroll down if you need to see more posts. Return done with the exact post id (e.g. p012) as soon as you know it."
)

TASKS = (
    "Open the most viral non-pinned post (most likes + reposts, ties broken by likes).",
    "Find the latest non-pinned post and return its id.",
    "Click the Open button of the post by @grace that is not pinned.",
    "Which visible post has the fewest reposts? Return its id.",
)


def data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def cua_messages(png: bytes, task: str, step: int = 1, history=()):
    hist = ("Previous actions: " + "; ".join(history[-4:]) + "\n") if history else ""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_1280},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri(png)}},
                {"type": "text", "text": f"{hist}Step {step}. Task: {task}"},
            ],
        },
    ]


def render(messages, effort=0.0):
    from sglang.srt.parser.inkling_renderer import render_inkling_messages

    return render_inkling_messages(messages, reasoning_effort=effort, spec="inkling2")


def encode_image(png: bytes):
    """-> (patches_bthwc bf16, grid [[1, h, w]], n_tokens) via the server's image processor."""
    from sglang.srt.multimodal.inkling2 import Inkling2ImageProcessor

    feats = Inkling2ImageProcessor().preprocess([png])
    return feats["vision_patches_bthwc"], feats["vision_grid_thw"].tolist(), feats["num_tokens"][0]


def expand(ids, png_list, device="cuda"):
    """Expand each -101 placeholder into its image rows.

    Returns (input_ids LongTensor, image_mask BoolTensor, images list for the model)."""
    out, mask, images = [], [], []
    it = iter(png_list)
    for tok in ids:
        if tok != IMAGE_PLACEHOLDER:
            out.append(tok)
            mask.append(False)
            continue
        patches, grid, n = encode_image(next(it))
        images.append((patches.to(device), grid))
        out.extend([0] * n)
        mask.extend([True] * n)
    return (
        torch.tensor(out, device=device),
        torch.tensor(mask, device=device),
        images,
    )

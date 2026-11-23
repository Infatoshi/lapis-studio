"""Build serving-benchmark inputs (run once in the sglang image; the client is stdlib-only).

  text_pool.json   natural-text token ids (cache-busting prefixes of exact length)
  prompts.json     rendered CUA prompt ids (JSON action, effort 0.0; micro-think, effort 0.05)
                   and a long-form text prompt for steady-state decode
  imgs/NNN.png     unique 1280x800 screenshot variants (defeat the vision-embedding cache)
"""

import io
import json
import os
import random

from PIL import Image

from .prompt import TASKS, cua_messages, render

OUT = "/work/bench_assets"


def main():
    os.makedirs(f"{OUT}/imgs", exist_ok=True)
    from sglang.srt.parser.inkling_renderer import load_tml_renderers

    tok = load_tml_renderers("inkling2").tokenizer
    corpus = []
    for root in ("/sgl-workspace/sglang/docs", "/sgl-workspace/sglang/python/sglang/srt/models"):
        for dirpath, _, files in os.walk(root):
            for f in sorted(files):
                if f.endswith((".md", ".py", ".rst")):
                    try:
                        corpus.append(open(os.path.join(dirpath, f), errors="ignore").read())
                    except OSError:
                        pass
    text = "\n".join(corpus)[:4_000_000]
    pool = list(tok.encode_ordinary(text))
    json.dump(pool, open(f"{OUT}/text_pool.json", "w"))
    base = open("/work/cua_1280x800.png", "rb").read()
    img = Image.open(io.BytesIO(base)).convert("RGB")
    rng = random.Random(0)
    for i in range(320):
        v = img.copy()
        px = v.load()
        for _ in range(64):  # a few imperceptible pixel edits in the margin
            x, y = rng.randrange(0, 200), rng.randrange(0, 800)
            px[x, y] = (rng.randrange(240, 256),) * 3
        buf = io.BytesIO()
        v.save(buf, format="PNG")
        open(f"{OUT}/imgs/{i:03d}.png", "wb").write(buf.getvalue())
    prompts = {
        "cua_json": [render(cua_messages(base, t), 0.0) for t in TASKS],
        "cua_micro": [render(cua_messages(base, t), 0.05) for t in TASKS],
        "longform": render([{"role": "user", "content": "Write a detailed, multi-section technical essay (at least 1500 words) on how GPU memory bandwidth limits LLM decoding, with examples."}], 0.0),
    }
    json.dump(prompts, open(f"{OUT}/prompts.json", "w"))
    print(f"pool tokens {len(pool)}; prompts {[(k, len(v)) for k, v in prompts.items()]}")


if __name__ == "__main__":
    main()

"""Bounded image diagnostics after the unchanged official Matplotlib command."""
import base64
import hashlib
import io
import json
from pathlib import Path

from PIL import Image

folder = Path("/testbed/result_images/test_image")
for name in (
    "interp_nearest_vs_none_pdf.png",
    "interp_nearest_vs_none-expected_pdf.png",
    "image_clip_pdf.png",
    "image_clip-expected_pdf.png",
):
    path = folder / name
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4000000:
        continue
    with Image.open(path) as source:
        dimensions = source.size
        thumbnail = source.convert("RGB")
        thumbnail.thumbnail((300, 300))
        content = io.BytesIO()
        thumbnail.save(content, format="PNG")
    print("__DYCON_IMAGE_DIAGNOSTIC__ " + json.dumps({
        "file": name, "size": dimensions, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "thumbnail_png": base64.b64encode(content.getvalue()).decode("ascii"),
    }))

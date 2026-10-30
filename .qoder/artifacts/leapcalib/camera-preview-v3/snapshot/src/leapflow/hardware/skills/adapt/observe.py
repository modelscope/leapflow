# Copyright (c) Alibaba, Inc. and its affiliates.
"""Camera frames fitted to the size a policy was trained at.

A model declares per camera ``{width, height, fit}`` under
``observation_space.images``; the node fits the frame before encoding it, so
what the model sees matches what it was trained on regardless of the sensor's
native resolution:

- ``stretch``: resize to exactly the declared size.
- ``crop``: centre-crop to the declared aspect ratio, then resize.
- ``pad``: resize to fit inside, then letterbox with black (aspect kept).

A camera the model declares nothing for is sent at full resolution. PIL is
imported lazily, as everywhere else colour frames are encoded.
"""

from __future__ import annotations

import io
from typing import Any

from leapflow.hardware.skills.adapt.spec import FIT_CROP, FIT_PAD, ImageSpec


def encode_image(frame: Any, spec: ImageSpec | None, *, quality: int = 90) -> bytes:
    """The frame's (``RgbdFrame``-like, BGR8) colour image fitted to *spec*, as JPEG."""
    from PIL import Image

    width, height = int(frame.width), int(frame.height)
    image = Image.frombytes("RGB", (width, height), frame.color, "raw", "BGR")
    image = fit_image(image, spec)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=min(100, max(1, int(quality))))
    return output.getvalue()


def fit_image(image: Any, spec: ImageSpec | None) -> Any:
    """Fit a PIL image exactly as inference does; return it unchanged without a declaration."""
    from PIL import Image

    if spec is None or image.size == (spec.width, spec.height):
        return image
    width, height = image.size
    if spec.fit == FIT_CROP:
        target = spec.width / spec.height
        if width / height > target:
            keep = round(height * target)
            left = (width - keep) // 2
            image = image.crop((left, 0, left + keep, height))
        else:
            keep = round(width / target)
            top = (height - keep) // 2
            image = image.crop((0, top, width, top + keep))
        return image.resize((spec.width, spec.height), Image.BILINEAR)
    if spec.fit == FIT_PAD:
        scale = min(spec.width / width, spec.height / height)
        inner = (max(1, round(width * scale)), max(1, round(height * scale)))
        canvas = Image.new("RGB", (spec.width, spec.height))
        offset = ((spec.width - inner[0]) // 2, (spec.height - inner[1]) // 2)
        canvas.paste(image.resize(inner, Image.BILINEAR), offset)
        return canvas
    return image.resize((spec.width, spec.height), Image.BILINEAR)


__all__ = ["encode_image", "fit_image"]

"""Generates 3la Rassi brand assets (TRK-60/61) into assets/brand/.

One-off design script, not part of the runtime pipeline — captioner.py and
publisher.py just consume the PNGs this writes. Re-run after editing the
palette/shapes below: `python assets/brand/generate.py` from the repo root.
Needs Pillow (assets/brand generation only, see requirements.txt).
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PURPLE = (45, 27, 78, 255)
ORANGE = (255, 107, 53, 255)
CREAM = (247, 243, 233, 255)

FONT_PATH = "/System/Library/Fonts/Avenir Next.ttc"
FONT_HEAVY = 8


def diagonal_gradient(size, color1, color2, angle=45):
    g = Image.linear_gradient("L").resize((size * 2, size * 2))
    g = g.rotate(angle, resample=Image.BICUBIC)
    w, h = g.size
    g = g.crop((w // 2 - size // 2, h // 2 - size // 2, w // 2 + size // 2, h // 2 + size // 2))
    solid1 = Image.new("RGBA", (size, size), color1)
    solid2 = Image.new("RGBA", (size, size), color2)
    return Image.composite(solid2, solid1, g)


def squircle_mask(size, radius_ratio=0.24):
    mask = Image.new("L", (size, size), 0)
    r = int(size * radius_ratio)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius=r, fill=255)
    return mask


def draw_icon(size=800):
    grad = diagonal_gradient(size, ORANGE, PURPLE)
    mask = squircle_mask(size)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(grad, (0, 0), mask)

    d = ImageDraw.Draw(out)
    font = ImageFont.truetype(FONT_PATH, int(size * 0.62), index=FONT_HEAVY)
    text = "3"
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(
        (size / 2 - tw / 2 - bbox[0], size / 2 - th / 2 - bbox[1]),
        text,
        font=font,
        fill=CREAM,
    )
    return out


def draw_wordmark(icon, height=300):
    icon_small = icon.resize((height, height), Image.LANCZOS)
    font = ImageFont.truetype(FONT_PATH, int(height * 0.4), index=FONT_HEAVY)
    tmp = Image.new("RGBA", (10, 10))
    bbox = ImageDraw.Draw(tmp).textbbox((0, 0), "3la Rassi", font=font)
    tw = bbox[2] - bbox[0]
    width = height + 40 + tw + 20
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    img.paste(icon_small, (0, 0), icon_small)
    d = ImageDraw.Draw(img)
    ty = (height - (bbox[3] - bbox[1])) // 2 - bbox[1]
    d.text((height + 40, ty), "3la Rassi", font=font, fill=ORANGE)
    return img


def draw_overlay_pill(icon, size=(440, 130)):
    """Modern rounded pill chip for the clip's top-left corner."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    pill = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(pill)
    d.rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], radius=size[1] // 2, fill=(22, 15, 38, 190))
    img = Image.alpha_composite(img, pill)

    pad = 10
    icon_small = icon.resize((size[1] - 2 * pad, size[1] - 2 * pad), Image.LANCZOS)
    img.paste(icon_small, (pad, pad), icon_small)

    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, 46, index=FONT_HEAVY)
    bbox = d.textbbox((0, 0), "3la Rassi", font=font)
    ty = (size[1] - (bbox[3] - bbox[1])) // 2 - bbox[1]
    d.text((size[1] + 10, ty), "3la Rassi", font=font, fill=CREAM)
    return img


def draw_banner(wordmark, size=(2560, 1440)):
    bg = diagonal_gradient(max(size), ORANGE, PURPLE).resize(size)
    img = bg.convert("RGBA")
    scale = 1500 / wordmark.width
    wm = wordmark.resize((1500, int(wordmark.height * scale)), Image.LANCZOS)
    img.paste(wm, ((size[0] - wm.width) // 2, (size[1] - wm.height) // 2), wm)
    return img.convert("RGB")


OUT_DIR = Path(__file__).parent

icon = draw_icon()
icon.save(OUT_DIR / "icon.png")

wordmark = draw_wordmark(icon)
wordmark.save(OUT_DIR / "wordmark.png")

pill = draw_overlay_pill(icon)
pill.save(OUT_DIR / "overlay_ribbon.png")

banner = draw_banner(wordmark)
banner.save(OUT_DIR / "banner_2560x1440.png")

icon.resize((800, 800), Image.LANCZOS).save(OUT_DIR / "profile_800.png")

print("done")

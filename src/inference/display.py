"""How results are shown to users (Phase 18 display rule, Phase 16 messages)."""

# gate status -> key of the headline text in the knowledge-base 'ui' block
STATUS_UI_KEYS = {
    "reliable": "status_reliable",
    "uncertain": "status_uncertain",
    "unsupported": "status_unsupported",
}

# quality reason (src/reliability/quality.py) -> 'ui' key
QUALITY_UI_KEYS = {
    "invalid_file": "quality_invalid_file",
    "too_small": "quality_too_small",
    "blank": "quality_blank",
    "too_dark": "quality_too_dark",
    "too_bright": "quality_too_bright",
    "too_blurry": "quality_blurry",
}


def confidence_text(conf):
    """Whole percent, never 100%: anything that rounds to 99 or more is shown as '99%+' (Phase 18)."""
    n = int(round(float(conf) * 100))
    return "99%+" if n >= 99 else f"{n}%"
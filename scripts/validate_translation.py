"""
Phase 21 – Translation parity validator.

Checks that knowledge_base/treatments_hi.json matches knowledge_base/treatments_en.json:
  - the usual structure checks (reuses validate_knowledge_base.validate on the Hindi file)
  - same disease IDs in the same order
  - fixed fields identical: id, pathogen, expert_review_flag, sources_used
  - same number of items in every list (symptoms, favorable_conditions, management lists)
  - same [S..] source tags, in the same order, in every translated string
  - translated strings are non-empty and contain Devanagari text
  - 'ui' blocks have identical keys, non-empty values, Hindi values contain Devanagari
  - metadata: language == 'hi', same source table

Usage:
    python scripts/validate_translation.py
    python scripts/validate_translation.py --en path/en.json --hi path/hi.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_knowledge_base as vkb  # noqa: E402

TAG_RE = re.compile(r"\[(S\d{2}(?:,\s*S\d{2})*)\]")
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")

FIXED_FIELDS = ["id", "pathogen", "expert_review_flag", "sources_used"]
STRING_FIELDS = ["display_name", "summary"]
LIST_FIELDS = ["symptoms", "favorable_conditions"]
MGMT_LISTS = ["cultural", "biological", "chemical_groups"]

problems = []


def bad(msg):
    problems.append(f"  ERROR: {msg}")


def check_text(where, en_text, hi_text, need_tags=True):
    if not isinstance(hi_text, str) or not hi_text.strip():
        bad(f"{where}: Hindi text is empty or not a string")
        return
    if not DEVANAGARI_RE.search(hi_text):
        bad(f"{where}: Hindi text has no Devanagari characters: \"{hi_text[:60]}\"")
    if need_tags and TAG_RE.findall(en_text) != TAG_RE.findall(hi_text):
        bad(f"{where}: source tags differ. EN {TAG_RE.findall(en_text)} vs HI {TAG_RE.findall(hi_text)}")


def check_list(where, en_list, hi_list):
    if len(en_list) != len(hi_list):
        bad(f"{where}: item count differs (EN {len(en_list)} vs HI {len(hi_list)})")
        return
    for i, (e, h) in enumerate(zip(en_list, hi_list), start=1):
        check_text(f"{where}[{i}]", e, h)


def parity(en, hi):
    print("\n[ Parity ] Hindi vs English")

    # metadata
    if hi["metadata"].get("language") != "hi":
        bad("metadata.language must be 'hi'")
    if hi["metadata"].get("sources") != en["metadata"].get("sources"):
        bad("metadata.sources differ between English and Hindi files")
    check_text("metadata.disclaimer", en["metadata"]["disclaimer"], hi["metadata"].get("disclaimer"), need_tags=False)

    # ui block
    en_ui, hi_ui = en.get("ui"), hi.get("ui")
    if not isinstance(en_ui, dict) or not isinstance(hi_ui, dict):
        bad("both files need a 'ui' block")
    else:
        if set(en_ui) != set(hi_ui):
            bad(f"ui keys differ. Only in EN: {sorted(set(en_ui) - set(hi_ui))}; only in HI: {sorted(set(hi_ui) - set(en_ui))}")
        for k in sorted(set(en_ui) & set(hi_ui)):
            if not str(en_ui[k]).strip():
                bad(f"ui.{k}: English text is empty")
            check_text(f"ui.{k}", en_ui[k], hi_ui[k], need_tags=False)

    # diseases
    en_ids = [d["id"] for d in en["diseases"]]
    hi_ids = [d["id"] for d in hi["diseases"]]
    if en_ids != hi_ids:
        bad(f"disease IDs/order differ. EN {en_ids} vs HI {hi_ids}")
        return
    for e, h in zip(en["diseases"], hi["diseases"]):
        did = e["id"]
        for f in FIXED_FIELDS:
            if e.get(f) != h.get(f):
                bad(f"{did}.{f}: must be identical in both files")
        for f in STRING_FIELDS:
            check_text(f"{did}.{f}", e[f], h.get(f), need_tags=False)
        for f in LIST_FIELDS:
            check_list(f"{did}.{f}", e[f], h.get(f, []))
        for m in MGMT_LISTS:
            check_list(f"{did}.management.{m}", e["management"][m], h["management"].get(m, []))
        check_text(f"{did}.management.resistant_varieties",
                   e["management"]["resistant_varieties"], h["management"].get("resistant_varieties"))
    if not problems:
        print("      OK — structure, tags, counts and Devanagari all match\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--en", default="knowledge_base/treatments_en.json")
    ap.add_argument("--hi", default="knowledge_base/treatments_hi.json")
    args = ap.parse_args()

    en_path, hi_path = Path(args.en), Path(args.hi)
    for p in (en_path, hi_path):
        if not p.exists():
            print(f"ERROR: file not found: {p}")
            sys.exit(1)

    print("=== Structure checks on the Hindi file ===")
    hi_ok = bool(vkb.validate(hi_path))

    with open(en_path, encoding="utf-8") as f:
        en = json.load(f)
    with open(hi_path, encoding="utf-8") as f:
        hi = json.load(f)
    parity(en, hi)

    print("=" * 60)
    if problems:
        print(f"PARITY FAILED — {len(problems)} problem(s):\n")
        for p in problems:
            print(p)
    else:
        print("PARITY PASSED — 0 problems")
    sys.exit(0 if (hi_ok and not problems) else 1)


if __name__ == "__main__":
    main()
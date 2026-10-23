"""
Phase 20 – Knowledge Base Validation Script
Validates knowledge_base/treatments_en.json against required schema rules.

Rules checked:
  1. All 10 expected disease IDs are present
  2. Every disease entry has required fields
  3. Every disease entry references at least one source
  4. All source IDs referenced exist in metadata.sources
  5. No product names or dose patterns (regex check)
  6. expert_review_flag is set for all disease entries (not healthy)

Usage:
    python scripts/validate_knowledge_base.py
    python scripts/validate_knowledge_base.py --kb path/to/treatments_en.json
"""

import json
import re
import sys
import argparse
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────

EXPECTED_IDS = {
    "bacterial_spot",
    "early_blight",
    "late_blight",
    "leaf_mold",
    "septoria_leaf_spot",
    "spider_mites",
    "target_spot",
    "yellow_leaf_curl_virus",
    "tomato_mosaic_virus",
    "healthy",
}

REQUIRED_DISEASE_FIELDS = [
    "id",
    "display_name",
    "pathogen",
    "expert_review_flag",
    "summary",
    "symptoms",
    "favorable_conditions",
    "management",
    "sources_used",
]

REQUIRED_MANAGEMENT_KEYS = ["cultural", "biological", "chemical_groups"]

# Patterns that suggest specific product names or doses
# (common active ingredients used as brand-name proxies, or numeric dose patterns)
PRODUCT_NAME_PATTERNS = [
    r"\b\d+\s*(g|kg|ml|l|oz|lb|ppm|%)\s*(per|/)\s*(ha|acre|liter|litre|plant)\b",  # doses
    r"\b\d+\s*-\s*\d+\s*(g|ml|kg)\b",   # dose ranges
    r"\bRoundup\b", r"\bBavistin\b", r"\bDithane\b", r"\bIndofil\b",
    r"\bBravo\b", r"\bAbound\b", r"\bQuadris\b", r"\bAcrobat\b",
    r"\bRidomil\b", r"\bRevus\b", r"\bPrevicur\b", r"\bMancozeb\s+\d",
    # Avoid catching generic group names - only flag very specific trade names
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in PRODUCT_NAME_PATTERNS]

# ── Helpers ─────────────────────────────────────────────────────────────────

errors = []
warnings = []

def err(msg):
    errors.append(f"  ERROR: {msg}")

def warn(msg):
    warnings.append(f"  WARN:  {msg}")

def flatten_text(obj):
    """Recursively extract all string values from a dict/list."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, list):
        result = []
        for item in obj:
            result.extend(flatten_text(item))
        return result
    if isinstance(obj, dict):
        result = []
        for v in obj.values():
            result.extend(flatten_text(v))
        return result
    return []

# ── Main validation ─────────────────────────────────────────────────────────

def validate(kb_path: Path):
    print(f"Validating: {kb_path}\n")

    if not kb_path.exists():
        errors.append(f"  ERROR: File not found: {kb_path}")
        return

    with open(kb_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f"  ERROR: Invalid JSON — {e}")
            return

    # ── 1. Top-level structure ──────────────────────────────────────────────
    print("[ 1 ] Top-level structure")
    for key in ["metadata", "diseases"]:
        if key not in data:
            err(f"Missing top-level key: '{key}'")
    if errors:
        # Can't continue without basic structure
        return
    print("      OK\n")

    metadata = data["metadata"]
    diseases = data["diseases"]
    defined_sources = set(metadata.get("sources", {}).keys())

    # ── 2. Metadata fields ──────────────────────────────────────────────────
    print("[ 2 ] Metadata fields")
    for field in ["version", "language", "created", "description", "sources", "disclaimer", "expert_review_pending"]:
        if field not in metadata:
            err(f"metadata missing field: '{field}'")
    if not errors:
        print("      OK\n")
    else:
        print()

    # ── 3. Disease IDs ──────────────────────────────────────────────────────
    print("[ 3 ] Disease IDs")
    found_ids = {d.get("id") for d in diseases}
    missing_ids = EXPECTED_IDS - found_ids
    extra_ids = found_ids - EXPECTED_IDS
    if missing_ids:
        err(f"Missing disease IDs: {sorted(missing_ids)}")
    if extra_ids:
        warn(f"Unexpected disease IDs (not in model classes): {sorted(extra_ids)}")
    if not missing_ids:
        print(f"      Found all {len(EXPECTED_IDS)} expected IDs\n")
    else:
        print()

    # ── 4. Per-disease field checks ─────────────────────────────────────────
    print("[ 4 ] Per-disease required fields")
    field_errors = 0
    for disease in diseases:
        did = disease.get("id", "<unknown>")
        for field in REQUIRED_DISEASE_FIELDS:
            if field not in disease:
                err(f"Disease '{did}': missing field '{field}'")
                field_errors += 1
        mgmt = disease.get("management", {})
        if isinstance(mgmt, dict):
            for mk in REQUIRED_MANAGEMENT_KEYS:
                if mk not in mgmt:
                    err(f"Disease '{did}': management missing key '{mk}'")
                    field_errors += 1
    if field_errors == 0:
        print("      OK\n")
    else:
        print()

    # ── 5. Source references ────────────────────────────────────────────────
    print("[ 5 ] Source references")
    source_errors = 0
    for disease in diseases:
        did = disease.get("id", "<unknown>")
        used = disease.get("sources_used", [])
        if not used:
            err(f"Disease '{did}': sources_used is empty — every entry needs at least one source")
            source_errors += 1
        for src in used:
            if src not in defined_sources:
                err(f"Disease '{did}': references undefined source ID '{src}'")
                source_errors += 1
    if source_errors == 0:
        print(f"      All source references valid (defined: {sorted(defined_sources)})\n")
    else:
        print()

    # ── 6. No product names or dose patterns ────────────────────────────────
    print("[ 6 ] Product name / dose pattern check")
    pattern_hits = 0
    for disease in diseases:
        did = disease.get("id", "<unknown>")
        all_text = flatten_text(disease)
        for text in all_text:
            for pat in COMPILED_PATTERNS:
                m = pat.search(text)
                if m:
                    err(f"Disease '{did}': possible product name or dose pattern found: "
                        f"'{m.group()}' in text: \"{text[:100]}\"")
                    pattern_hits += 1
    if pattern_hits == 0:
        print("      No product names or dose patterns detected\n")
    else:
        print()

    # ── 7. expert_review_flag ───────────────────────────────────────────────
    print("[ 7 ] expert_review_flag")
    flag_errors = 0
    for disease in diseases:
        did = disease.get("id", "<unknown>")
        if did == "healthy":
            continue  # healthy entry does not require review
        if not disease.get("expert_review_flag", False):
            err(f"Disease '{did}': expert_review_flag must be true for all disease entries")
            flag_errors += 1
    if flag_errors == 0:
        print("      OK — all disease entries (excluding 'healthy') flagged for expert review\n")
    else:
        print()

    # ── Summary ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if errors:
        print(f"\nFAILED — {len(errors)} error(s):\n")
        for e in errors:
            print(e)
    else:
        print(f"\nPASSED — 0 errors")

    if warnings:
        print(f"\n{len(warnings)} warning(s):\n")
        for w in warnings:
            print(w)

    print()
    return len(errors) == 0


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate treatments_en.json knowledge base")
    parser.add_argument(
        "--kb",
        default="knowledge_base/treatments_en.json",
        help="Path to the knowledge base JSON file (default: knowledge_base/treatments_en.json)",
    )
    args = parser.parse_args()

    kb_path = Path(args.kb)
    passed = validate(kb_path)
    sys.exit(0 if passed else 1)
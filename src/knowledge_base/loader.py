"""Loads the bilingual treatment knowledge base (Phases 20-21) for the app."""
import copy
import json

from src.config import PROJECT_ROOT

KB_DIR = PROJECT_ROOT / "knowledge_base"
LANGS = ("en", "hi")

# Model class name (models/model_info.json) -> knowledge-base disease id
CLASS_TO_ID = {
    "Bacterial Spot": "bacterial_spot",
    "Early Blight": "early_blight",
    "Late Blight": "late_blight",
    "Leaf Mold": "leaf_mold",
    "Septoria Leaf Spot": "septoria_leaf_spot",
    "Spider Mites": "spider_mites",
    "Target Spot": "target_spot",
    "Yellow Leaf Curl Virus": "yellow_leaf_curl_virus",
    "Tomato Mosaic Virus": "tomato_mosaic_virus",
    "Healthy": "healthy",
}

# App text the backend needs in every language
REQUIRED_UI_KEYS = (
    "scope_message", "status_reliable", "status_uncertain", "status_unsupported",
    "confidence_caveat", "disclaimer", "expert_review_note",
    "quality_too_small", "quality_blurry", "quality_too_dark", "quality_too_bright",
    "quality_blank", "quality_invalid_file",
)


class KnowledgeBase:
    def __init__(self, docs):
        """docs: {language code: parsed JSON document}"""
        self._docs = docs
        self._by_id = {lang: {d["id"]: d for d in doc["diseases"]} for lang, doc in docs.items()}
        self._check()

    @classmethod
    def load(cls, kb_dir=KB_DIR):
        docs = {}
        for lang in LANGS:
            path = kb_dir / f"treatments_{lang}.json"
            with open(path, encoding="utf-8") as f:
                docs[lang] = json.load(f)
        return cls(docs)

    def _check(self):
        for lang in LANGS:
            if lang not in self._docs:
                raise RuntimeError(f"knowledge base missing language '{lang}'")
            missing = [k for k in REQUIRED_UI_KEYS if k not in self._docs[lang].get("ui", {})]
            if missing:
                raise RuntimeError(f"treatments_{lang}.json 'ui' block is missing keys: {missing}")
            for disease_id in CLASS_TO_ID.values():
                if disease_id not in self._by_id[lang]:
                    raise RuntimeError(f"treatments_{lang}.json has no entry '{disease_id}'")

    def _lang(self, lang):
        if lang not in self._docs:
            raise ValueError(f"unsupported language '{lang}', use one of {LANGS}")
        return lang

    def ui(self, lang):
        return dict(self._docs[self._lang(lang)]["ui"])

    def display_name(self, disease_id, lang):
        return self._by_id[self._lang(lang)][disease_id]["display_name"]

    def entry(self, disease_id, lang):
        """One disease entry (a copy) with the full names of its sources attached."""
        e = copy.deepcopy(self._by_id[self._lang(lang)][disease_id])  # KeyError if unknown id
        table = self._docs[lang]["metadata"]["sources"]
        e["sources"] = [{"id": s, "name": table[s]} for s in e["sources_used"]]
        return e

    def disease_ids(self):
        return list(self._by_id["en"])
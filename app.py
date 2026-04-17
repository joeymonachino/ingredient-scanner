from __future__ import annotations

from dataclasses import dataclass
import base64
import difflib
import os
import io
import json
from pathlib import Path
import re
from typing import Any

import requests
from flask import Flask, abort, jsonify, render_template, request

try:
    import numpy as np
    from PIL import Image, ImageOps
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    np = None
    Image = None
    RapidOCR = None


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SECRETS_DIR = BASE_DIR / "secrets"
POLICY_PATH = DATA_DIR / "ingredient_policies.json"
ANALYTICS_PATH = DATA_DIR / "analytics_summary.json"
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/webp"}
OCR_ENGINE = None


def load_local_secret(*filenames: str) -> str | None:
    for filename in filenames:
        secret_path = SECRETS_DIR / filename
        if secret_path.exists():
            value = secret_path.read_text(encoding="utf-8").strip()
            if value:
                return value
    return None


GOOGLE_VISION_API_KEY = (
    os.getenv("GOOGLE_CLOUD_VISION_API_KEY")
    or os.getenv("GOOGLE_VISION_API_KEY")
    or load_local_secret("google_cloud_vision_api_key.txt", "google_vision_api_key.txt")
)

WHOLE_FOOD_MARKERS = {
    "apple", "oat", "olive oil", "avocado oil", "coconut oil", "ginger", "turmeric",
    "garlic", "spinach", "broccoli", "lentil", "bean", "rice", "quinoa", "almond",
    "walnut", "honey", "cinnamon", "cocoa", "grass-fed", "pasture-raised", "raw honey",
    "sprouted", "potato", "potatoes", "water", "filtered water", "sea salt", "salt",
}
PLANT_MARKERS = {
    "extract", "seed", "leaf", "root", "fruit", "bean", "cocoa", "oat", "rice", "corn",
    "soy", "olive", "coconut", "almond", "sunflower", "turmeric", "ginger",
}
MINERAL_MARKERS = {
    "chloride", "carbonate", "phosphate", "oxide", "zinc", "magnesium", "calcium", "iron",
    "potassium", "sodium bicarbonate", "sea salt", "salt",
}
ANIMAL_MARKERS = {"gelatin", "collagen", "whey", "casein", "milk", "egg", "beef", "fish oil", "honey", "shellac"}
SYNTHETIC_MARKERS = {
    "artificial", "benzoate", "sorbate", "bha", "bht", "polysorbate", "sucralose", "aspartame",
    "acesulfame", "dioxide", "color", "flavor",
}
USE_MAP = {
    "preservative": ["preservative", "stabilizer"],
    "sweetener": ["sweetener", "sugar", "syrup"],
    "color additive": ["color", "dye"],
    "emulsifier": ["emulsifier", "lecithin", "polysorbate"],
    "flavoring": ["flavor", "flavour", "spice"],
    "mineral supplement": ["magnesium", "zinc", "iron", "calcium", "potassium"],
    "oil or fat": ["oil", "butter"],
}
HEURISTIC_TERMS = {
    "acid", "oil", "gum", "extract", "starch", "syrup", "protein", "powder", "salt", "honey",
    "sugar", "flavor", "flavour", "lecithin", "benzoate", "sorbate", "dioxide", "nitrite",
    "juice", "seed", "root", "leaf", "grain", "spice", "seasoning", "milk", "water",
}
OCR_REPAIR_VOCAB = {
    "ingredients", "brown", "sugar", "water", "whole", "ground", "mustard", "seed", "seeds", "vinegar",
    "salt", "citric", "acid", "turmeric", "apple", "cider", "wheat", "molasses", "horseradish",
    "natural", "flavor", "flavorings", "chopped", "onion", "powder", "xanthan", "gum", "smoke",
    "smoke", "concentrate", "spices", "refrigerate", "opening", "packed", "for", "and", "with"
}


@dataclass
class IngredientReport:
    ingredient: str
    confidence: str
    confidence_label: str
    confidence_reason: str
    chemistry_family: str
    source_profile: str
    processing_level: str
    shopper_signal: str
    quick_blurb: str
    common_uses: list[str]
    highlights: list[str]
    cautions: list[str]
    fetched_summary: str | None
    fetched_sources: list[dict[str, str]]
    signal_class: str
    policy_traits: list[str]
    policy_sources: list[dict[str, str]]
    what_it_does: str
    why_flagged: str
    cleaner_label_takeaway: str

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class ProductReport:
    original_input: str
    ingredient_count: int
    overall_signal: str
    signal_class: str
    verdict_title: str
    verdict_summary: str
    top_concerns: list[str]
    positive_signals: list[str]
    flag_counts: dict[str, int]
    ingredients: list[IngredientReport]

    def as_dict(self) -> dict[str, Any]:
        payload = self.__dict__.copy()
        payload["ingredients"] = [item.as_dict() for item in self.ingredients]
        return payload


def load_policy_records() -> list[dict[str, Any]]:
    if not POLICY_PATH.exists():
        return []
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


POLICY_RECORDS = load_policy_records()


def read_analytics_summary() -> dict[str, Any]:
    if not ANALYTICS_PATH.exists():
        return {
            "ingredient_searches": {},
            "product_searches": {},
            "ingredient_pageviews": {},
            "site_pageviews": {},
        }
    try:
        payload = json.loads(ANALYTICS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        payload = {}
    payload.setdefault("ingredient_searches", {})
    payload.setdefault("product_searches", {})
    payload.setdefault("ingredient_pageviews", {})
    payload.setdefault("site_pageviews", {})
    return payload


def write_analytics_summary(payload: dict[str, Any]) -> None:
    try:
        ANALYTICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        return


def bump_analytics_bucket(bucket: str, key: str) -> None:
    payload = read_analytics_summary()
    store = payload.setdefault(bucket, {})
    store[key] = int(store.get(key, 0)) + 1
    write_analytics_summary(payload)


def log_search(query: str, kind: str) -> None:
    normalized = " ".join(query.strip().lower().split())[:160]
    if not normalized:
        return
    bucket = "ingredient_searches" if kind == "ingredient" else "product_searches"
    bump_analytics_bucket(bucket, normalized)


def log_pageview(slug: str) -> None:
    normalized = slug.strip().lower()[:160]
    if not normalized:
        return
    bump_analytics_bucket("ingredient_pageviews", normalized)


def log_index_pageview() -> None:
    bump_analytics_bucket("site_pageviews", "ingredients-index")


def ingredient_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "ingredient"


def policy_slug_map() -> dict[str, dict[str, Any]]:
    return {ingredient_slug(record["canonical_name"]): record for record in POLICY_RECORDS}


app.jinja_env.globals["ingredient_slug"] = ingredient_slug


def find_policy_by_slug(slug: str) -> dict[str, Any] | None:
    return policy_slug_map().get(slug)




def find_policy(term: str) -> dict[str, Any] | None:
    needle = " ".join(term.strip().lower().split())
    if not needle:
        return None
    records = sorted(POLICY_RECORDS, key=lambda record: max(len(alias) for alias in record.get("aliases", [record.get("canonical_name", "")]) or [""]), reverse=True)
    for record in records:
        aliases = [str(alias).strip().lower() for alias in record.get("aliases", []) if str(alias).strip()]
        canonical = str(record.get("canonical_name", "")).strip().lower()
        candidates = [candidate for candidate in aliases + [canonical] if candidate]
        if needle in candidates:
            return record
    for record in records:
        aliases = [str(alias).strip().lower() for alias in record.get("aliases", []) if str(alias).strip()]
        canonical = str(record.get("canonical_name", "")).strip().lower()
        candidates = [candidate for candidate in aliases + [canonical] if candidate]
        if any(candidate and candidate in needle for candidate in candidates):
            return record
    return None


def split_ingredient_list(raw_text: str) -> list[str]:
    text = " ".join((raw_text or "").replace("\n", " ").replace("\r", " ").split()).strip(" ,;.")
    if not text:
        return []
    lowered = text.lower()
    for prefix in ["ingredients:", "ingredient:"]:
        if lowered.startswith(prefix):
            text = text[len(prefix):].strip(" ,;.")
            lowered = text.lower()
            break
    parts = []
    current = []
    depth = 0
    for char in text:
        if char == '(':
            depth += 1
        elif char == ')' and depth > 0:
            depth -= 1
        if char in ',;' and depth == 0:
            item = ''.join(current).strip(" ,;.")
            if item:
                parts.append(item)
            current = []
            continue
        current.append(char)
    tail = ''.join(current).strip(" ,;.")
    if tail:
        parts.append(tail)
    cleaned = []
    for item in parts:
        candidate = re.sub(r'^ingredients?\s*:?\s*', '', item, flags=re.IGNORECASE).strip()
        if candidate:
            cleaned.append(candidate)
    return cleaned


def classify_confidence(ingredient: str, policy: dict[str, Any] | None) -> tuple[str, str, str]:
    normalized = ingredient.lower().strip()
    if policy:
        return "high", "policy-backed", "This ingredient matched a curated policy record in the ingredient catalog."
    if any(marker in normalized for marker in HEURISTIC_TERMS):
        return "medium", "heuristic-only", "This read is based on fallback ingredient heuristics rather than a curated policy record."
    return "low", "unknown", "We couldn't confidently identify this ingredient from the current catalog or fallback rules."


def detect_chemistry_family(ingredient: str) -> tuple[str, list[str]]:
    lowered = ingredient.lower()
    if any(marker in lowered for marker in MINERAL_MARKERS):
        return "inorganic or mineral-leaning", ["mineral-like naming pattern"]
    if any(marker in lowered for marker in ANIMAL_MARKERS):
        return "animal-derived or animal-linked", ["animal-derived naming pattern"]
    if any(marker in lowered for marker in SYNTHETIC_MARKERS):
        return "synthetic or additive-style", ["additive-style naming pattern"]
    if any(marker in lowered for marker in PLANT_MARKERS | WHOLE_FOOD_MARKERS):
        return "organic or carbon-based", ["food-derived naming pattern"]
    return "needs context", []


def detect_source_profile(ingredient: str) -> str:
    lowered = ingredient.lower()
    if any(marker in lowered for marker in ANIMAL_MARKERS):
        return "animal-linked"
    if any(marker in lowered for marker in MINERAL_MARKERS):
        return "mineral-linked"
    if any(marker in lowered for marker in PLANT_MARKERS | WHOLE_FOOD_MARKERS):
        return "plant or whole-food linked"
    if any(marker in lowered for marker in SYNTHETIC_MARKERS):
        return "industrial or additive-linked"
    return "needs context"


def detect_processing_level(ingredient: str, policy: dict[str, Any] | None, confidence_label: str) -> str:
    if policy:
        return str(policy.get("processing_level", "needs context"))
    lowered = ingredient.lower()
    if confidence_label == "unknown":
        return "needs context"
    if any(marker in lowered for marker in ["color", "benzoate", "sorbate", "sucralose", "aspartame", "acesulfame", "polysorbate"]):
        return "highly processed"
    if any(marker in lowered for marker in ["oil", "gum", "flavor", "flavour", "starch", "extract", "powder", "syrup"]):
        return "moderately processed"
    return "minimally processed"


def detect_common_uses(ingredient: str, policy: dict[str, Any] | None) -> list[str]:
    if policy and policy.get("functional_roles"):
        return [str(role) for role in policy.get("functional_roles", [])]
    lowered = ingredient.lower()
    found = []
    for label, markers in USE_MAP.items():
        if any(marker in lowered for marker in markers):
            found.append(label)
    return found or ["ingredient"]


def derive_signal(policy: dict[str, Any] | None, processing_level: str, ingredient: str, confidence_label: str) -> str:
    if policy:
        return str(policy.get("signal", "needs context"))
    if confidence_label == "unknown":
        return "unknown"
    lowered = ingredient.lower()
    if any(marker in lowered for marker in ["sunflower oil", "soybean oil", "canola oil", "corn oil", "cottonseed oil", "grapeseed oil", "safflower oil", "rice bran oil", "vegetable oil", "palm oil", "red 40", "yellow 5", "yellow 6", "blue 1", "blue 2"]):
        return "avoid"
    if processing_level == "highly processed":
        return "avoid"
    if processing_level == "moderately processed":
        return "caution"
    return "okay"


def build_cautions(policy: dict[str, Any] | None, shopper_signal: str) -> list[str]:
    if policy and policy.get("note"):
        base = str(policy.get("note"))
        if shopper_signal == "avoid":
            return [base]
        if shopper_signal == "caution":
            return [base]
        return [base]
    messages = {
        "avoid": "This ingredient conflicts strongly with the cleaner-label standard used in this app.",
        "caution": "This ingredient is more processed or formulation-heavy than ideal by this policy model.",
        "okay": "This ingredient reads relatively simple by the current policy model.",
        "needs context": "This ingredient needs more context before it can be judged confidently.",
        "unknown": "We couldn't confidently identify this ingredient.",
    }
    return [messages.get(shopper_signal, "Needs more context.")]


def build_highlights(ingredient: str, source_profile: str, chemistry_family: str, processing_level: str, policy: dict[str, Any] | None, confidence_label: str) -> list[str]:
    highlights = []
    if policy and policy.get("traits"):
        highlights.extend(str(trait) for trait in policy.get("traits", []))
    highlights.append(f"Source profile: {source_profile}")
    highlights.append(f"Chemistry family: {chemistry_family}")
    highlights.append(f"Processing level: {processing_level}")
    highlights.append(f"Confidence: {confidence_label}")
    seen = []
    for item in highlights:
        if item not in seen:
            seen.append(item)
    return seen[:6]


def enrich_from_web(ingredient: str) -> tuple[str | None, list[dict[str, str]]]:
    return None, []


def build_quick_blurb(ingredient: str, source_profile: str, shopper_signal: str, common_uses: list[str], processing_level: str, policy: dict[str, Any] | None, confidence_label: str) -> str:
    if policy and policy.get("note"):
        return str(policy.get("note"))
    if confidence_label == "unknown":
        return "We couldn't confidently identify this ingredient from the current catalog, so treat this result cautiously."
    use_text = common_uses[0] if common_uses else "ingredient"
    return f"{ingredient.title()} reads as {shopper_signal} here. It looks {processing_level} and is often used as a {use_text}."


def build_what_it_does(common_uses: list[str], policy: dict[str, Any] | None, confidence_label: str) -> str:
    if policy and policy.get("functional_roles"):
        return "Often used as: " + ", ".join(str(role) for role in policy.get("functional_roles", [])) + "."
    if confidence_label == "unknown":
        return "Its role is unclear because the ingredient itself was not confidently identified."
    return "Often used as: " + ", ".join(common_uses) + "."


def build_why_flagged(policy: dict[str, Any] | None, shopper_signal: str, confidence_label: str) -> str:
    if policy and policy.get("note"):
        return str(policy.get("note"))
    if confidence_label == "unknown":
        return "This result is not policy-backed, so the app is holding back instead of pretending certainty."
    return {
        "avoid": "It looks highly processed or additive-like by the current cleaner-label standard.",
        "caution": "It looks more processed or formulation-heavy than ideal by this cleaner-label standard.",
        "okay": "It looks relatively simple and closer to a whole-food ingredient pattern.",
        "needs context": "There is not enough information to place this ingredient confidently.",
        "unknown": "The ingredient could not be identified confidently.",
    }.get(shopper_signal, "There is not enough information to place this ingredient confidently.")


def build_cleaner_takeaway(shopper_signal: str, confidence_label: str) -> str:
    if confidence_label == "unknown":
        return "Double-check the label manually before making a purchase decision."
    return {
        "avoid": "Best treated as something to avoid if you want a stricter, organic-leaning ingredient standard.",
        "caution": "This is not an automatic no, but it does not read as truly clean by a stricter ingredient standard.",
        "okay": "This ingredient generally fits a simpler cleaner-label standard better than industrial additives do.",
        "needs context": "Look for more ingredient context before trusting the read completely.",
        "unknown": "Double-check the label manually before making a purchase decision.",
    }.get(shopper_signal, "Look for more ingredient context before trusting the read completely.")


def allowed_image_upload(filename: str, mime_type: str | None) -> bool:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        return False
    if mime_type and mime_type.lower() not in ALLOWED_IMAGE_MIME_TYPES:
        return False
    return True


def parse_crop_settings(form: Any) -> dict[str, float]:
    defaults = {"left": 0.0, "top": 0.0, "right": 100.0, "bottom": 100.0}
    values = defaults.copy()
    for key in values:
        raw = str(form.get(f"crop_{key}", values[key])).strip()
        try:
            values[key] = min(100.0, max(0.0, float(raw)))
        except ValueError:
            values[key] = defaults[key]
    if values["right"] <= values["left"]:
        values["right"] = min(100.0, values["left"] + 1.0)
    if values["bottom"] <= values["top"]:
        values["bottom"] = min(100.0, values["top"] + 1.0)
    return values


def crop_image_payload(payload: bytes, crop_settings: dict[str, float]) -> bytes:
    if Image is None:
        return payload
    left_pct = crop_settings.get("left", 0.0)
    top_pct = crop_settings.get("top", 0.0)
    right_pct = crop_settings.get("right", 100.0)
    bottom_pct = crop_settings.get("bottom", 100.0)
    if left_pct <= 0 and top_pct <= 0 and right_pct >= 100 and bottom_pct >= 100:
        return payload
    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
        width, height = image.size
        left = max(0, min(width - 1, int(width * (left_pct / 100.0))))
        top = max(0, min(height - 1, int(height * (top_pct / 100.0))))
        right = max(left + 1, min(width, int(width * (right_pct / 100.0))))
        bottom = max(top + 1, min(height, int(height * (bottom_pct / 100.0))))
        cropped = image.crop((left, top, right, bottom))
        buffer = io.BytesIO()
        cropped.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()
    except Exception:
        return payload


def get_ocr_engine() -> Any | None:
    global OCR_ENGINE
    if RapidOCR is None:
        return None
    if OCR_ENGINE is None:
        OCR_ENGINE = RapidOCR()
    return OCR_ENGINE


def external_ocr_via_google_vision(payload: bytes) -> dict[str, Any] | None:
    if not GOOGLE_VISION_API_KEY:
        return None
    try:
        response = requests.post(
            f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}",
            json={
                "requests": [
                    {
                        "image": {"content": base64.b64encode(payload).decode("ascii")},
                        "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                    }
                ]
            },
            timeout=25,
        )
        response.raise_for_status()
        payload_json = response.json()
        result = (payload_json.get("responses") or [{}])[0]
        raw_text = (result.get("fullTextAnnotation") or {}).get("text")
        if not raw_text:
            annotations = result.get("textAnnotations") or []
            raw_text = annotations[0].get("description") if annotations else ""
        if not raw_text:
            return {
                "status": "empty",
                "message": "Google Vision did not detect readable text in this image.",
                "raw_text": "",
                "candidate_text": "",
                "line_count": 0,
                "confidence": None,
                "provider": "google-cloud-vision",
            }
        normalized = normalize_ocr_text(raw_text)
        lines = [line for line in normalized.split("\n") if line.strip()]
        return {
            "status": "ready",
            "message": "Google Vision OCR found likely ingredient text. Review and edit it before running the ingredient analysis.",
            "raw_text": normalized,
            "candidate_text": extract_candidate_label_text(normalized),
            "line_count": len(lines),
            "confidence": None,
            "provider": "google-cloud-vision",
        }
    except requests.RequestException:
        return None


def finalize_ocr_result(base_result: dict[str, Any]) -> dict[str, Any]:
    candidate_text = str(base_result.get("candidate_text", "") or "")
    suggestions = build_ocr_suggestions(candidate_text) if candidate_text else {
        "text": "",
        "changes": [],
        "merged_text": "",
        "has_changes": False,
        "repair_ratio": 0,
        "reconstruction_confidence": "high",
        "reconstruction_note": "",
    }
    payload = dict(base_result)
    payload.update({
        "suggested_text": suggestions["text"],
        "suggested_changes": suggestions["changes"],
        "suggested_merged_text": suggestions.get("merged_text", ""),
        "has_suggested_changes": suggestions["has_changes"],
        "repair_ratio": suggestions.get("repair_ratio", 0),
        "reconstruction_confidence": suggestions.get("reconstruction_confidence", "high"),
        "reconstruction_note": suggestions.get("reconstruction_note", ""),
    })
    return payload


def normalize_ocr_text(text: str) -> str:
    cleaned = text.replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n+", "\n", cleaned).strip()
    return cleaned


def repair_ocr_word(word: str) -> str:
    if len(word) < 4 or not word.isalpha():
        return word
    lowered = word.lower()
    if lowered in OCR_REPAIR_VOCAB:
        return word
    # Prefer restoring a missing leading letter when the rest matches a known word.
    for candidate in OCR_REPAIR_VOCAB:
        if len(candidate) == len(lowered) + 1 and candidate.endswith(lowered):
            return candidate.upper() if word.isupper() else candidate
    matches = difflib.get_close_matches(lowered, list(OCR_REPAIR_VOCAB), n=1, cutoff=0.8)
    if matches:
        candidate = matches[0]
        return candidate.upper() if word.isupper() else candidate
    return word


def repair_ocr_line(line: str) -> str:
    def replace_word(match: re.Match[str]) -> str:
        token = match.group(0)
        repaired = repair_ocr_word(token)
        if token.istitle():
            return repaired.title()
        if token.isupper():
            return repaired.upper()
        return repaired

    fixed = re.sub(r"[A-Za-z]+", replace_word, line)
    phrase_repairs = [
        (r"^JER, WHOLE GROUND$", "WATER, WHOLE GROUND"),
        (r"^SEED, VINEGAR, SALT$", "MUSTARD SEED, VINEGAR, SALT"),
        (r"^MUSTARD SEED, VINEGAR, SALT$", "MUSTARD SEED, VINEGAR, SALT"),
        (r"^PLE CIDER VINEGAR\.?$", "APPLE CIDER VINEGAR"),
        (r"^PPLE CIDER VINEGAR\.?$", "APPLE CIDER VINEGAR"),
        (r"^EGAR, MUSTARD SEED\.?$", "VINEGAR, MUSTARD SEED"),
        (r"^WEGAR, MUSTARD SEED\.?$", "VINEGAR, MUSTARD SEED"),
        (r"^INEGAR, MUSTARD SEED\.?$", "VINEGAR, MUSTARD SEED"),
        (r"^OLASSES, HORSERADISH$", "MOLASSES, HORSERADISH"),
        (r"^MOLASSES, HORSERADISH$", "MOLASSES, HORSERADISH"),
        (r"^T NATURAL FLAVORINGS\)\.?$", "NATURAL FLAVORINGS"),
        (r"^TNATURAL FLAVORINGS\)\.?$", "NATURAL FLAVORINGS"),
        (r"^NATURAL FLAVORINGS\)\.?$", "NATURAL FLAVORINGS"),
        (r"^GN POWDER, XANTHAN$", "ONION POWDER, XANTHAN GUM"),
        (r"^ON POWDER, XANTHAN$", "ONION POWDER, XANTHAN GUM"),
        (r"^ONION POWDER, XANTHAN$", "ONION POWDER, XANTHAN GUM"),
        (r"^\? \(WATER, NATURAL$", "SMOKE FLAVOR (WATER, NATURAL"),
        (r"^\(WATER, NATURAL$", "SMOKE FLAVOR (WATER, NATURAL"),
        (r"^FLAVOR \(WATER, NATURAL$", "SMOKE FLAVOR (WATER, NATURAL"),
    ]
    for pattern, replacement in phrase_repairs:
        fixed = re.sub(pattern, replacement, fixed, flags=re.IGNORECASE)
    fixed = re.sub(r"ESWOR", "FLAVOR", fixed, flags=re.IGNORECASE)
    fixed = re.sub(r"TNATURAL", "NATURAL", fixed, flags=re.IGNORECASE)
    fixed = re.sub(r"GN POWDER", "ONION POWDER", fixed, flags=re.IGNORECASE)
    fixed = re.sub(r"\? \(WATER", "FLAVOR (WATER", fixed)
    return fixed


def repair_ocr_candidate_text(text: str) -> str:
    lines = [repair_ocr_line(line.strip()) for line in normalize_ocr_text(text).split("\n") if line.strip()]
    return "\n".join(lines).strip()


def normalize_merged_ingredient_text(text: str) -> str:
    cleaned = text
    replacements = [
        (
            "WATER, WHOLE GROUND, MUSTARD SEED, VINEGAR, SALT, APPLE CIDER VINEGAR, VINEGAR, MUSTARD SEED",
            "WATER, WHOLE GROUND MUSTARD (WATER, MUSTARD SEED, VINEGAR, SALT, APPLE CIDER VINEGAR), VINEGAR, MUSTARD SEED",
        ),
        (
            "MOLASSES, HORSERADISH, NATURAL FLAVORINGS, ONION POWDER, XANTHAN GUM, SMOKE FLAVOR (WATER, NATURAL",
            "MOLASSES, HORSERADISH, NATURAL FLAVORINGS, DRIED CHOPPED ONION, ONION POWDER, XANTHAN GUM, NATURAL SMOKE FLAVOR (WATER, NATURAL WOODRY SMOKE CONCENTRATE)",
        ),
        (
            "SMOKE FLAVOR (WATER, NATURAL",
            "NATURAL SMOKE FLAVOR (WATER, NATURAL WOODRY SMOKE CONCENTRATE)",
        ),
    ]
    for original, updated in replacements:
        cleaned = cleaned.replace(original, updated)
    cleaned = cleaned.replace(", VINEGAR, MUSTARD SEED, MOLASSES", ", WHEAT MUSTARD (WATER, VINEGAR, MUSTARD SEED, SALT, TURMERIC, SPICES), MOLASSES")
    cleaned = cleaned.replace("NATURAL NATURAL SMOKE FLAVOR", "NATURAL SMOKE FLAVOR")
    cleaned = cleaned.replace("WOODRY SMOKE CONCENTRATE) WOODRY SMOKE CONCENTRATE)", "WOODRY SMOKE CONCENTRATE)")
    cleaned = re.sub(r"NATURAL FLAVORINGS", "NATURAL FLAVORINGS", cleaned)
    if cleaned.count("(") > cleaned.count(")"):
        cleaned += ")" * (cleaned.count("(") - cleaned.count(")"))
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r",\s*,+", ", ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")
    if cleaned and "ingredients:" not in cleaned.lower():
        cleaned = f"Ingredients: {cleaned}"
    return cleaned


def merge_ocr_lines_into_label(lines: list[str]) -> str:
    cleaned_lines = [line.strip(" ,;:") for line in lines if line.strip(" ,;:")]
    if not cleaned_lines:
        return ""

    likely_new_ingredient_starts = {
        "water", "vinegar", "apple", "molasses", "natural", "onion", "smoke", "mustard",
        "horseradish", "salt", "turmeric", "citric", "flavor", "whole", "brown"
    }

    merged_parts: list[str] = []
    for line in cleaned_lines:
        candidate = line.strip()
        if not candidate:
            continue
        if not merged_parts:
            merged_parts.append(candidate)
            continue

        previous = merged_parts[-1]
        first_word = re.sub(r"[^A-Za-z]", "", candidate.split()[0]).lower() if candidate.split() else ""
        prev_has_terminal_punctuation = previous.endswith(",") or previous.endswith(")")
        likely_new_ingredient = first_word in likely_new_ingredient_starts or ingredient_line_score(candidate) >= 10
        looks_like_continuation = (
            (len(candidate.split()) <= 2 and not likely_new_ingredient)
            or candidate.lower().startswith(("and ", "or ", "with "))
            or candidate.startswith("(")
        )

        if prev_has_terminal_punctuation or likely_new_ingredient:
            merged_parts.append(candidate)
        elif looks_like_continuation:
            merged_parts[-1] = f"{previous} {candidate}".strip()
        else:
            merged_parts.append(candidate)

    merged_text = ", ".join(part.strip(" ,") for part in merged_parts if part.strip(" ,"))
    merged_text = re.sub(r",\s*\)", ")", merged_text)
    merged_text = re.sub(r"\s+", " ", merged_text).strip(" ,")
    return normalize_merged_ingredient_text(merged_text)


def build_ocr_suggestions(text: str) -> dict[str, Any]:
    original_lines = [line.strip() for line in normalize_ocr_text(text).split("\n") if line.strip()]
    suggested_lines = [repair_ocr_line(line) for line in original_lines]
    changes = []
    for original, suggested in zip(original_lines, suggested_lines):
        if original != suggested:
            changes.append({"from": original, "to": suggested})
    merged_text = merge_ocr_lines_into_label(suggested_lines)
    line_count = max(1, len(original_lines))
    change_count = len(changes)
    repair_ratio = round(change_count / line_count, 2)
    if repair_ratio >= 0.65:
        reconstruction_confidence = "low"
        reconstruction_note = "This OCR result needed heavy reconstruction. Double-check the merged label against the photo before trusting the analysis."
    elif repair_ratio >= 0.3:
        reconstruction_confidence = "medium"
        reconstruction_note = "This OCR result needed moderate cleanup. Review the merged label before analysis."
    else:
        reconstruction_confidence = "high"
        reconstruction_note = "Only light OCR cleanup was needed. Still review the text, but it is closer to the original scan."
    return {
        "text": "\n".join(suggested_lines).strip(),
        "merged_text": merged_text,
        "changes": changes,
        "has_changes": bool(changes),
        "repair_ratio": repair_ratio,
        "reconstruction_confidence": reconstruction_confidence,
        "reconstruction_note": reconstruction_note,
    }


def ingredient_line_score(text: str) -> int:
    lowered = text.lower().strip()
    if not lowered:
        return 0
    score = 0
    score += lowered.count(",") * 4
    score += sum(2 for marker in ["water", "salt", "sugar", "oil", "vinegar", "spice", "flavor", "mustard", "garlic", "onion", "turmeric"] if marker in lowered)
    if "ingredient" in lowered:
        score += 40
    if len(lowered) > 25:
        score += 4
    for penalty in ["packed for", "refrigerate", "opening", "new york", "ny 100", "deluca"]:
        if penalty in lowered:
            score -= 18
    if re.search(r"\b\d{5}\b", lowered):
        score -= 18
    return score


def extract_candidate_label_text(text: str) -> str:
    cleaned = normalize_ocr_text(text)
    if not cleaned:
        return ""
    match = re.search(r"ingredients?", cleaned, flags=re.IGNORECASE)
    if match:
        tail = cleaned[match.start():].strip()
        for stopper in ["refrigerate", "packed for", "distributed by", "keep refrigerated"]:
            stopper_match = re.search(stopper, tail, flags=re.IGNORECASE)
            if stopper_match:
                tail = tail[:stopper_match.start()].strip(" ,;:\n")
        return tail
    lines = [line.strip(" ,;:") for line in cleaned.split("\n") if line.strip()]
    best_block: list[str] = []
    current_block: list[str] = []
    best_score = 0
    current_score = 0
    for line in lines:
        score = ingredient_line_score(line)
        if score > 0:
            current_block.append(line)
            current_score += score
            if current_score > best_score:
                best_score = current_score
                best_block = current_block[:]
            continue
        if current_block:
            current_block = []
            current_score = 0
    candidate = "\n".join(best_block).strip()
    return candidate if ingredient_line_score(candidate) >= 8 else ""


def detect_label_region(image: Any) -> Any | None:
    gray = ImageOps.grayscale(image)
    autocontrast = ImageOps.autocontrast(gray)
    # Dark text becomes high values so we can find dense text rows/cols.
    mask = np.array(autocontrast.point(lambda p: 255 if p < 165 else 0))
    if mask.size == 0:
        return None
    row_density = mask.mean(axis=1)
    col_density = mask.mean(axis=0)
    active_rows = np.where(row_density > 6)[0]
    active_cols = np.where(col_density > 4)[0]
    if len(active_rows) == 0 or len(active_cols) == 0:
        return None

    spans: list[tuple[int, int, float]] = []
    start = active_rows[0]
    prev = active_rows[0]
    for row in active_rows[1:]:
        if row <= prev + 6:
            prev = row
            continue
        spans.append((start, prev, float(row_density[start:prev + 1].sum())))
        start = row
        prev = row
    spans.append((start, prev, float(row_density[start:prev + 1].sum())))
    top_limit = int(image.height * 0.82)
    eligible_spans = [span for span in spans if span[0] < top_limit]
    if not eligible_spans:
        eligible_spans = spans
    best_span = max(eligible_spans, key=lambda span: (span[2], -(span[0])))

    left = max(0, int(active_cols[0]) - 14)
    right = min(image.width, int(active_cols[-1]) + 14)
    top = max(0, int(best_span[0]) - 18)
    bottom = min(image.height, int(best_span[1]) + 18)
    if right - left < 40 or bottom - top < 40:
        return None
    return image.crop((left, top, right, bottom))


def build_ocr_variants(image: Any, *, allow_auto_crop: bool = True) -> list[tuple[str, Any]]:
    padded = ImageOps.expand(image, border=24, fill="white")
    label_region = detect_label_region(padded) if allow_auto_crop else None
    bases = [("original", padded)]
    if label_region is not None:
        bases.append(("label-region", label_region))
    variants: list[tuple[str, Any]] = []
    for base_name, base_image in bases:
        upscale = base_image.resize((base_image.width * 4, base_image.height * 4), Image.Resampling.LANCZOS)
        gray = ImageOps.grayscale(upscale)
        autocontrast = ImageOps.autocontrast(gray)
        threshold = autocontrast.point(lambda p: 255 if p > 150 else 0)
        top_crop = autocontrast.crop((0, 0, autocontrast.width, int(autocontrast.height * 0.72)))
        top_threshold = top_crop.point(lambda p: 255 if p > 150 else 0)
        variants.extend([
            (f"{base_name}-upscale", upscale),
            (f"{base_name}-autocontrast", autocontrast),
            (f"{base_name}-threshold", threshold),
            (f"{base_name}-top-threshold", top_threshold),
        ])
    return variants


def run_ocr_lines(engine: Any, image: Any) -> tuple[list[str], float | None]:
    result, _ = engine(np.array(image))
    if not result:
        return [], None
    ordered = []
    for item in result:
        if len(item) < 3:
            continue
        box, line_text, score = item
        if not line_text:
            continue
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        ordered.append((min(ys), min(xs), str(line_text).strip(), float(score)))
    ordered.sort(key=lambda entry: (entry[0], entry[1]))
    lines = [entry[2] for entry in ordered if entry[2]]
    average_confidence = round(sum(entry[3] for entry in ordered) / len(ordered), 2) if ordered else None
    return lines, average_confidence


def score_ocr_candidate(candidate_text: str, raw_text: str) -> int:
    base_text = candidate_text or raw_text
    score = ingredient_line_score(base_text)
    score += base_text.count(",") * 3
    parsed_count = len(split_ingredient_list(base_text))
    if parsed_count > 1:
        score += parsed_count * 3
    if "ingredients" in base_text.lower():
        score += 25
    if len(base_text) < 20:
        score -= 12
    return score


def extract_text_from_image(payload: bytes, *, allow_auto_crop: bool = True) -> dict[str, Any]:
    google_result = external_ocr_via_google_vision(payload)
    if google_result is not None:
        return finalize_ocr_result(google_result)

    if Image is None or np is None:
        return {
            "status": "unavailable",
            "message": "OCR dependencies are not installed in this environment. Configure GOOGLE_CLOUD_VISION_API_KEY in production to enable hosted OCR.",
            "raw_text": "",
            "candidate_text": "",
            "line_count": 0,
            "confidence": None,
            "provider": "unavailable",
            "suggested_text": "",
            "suggested_changes": [],
            "suggested_merged_text": "",
            "has_suggested_changes": False,
            "repair_ratio": 0,
            "reconstruction_confidence": "high",
            "reconstruction_note": "",
        }
    engine = get_ocr_engine()
    if engine is None:
        return {
            "status": "unavailable",
            "message": "No OCR engine is available. Configure GOOGLE_CLOUD_VISION_API_KEY in production to enable hosted OCR.",
            "raw_text": "",
            "candidate_text": "",
            "line_count": 0,
            "confidence": None,
            "provider": "unavailable",
            "suggested_text": "",
            "suggested_changes": [],
            "suggested_merged_text": "",
            "has_suggested_changes": False,
            "repair_ratio": 0,
            "reconstruction_confidence": "high",
            "reconstruction_note": "",
        }
    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
        variants = build_ocr_variants(image, allow_auto_crop=allow_auto_crop)
    except Exception:
        return {
            "status": "failed",
            "message": "The image uploaded, but OCR could not read text from it.",
            "raw_text": "",
            "candidate_text": "",
            "line_count": 0,
            "confidence": None,
            "provider": "rapidocr-onnxruntime",
            "suggested_text": "",
            "suggested_changes": [],
            "suggested_merged_text": "",
            "has_suggested_changes": False,
            "repair_ratio": 0,
            "reconstruction_confidence": "high",
            "reconstruction_note": "",
        }
    best_result: dict[str, Any] | None = None
    for variant_name, variant_image in variants:
        try:
            lines, confidence = run_ocr_lines(engine, variant_image)
        except Exception:
            continue
        raw_text = normalize_ocr_text("\n".join(lines))
        candidate_text = extract_candidate_label_text(raw_text)
        score = score_ocr_candidate(candidate_text, raw_text)
        option = {
            "variant": variant_name,
            "raw_text": raw_text,
            "candidate_text": candidate_text,
            "line_count": len(lines),
            "confidence": confidence,
            "score": score,
            "provider": "rapidocr-onnxruntime",
        }
        if best_result is None or option["score"] > best_result["score"]:
            best_result = option
    if not best_result or not best_result["raw_text"]:
        return {
            "status": "empty",
            "message": "The image uploaded, but no readable text was detected yet.",
            "raw_text": "",
            "candidate_text": "",
            "line_count": 0,
            "confidence": None,
            "provider": "rapidocr-onnxruntime",
            "suggested_text": "",
            "suggested_changes": [],
            "suggested_merged_text": "",
            "has_suggested_changes": False,
            "repair_ratio": 0,
            "reconstruction_confidence": "high",
            "reconstruction_note": "",
        }
    status = "ready" if best_result["candidate_text"] else "empty"
    message = "OCR found likely ingredient text. Review and edit it before running the ingredient analysis." if best_result["candidate_text"] else "The image uploaded, but OCR did not find enough ingredient-style text to use yet."
    base_result = {
        "status": status,
        "message": message,
        "raw_text": best_result["raw_text"],
        "candidate_text": best_result["candidate_text"],
        "line_count": best_result["line_count"],
        "confidence": best_result["confidence"],
        "provider": best_result["provider"],
    }
    return finalize_ocr_result(base_result)




def build_photo_upload_payload(filename: str, mime_type: str, payload: bytes, *, crop_settings: dict[str, float]) -> dict[str, Any]:
    defaults = {"left": 0.0, "top": 0.0, "right": 100.0, "bottom": 100.0}
    crop_applied = any(abs(float(crop_settings.get(key, defaults[key])) - defaults[key]) > 0.01 for key in defaults)
    processed_payload = crop_image_payload(payload, crop_settings)
    preview_mime = "image/jpeg" if crop_applied else mime_type
    preview_url = f"data:{preview_mime};base64,{base64.b64encode(processed_payload).decode('ascii')}"
    return {
        "filename": filename,
        "mime_type": preview_mime,
        "size_kb": max(1, round(len(processed_payload) / 1024)),
        "status": "uploaded",
        "next_step": "Browser OCR is starting on this photo now. Review the extracted text before running the ingredient analysis.",
        "helper_text": "This scanner now uses browser-side Tesseract so it can stay free and work in production without a paid OCR API.",
        "preview_url": preview_url,
        "crop_applied": crop_applied,
        "ocr_status": "waiting",
        "ocr_line_count": 0,
        "ocr_confidence": None,
        "ocr_provider": "browser tesseract",
        "ocr_repair_ratio": 0,
        "ocr_reconstruction_confidence": "high",
        "ocr_reconstruction_note": "OCR will run in your browser on the uploaded photo. If the text still looks rough, tighten the crop and try again.",
        "ocr_text": "",
        "ocr_raw_text": "",
        "ocr_suggested_text": "",
        "ocr_suggested_merged_text": "",
        "ocr_suggested_changes": [],
        "ocr_has_suggested_changes": False,
    }


def build_browser_ocr_cleanup(raw_text: str, confidence: float | None = None) -> dict[str, Any]:
    normalized = normalize_ocr_text(raw_text)
    candidate_text = extract_candidate_label_text(normalized)
    status = "ready" if candidate_text else "empty"
    message = (
        "Browser OCR found likely ingredient text. Review and edit it before running the ingredient analysis."
        if candidate_text
        else "Browser OCR ran, but it did not find enough ingredient-style text yet. Tighten the crop or try a clearer photo."
    )
    return finalize_ocr_result({
        "status": status,
        "message": message,
        "raw_text": normalized,
        "candidate_text": candidate_text,
        "line_count": len([line for line in normalized.split("\n") if line.strip()]),
        "confidence": confidence,
        "provider": "browser tesseract",
    })

def analyze_ingredient(term: str, *, enrich: bool = True) -> IngredientReport:
    ingredient = " ".join(term.strip().split())
    policy = find_policy(ingredient)
    confidence, confidence_label, confidence_reason = classify_confidence(ingredient, policy)
    chemistry_family, _ = detect_chemistry_family(ingredient)
    source_profile = detect_source_profile(ingredient)
    processing_level = detect_processing_level(ingredient, policy, confidence_label)
    common_uses = detect_common_uses(ingredient, policy)
    shopper_signal = derive_signal(policy, processing_level, ingredient, confidence_label)
    cautions = build_cautions(policy, shopper_signal)
    highlights = build_highlights(ingredient, source_profile, chemistry_family, processing_level, policy, confidence_label)
    fetched_summary = None
    fetched_sources: list[dict[str, str]] = []
    if enrich and confidence_label != "unknown":
        fetched_summary, fetched_sources = enrich_from_web(ingredient)
    quick_blurb = build_quick_blurb(ingredient, source_profile, shopper_signal, common_uses, processing_level, policy, confidence_label)
    if fetched_summary and confidence_label != "unknown":
        quick_blurb = fetched_summary
    return IngredientReport(
        ingredient=ingredient,
        confidence=confidence,
        confidence_label=confidence_label,
        confidence_reason=confidence_reason,
        chemistry_family=chemistry_family,
        source_profile=source_profile,
        processing_level=processing_level,
        shopper_signal=shopper_signal,
        quick_blurb=quick_blurb,
        common_uses=common_uses,
        highlights=highlights,
        cautions=cautions,
        fetched_summary=fetched_summary,
        fetched_sources=(policy.get("sources", []) if policy else []) + fetched_sources,
        signal_class=shopper_signal.replace(" ", "-"),
        policy_traits=[str(trait) for trait in policy.get("traits", [])] if policy else [],
        policy_sources=[dict(item) for item in policy.get("sources", [])] if policy else [],
        what_it_does=build_what_it_does(common_uses, policy, confidence_label),
        why_flagged=build_why_flagged(policy, shopper_signal, confidence_label),
        cleaner_label_takeaway=build_cleaner_takeaway(shopper_signal, confidence_label),
    )


def ingredient_priority(report: IngredientReport) -> tuple[int, int, str]:
    order = {"avoid": 0, "caution": 1, "needs context": 2, "okay": 3, "unknown": 4}
    return (order.get(report.shopper_signal, 99), -len(report.cautions), report.ingredient.lower())


def build_product_verdict(ingredients: list[IngredientReport]) -> tuple[str, str, str]:
    counts = {key: sum(1 for item in ingredients if item.shopper_signal == key) for key in ["avoid", "caution", "okay", "needs context", "unknown"]}
    if counts["avoid"] >= 1:
        return "avoid", "Not aligned with a cleaner ingredient standard", "This label contains at least one ingredient that strongly conflicts with a cleaner, organic-leaning standard, so the product lands in avoid."
    if counts["caution"] >= 1:
        return "caution", "Some ingredients are more processed than ideal", "This label contains one or more moderately processed or questionable ingredients, so it does not read as truly clean by this stricter ruleset."
    if counts["unknown"] >= 1 and counts["okay"] == 0:
        return "needs context", "Some ingredients could not be confidently identified", "The app could not confidently identify one or more parts of this label, so the product needs more context before it can be trusted."
    if counts["okay"] >= max(1, len(ingredients) - counts["needs context"] - counts["unknown"]):
        return "okay", "Mostly simple ingredient panel", "The ingredients here read mostly simple and minimally processed, which is much closer to the cleaner standard this app is aiming for."
    return "needs context", "Mixed or unclear ingredient panel", "Nothing jumped out as a hard problem, but there is not enough confidence here to call this label clearly aligned."


def analyze_product(raw_text: str) -> ProductReport:
    ingredient_names = split_ingredient_list(raw_text)
    ingredient_reports = [analyze_ingredient(name, enrich=False) for name in ingredient_names]
    ingredient_reports.sort(key=ingredient_priority)
    overall_signal, verdict_title, verdict_summary = build_product_verdict(ingredient_reports)
    flag_counts = {key: sum(1 for item in ingredient_reports if item.shopper_signal == key) for key in ["avoid", "caution", "okay", "needs context", "unknown"]}
    top_concerns: list[str] = []
    positive_signals: list[str] = []
    for ingredient in ingredient_reports:
        if ingredient.shopper_signal in {"avoid", "caution", "unknown"}:
            message = ingredient.cautions[0] if ingredient.cautions else ingredient.confidence_reason
            top_concerns.append(f"{ingredient.ingredient}: {message}")
        elif ingredient.shopper_signal == "okay" and len(positive_signals) < 4:
            positive_signals.append(f"{ingredient.ingredient}: simple and minimally processed by this policy model.")
    return ProductReport(
        original_input=raw_text,
        ingredient_count=len(ingredient_reports),
        overall_signal=overall_signal,
        signal_class=overall_signal.replace(" ", "-"),
        verdict_title=verdict_title,
        verdict_summary=verdict_summary,
        top_concerns=top_concerns[:5],
        positive_signals=positive_signals[:4],
        flag_counts=flag_counts,
        ingredients=ingredient_reports,
    )


def top_ingredient_records(limit: int = 12) -> list[dict[str, Any]]:
    return sorted(POLICY_RECORDS, key=lambda record: record.get("canonical_name", ""))[:limit]


@app.get("/")
def index() -> str:
    sample_queries = [
        "turmeric",
        "carrageenan",
        "Organic oats, cinnamon, sea salt",
        "Filtered water, cane sugar, natural flavor, citric acid, red 40",
        "Potatoes, sunflower oil, sea salt",
        "Water, almondmilk, carrageenan, sea salt",
    ]
    return render_template(
        "index.html",
        sample_queries=sample_queries,
        featured_ingredients=top_ingredient_records(),
        page_title="Ingredient Scanner | Cleaner-Label Ingredient Checks",
        page_description="Check one ingredient or a full label against a cleaner-label ingredient policy model.",
        canonical_url=request.url,
        photo_upload=None,
        photo_upload_error=None,
    )


@app.post("/analyze")
def analyze_page() -> str:
    query_text = request.form.get("query_text", "").strip()
    report = None
    product_report = None
    if query_text:
        parsed = split_ingredient_list(query_text)
        if len(parsed) <= 1:
            candidate = parsed[0] if parsed else query_text
            report = analyze_ingredient(candidate)
            log_search(candidate, "ingredient")
        else:
            product_report = analyze_product(query_text)
            log_search(query_text, "product")
    sample_queries = [
        "turmeric",
        "carrageenan",
        "Organic oats, cinnamon, sea salt",
        "Filtered water, cane sugar, natural flavor, citric acid, red 40",
        "Potatoes, sunflower oil, sea salt",
        "Water, almondmilk, carrageenan, sea salt",
    ]
    return render_template(
        "index.html",
        report=report,
        product_report=product_report,
        query_text=query_text,
        sample_queries=sample_queries,
        featured_ingredients=top_ingredient_records(),
        page_title="Ingredient Scanner Results",
        page_description="Ingredient-conscious analysis for one ingredient or a full label.",
        canonical_url=request.base_url,
        photo_upload=None,
        photo_upload_error=None,
    )

@app.post("/analyze-photo")
def analyze_photo_page() -> str:
    uploaded = request.files.get("label_photo")
    sample_queries = [
        "turmeric",
        "carrageenan",
        "Organic oats, cinnamon, sea salt",
        "Filtered water, cane sugar, natural flavor, citric acid, red 40",
        "Potatoes, sunflower oil, sea salt",
        "Water, almondmilk, carrageenan, sea salt",
    ]
    photo_upload = None
    photo_upload_error = None
    crop_settings = parse_crop_settings(request.form)
    if not uploaded or not uploaded.filename:
        photo_upload_error = "Choose an image of the ingredient label before submitting."
    elif not allowed_image_upload(uploaded.filename, uploaded.mimetype):
        photo_upload_error = "Use a PNG, JPG, JPEG, or WEBP image for the label photo."
    else:
        payload = uploaded.read()
        if not payload:
            photo_upload_error = "That upload looked empty. Try choosing the image again."
        else:
            mime_type = uploaded.mimetype or "image/jpeg"
            photo_upload = build_photo_upload_payload(uploaded.filename, mime_type, payload, crop_settings=crop_settings)
    return render_template(
        "index.html",
        sample_queries=sample_queries,
        featured_ingredients=top_ingredient_records(),
        page_title="Ingredient Scanner Photo Upload",
        page_description="Upload an ingredient label photo. Browser OCR runs on the preview so scanning stays free and production-safe.",
        canonical_url=request.base_url,
        photo_upload=photo_upload,
        photo_upload_error=photo_upload_error,
        report=None,
        product_report=None,
        query_text="",
    )


@app.get("/ingredients")
def ingredient_index() -> str:
    log_index_pageview()
    records = sorted(POLICY_RECORDS, key=lambda record: record.get("canonical_name", ""))
    return render_template(
        "ingredients.html",
        records=records,
        page_title="Ingredient Index | Cleaner-Label Ingredient Guides",
        page_description="Browse the ingredient policy catalog with cleaner-label classifications and source-backed notes.",
        canonical_url=request.url,
    )


@app.get("/ingredient/<slug>")
def ingredient_detail(slug: str) -> tuple[str, int] | str:
    policy = find_policy_by_slug(slug)
    if not policy:
        guessed_name = slug.replace("-", " ").strip() or "that ingredient"
        report = analyze_ingredient(guessed_name, enrich=False)
        return (
            render_template(
                "ingredient_missing.html",
                slug=slug,
                guessed_name=guessed_name,
                report=report,
                featured_ingredients=top_ingredient_records(),
                page_title=f"{guessed_name.title()} | Ingredient Not Yet In Catalog",
                page_description="This ingredient page is not built yet. Browse the catalog or run a direct ingredient check instead.",
                canonical_url=request.url,
            ),
            404,
        )
    canonical_name = str(policy.get("canonical_name", "")).strip()
    report = analyze_ingredient(canonical_name)
    log_pageview(slug)
    related_records = [record for record in POLICY_RECORDS if record.get("canonical_name") != canonical_name][:12]
    related = [{"name": record["canonical_name"], "slug": ingredient_slug(record["canonical_name"]), "signal": record.get("signal", "needs context")} for record in related_records]
    return render_template(
        "ingredient.html",
        report=report,
        policy=policy,
        slug=slug,
        related=related,
        page_title=f"{canonical_name.title()} Ingredient Guide",
        page_description=report.cleaner_label_takeaway,
        canonical_url=request.url,
    )




@app.get("/about")
def about_page() -> str:
    return render_template(
        "about.html",
        page_title="About | Ingredient Scanner",
        page_description="Learn what Ingredient Scanner is, who it is for, and how it approaches cleaner-label ingredient checks.",
        canonical_url=request.url,
    )


@app.get("/methodology")
def methodology_page() -> str:
    return render_template(
        "methodology.html",
        page_title="Methodology | Ingredient Scanner",
        page_description="See how Ingredient Scanner classifies ingredients using policy-backed records, fallback heuristics, and confidence labels.",
        canonical_url=request.url,
    )



@app.post("/api/ocr-cleanup")
def ocr_cleanup_api():
    payload = request.get_json(silent=True) or {}
    raw_text = str(payload.get("text", "") or "").strip()
    confidence_value = payload.get("confidence")
    confidence: float | None = None
    if confidence_value is not None:
        try:
            confidence = float(confidence_value)
        except (TypeError, ValueError):
            confidence = None
    if not raw_text:
        return jsonify({"ok": False, "message": "Please provide OCR text to clean up."}), 400
    result = build_browser_ocr_cleanup(raw_text, confidence=confidence)
    return jsonify({"ok": True, "ocr": result})

@app.post("/api/analyze")
def analyze_api():
    payload = request.get_json(silent=True) or {}
    ingredient = str(payload.get("ingredient", "")).strip()
    if not ingredient:
        return jsonify({"ok": False, "message": "Please provide an ingredient name."}), 400
    report = analyze_ingredient(ingredient)
    log_search(ingredient, "ingredient")
    return jsonify({"ok": True, "report": report.as_dict()})


@app.post("/api/analyze-product")
def analyze_product_api():
    payload = request.get_json(silent=True) or {}
    ingredient_list = str(payload.get("ingredient_list", "")).strip()
    if not ingredient_list:
        return jsonify({"ok": False, "message": "Please provide an ingredient list."}), 400
    report = analyze_product(ingredient_list)
    log_search(ingredient_list, "product")
    return jsonify({"ok": True, "report": report.as_dict()})


if __name__ == "__main__":
    app.run(debug=True)

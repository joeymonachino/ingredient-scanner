from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import requests
from flask import Flask, jsonify, render_template, request


app = Flask(__name__)


HIGH_SCRUTINY_MARKERS = {
    "sodium nitrite": "Commonly scrutinized as a preservative in processed meats.",
    "nitrites": "Nitrites are often flagged when people try to reduce heavily processed additives.",
    "bha": "BHA is a synthetic preservative that many label-conscious shoppers avoid.",
    "bht": "BHT is a synthetic preservative that often triggers a caution review.",
    "titanium dioxide": "Titanium dioxide is controversial in food and supplement discussions.",
    "partially hydrogenated oil": "This points to trans-fat style processing and deserves extra caution.",
    "red 40": "Artificial dye that many shoppers intentionally avoid.",
    "yellow 5": "Artificial dye that many shoppers intentionally avoid.",
    "yellow 6": "Artificial dye that many shoppers intentionally avoid.",
    "blue 1": "Artificial dye that many shoppers intentionally avoid.",
    "blue 2": "Artificial dye that many shoppers intentionally avoid.",
    "green 3": "Artificial dye that many shoppers intentionally avoid.",
    "aspartame": "Artificial sweetener that some people prefer to avoid.",
    "sucralose": "Artificial sweetener that some people prefer to avoid.",
    "acesulfame potassium": "Artificial sweetener frequently grouped into a caution bucket.",
    "high fructose corn syrup": "Often treated as a major red flag in ingredient-conscious shopping.",
    "corn syrup": "Often treated as a major red flag in ingredient-conscious shopping.",
    "artificial color": "Artificial coloring is commonly treated as a hard avoid.",
    "fd&c": "Synthetic food coloring is commonly treated as a hard avoid.",
}

CAUTION_MARKERS = {
    "natural flavor": "Broad catch-all labeling that can be hard for shoppers to interpret.",
    "artificial flavor": "Artificial flavoring is usually a sign of a more processed formula.",
    "carrageenan": "Often debated in ingredient-conscious communities.",
    "polysorbate": "Emulsifier that usually signals higher processing.",
    "maltodextrin": "Highly processed carbohydrate ingredient.",
    "monosodium glutamate": "Flavor enhancer that some people choose to limit.",
    "msg": "Flavor enhancer that some people choose to limit.",
    "sodium benzoate": "Preservative ingredient that may trigger a caution review.",
    "potassium sorbate": "Preservative ingredient that may trigger a caution review.",
    "sunflower oil": "Seed oils are often flagged by shoppers trying to stay closer to minimally processed foods.",
    "soybean oil": "Seed oils are often flagged by shoppers trying to stay closer to minimally processed foods.",
    "canola oil": "Seed oils are often flagged by shoppers trying to stay closer to minimally processed foods.",
    "safflower oil": "Seed oils are often flagged by shoppers trying to stay closer to minimally processed foods.",
    "corn oil": "Seed oils are often flagged by shoppers trying to stay closer to minimally processed foods.",
    "cottonseed oil": "Seed oils are often flagged by shoppers trying to stay closer to minimally processed foods.",
    "grapeseed oil": "Seed oils are often flagged by shoppers trying to stay closer to minimally processed foods.",
    "rice bran oil": "Seed oils are often flagged by shoppers trying to stay closer to minimally processed foods.",
    "vegetable oil": "Generic blended oils are usually treated cautiously because sourcing and processing are unclear.",
    "palm oil": "Often reviewed carefully due to processing and sourcing concerns.",
}

WHOLE_FOOD_MARKERS = {
    "apple",
    "oat",
    "olive oil",
    "avocado oil",
    "coconut oil",
    "ginger",
    "turmeric",
    "garlic",
    "spinach",
    "broccoli",
    "lentil",
    "bean",
    "rice",
    "quinoa",
    "almond",
    "walnut",
    "honey",
    "cinnamon",
    "cocoa",
    "grass-fed",
    "pasture-raised",
    "raw honey",
    "sprouted",
}

PLANT_MARKERS = {
    "extract",
    "seed",
    "leaf",
    "root",
    "fruit",
    "bean",
    "cocoa",
    "oat",
    "rice",
    "corn",
    "soy",
    "olive",
    "coconut",
    "almond",
    "sunflower",
    "turmeric",
    "ginger",
}

MINERAL_MARKERS = {
    "chloride",
    "carbonate",
    "phosphate",
    "oxide",
    "zinc",
    "magnesium",
    "calcium",
    "iron",
    "potassium",
    "sodium bicarbonate",
    "sea salt",
    "salt",
}

ANIMAL_MARKERS = {
    "gelatin",
    "collagen",
    "whey",
    "casein",
    "milk",
    "egg",
    "beef",
    "fish oil",
    "honey",
    "shellac",
}

SYNTHETIC_MARKERS = {
    "artificial",
    "benzoate",
    "sorbate",
    "bha",
    "bht",
    "polysorbate",
    "sucralose",
    "aspartame",
    "acesulfame",
    "dioxide",
    "color",
    "flavor",
}

USE_MAP = {
    "preservative": ["preservative", "stabilizer"],
    "sweetener": ["sweetener"],
    "color additive": ["color", "dye"],
    "emulsifier": ["emulsifier", "lecithin", "polysorbate"],
    "flavoring": ["flavor", "flavour", "spice"],
    "mineral supplement": ["magnesium", "zinc", "iron", "calcium", "potassium"],
    "oil or fat": ["oil", "butter"],
}

PREFERRED_MARKERS = {
    "avocado oil",
    "olive oil",
    "coconut oil",
    "grass-fed",
    "pasture-raised",
    "organic",
    "sprouted",
    "raw honey",
}

SEED_OIL_MARKERS = {
    "sunflower oil",
    "soybean oil",
    "canola oil",
    "safflower oil",
    "corn oil",
    "cottonseed oil",
    "grapeseed oil",
    "rice bran oil",
    "vegetable oil",
}

COLOR_MARKERS = {
    "red 40",
    "yellow 5",
    "yellow 6",
    "blue 1",
    "blue 2",
    "green 3",
    "artificial color",
    "fd&c",
}


@dataclass
class IngredientReport:
    ingredient: str
    confidence: str
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "ingredient": self.ingredient,
            "confidence": self.confidence,
            "chemistry_family": self.chemistry_family,
            "source_profile": self.source_profile,
            "processing_level": self.processing_level,
            "shopper_signal": self.shopper_signal,
            "quick_blurb": self.quick_blurb,
            "common_uses": self.common_uses,
            "highlights": self.highlights,
            "cautions": self.cautions,
            "fetched_summary": self.fetched_summary,
            "fetched_sources": self.fetched_sources,
            "signal_class": self.signal_class,
        }


def contains_any(text: str, phrases: set[str]) -> bool:
    return any(matches_phrase(text, phrase) for phrase in phrases)


def matches_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text))


def detect_chemistry_family(name: str) -> tuple[str, str]:
    lowered = name.lower()
    if contains_any(lowered, MINERAL_MARKERS):
        return "Likely inorganic or mineral-based", "medium"
    if contains_any(lowered, WHOLE_FOOD_MARKERS | PLANT_MARKERS | ANIMAL_MARKERS):
        return "Likely organic compound or food-derived material", "medium"
    if contains_any(lowered, SYNTHETIC_MARKERS):
        return "Likely synthetic organic compound", "low"
    return "Unclear from name alone", "low"


def detect_source_profile(name: str) -> str:
    lowered = name.lower()
    if contains_any(lowered, ANIMAL_MARKERS):
        return "animal-derived or may be animal-derived"
    if contains_any(lowered, MINERAL_MARKERS):
        return "mineral-derived"
    if contains_any(lowered, WHOLE_FOOD_MARKERS | PLANT_MARKERS):
        return "plant-derived or botanical"
    if contains_any(lowered, SYNTHETIC_MARKERS):
        return "synthetic or highly modified"
    return "mixed or unclear origin"


def detect_processing_level(name: str) -> str:
    lowered = name.lower()
    if contains_any(lowered, WHOLE_FOOD_MARKERS | PREFERRED_MARKERS):
        return "minimally processed"
    if "extract" in lowered or "protein" in lowered or "concentrate" in lowered:
        return "moderately processed"
    if contains_any(lowered, HIGH_SCRUTINY_MARKERS.keys() | CAUTION_MARKERS.keys() | SYNTHETIC_MARKERS):
        return "highly processed"
    return "moderately processed"


def detect_common_uses(name: str) -> list[str]:
    lowered = name.lower()
    uses = [label for label, markers in USE_MAP.items() if any(marker in lowered for marker in markers)]
    if not uses and "acid" in lowered:
        uses.append("acidity regulator")
    if not uses and "extract" in lowered:
        uses.append("functional botanical ingredient")
    if not uses:
        uses.append("general formulation ingredient")
    return uses


def build_cautions(name: str) -> list[str]:
    lowered = name.lower()
    cautions: list[str] = []
    for marker, note in HIGH_SCRUTINY_MARKERS.items():
        if matches_phrase(lowered, marker):
            cautions.append(note)
    for marker, note in CAUTION_MARKERS.items():
        if matches_phrase(lowered, marker):
            cautions.append(note)
    if "organic" in lowered:
        cautions.append("The word 'organic' here may describe chemistry, not USDA-certified organic farming.")
    return cautions


def build_highlights(name: str, source_profile: str, chemistry_family: str, processing_level: str) -> list[str]:
    highlights = [
        f"Source profile: {source_profile}.",
        f"Chemistry family: {chemistry_family}.",
        f"Processing level: {processing_level}.",
    ]
    lowered = name.lower()
    if contains_any(lowered, WHOLE_FOOD_MARKERS):
        highlights.append("Looks closer to a recognizable food ingredient than a synthetic additive.")
    if contains_any(lowered, MINERAL_MARKERS):
        highlights.append("May function more like a mineral salt or fortification ingredient than a whole-food component.")
    if contains_any(lowered, SYNTHETIC_MARKERS):
        highlights.append("Signals a more formulation-driven ingredient rather than a kitchen pantry staple.")
    if contains_any(lowered, SEED_OIL_MARKERS):
        highlights.append("This matches a seed-oil pattern that ingredient-conscious shoppers often flag.")
    if contains_any(lowered, COLOR_MARKERS):
        highlights.append("This matches an added-color pattern that many shoppers prefer to avoid entirely.")
    if contains_any(lowered, PREFERRED_MARKERS):
        highlights.append("Contains terms that usually read cleaner to shoppers looking for simpler ingredient panels.")
    return highlights


def decide_shopper_signal(name: str, cautions: list[str], processing_level: str) -> str:
    lowered = name.lower()
    if any(matches_phrase(lowered, marker) for marker in HIGH_SCRUTINY_MARKERS):
        return "avoid"
    if cautions or processing_level == "highly processed":
        return "caution"
    if contains_any(lowered, WHOLE_FOOD_MARKERS | PLANT_MARKERS | MINERAL_MARKERS | PREFERRED_MARKERS):
        return "okay"
    return "needs context"


def build_quick_blurb(name: str, source_profile: str, shopper_signal: str, common_uses: list[str]) -> str:
    use_text = ", ".join(common_uses[:2])
    return (
        f"{name} appears to be {source_profile}. In a product label, it is often acting as "
        f"{use_text}. For ingredient-conscious shoppers, it currently lands in the "
        f"'{shopper_signal}' bucket based on a more ingredient-conscious ruleset and any fetched reference data."
    )


def wikipedia_summary(term: str) -> tuple[str | None, list[dict[str, str]]]:
    try:
        search_response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": term,
                "utf8": 1,
                "format": "json",
            },
            timeout=8,
        )
        search_response.raise_for_status()
        hits = search_response.json().get("query", {}).get("search", [])
        if not hits:
            return None, []

        title = hits[0]["title"]
        summary_response = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
            timeout=8,
        )
        summary_response.raise_for_status()
        payload = summary_response.json()
        extract = payload.get("extract")
        if not extract:
            return None, []
        return extract, [{"label": "Wikipedia", "url": payload.get("content_urls", {}).get("desktop", {}).get("page", "")}]
    except requests.RequestException:
        return None, []


def pubchem_summary(term: str) -> tuple[str | None, list[dict[str, str]]]:
    try:
        response = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{term}/property/IUPACName,MolecularFormula/JSON",
            timeout=8,
        )
        response.raise_for_status()
        properties = response.json().get("PropertyTable", {}).get("Properties", [])
        if not properties:
            return None, []
        first = properties[0]
        iupac = first.get("IUPACName")
        formula = first.get("MolecularFormula")
        bits = []
        if iupac:
            bits.append(f"IUPAC name: {iupac}.")
        if formula:
            bits.append(f"Molecular formula: {formula}.")
        if not bits:
            return None, []
        return " ".join(bits), [{"label": "PubChem", "url": f"https://pubchem.ncbi.nlm.nih.gov/#query={term}"}]
    except requests.RequestException:
        return None, []


def enrich_from_web(term: str) -> tuple[str | None, list[dict[str, str]]]:
    wiki_text, wiki_sources = wikipedia_summary(term)
    if wiki_text:
        return wiki_text, wiki_sources

    pubchem_text, pubchem_sources = pubchem_summary(term)
    if pubchem_text:
        return pubchem_text, pubchem_sources

    return None, []


def analyze_ingredient(term: str) -> IngredientReport:
    ingredient = " ".join(term.strip().split())
    chemistry_family, confidence = detect_chemistry_family(ingredient)
    source_profile = detect_source_profile(ingredient)
    processing_level = detect_processing_level(ingredient)
    common_uses = detect_common_uses(ingredient)
    cautions = build_cautions(ingredient)
    shopper_signal = decide_shopper_signal(ingredient, cautions, processing_level)
    signal_class = shopper_signal.replace(" ", "-").replace("/", "").replace("--", "-")
    highlights = build_highlights(ingredient, source_profile, chemistry_family, processing_level)
    fetched_summary, fetched_sources = enrich_from_web(ingredient)
    quick_blurb = build_quick_blurb(ingredient, source_profile, shopper_signal, common_uses)

    if fetched_summary:
        quick_blurb = fetched_summary

    return IngredientReport(
        ingredient=ingredient,
        confidence=confidence,
        chemistry_family=chemistry_family,
        source_profile=source_profile,
        processing_level=processing_level,
        shopper_signal=shopper_signal,
        quick_blurb=quick_blurb,
        common_uses=common_uses,
        highlights=highlights,
        cautions=cautions,
        fetched_summary=fetched_summary,
        fetched_sources=fetched_sources,
        signal_class=signal_class,
    )


@app.get("/")
def index() -> str:
    sample_terms = [
        "turmeric",
        "citric acid",
        "sodium benzoate",
        "sucralose",
        "magnesium stearate",
    ]
    return render_template("index.html", sample_terms=sample_terms)


@app.post("/analyze")
def analyze_page() -> str:
    ingredient = request.form.get("ingredient", "").strip()
    report = analyze_ingredient(ingredient) if ingredient else None
    sample_terms = [
        "turmeric",
        "citric acid",
        "sodium benzoate",
        "sucralose",
        "magnesium stearate",
    ]
    return render_template("index.html", report=report, ingredient=ingredient, sample_terms=sample_terms)


@app.post("/api/analyze")
def analyze_api():
    payload = request.get_json(silent=True) or {}
    ingredient = str(payload.get("ingredient", "")).strip()
    if not ingredient:
        return jsonify({"ok": False, "message": "Please provide an ingredient name."}), 400
    report = analyze_ingredient(ingredient)
    return jsonify({"ok": True, "report": report.as_dict()})


if __name__ == "__main__":
    app.run(debug=True)

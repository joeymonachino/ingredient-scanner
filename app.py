from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import requests
from flask import Flask, abort, jsonify, render_template, request


app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
POLICY_PATH = DATA_DIR / "ingredient_policies.json"
ANALYTICS_PATH = DATA_DIR / "analytics_summary.json"

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


def ingredient_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "ingredient"


def policy_slug_map() -> dict[str, dict[str, Any]]:
    return {ingredient_slug(record["canonical_name"]): record for record in POLICY_RECORDS}


def find_policy_by_slug(slug: str) -> dict[str, Any] | None:
    return policy_slug_map().get(slug)


@app.context_processor
def inject_template_helpers() -> dict[str, Any]:
    return {"ingredient_slug": ingredient_slug}


def matches_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(phrase.lower())}(?![a-z0-9])", text.lower()))


def contains_any(text: str, phrases: set[str]) -> bool:
    return any(matches_phrase(text, phrase) for phrase in phrases)


def normalize_input(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\r", " ").replace("\n", " ")).strip(" ,;.")


def clean_ingredient_name(name: str) -> str:
    cleaned = normalize_input(name)
    cleaned = re.sub(r"^\s*(ingredients?|contains)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\[[^\]]*\]\s*$", "", cleaned)
    cleaned = re.sub(r"\s*\([^)]*\)\s*$", "", cleaned)
    return cleaned.strip(" ,;.")


def split_ingredient_list(raw_text: str) -> list[str]:
    text = normalize_input(raw_text)
    text = re.sub(r"^\s*ingredients?\s*:\s*", "", text, flags=re.IGNORECASE)
    if not text:
        return []
    items: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char in "([":
            depth += 1
        elif char in ")]" and depth > 0:
            depth -= 1
        if char in ",;" and depth == 0:
            piece = clean_ingredient_name("".join(current))
            if piece:
                items.append(piece)
            current = []
            continue
        current.append(char)
    tail = clean_ingredient_name("".join(current))
    if tail:
        items.append(tail)
    return [item for index, item in enumerate(items) if item and item not in items[:index]]

def empty_analytics_payload() -> dict[str, Any]:
    return {
        "totals": {"searches": 0, "ingredient_pages": 0, "index_pages": 0},
        "ingredient_queries": {},
        "product_queries": {},
        "pageviews": {},
        "recent_events": [],
    }


def load_analytics() -> dict[str, Any]:
    if not ANALYTICS_PATH.exists():
        return empty_analytics_payload()
    try:
        return json.loads(ANALYTICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_analytics_payload()


def save_analytics(payload: dict[str, Any]) -> None:
    try:
        DATA_DIR.mkdir(exist_ok=True)
        ANALYTICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        # Serverless hosts such as Vercel may expose a read-only filesystem.
        return


def log_search(query_text: str, mode: str) -> None:
    analytics = load_analytics()
    analytics["totals"]["searches"] = analytics["totals"].get("searches", 0) + 1
    bucket = "ingredient_queries" if mode == "ingredient" else "product_queries"
    normalized = normalize_input(query_text).lower()
    analytics.setdefault(bucket, {})
    analytics[bucket][normalized] = analytics[bucket].get(normalized, 0) + 1
    analytics.setdefault("recent_events", [])
    analytics["recent_events"].insert(0, {"kind": f"search:{mode}", "value": normalized})
    analytics["recent_events"] = analytics["recent_events"][:80]
    save_analytics(analytics)


def log_pageview(slug: str) -> None:
    analytics = load_analytics()
    analytics["totals"]["ingredient_pages"] = analytics["totals"].get("ingredient_pages", 0) + 1
    analytics.setdefault("pageviews", {})
    analytics["pageviews"][slug] = analytics["pageviews"].get(slug, 0) + 1
    save_analytics(analytics)


def log_index_pageview() -> None:
    analytics = load_analytics()
    analytics["totals"]["index_pages"] = analytics["totals"].get("index_pages", 0) + 1
    save_analytics(analytics)


def find_policy(name: str) -> dict[str, Any] | None:
    lowered = name.lower()
    best_match: dict[str, Any] | None = None
    best_alias_length = -1
    for record in POLICY_RECORDS:
        aliases = [str(alias).lower() for alias in record.get("aliases", [])]
        for alias in aliases:
            if matches_phrase(lowered, alias) and len(alias) > best_alias_length:
                best_match = record
                best_alias_length = len(alias)
    return best_match


def looks_like_known_ingredient(name: str) -> bool:
    lowered = name.lower().strip()
    if not lowered or len(lowered) < 3:
        return False
    if contains_any(lowered, WHOLE_FOOD_MARKERS | PLANT_MARKERS | MINERAL_MARKERS | ANIMAL_MARKERS | SYNTHETIC_MARKERS):
        return True
    if any(matches_phrase(lowered, term) for term in HEURISTIC_TERMS):
        return True
    if re.search(r"\b(acid|oil|gum|extract|starch|syrup|protein|powder|salt|sugar|flavor|flavour|juice|milk|seed|root|leaf)\b", lowered):
        return True
    return False


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


def detect_processing_level(name: str, policy: dict[str, Any] | None, confidence_label: str) -> str:
    if confidence_label == "unknown":
        return "unknown"
    if policy and policy.get("processing_level"):
        return str(policy["processing_level"])
    lowered = name.lower()
    if contains_any(lowered, WHOLE_FOOD_MARKERS):
        return "minimally processed"
    if contains_any(lowered, SYNTHETIC_MARKERS):
        return "highly processed"
    return "moderately processed"


def detect_common_uses(name: str, policy: dict[str, Any] | None) -> list[str]:
    if policy and policy.get("functional_roles"):
        return [str(role) for role in policy["functional_roles"]]
    lowered = name.lower()
    uses = [label for label, markers in USE_MAP.items() if any(marker in lowered for marker in markers)]
    if not uses and "acid" in lowered:
        uses.append("acidity regulator")
    if not uses and "extract" in lowered:
        uses.append("functional botanical ingredient")
    return uses or ["general formulation ingredient"]


def derive_signal(policy: dict[str, Any] | None, processing_level: str, name: str, confidence_label: str) -> str:
    if confidence_label == "unknown":
        return "unknown"
    if policy and policy.get("signal"):
        return str(policy["signal"])
    lowered = name.lower()
    if contains_any(lowered, SYNTHETIC_MARKERS):
        return "avoid"
    if processing_level == "highly processed":
        return "avoid"
    if processing_level == "moderately processed":
        return "caution"
    if processing_level == "minimally processed":
        return "okay"
    return "needs context"


def classify_confidence(name: str, policy: dict[str, Any] | None) -> tuple[str, str, str]:
    if policy:
        return "high", "policy-backed", "Matched a curated ingredient policy record."
    if looks_like_known_ingredient(name):
        return "medium", "heuristic-only", "Looks like a plausible ingredient or additive, but is not yet in the curated policy catalog."
    return "low", "unknown", "We couldn't confidently identify this ingredient."

def build_cautions(policy: dict[str, Any] | None, signal: str) -> list[str]:
    if not policy:
        return []
    note = str(policy.get("note", "")).strip()
    if signal in {"avoid", "caution"} and note:
        return [note]
    return []


def build_highlights(name: str, source_profile: str, chemistry_family: str, processing_level: str, policy: dict[str, Any] | None, confidence_label: str) -> list[str]:
    highlights = [
        f"Source profile: {source_profile}.",
        f"Chemistry family: {chemistry_family}.",
        f"Processing level: {processing_level}.",
        f"Confidence mode: {confidence_label}.",
    ]
    if policy:
        if policy.get("whole_food_alignment"):
            highlights.append(f"Whole-food alignment: {policy['whole_food_alignment']}.")
        if policy.get("organic_alignment"):
            highlights.append(f"Organic alignment: {policy['organic_alignment']}.")
        traits = [str(trait) for trait in policy.get("traits", [])]
        if traits:
            highlights.append(f"Policy traits: {', '.join(traits[:3])}.")
        note = str(policy.get("note", "")).strip()
        if note:
            highlights.append(note)
    elif confidence_label == "unknown":
        highlights.append("The current app does not have enough evidence to classify this text as a real ingredient with confidence.")
    return highlights


def build_quick_blurb(name: str, source_profile: str, signal: str, common_uses: list[str], processing_level: str, policy: dict[str, Any] | None, confidence_label: str) -> str:
    if confidence_label == "unknown":
        return "We couldn't confidently identify this ingredient. Try checking the spelling or pasting the exact wording from the label."
    use_text = ", ".join(common_uses[:2])
    if policy:
        rationale = str(policy.get("note", "")).strip()
        if rationale:
            return (
                f"{name} appears to be {source_profile}. In products it often acts as {use_text}. "
                f"It lands in '{signal}' because it reads as {processing_level} and matches this policy basis: {rationale}"
            )
    return (
        f"{name} appears to be {source_profile}. In products it often acts as {use_text}. "
        f"It lands in '{signal}' because it reads as {processing_level} by the current heuristic cleaner-label ruleset."
    )


def build_what_it_does(common_uses: list[str], policy: dict[str, Any] | None, confidence_label: str) -> str:
    if confidence_label == "unknown":
        return "The app could not determine a reliable role for this input."
    if policy and policy.get("functional_roles"):
        return f"This ingredient is commonly used as {', '.join(policy['functional_roles'])}."
    return f"This ingredient most likely functions as {', '.join(common_uses[:2])}."


def build_why_flagged(policy: dict[str, Any] | None, signal: str, confidence_label: str) -> str:
    if confidence_label == "unknown":
        return "No flag is being assigned with confidence because the ingredient itself is not confidently identified."
    if policy:
        traits = ', '.join(policy.get('traits', [])[:3])
        note = str(policy.get('note', '')).strip()
        return f"The app marks this as '{signal}' based on the policy record traits: {traits}. {note}".strip()
    return f"The app marks this as '{signal}' using fallback heuristic rules because it is not yet in the curated catalog."


def build_cleaner_takeaway(signal: str, confidence_label: str) -> str:
    if confidence_label == "unknown":
        return "Treat this as unclassified until the app has a verified policy record or you confirm the exact ingredient name."
    if signal == "okay":
        return "This reads broadly aligned with a simpler, cleaner ingredient standard."
    if signal == "caution":
        return "This is more processed or formulation-oriented than ideal for a strict cleaner-label approach."
    if signal == "avoid":
        return "This conflicts strongly with the cleaner-label standard this app is built around."
    return "This needs more context before it can be treated as clearly aligned."


def wikipedia_summary(term: str) -> tuple[str | None, list[dict[str, str]]]:
    try:
        search_response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": term, "utf8": 1, "format": "json"},
            timeout=8,
        )
        search_response.raise_for_status()
        hits = search_response.json().get("query", {}).get("search", [])
        if not hits:
            return None, []
        title = hits[0]["title"]
        summary_response = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}", timeout=8)
        summary_response.raise_for_status()
        payload = summary_response.json()
        extract = payload.get("extract")
        if not extract:
            return None, []
        source = payload.get("content_urls", {}).get("desktop", {}).get("page", "")
        return extract, ([{"label": "Wikipedia", "url": source}] if source else [])
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
        bits = []
        if first.get("IUPACName"):
            bits.append(f"IUPAC name: {first['IUPACName']}.")
        if first.get("MolecularFormula"):
            bits.append(f"Molecular formula: {first['MolecularFormula']}.")
        return (" ".join(bits) if bits else None), ([{"label": "PubChem", "url": f"https://pubchem.ncbi.nlm.nih.gov/#query={term}"}] if bits else [])
    except requests.RequestException:
        return None, []


def enrich_from_web(term: str) -> tuple[str | None, list[dict[str, str]]]:
    wiki_text, wiki_sources = wikipedia_summary(term)
    if wiki_text:
        return wiki_text, wiki_sources
    return pubchem_summary(term)

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

"""Maps radiology-report language to the 12 target labels via regex synonym
lists.

Coverage is English plus partial Spanish/Dutch, inferred from the report
text actually observed in `data/train.csv` (the competition reports are
multilingual). This is a known, stated limitation, not exhaustive
multilingual coverage -- an unmapped language simply yields fewer mentions
(lower recall) rather than raising an error. See README.md "Known
limitations".

Callers should match against text that has already been normalized (see
`src/reports/weak_labels.py::normalize_report_text`), which lowercases and
strips diacritics -- so patterns below are written in their unaccented form
(e.g. "contusion osea", not "contusi[oó]n [oó]sea").
"""

from __future__ import annotations

import re

TARGETS: list[str] = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

# Each pattern is matched case-insensitively against normalized report text.
# Patterns are deliberately conservative (word boundaries, specific phrases)
# to avoid false mentions -- recall is improved by adding more synonyms, not
# by loosening existing ones.
SYNONYMS: dict[str, list[str]] = {
    "ACL": [
        r"\bacl\b",
        r"anterior cruciate ligament",
        r"ligamento cruzado anterior",
        r"voorste kruisband",
    ],
    "MCL": [
        r"\bmcl\b",
        r"medial collateral ligament",
        r"ligamento colateral medial",
        r"mediale collaterale band",
        r"\bbinnenband\b",
    ],
    "Medial Meniscus": [
        r"medial meniscus",
        r"menisco (medial|interno)",
        r"binnenmeniscus",
        r"mediale meniscus",
    ],
    "Lateral Meniscus": [
        r"lateral meniscus",
        r"menisco (lateral|externo)",
        r"buitenmeniscus",
        r"laterale meniscus",
    ],
    "Medial OA": [
        r"medial (compartment )?osteoarthrit",
        r"medial\w*.{0,20}(joint space narrowing|degenerative change)",
        r"artrosis (femorotibial )?medial",
        r"mediale (femorotibiale )?artrose",
    ],
    "Lateral OA": [
        r"lateral (compartment )?osteoarthrit",
        r"lateral\w*.{0,20}(joint space narrowing|degenerative change)",
        r"artrosis (femorotibial )?lateral",
        r"laterale (femorotibiale )?artrose",
    ],
    "PF OA": [
        r"patellofemoral osteoarthrit",
        r"patello ?femoral\w*.{0,20}(osteoarthrit|degenerative change)",
        r"artrosis (femoro)?patelar",
        r"patellofemorale artrose",
    ],
    "Effusion": [
        r"\beffusion\b",
        r"joint effusion",
        r"\bderrame\b",
        r"gewrichtsvocht",
        r"gewrichtseffusie",
    ],
    "Synovitis": [
        r"synoviti",
        r"sinoviti",
    ],
    "Baker's": [
        r"baker'?s? cyst",
        r"popliteal cyst",
        r"quiste (de baker|poplite[oa])",
        r"bakercyste",
    ],
    "Contusion": [
        r"bone (bruise|contusion)",
        r"marrow edema pattern",
        r"contusion osea",
        r"botkneuzing",
        r"beenmergoedeem",
    ],
    "Fracture": [
        r"\bfractur\w*",
        r"\bfractura\b",
        r"\bfractuur\b",
    ],
}

assert set(SYNONYMS.keys()) == set(TARGETS), "SYNONYMS must cover exactly the 12 TARGETS"


def compile_patterns(synonyms: dict[str, list[str]] = SYNONYMS) -> dict[str, list[re.Pattern]]:
    """Compile every synonym pattern once, case-insensitive. Cheap enough to
    call per-import; cached at module load as `_COMPILED` below."""
    return {
        target: [re.compile(p, re.IGNORECASE) for p in patterns]
        for target, patterns in synonyms.items()
    }


_COMPILED = compile_patterns()


def find_target_mentions(text: str) -> dict[str, list[re.Match]]:
    """Find every synonym-pattern match for every target in `text`.

    Returns a dict from target name to a (possibly empty) list of
    `re.Match` objects, each carrying `.start()`/`.end()` character offsets
    into `text` for downstream negation-scope analysis.
    """
    mentions: dict[str, list[re.Match]] = {target: [] for target in TARGETS}
    for target, patterns in _COMPILED.items():
        for pattern in patterns:
            mentions[target].extend(pattern.finditer(text))
    return mentions

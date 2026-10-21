# Stage 3 prompts: understanding judge, GEMBA-DA (plain + definition-anchored),
# and response-parsing helpers.
#
# The understanding rubric is authored for this benchmark. GEMBA-DA wording and
# validation helpers come from Freitag et al. (2022, arXiv:2205.13748), with JSON
# output and dependency-free validation.

from __future__ import annotations

import re

LANGUAGE_NAMES = {"zh": "Chinese", "ru": "Russian", "en": "English"}


# ─────────────────────────────────────────────────────────────────────────────
# Binary understanding judge for the two identify paths.
# ─────────────────────────────────────────────────────────────────────────────

_UNDERSTANDING_SCHEMA = '{\n  "score": 0,\n  "reason": "one sentence"\n}'


def build_understanding_judge_prompt(
    term: str, reference_definition: str, mt_understanding: str
) -> str:
    """Binary judge: does the MT's context-inferred definition match the gold meaning?"""
    return (
        f'Term: "{term}"\n\n'
        f"Definition A: {reference_definition}\n"
        f"Definition B: {mt_understanding}\n\n"
        "Do these two definitions convey the same core meaning?\n"
        "- 1 = yes: the essential meaning matches (allow paraphrase; minor gaps are fine)\n"
        "- 0 = no: the meaning is wrong, inverted, or a key component is missing\n\n"
        f"Return JSON:\n{_UNDERSTANDING_SCHEMA}"
    )


# GEMBA-DA (Freitag et al. 2022), adapted to JSON output.

def parse_numerical_answer(answer: str, min: int | None = None, max: int | None = None) -> int | None:
    """Extract a single integer from a model response string."""
    numbers = re.findall(r'\d+', answer)
    if len(numbers) == 1:
        return int(numbers[0])
    # handle ['100'] form
    r1 = re.match(r"^\[['\"][0-9]*['\"]\]$", answer)
    if r1 is not None:
        return int(answer[2:-2])
    if max is not None:
        # handle 0/100 form
        r2 = re.match(rf"^[0-9]*/{max}$", answer)
        if r2 is not None:
            return int(answer.split("/")[0])
    return None


def validate_number(x: str, min: int = 0, max: int = 100) -> int | None:
    """Parse and range-check a numerical score. Returns None if unparseable or out of range."""
    n = parse_numerical_answer(x, min, max)
    if n is not None and min <= n <= max:
        return n
    return None


def validate_stars(x: str) -> int | None:
    """Parse a 1–5 star rating from a model response."""
    x = x.lower()
    possible: set[int] = set()
    if "*" in x:
        possible.add(x.count("*"))
    if "★" in x:
        possible.add(x.count("★"))
    x = f" {x} ".replace("\n", " ")
    if " one " in x or "1 star" in x:
        possible.add(1)
    if " two " in x or "2 star" in x:
        possible.add(2)
    if " three " in x or "3 star" in x:
        possible.add(3)
    if " four " in x or "4 star" in x:
        possible.add(4)
    if " five " in x or "5 star" in x:
        possible.add(5)
    n = parse_numerical_answer(x)
    if n is not None:
        possible.add(n)
    if len(possible) == 1:
        answer = possible.pop()
        if 1 <= answer <= 5:
            return answer
    return None


def parse_classes(answer: str, classes: list[str]) -> int | None:
    """Return the index of the matched class label, or None if zero or multiple match."""
    final: int | None = None
    for i, cls in enumerate(classes):
        if cls.lower() in answer.lower():
            if final is None:
                final = i
            else:
                print(f"[parse_classes] two classes found in answer: {answer!r}")
                return None
    return final


_GEMBA_DA_SCHEMA = '{\n  "score": 75\n}'  # 0–100; example value only


def build_gemba_da_prompt(
    source: str,
    mt_translation: str,
    src_lang: str = "zh",
    tgt_lang: str = "en",
) -> str:
    """GEMBA-DA: reference-free direct assessment, 0–100."""
    src = LANGUAGE_NAMES.get(src_lang, src_lang)
    tgt = LANGUAGE_NAMES.get(tgt_lang, tgt_lang)
    return (
        f'Score the following translation from {src} to {tgt} on a continuous scale '
        'from 0 to 100, where a score of zero means "no meaning preserved" and score '
        'of one hundred means "perfect meaning and grammar".\n\n'
        f'{src} source: "{source}"\n'
        f'{tgt} translation: "{mt_translation}"\n\n'
        f"Return JSON:\n{_GEMBA_DA_SCHEMA}"
    )


def build_gemba_da_def_prompt(
    source: str,
    mt_translation: str,
    term: str,
    definition: str,
    src_lang: str = "zh",
    tgt_lang: str = "en",
) -> str:
    """GEMBA-DA anchored on the slang term and its definition instead of a reference translation."""
    src = LANGUAGE_NAMES.get(src_lang, src_lang)
    tgt = LANGUAGE_NAMES.get(tgt_lang, tgt_lang)
    return (
        f'Score the following translation from {src} to {tgt} on a continuous scale '
        'from 0 to 100, where a score of zero means "no meaning preserved" and score '
        'of one hundred means "perfect meaning and grammar".\n\n'
        f'Note: the source contains the slang term "{term}", which means: {definition}\n\n'
        f'{src} source: "{source}"\n'
        f'{tgt} translation: "{mt_translation}"\n\n'
        f"Return JSON:\n{_GEMBA_DA_SCHEMA}"
    )

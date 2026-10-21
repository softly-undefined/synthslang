# Stage 2 prompts: Direct, DirectWithHint, SlangDITIdentify, and the two
# paragraph-context paths (DirectContext, SlangDITIdentifyContext).
#
# The identify path adapts SlangDIT (arXiv:2505.14181). Direct uses Andrew Ng's
# translation-agent initial-translation prompt. Hint paths add the gold definition;
# context paths add a source-language paragraph. Identified definitions are English
# and are scored against the intended meaning in Stage 3.

from __future__ import annotations

import json
import re

LANGUAGE_NAMES = {"zh": "Chinese", "ru": "Russian", "fa": "Persian", "en": "English"}

# Few-shot examples for SlangDITIdentify / SlangDITIdentifyContext, keyed by source
# language: the slang term is PROVIDED (not discovered) and the model defines it
# in English (no translation). These are well-known Chinese internet-slang terms
# used purely as format demonstrations, not part of the benchmark item set. No
# no-slang example — a term is always given.
_FEW_SHOT_IDENTIFY_EXAMPLES: dict[str, str] = {
    "zh": (
        "- Input Sentence: 一次牛油果没吃过的有没有，吃过的朋友告诉我是什么味道，我脑补一下\n"
        "Slang term: 脑补\n"
        'Output:\n{"slang": "脑补", "definition": "Literally \\"brain-supplement\\"; internet slang for mentally imagining or filling in the details of something one hasn\'t directly experienced."}\n\n'
        "- Input Sentence: 那太有趣了！这个应用上的梗图太棒了\n"
        "Slang term: 梗图\n"
        'Output:\n{"slang": "梗图", "definition": "Internet slang for \\"meme\\" — a humorous image shared online (梗 = a comedic gag/reference, 图 = picture)."}'
    ),
    "ru": (
        "- Input Sentence: [RU_SLANG_SENTENCE]\n"
        "Slang term: [RU_SLANG_TERM]\n"
        'Output:\n{"slang": "[RU_SLANG_TERM]", "definition": "[RU_SLANG_DEFINITION]"}'
    ),
    # These are two well-known Persian colloquial-slang terms, used purely as
    # format demonstrations. The benchmark item set is synthetic-slang-only, so
    # these demonstration terms cannot collide with it.
    "fa": (
        "- Input Sentence: این آهنگ جدیدش واقعاً خفنه!\n"
        "Slang term: خفن\n"
        'Output:\n{"slang": "خفن", "definition": "Colloquial/internet slang for \\"awesome,\\" \\"cool,\\" or impressively intense; describes something excellent or badass."}\n\n'
        "- Input Sentence: اینقدر تابلو نگاش نکن، می‌فهمه!\n"
        "Slang term: تابلو\n"
        'Output:\n{"slang": "تابلو", "definition": "Literally \\"billboard/sign\\"; slang for \\"obvious\\" or \\"conspicuous\\" — behavior or feelings so glaringly evident that everyone notices."}'
    ),
}


def build_translate_direct_prompt(source: str, src_lang: str = "zh", tgt_lang: str = "en") -> tuple[str, str]:
    """Direct path: Andrew Ng translation-agent initial-translation prompt.
    Returns (system_message, user_prompt). Response is plain text — store raw as mt_translation.
    Source: https://github.com/andrewyng/translation-agent/blob/main/src/translation_agent/utils.py#L88
    """
    src = LANGUAGE_NAMES.get(src_lang, src_lang)
    tgt = LANGUAGE_NAMES.get(tgt_lang, tgt_lang)
    system = f"You are an expert linguist, specializing in translation from {src} to {tgt}."
    prompt = (
        f"This is an {src} to {tgt} translation, please provide the {tgt} translation "
        f"for this text. Do not provide any explanations or text apart from the translation.\n"
        f"{src}: {source}\n\n"
        f"{tgt}:"
    )
    return system, prompt


def build_translate_direct_with_hint_prompt(
    source: str, term: str, definition: str, src_lang: str = "zh", tgt_lang: str = "en"
) -> tuple[str, str]:
    """DirectWithHint path: Andrew Ng prompt + gold slang term and definition appended.
    Upper-bound comparison — model knows exactly what the slang means before translating.
    """
    src = LANGUAGE_NAMES.get(src_lang, src_lang)
    tgt = LANGUAGE_NAMES.get(tgt_lang, tgt_lang)
    system = f"You are an expert linguist, specializing in translation from {src} to {tgt}."
    prompt = (
        f"This is an {src} to {tgt} translation, please provide the {tgt} translation "
        f"for this text. Do not provide any explanations or text apart from the translation.\n"
        f"{src}: {source}\n\n"
        f"Note: The text contains the following slang: \"{term}\" — {definition}\n\n"
        f"{tgt}:"
    )
    return system, prompt


def build_translate_direct_context_prompt(
    source: str, context: str, src_lang: str = "zh", tgt_lang: str = "en"
) -> tuple[str, str]:
    """Build a sentence-translation prompt with a source-language context paragraph."""
    src = LANGUAGE_NAMES.get(src_lang, src_lang)
    tgt = LANGUAGE_NAMES.get(tgt_lang, tgt_lang)
    system = f"You are an expert linguist, specializing in translation from {src} to {tgt}."
    prompt = (
        f"This is an {src} to {tgt} translation, please provide the {tgt} translation "
        f"for this text. Do not provide any explanations or text apart from the translation.\n"
        f"For reference, here is a {src} paragraph using similar terminology in context "
        f"(provided only to help you understand the usage — do not translate it):\n"
        f"{context}\n\n"
        f"{src}: {source}\n\n"
        f"{tgt}:"
    )
    return system, prompt


def build_identify_prompt(source: str, term: str, src_lang: str = "zh") -> str:
    """SlangDITIdentify path: the slang term is provided; the model defines its
    meaning and usage in context, in English. No translation."""
    return (
        "Your task is to analyze the given sentence, which contains a slang term. "
        "Follow the steps below to provide your answer.\n\n"
        "1. Slang Term:\n\n"
        f'- The sentence contains the slang term: "{term}".\n\n'
        "2. Provide Definition:\n\n"
        "- Provide a brief definition, in English, of the slang term's meaning and usage.\n\n"
        "3. Output the Results as JSON with the following keys:\n"
        '- "slang": the slang term (string)\n'
        '- "definition": brief definition of the slang term, in English (string)\n\n'
        f"Examples:\n{_FEW_SHOT_IDENTIFY_EXAMPLES.get(src_lang, _FEW_SHOT_IDENTIFY_EXAMPLES['zh'])}\n\n"
        f"Please analyze the following sentence:\nInput Sentence: {source}\nSlang term: {term}"
    )


def build_identify_context_prompt(
    source: str, term: str, context: str, src_lang: str = "zh"
) -> str:
    """Build an English slang-definition prompt with source-language context."""
    return (
        "Your task is to analyze the given sentence, which contains a slang term. "
        "Follow the steps below to provide your answer.\n\n"
        "1. Slang Term:\n\n"
        f'- The sentence contains the slang term: "{term}".\n\n'
        "2. Provide Definition:\n\n"
        "- Provide a brief definition, in English, of the slang term's meaning and usage.\n\n"
        "3. Output the Results as JSON with the following keys:\n"
        '- "slang": the slang term (string)\n'
        '- "definition": brief definition of the slang term, in English (string)\n\n'
        f"Examples:\n{_FEW_SHOT_IDENTIFY_EXAMPLES.get(src_lang, _FEW_SHOT_IDENTIFY_EXAMPLES['zh'])}\n\n"
        "For reference, here is a paragraph that uses the same term in context "
        "(provided only to help you understand its usage):\n"
        f"{context}\n\n"
        f"Please analyze the following sentence:\nInput Sentence: {source}\nSlang term: {term}"
    )


def parse_json_response(raw: str) -> dict:
    """Extract a JSON object from a model response, handling fenced blocks and leading
    text. Raises ValueError if nothing parses, so run_task retries the call rather
    than silently recording a broken result."""
    text = (raw or "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"no JSON object in model response: {text[:200]!r}")


def collapse_repeated_tail(text: str, *, min_period: int = 8,
                           min_repeats: int = 3, keep: int = 2) -> str:
    """Reduce a repeated response tail to at most `keep` copies of its shortest period."""
    if not text:
        return text
    n = len(text)
    for period in range(min_period, n // min_repeats + 1):
        unit = text[n - period:]
        count = 1
        while (start := n - period * (count + 1)) >= 0 and text[start:start + period] == unit:
            count += 1
        if count >= min_repeats:
            drop = count - keep
            return text[: n - period * drop] if drop > 0 else text
    return text


def parse_identify_response(raw: str) -> dict:
    """Parse an identify response, salvaging malformed fields and repeated tails."""
    def clean(definition):
        return collapse_repeated_tail(definition) if isinstance(definition, str) else definition

    try:
        data = parse_json_response(raw)
        return {"slang": data.get("slang"), "definition": clean(data.get("definition")),
                "salvaged": False}
    except ValueError:
        pass

    text = (raw or "").strip()
    # slang: first "..."-quoted value after the key up to the next comma (terms are
    # short and rarely contain quotes, so a non-greedy match is safe here).
    m = re.search(r'"slang"\s*:\s*"(.*?)"\s*,', text, re.DOTALL)
    slang = m.group(1).strip() if m else None
    # definition: everything after the key to the final closing quote+brace, or to
    # end-of-text when truncated. Keeps unescaped inner quotes verbatim.
    m = re.search(r'"definition"\s*:\s*"(.*)', text, re.DOTALL)
    definition = re.sub(r'"\s*}?\s*$', "", m.group(1)).strip() if m else None
    return {"slang": slang, "definition": clean(definition), "salvaged": True}

# Stage 1 prompts: slang generation (step 1), in-context generation (step 2),
# and the contamination gate (LLM guessability check).
#
# SlangDIT paper (translation methodology): https://arxiv.org/pdf/2505.14181
#
# [VERBATIM] marks source wording; [AUTHORED] marks prompts derived from the
# protocol; [ADAPTED] marks restructured source prompts. Few-shot examples were
# web-checked for collisions on June 9, 2026; replacements are noted inline.

from __future__ import annotations

import json
from dataclasses import dataclass

LANGUAGE_NAMES = {"zh": "Chinese", "ru": "Russian", "ar": "Arabic", "fa": "Farsi (Persian)", "en": "English"}

# Where each Chinese type's component-form sits, used to phrase the gate's
# "guess from form alone" instruction per language.
_GUESS_FORM_BASIS = {
    "zh": "character components",
    "ru": "morphology and root",
    "ar": "root and morphological pattern",
    "fa": "morphology, roots, and component words",
}


# Step 1: generate terms and definitions. __N__ is replaced by TERMS_PER_PROMPT.

TERMS_PER_PROMPT = 20  # how many candidate terms each generation call asks for

# Chinese semantic shift. Domain steering was removed to reduce thematic clustering;
# real-slang examples were replaced with collision-checked invented examples.
GEN_ZH_SEMANTIC_SHIFT = """Invent fake-but-plausible Chinese internet slang using SEMANTIC SHIFT: take an ordinary Chinese word and assign it a completely new figurative meaning through metaphor or metonymy.

Examples:
- 借火 ("borrow a light for a cigarette") → briefly latching onto a hot topic or famous person to catch some of their visibility
- 醒面 ("let dough rest") → deliberately leaving a drafted message or post unsent overnight to see whether you still mean it in the morning
- 封罐 ("seal the jar/can") → sitting on a piece of big personal news so you can deliver it in person instead of over text

Rules:
- The new meaning must be non-obvious from the characters alone.
- The meaning should name a recognizable modern phenomenon, behavior, feeling, or situation.
- Do not use words already known to have slang meanings.

Generate exactly __N__ fake slang terms as a JSON array:

[
  {
    "term": "Chinese characters",
    "base": "literal meaning",
    "definition_tar_lang": "the new slang meaning, in English, 1-2 sentences"
  }
]"""

# Chinese morphological blend, with collision-checked invented examples.
GEN_ZH_MORPH_BLEND = """Invent fake-but-plausible Chinese internet slang using MORPHOLOGICAL BLEND (混成型): fuse two existing words or morphemes into a new compound whose meaning goes beyond either component.

Examples:
- 赞饿 = the restless itch to keep reopening an app to check how a fresh post is performing, from 点赞 (likes) + 饥饿 (hunger)
- 夸疲 = burnout from constant performative praise and hype, from 夸奖 (praise) + 疲惫 (fatigued)
- 爽班 = mentally rewriting your dull workday as a fantasy where you're the genius protagonist, from 爽文 (wish-fulfillment fantasy) + 上班 (go to work)

Rules:
- The result should feel like a single new concept, not just a noun phrase.
- The meaning should not be fully predictable from the parts — there should be a conceptual leap.
- Aim for 2–3 character compounds.
- It should name something people recognize but didn't have a word for.

Generate exactly __N__ fake slang terms as a JSON array:

[
  {
    "term": "Chinese characters",
    "components": "Word A (meaning) + Word B (meaning) → blend logic",
    "definition_tar_lang": "what the compound means, in English, 1-2 sentences"
  }
]"""

# Chinese affixation. Narrow audience steering and real-slang examples were removed.
GEN_ZH_AFFIXATION = """Invent fake-but-plausible Chinese internet slang using AFFIXATION (派生型): attach a productive suffix or prefix to a novel base to coin a new term for a person-type, social group, or behavior.

Affixes (use a mix of all 6 across your __N__ examples):
- -族: social tribe defined by a shared situation
- -控: someone obsessively into X
- -狗: self-deprecating in-group label
- -系: a style or aesthetic vibe
- 裸-: doing something stripped-down, without backup
- 被-: passive/"reportedly" — done to you without consent, often ironic

Examples:
- 抢座族 (-族): people with a whole survival system for grabbing seats the instant they open up
- 截图控 (-控): someone who compulsively screenshots everything — chats, receipts, fleeting stories — as evidence and backup
- 裸聚餐 (裸-): showing up to a group dinner with no idea of the budget, menu, or who's splitting the bill

Rules:
- The coined term must describe something recognizable but not-yet-named.
- The definition should describe a recognizable kind of person, group, or behavior.
- Do not reuse existing well-known slang.

Generate exactly __N__ fake slang terms as a JSON array:

[
  {
    "term": "Chinese characters",
    "affix": "which affix and its meaning",
    "base": "root and its meaning",
    "definition_tar_lang": "what the term means, in English, 2-3 sentences"
  }
]"""

# Chinese sentence compression. Narrow domain steering was removed.
GEN_ZH_SENTENCE_COMPRESSION = """Invent fake-but-plausible Chinese "internet chengyu" using SENTENCE COMPRESSION (缩略型): take a multi-clause colloquial expression and extract one key character from each clause to produce a memorable 4-character idiom mimicking the classical chengyu form.

Examples:
- 社恐硬聊: compressed from 社恐到想逃，但场面冷下来了，只能硬着头皮尬聊 → forcing small talk through intense social anxiety once silence gets too awkward
- 撤回重编: from 消息刚发出去就觉得不对，赶紧撤回又重新编辑半天 → regretting a message the instant you send it, then retracting and rewriting it over and over
- 拍修判若: from 照片拍的时候还行，修图修了半天，发出来判若两人 → editing a photo so heavily the posted version looks like a different person

Construction: start with a natural colloquial expression (2–4 clauses, 20–50 chars), extract the most semantically loaded character from each clause, string them together so they read well aloud.

Rules:
- Source expression must be colloquial modern Chinese, not literary.
- The 4-char form must be traceable back to the source.
- The meaning should reflect a recognizable modern phenomenon, behavior, feeling, or situation.

Generate exactly __N__ fake internet chengyu as a JSON array:

[
  {
    "term": "4 Chinese characters",
    "source": "full source expression",
    "source_gloss": "English translation of source expression",
    "compression_map": "char1 from clause 1 + char2 from clause 2 + ...",
    "definition_tar_lang": "what it means and when you'd use it, in English"
  }
]"""

# Russian formation rules are split into four prompts matching the Chinese track.
# Novelty is enforced downstream; transliteration is retained for human review.
# Few-shot examples were collision-checked and conflicting examples replaced.

# Russian semantic shift: an existing word receives a new figurative meaning.
GEN_RU_SEMANTIC_SHIFT = """Invent fake-but-plausible Russian internet slang using SEMANTIC SHIFT: take a real, ordinary Russian word and assign it a radically new colloquial meaning through metaphor or metonymy, unrelated to (or only obliquely connected with) its dictionary definition.

Examples:
- штопать ("to darn socks") → to frantically fill the awkward silences in a conversation with filler chatter, patching the holes as they open
- залежь ("mineral deposit; untouched stock") → the ever-growing pile of saved posts, open tabs, and watch-later videos you will never actually return to
- грунтовать ("to prime a canvas or wall") → to drop casual, friendly groundwork messages in the days before asking someone for a big favor

Rules:
- The base must be a real Russian word, but one with NO documented slang meaning.
- The new meaning must not be predictable from the literal meaning alone — there should be a figurative leap, not a paraphrase.
- The meaning should name a recognizable modern phenomenon, behavior, feeling, or situation.
- The word should keep its normal Russian inflection (conjugation/declension) in the new sense.

Generate exactly __N__ fake slang terms as a JSON array:

[
  {
    "term": "Cyrillic term",
    "translit": "Latin transliteration of the term",
    "pos": "part of speech",
    "base": "literal/dictionary meaning of the word",
    "definition_tar_lang": "the new slang meaning, in English, 1-2 sentences"
  }
]"""

# Russian morphological blend, allowing Russian and foreign roots.
GEN_RU_MORPH_BLEND = """Invent fake-but-plausible Russian internet slang using MORPHOLOGICAL BLEND: fuse two existing roots (Russian + Russian, or Russian + assimilated foreign) into a single new word, portmanteau-style, whose meaning goes beyond either component.

Examples:
- стыдоскоп = the involuntary late-night mental slideshow of your most embarrassing moments, from стыд (shame) + калейдоскоп (kaleidoscope)
- дофамель = the cheap, sticky-sweet little hit of junk content you know is empty but keep reaching for, from дофамин (dopamine) + карамель (caramel)
- лайколомка = the twitchy withdrawal of watching a fresh post sit at zero reactions, from лайк (like) + ломка (withdrawal pangs)

Rules:
- The result should read as a single new word, not a noun phrase or hyphenated pair.
- The meaning should not be fully predictable from the parts — there should be a conceptual leap.
- The blend must be phonologically tight and pronounceable, and must inflect like a normal Russian word.
- It should name something people recognize but didn't have a word for, and must not be an existing word or known slang.

Generate exactly __N__ fake slang terms as a JSON array:

[
  {
    "term": "Cyrillic term",
    "translit": "Latin transliteration of the term",
    "pos": "part of speech",
    "components": "Root A (meaning) + Root B (meaning) → blend logic",
    "definition_tar_lang": "what the blend means, in English, 1-2 sentences"
  }
]"""

# Russian suffixation, using the source taxonomy's affix inventory.
GEN_RU_SUFFIXATION = """Invent fake-but-plausible Russian internet slang using SUFFIXATION: attach a productive colloquial Russian suffix to an existing or invented root to coin an expressive new noun, verb, or adjective for a state, person-type, or behavior.

Suffixes (use a mix across your __N__ examples):
- -яга, -ак / -як, -ок: expressive personal/object nouns
- -няк: a chronic state or condition (cf. бардак-like expressive nouns)
- -уха: expressive feminine nouns for states or things
- -ня, -ёж, -лово: collective/process nouns, often dismissive
- -ота: a quality or atmosphere, with an unexpectedly warm or ironic tone

Examples:
- глючняк (-няк): a state of total mental fog, as if the brain is crashing — root глюч- (glitch) + -няк (chronic unpleasant state)
- слипуха (-уха): the groggy, barely-conscious state right after waking when you can't think straight — root слип- (слипаться, eyes glued shut)
- тыкалово (-лово): the futile ritual of jabbing at a frozen app or unresponsive interface over and over, hoping it will come back to life — root тык- (тыкать, to poke) + dismissive process suffix -лово

Rules:
- The coined word must describe something recognizable but not-yet-named.
- The suffix's register and the root's meaning must combine naturally — the word should sound like it could be real.
- The result must not be an existing word or known slang, and the meaning must not be transparently obvious from the root alone.

Generate exactly __N__ fake slang terms as a JSON array:

[
  {
    "term": "Cyrillic term",
    "translit": "Latin transliteration of the term",
    "pos": "part of speech",
    "affix": "which suffix and the flavor it adds",
    "base": "root and its meaning (real or invented)",
    "definition_tar_lang": "what the term means, in English, 1-2 sentences"
  }
]"""

# Russian verbalization of foreign loanwords; examples are authored and web-checked.
GEN_RU_FOREIGN_VERB = """Invent fake-but-plausible Russian internet slang using FOREIGN WORD RUSSIFICATION: take a non-Russian word (English, French, German, etc.) and russify it by adding native verbal or nominal morphology (-ить, -овать, -нуть, -ануть, -ировать, or noun endings) so it sounds and inflects like a native Russian word.

Examples (form only — invent your own, do not reuse these):
- мингловать (from English "mingle" + -овать) — to drift through a party or event making deliberately shallow small talk with everyone and real conversation with no one
- дедлайнуть (from "deadline" + -нуть) — to spring a sudden, non-negotiable deadline on someone at the last moment
- брунчевать (from "brunch" + -евать) — to stretch a late breakfast into an all-afternoon, fully legitimate excuse to do nothing else with the day

Rules:
- The foreign source word must NOT already be established in Russian slang in this form (avoid well-worn borrowings like чилить, флексить, кринжовать).
- The russified form must inflect naturally (conjugate/decline like a native word) and be phonologically comfortable in Russian.
- Avoid forms that sit close to existing Russian words — especially vulgar ones — in sound or spelling.
- The meaning should add a specific twist or scenario beyond the plain meaning of the source word — not just a translation of it.
- The meaning should reflect a recognizable modern phenomenon, behavior, feeling, or situation.

Generate exactly __N__ fake slang terms as a JSON array:

[
  {
    "term": "Cyrillic term",
    "translit": "Latin transliteration of the term",
    "pos": "part of speech",
    "source_word": "foreign source word and its language",
    "morphology": "which Russian affix/ending was added",
    "definition_tar_lang": "what it means, in English, 1-2 sentences"
  }
]"""


# Arabic semantic shift, with diacritics and Latin transliteration fields.

GEN_AR_SEMANTIC_SHIFT = """Invent fake-but-plausible Arabic internet slang using SEMANTIC SHIFT: take a real, ordinary Arabic word and assign it a radically new colloquial meaning through metaphor or metonymy, unrelated to (or only obliquely connected with) its dictionary definition.

Examples:
- عَتَبة (ʿataba, "threshold/doorstep") → the awkward moment when you have opened an app, chat, or email but still cannot bring yourself to actually enter or respond
- حِبر (ḥibr, "ink") → the emotional residue left after writing a long message, even if you delete it before sending
- غُبار (ghubār, "dust") → old notifications, unread chats, or saved posts that have piled up for so long that replying now feels embarrassing

Rules:
- The base must be a real Arabic word, but one with NO documented slang meaning.
- The new meaning must not be predictable from the literal meaning alone — there should be a figurative leap, not a paraphrase.
- The meaning should name a recognizable modern phenomenon, behavior, feeling, or situation.
- The term should be written in Arabic script and include full or partial diacritics when useful for disambiguation.
- The transliteration should use a clear scholarly-style Latin transliteration.

Generate exactly __N__ fake slang terms as a JSON array:

[
  {
    "term": "Arabic-script term",
    "translit": "Latin transliteration of the term",
    "base": "literal/dictionary meaning of the word",
    "definition_tar_lang": "the new slang meaning, in English, 1-2 sentences"
  }
]"""

# Arabic morphological blend, allowing Arabic and assimilated foreign roots.
# Because ـجي is productive, outputs using it need careful collision checks.
GEN_AR_MORPH_BLEND = """Invent fake-but-plausible Arabic internet/youth slang using MORPHOLOGICAL TRUNCATION-BLEND (النحت المعاصر): consciously chop and fuse two existing words or morphemes (Arabic + Arabic, or Arabic + assimilated foreign) into a single new pseudo-noun or expression whose meaning goes beyond either component.
Examples:
- بلوكتئاب = the specific low mood that sets in after you block someone (or get blocked), when the finality of the severed connection actually lands, from بلوك (to block, social-media sense) + اكتئاب (depression) → block-induced melancholy
- سكرولان = the horizontal paralysis where you lie down "for one minute," open your phone, and scroll for an hour physically unable to get up, from سكرول (to scroll) + كسلان (sluggish/lazy) → scroll-induced inertia
- ديليفرجي = someone who has routed all of daily life through delivery apps and feels low-grade dread at the idea of physically going to a shop, from ديليفري (delivery) + ـجي (occupational/trade suffix) → the delivery-dependent person as a "profession"
Rules:
- The result should read as a single new concept, not just a noun phrase.
- The meaning should not be fully predictable from the parts — there should be a conceptual leap.
- Favor fusing modern tech, commercial, or social phrases; the ـجي suffix and short, rapid-fire compounds are characteristic but not required.
- The blend should be phonologically tight and pronounceable.
- It should name something people recognize but didn't have a word for, and must not be an existing word or known slang.
Generate exactly __N__ fake slang terms as a JSON array:
[
  {
    "term": "Arabic script",
    "translit": "Latin transliteration of the term",
    "pos": "part of speech",
    "components": "Word A (meaning) + Word B (meaning) → blend logic",
    "definition_tar_lang": "what the blend means, in English, 1-2 sentences"
  }
]"""


# Farsi types derive from a native-speaker contributor's taxonomy. Transliteration
# is retained for reviewers. Few-shot examples must still pass the existence gate.

# Farsi light-verb construction with a non-compositional slang meaning.
GEN_FA_LIGHT_VERB = """Invent fake-but-plausible Farsi (Persian) slang using LIGHT VERB CONSTRUCTION: pair a noun or adjective (native, or an assimilated English/foreign loanword) with a native Persian light verb to coin a new compound verb whose slang meaning goes beyond the literal parts.
Light verbs to draw from (use a mix across your __N__ examples): کردن (to do/make), شدن (to become), زدن (to strike), گرفتن (to take/catch), دادن (to give), آوردن (to bring), کشیدن (to drag/pull), بستن (to tie/close), خوردن (to eat).
Examples:
- لاک‌پشت شدن (lâkposht shodan – literally "to become a turtle") → going completely dark on all messaging for days, pulling into your shell after being socially overwhelmed
- تیتر زدن (titr zadan – literally "to strike a headline") → hyping up a totally mundane personal update as if it were breaking, headline-worthy news
- گردگیری کردن (gardgiri kardan – literally "to do dusting") → belatedly combing back through your own old posts to quietly scrub embarrassing likes and comments before someone important looks you up
Rules:
- The compound's meaning must NOT be predictable from the noun/adjective + verb literally — there must be a figurative leap.
- The light verb must be one of the native light verbs above (not an invented verb).
- The meaning should name a recognizable modern phenomenon, behavior, feeling, or situation.
- The compound must conjugate naturally like a normal Persian compound verb.
- Do not reuse existing well-known compound-verb slang (e.g. مخ زدن, تیغ زدن, خالی بستن).
Generate exactly __N__ fake slang terms as a JSON array:
[
  {
    "term": "Persian term (noun/adjective + light verb)",
    "translit": "Latin transliteration of the term",
    "light_verb": "which light verb and its literal meaning",
    "base": "the noun/adjective and its literal meaning",
    "definition_tar_lang": "the new slang meaning, in English, 1-2 sentences"
  }
]"""

# Farsi affixation using the contributor's productive-affix inventory.
GEN_FA_AFFIXATION = """Invent fake-but-plausible Farsi (Persian) slang using AFFIXATION: attach a productive Persian prefix or suffix to a novel base (native or an assimilated loanword) to coin a new term for a person-type, social group, style, or behavior.
Affixes (use a mix across your __N__ examples):
- باز- (-bâz): someone obsessed with, playing at, or generating drama around the root
- خور- (-khor, "eater"): someone who suffers from or greedily consumes the root
- طور- (-toor): doing something in the manner / vibe / style of the root
- مال- (-mâl, "rubber"): one who flatters or sucks up via the root
- بی- (bi-, "without"): a cutting character judgment — lacking the root quality
- گنده- (gondeh-, "huge"): a false, inflated sense of grandeur about the root
- سه- (se-, "three"): doing the root to the utmost / in a flash
- چهل- (chehel-, "forty"): an abundant, chaotic, or excessive amount of the root
Examples:
- استوری‌باز (-bâz): someone addicted to documenting every trivial moment of their day in their stories, narrating life as if always on air
- لایک‌خور (-khor): a person who constantly fishes for validation, posting obvious bait just to "feed" on the likes and comments
- قهرطور (-toor): doing something in a pointedly sulky, passive-aggressive manner so everyone can tell you're offended without you saying it
Rules:
- The coined term must describe something recognizable but not-yet-named.
- The affix's flavor and the base's meaning must combine naturally — it should sound like it could be real.
- The meaning should describe a recognizable kind of person, group, style, or behavior.
- Do not reuse existing well-known slang (e.g. دخترباز, غصه‌خور, خایه‌مال, بی‌جنبه).
Generate exactly __N__ fake slang terms as a JSON array:
[
  {
    "term": "Persian term",
    "translit": "Latin transliteration of the term",
    "affix": "which affix and the flavor it adds",
    "base": "root and its meaning (native or loanword)",
    "definition_tar_lang": "what the term means, in English, 2-3 sentences"
  }
]"""

# Farsi semantic shift of an ordinary word.
GEN_FA_SEMANTIC_SHIFT = """Invent fake-but-plausible Farsi (Persian) slang using SEMANTIC SHIFT: take an ordinary, everyday Persian word and assign it a radically new colloquial meaning through metaphor, metonymy, or subversion, unrelated to (or only obliquely connected with) its normal meaning.
Examples:
- آباژور ("a lampshade") → the person who lingers at the edge of a group hoping to be included but contributes nothing, just quietly soaking up the warmth and light
- زیرنویس ("subtitle") → the quiet running commentary a friend gives you about the real subtext of what's happening; "بی‌زیرنویس نمی‌فهمم" = I can't follow the situation without the subtitles
- پریز ("electrical outlet/socket") → a friend you only ever plug into when you need something — energy, a favor, a ride — and unplug the moment you're done
Rules:
- The base must be a real, ordinary Persian word with NO documented slang meaning.
- The new meaning must not be predictable from the literal meaning alone — there should be a figurative leap, not a paraphrase.
- The meaning should name a recognizable modern phenomenon, behavior, feeling, or situation.
- The word should keep its normal Persian grammar in the new sense.
Generate exactly __N__ fake slang terms as a JSON array:
[
  {
    "term": "Persian word",
    "translit": "Latin transliteration of the term",
    "base": "the ordinary/literal meaning of the word",
    "definition_tar_lang": "the new slang meaning, in English, 1-2 sentences"
  }
]"""


@dataclass(frozen=True)
class GenPrompt:
    """One generation-prompt variant for a language and slang type."""
    lang: str    # "zh" | "ru" | "ar" | "fa"
    type: str    # "semantic_shift" | "morph_blend" | "affixation" | "sentence_compression" | "suffixation" | "foreign_verb" | "light_verb" | ...
    name: str    # short id, unique within a (lang, type)
    prompt: str


# The registry is a flat array — EDIT HERE to add or remove generation prompts.
GENERATION_PROMPTS: list[GenPrompt] = [
    GenPrompt("zh", "semantic_shift", "default", GEN_ZH_SEMANTIC_SHIFT),
    GenPrompt("zh", "morph_blend", "default", GEN_ZH_MORPH_BLEND),
    GenPrompt("zh", "affixation", "default", GEN_ZH_AFFIXATION),
    GenPrompt("zh", "sentence_compression", "default", GEN_ZH_SENTENCE_COMPRESSION),
    # Shared category names support cross-language comparisons.
    GenPrompt("ru", "semantic_shift", "default", GEN_RU_SEMANTIC_SHIFT),
    GenPrompt("ru", "morph_blend", "default", GEN_RU_MORPH_BLEND),
    GenPrompt("ru", "suffixation", "default", GEN_RU_SUFFIXATION),
    GenPrompt("ru", "foreign_verb", "default", GEN_RU_FOREIGN_VERB),
    # Arabic taxonomy — types 1–2 so far; types 3–4 to be added.
    GenPrompt("ar", "semantic_shift", "default", GEN_AR_SEMANTIC_SHIFT),
    GenPrompt("ar", "morph_blend", "default", GEN_AR_MORPH_BLEND),
    # Farsi uses shared semantic_shift/affixation labels plus light_verb.
    GenPrompt("fa", "light_verb", "default", GEN_FA_LIGHT_VERB),
    GenPrompt("fa", "affixation", "default", GEN_FA_AFFIXATION),
    GenPrompt("fa", "semantic_shift", "default", GEN_FA_SEMANTIC_SHIFT),
    # Add specialized variants or whole new types here as they are written, e.g.:
    #   GenPrompt("ru", "semantic_shift", "misdirect", GEN_RU_SEMANTIC_SHIFT_MISDIRECT),
    #   GenPrompt("ru", "clipping", "default", GEN_RU_CLIPPING),
]


def select_generation_prompts(
    lang: str, type_: str | None = None, name: str | None = None
) -> list[GenPrompt]:
    """Resolve prompts by language, optional type, and optional variant name."""
    pool = [g for g in GENERATION_PROMPTS if g.lang == lang]
    if not pool:
        langs = ", ".join(sorted({g.lang for g in GENERATION_PROMPTS}))
        raise ValueError(f"no generation prompts for lang '{lang}'; have: {langs}")
    if type_ is not None:
        pool = [g for g in pool if g.type == type_]
        if not pool:
            types = ", ".join(sorted({g.type for g in GENERATION_PROMPTS if g.lang == lang}))
            raise ValueError(f"no '{type_}' prompt for {lang}; have types: {types}")
    if name is not None:
        pool = [g for g in pool if g.name == name]
        if not pool:
            raise ValueError(f"no variant '{name}' for {lang}/{type_}")
    return pool


def get_generation_prompt(lang: str, type_: str, name: str | None = None) -> str:
    """Return a prompt with its per-call item count filled in."""
    prompt = select_generation_prompts(lang, type_, name)[0].prompt
    return prompt.replace("__N__", str(TERMS_PER_PROMPT))


def list_generation_prompts(lang: str | None = None) -> list[tuple[str, str, str]]:
    """(lang, type, name) for each registered prompt — for CLI discovery / --list."""
    return [(g.lang, g.type, g.name) for g in GENERATION_PROMPTS
            if lang is None or g.lang == lang]


def with_avoid_list(prompt: str, terms: list[str]) -> str:
    """Append prior terms so later batches avoid duplicates in form and meaning."""
    terms = [t for t in terms if t]
    if not terms:
        return prompt
    listed = "\n".join(f"- {t}" for t in terms)
    avoid = (
        "\n\nIMPORTANT — the terms below have ALREADY been generated. Do NOT repeat "
        "any of them, and do NOT coin new terms that are close to them in form OR in "
        "meaning. Every term you return must be clearly distinct from all of these:\n"
        f"{listed}"
    )
    return prompt + avoid


# Step 2: generate sentence and paragraph contexts. Chinese uses character limits;
# Russian uses equivalent word-count limits.

_SENTENCE_PARAGRAPH_INSTRUCTION_ZH = """Write two pieces of natural Chinese social media text using this term. Return JSON:

{
  "sentence_in_context": "single comment or post, 15–30 characters, meaning approximately inferable from context — do not paraphrase the definition directly",
  "sentence_in_context_gloss": "English translation",
  "paragraph_in_context": "short post or thread excerpt, 3–5 sentences, 60–120 characters total, term appears naturally without explanation",
  "paragraph_in_context_gloss": "English translation"
}"""

_SENTENCE_PARAGRAPH_INSTRUCTION_RU = """Write two pieces of natural Russian social media text using this term. Return JSON:

{
  "sentence_in_context": "single comment or post, 8–18 words, meaning approximately inferable from context — do not paraphrase the definition directly",
  "sentence_in_context_gloss": "English translation",
  "paragraph_in_context": "short post or thread excerpt, 3–5 sentences, 35–70 words total, term appears naturally (and inflected naturally) without explanation",
  "paragraph_in_context_gloss": "English translation"
}"""

_SENTENCE_PARAGRAPH_INSTRUCTION_AR = """Write two pieces of natural Arabic social media text using this term. Return JSON:

{
  "sentence_in_context": "single comment or post, 8–16 words, meaning approximately inferable from context — do not paraphrase the definition directly",
  "sentence_in_context_gloss": "English translation",
  "paragraph_in_context": "short post or thread excerpt, 3–5 sentences, 30–60 words total, term appears naturally without explanation",
  "paragraph_in_context_gloss": "English translation"
}"""

_SENTENCE_PARAGRAPH_INSTRUCTION_FA = """Write two pieces of natural Farsi (Persian) social media text using this term. Return JSON:
{
  "sentence_in_context": "single comment or post, 8–18 words, meaning approximately inferable from context — do not paraphrase the definition directly",
  "sentence_in_context_gloss": "English translation",
  "paragraph_in_context": "short post or thread excerpt, 3–5 sentences, 35–70 words total, term appears naturally (and conjugated/inflected naturally) without explanation",
  "paragraph_in_context_gloss": "English translation"
}"""

_SENTENCE_PARAGRAPH_INSTRUCTIONS = {
    "zh": _SENTENCE_PARAGRAPH_INSTRUCTION_ZH,
    "ru": _SENTENCE_PARAGRAPH_INSTRUCTION_RU,
    "ar": _SENTENCE_PARAGRAPH_INSTRUCTION_AR,
    "fa": _SENTENCE_PARAGRAPH_INSTRUCTION_FA,
}

# Backwards-compatible alias (pre-Russian code referenced the unsuffixed name).
_SENTENCE_PARAGRAPH_INSTRUCTION = _SENTENCE_PARAGRAPH_INSTRUCTION_ZH


def build_sentence_paragraph_prompt(
    term: str,
    definition: str,
    *,
    lang: str = "zh",
    source: str | None = None,
    source_gloss: str | None = None,
) -> str:
    """Build a context-generation prompt from an item and language template."""
    try:
        instruction = _SENTENCE_PARAGRAPH_INSTRUCTIONS[lang]
    except KeyError:
        have = ", ".join(sorted(_SENTENCE_PARAGRAPH_INSTRUCTIONS))
        raise ValueError(f"no step-2 instruction for lang '{lang}'; have: {have}")
    payload: dict[str, str] = {"term": term, "definition": definition}
    if source is not None:
        payload["source"] = source
        payload["source_gloss"] = source_gloss or ""
    block = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"{block}\n\n{instruction}"


# Gate 2 prompt: guess meaning from the bare term.

_GATE_GUESS_SCHEMA = '{\n  "guess": "your best guess at the meaning, in English"\n}'


def build_gate_guess_prompt(term: str, lang: str, pos: str | None = None) -> str:
    """Show the term with NO context; ask the model to guess from form alone."""
    form_basis = _GUESS_FORM_BASIS.get(lang, "form")
    lang_name = LANGUAGE_NAMES.get(lang, lang)
    pos_line = f"Part of speech: {pos}\n" if pos else ""
    return (
        f"You will be shown a {lang_name} slang term with NO context. Guess what it "
        f"most likely means, reasoning only from its {form_basis} — its form, "
        "components, and sound. Do not search; give your single best interpretation.\n\n"
        f"Term: {term}\n{pos_line}\n"
        f"Return JSON:\n{_GATE_GUESS_SCHEMA}"
    )


# Gate 1 prompt: classify search snippets by sense and detect form collisions.

_GATE1_SENSE_SCHEMA = """{
  "verdicts": [{"index": 1, "verdict": "in_sense | different_sense | noise"}],
  "form_collision": false,
  "collision_sense": "if the term exists as a real word/name/idiom, what it means there (else null)",
  "collision_adjacent": false
}"""


def build_gate1_sense_judge_prompt(
    term: str, definition: str, hits: list[dict[str, str]]
) -> str:
    """Adjudicate search snippets: does each one use the term IN ITS ASSIGNED slang
    sense, in some different sense, or is it noise (username, typo, unrelated string
    match)? Separately: does the term collide with a real word/name, and if so is
    that colliding sense adjacent to the assigned one?"""
    listed = "\n".join(
        f"[{i + 1}] {h.get('title', '')} ({h.get('url', '')})\n    {h.get('snippet', '')}"
        for i, h in enumerate(hits)
    )
    return (
        "A candidate slang term was web-searched to check whether it already exists "
        "in its assigned sense. Judge each search result snippet below.\n\n"
        f"Term: {term}\n"
        f"Assigned slang sense: {definition}\n\n"
        f"Search results:\n{listed}\n\n"
        "For each result, give a verdict:\n"
        "- in_sense: the page uses or defines the term in (approximately) the assigned sense\n"
        "- different_sense: the term appears as a real word, name, title, or other slang "
        "with a DIFFERENT meaning\n"
        "- noise: incidental string match (username, typo, tokenization accident, term "
        "absent from snippet)\n\n"
        "Also report form_collision: does the term exist as an established word, name, "
        "idiom, or product? If yes, give that meaning in collision_sense and set "
        "collision_adjacent=true ONLY if that existing meaning is close enough to the "
        "assigned sense that knowing the existing word would let someone decode the "
        "slang meaning.\n\n"
        f"Return JSON:\n{_GATE1_SENSE_SCHEMA}"
    )

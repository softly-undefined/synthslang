# Gate 1 tests time-stamped public-web novelty by sense, not raw string occurrence.
# DuckDuckGo is the default retriever, with Brave as fallback. Local embeddings or
# an optional LLM judge adjudicate results; evidence is cached for resumed runs.

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests
from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

SYNTHSLANG_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SYNTHSLANG_DIR))

import inference  # noqa: E402  (after sys.path insert)
import prompts  # noqa: E402

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
BRAVE_PAUSE = 1.1  # seconds between queries — free tier is rate-limited to ~1 req/s

# "Does X mean something" query suffixes, per language — these surface the
# definitional / meme-explainer pages where a real slang sense would live.
SENSE_QUERY_SUFFIXES = {
    "zh": ["什么意思", "梗", "是什么"],
    "ru": ["значит", "это что", "сленг"],
    "ar": ["معنى", "وش يعني", "سلانج"],
    "fa": ["یعنی چی", "اصطلاح", "معنی"],
}


def sense_keywords(definition: str, n: int = 3) -> list[str]:
    """Return the longest definition words for sense-specific search queries."""
    words = [w.strip(".,;:()\"'") for w in definition.split()]
    words = [w for w in words if w.isalpha()]
    return sorted(set(words), key=len, reverse=True)[:n]


def build_query_set(term: str, definition: str, lang: str) -> list[str]:
    """The documented query templates: bare term, quoted term, term + per-language
    'what does it mean' suffixes, term + sense keywords from the definition."""
    queries = [term, f'"{term}"']
    queries += [f"{term} {suffix}" for suffix in SENSE_QUERY_SUFFIXES.get(lang, [])]
    kws = sense_keywords(definition)
    if kws:
        queries.append(f"{term} " + " ".join(kws))
    return queries


def require_brave_key() -> str:
    key = os.getenv("BRAVE_API_KEY")
    if not key:
        raise SystemExit(
            "BRAVE_API_KEY is not set. Gate 1 retrieves through the Brave Search API:\n"
            "  export BRAVE_API_KEY=...   (free tier: https://api-dashboard.search.brave.com)"
        )
    return key


def brave_search(query: str, api_key: str, count: int = 10) -> list[dict[str, str]]:
    """One Brave web search → [{title, url, snippet}]. Retries once on 429."""
    headers = {"X-Subscription-Token": api_key, "Accept": "application/json"}
    params = {"q": query, "count": count}
    for attempt in range(2):
        resp = requests.get(BRAVE_ENDPOINT, headers=headers, params=params, timeout=30)
        if resp.status_code == 429 and attempt == 0:
            time.sleep(5)
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"Brave search failed ({resp.status_code}) for {query!r}: "
                               f"{resp.text[:200]}")
        break
    results = (resp.json().get("web") or {}).get("results") or []
    return [{"title": r.get("title", ""), "url": r.get("url", ""),
             "snippet": r.get("description", "")} for r in results]


def retrieve_brave(term: str, definition: str, lang: str, *,
                   count: int = 10) -> dict[str, Any]:
    """Run the full query set; hits deduped by URL, each tagged with the query that
    found it. Returns {"queries": [...], "hits": [...]}."""
    api_key = require_brave_key()
    queries = build_query_set(term, definition, lang)
    hits: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for q in queries:
        for hit in brave_search(q, api_key, count=count):
            if hit["url"] and hit["url"] not in seen_urls:
                seen_urls.add(hit["url"])
                hits.append({**hit, "query": q})
        time.sleep(BRAVE_PAUSE)
    return {"queries": queries, "hits": hits}


DDG_PAUSE = 2.5  # seconds between queries — DDG blocks by IP; pause hard, not 1.1s
DDG_BACKOFF = (15, 45, 90)  # escalating sleeps on rate-limit before giving up the query

# DDG region codes per source-language track — steer results to the zh / ru web, where
# a real slang sense would actually be discussed.
DDG_REGIONS = {"zh": "cn-zh", "ru": "ru-ru", "ar": "xa-ar", "fa": "ir-fa"}


def ddg_search(query: str, *, count: int = 10, region: str = "wt-wt") -> list[dict[str, str]]:
    """One DuckDuckGo web search → [{title, url, snippet}]. Escalating backoff on
    rate-limit / timeout; raises once exhausted so the run stops cleanly and can be
    resumed (the per-term Gate 1 cache skips finished work)."""
    results: list[dict[str, Any]] = []
    for attempt, wait in enumerate((0, *DDG_BACKOFF)):
        if wait:
            time.sleep(wait)
        try:
            results = DDGS().text(query, region=region, max_results=count)
            break
        except (RatelimitException, TimeoutException) as exc:  # subclasses of DDGSException
            if attempt == len(DDG_BACKOFF):
                raise RuntimeError(
                    f"DDG gave up on {query!r} after {attempt + 1} tries ({exc}). "
                    "Raise DDG_PAUSE or resume later — the per-term cache skips done work."
                ) from exc
        except DDGSException as exc:
            # ddgs reports genuine empty results as an exception; rate limits use a
            # different exception and continue through the retry path.
            if "No results found" in str(exc):
                return []
            if attempt == len(DDG_BACKOFF):
                raise RuntimeError(
                    f"DDG gave up on {query!r} after {attempt + 1} tries ({exc}). "
                    "Raise DDG_PAUSE or resume later — the per-term cache skips done work."
                ) from exc
    return [{"title": r.get("title", ""), "url": r.get("href", ""),
             "snippet": r.get("body", "")} for r in results]


def retrieve_ddg(term: str, definition: str, lang: str, *,
                 count: int = 10) -> dict[str, Any]:
    """DuckDuckGo retriever — free, keyless drop-in for Brave. Same query set; hits
    deduped by URL and tagged with the query that found them. Region-steered per track."""
    queries = build_query_set(term, definition, lang)
    region = DDG_REGIONS.get(lang, "wt-wt")
    hits: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for q in queries:
        for hit in ddg_search(q, count=count, region=region):
            if hit["url"] and hit["url"] not in seen_urls:
                seen_urls.add(hit["url"])
                hits.append({**hit, "query": q})
        time.sleep(DDG_PAUSE)
    return {"queries": queries, "hits": hits}


# Pluggable retrievers — add a local corpus index here later without touching the
# decision logic below. Signature: (term, definition, lang, *, count) -> {queries, hits}.
RETRIEVERS = {"ddg": retrieve_ddg, "brave": retrieve_brave}


def domain_of(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def adjudicate_sense(client, judge_model: str, term: str, definition: str,
                     hits: list[dict[str, str]]) -> dict[str, Any]:
    """The precision step: one LLM call sorts every snippet into in_sense /
    different_sense / noise and flags form collisions. Returns the parsed judgment
    plus the per-hit verdicts zipped back onto the hits."""
    prompt = prompts.build_gate1_sense_judge_prompt(term, definition, hits)
    # Omit temperature for reasoning models that reject an explicit value.
    raw = inference.chat(client, judge_model, prompt, temperature=None)
    data = inference.extract_json(raw)
    if not isinstance(data, dict) or "verdicts" not in data:
        raise ValueError(f"sense judge returned no verdicts: {raw[:200]!r}")
    verdict_by_index = {int(v["index"]): v.get("verdict", "noise")
                        for v in data["verdicts"] if "index" in v}
    missing = [i + 1 for i in range(len(hits)) if i + 1 not in verdict_by_index]
    if missing:
        # Retry missing verdicts instead of treating them as evidence for a pass.
        raise ValueError(f"sense judge skipped snippet(s) {missing}: {raw[:200]!r}")
    judged = [{**hit, "sense_verdict": verdict_by_index[i + 1]}
              for i, hit in enumerate(hits)]
    return {
        "judged_hits": judged,
        "form_collision": bool(data.get("form_collision")),
        "collision_sense": data.get("collision_sense"),
        "collision_adjacent": bool(data.get("collision_adjacent")),
    }


def load_embedder(model_name: str):
    """Load a reusable local embedder that returns normalized sentence vectors."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    def embed(texts: list[str]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), 32):
                chunk = texts[start : start + 32]
                enc = tok(chunk, padding=True, truncation=True, max_length=256,
                          return_tensors="pt").to(device)
                hidden = model(**enc).last_hidden_state            # (B, T, H)
                mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
                mean = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                mean = torch.nn.functional.normalize(mean, p=2, dim=1)
                vectors.append(mean.cpu().numpy().astype(np.float32))
        return np.vstack(vectors)

    return embed


def embed_texts(texts: list[str], model_name: str) -> np.ndarray:
    """One-shot load + embed (near-dup runs this once over the whole batch)."""
    return load_embedder(model_name)(texts)


def adjudicate_sense_embedding(embed, definition: str,
                               hits: list[dict[str, str]]) -> dict[str, Any]:
    """Attach cross-lingual definition/snippet cosine scores to search hits.

    This topical signal is weaker than LLM adjudication and cannot detect form
    collisions. Decisions are applied later so the cutoff remains tunable.
    """
    texts = [definition] + [f"{h.get('title', '')} {h.get('snippet', '')}".strip()
                            for h in hits]
    vecs = embed(texts)
    judged = [{**hit, "sense_sim": float(vecs[0] @ vecs[i + 1])}
              for i, hit in enumerate(hits)]
    return {"judged_hits": judged, "form_collision": False,
            "collision_sense": None, "collision_adjacent": False}


def gather_gate1(record: dict[str, Any], *, retriever: str, count: int, snapshot_date: str,
                 sense_judge: str = "embedding", embed=None,
                 client=None, judge_model: str | None = None) -> dict[str, Any]:
    """Retrieve and adjudicate one term, returning cacheable Gate 1 evidence."""
    term = record["term"]
    definition = record.get("definition_tar_lang") or record.get("explanation_tar_lang") or ""
    lang = record.get("lang", "zh")
    retrieved = RETRIEVERS[retriever](term, definition, lang, count=count)

    if not retrieved["hits"]:  # nothing retrieved — trivially unseen, no judge needed
        judgment = {"judged_hits": [], "form_collision": False,
                    "collision_sense": None, "collision_adjacent": False}
    elif sense_judge == "embedding":
        judgment = adjudicate_sense_embedding(embed, definition, retrieved["hits"])
    else:
        judgment = adjudicate_sense(client, judge_model, term, definition, retrieved["hits"])

    return {
        "gate1_query_set": retrieved["queries"],
        "gate1_raw_hits": len(retrieved["hits"]),
        "gate1_snippets": judgment["judged_hits"],
        "gate1_form_collision": judgment["form_collision"],
        "gate1_collision_sense": judgment["collision_sense"],
        "gate1_collision_adjacent": judgment["collision_adjacent"],
        "gate1_retriever": retriever,
        "gate1_sense_judge": sense_judge,
        "gate1_snapshot_date": snapshot_date,
    }


def decide_gate1(evidence: dict[str, Any], threshold: int, *,
                 snippet_sim_threshold: float = 0.5) -> dict[str, Any]:
    """Apply source-count, similarity, and form-collision rules to cached evidence."""
    snippets = evidence.get("gate1_snippets", [])
    if snippets and "sense_sim" in snippets[0]:        # embedding mode
        in_sense = [h for h in snippets if h["sense_sim"] >= snippet_sim_threshold]
    else:                                              # llm mode (or no hits)
        in_sense = [h for h in snippets if h.get("sense_verdict") == "in_sense"]
    sources = sorted({domain_of(h["url"]) for h in in_sense if h.get("url")})
    n_in = len(in_sense)

    if len(sources) >= threshold:
        passed, reason = False, (f"{n_in} in-sense hit(s) from {len(sources)} "
                                 f"independent source(s) >= threshold {threshold}")
    elif evidence.get("gate1_form_collision") and evidence.get("gate1_collision_adjacent"):
        passed, reason = False, (f"form collision adjacent to assigned sense: "
                                 f"{evidence.get('gate1_collision_sense')}")
    else:
        bits = [f"{n_in} in-sense / {evidence.get('gate1_raw_hits', 0)} hits"]
        if evidence.get("gate1_form_collision"):
            bits.append(f"non-adjacent form collision noted: {evidence.get('gate1_collision_sense')}")
        passed, reason = True, "; ".join(bits)
    return {
        "gate1_in_sense_hits": n_in,
        "gate1_sources": sources,
        "gate1_threshold": threshold,
        "gate1_snippet_sim_threshold": snippet_sim_threshold,
        "gate1_pass": passed,
        "gate1_reason": reason,
    }


def run_gate1(record: dict[str, Any], *, retriever: str, threshold: int, count: int,
              snapshot_date: str, sense_judge: str = "embedding", embed=None,
              snippet_sim_threshold: float = 0.5,
              client=None, judge_model: str | None = None) -> dict[str, Any]:
    """Convenience wrapper: gather evidence, then decide. The contamination-gate
    pipeline caches gather_gate1 and calls decide_gate1 each run so both thresholds
    stay tunable."""
    evidence = gather_gate1(record, retriever=retriever, count=count,
                            snapshot_date=snapshot_date, sense_judge=sense_judge,
                            embed=embed, client=client, judge_model=judge_model)
    return {**evidence, **decide_gate1(evidence, threshold,
                                       snippet_sim_threshold=snippet_sim_threshold)}


# ── consensus: both sense judges over one retrieval ──────────────────────────

def _judge_slice(judgment: dict[str, Any], raw_hits: int) -> dict[str, Any]:
    """Reshape one judge's output into the keys decide_gate1 reads, so each judge can be
    decided independently off the shared evidence."""
    return {"gate1_snippets": judgment["judged_hits"],
            "gate1_form_collision": judgment["form_collision"],
            "gate1_collision_sense": judgment["collision_sense"],
            "gate1_collision_adjacent": judgment["collision_adjacent"],
            "gate1_raw_hits": raw_hits}


def gather_gate1_both(record: dict[str, Any], *, retriever: str, count: int,
                      snapshot_date: str, embed, client,
                      judge_model: str | None) -> dict[str, Any]:
    """Run both sense judges over one shared retrieval result."""
    term = record["term"]
    definition = record.get("definition_tar_lang") or record.get("explanation_tar_lang") or ""
    lang = record.get("lang", "zh")
    retrieved = RETRIEVERS[retriever](term, definition, lang, count=count)
    hits = retrieved["hits"]
    if not hits:  # nothing retrieved — trivially unseen for both judges
        empty = {"judged_hits": [], "form_collision": False,
                 "collision_sense": None, "collision_adjacent": False}
        emb_j, llm_j = empty, dict(empty)
    else:
        emb_j = adjudicate_sense_embedding(embed, definition, hits)
        llm_j = adjudicate_sense(client, judge_model, term, definition, hits)
    return {
        "gate1_query_set": retrieved["queries"],
        "gate1_raw_hits": len(hits),
        "gate1_retriever": retriever,
        "gate1_sense_judge": "consensus",
        "gate1_snapshot_date": snapshot_date,
        "gate1_embedding": _judge_slice(emb_j, len(hits)),
        "gate1_llm": _judge_slice(llm_j, len(hits)),
    }


def decide_consensus(evidence: dict[str, Any], threshold: int, *,
                     snippet_sim_threshold: float = 0.5) -> dict[str, Any]:
    """Combine embedding and LLM decisions into pass, remove, or human escalation."""
    emb = decide_gate1(evidence["gate1_embedding"], threshold,
                       snippet_sim_threshold=snippet_sim_threshold)
    llm = decide_gate1(evidence["gate1_llm"], threshold,
                       snippet_sim_threshold=snippet_sim_threshold)
    emb_pass, llm_pass = emb["gate1_pass"], llm["gate1_pass"]

    if not emb_pass and not llm_pass:
        consensus, passed, human_evidence = "remove", False, ""
        reason = (f"both judges flag existence (embedding: {emb['gate1_reason']}; "
                  f"llm: {llm['gate1_reason']})")
    elif emb_pass and not llm_pass:
        consensus, passed, human_evidence = "escalate", True, llm["gate1_reason"]
        reason = f"escalate to human — embedding keeps, llm flags: {llm['gate1_reason']}"
    elif not emb_pass and llm_pass:
        consensus, passed, human_evidence = "llm_kept", True, ""
        reason = f"kept on llm judge; embedding flagged (not escalated): {emb['gate1_reason']}"
    else:
        consensus, passed, human_evidence = "pass", True, ""
        reason = "both judges keep"

    return {
        "gate1_embedding_pass": emb_pass,
        "gate1_llm_pass": llm_pass,
        "gate1_embedding_reason": emb["gate1_reason"],
        "gate1_llm_reason": llm["gate1_reason"],
        "gate1_llm_collision_sense": evidence["gate1_llm"].get("gate1_collision_sense"),
        "gate1_embedding_sources": emb["gate1_sources"],
        "gate1_llm_sources": llm["gate1_sources"],
        "gate1_consensus": consensus,
        "gate1_pass": passed,
        "gate1_reason": reason,
        "gate1_llm_evidence": human_evidence,
    }

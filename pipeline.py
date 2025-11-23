#!/usr/bin/env python3
"""
Schema matching CLI: loads CSV/JSON datasets, auto-generates neighbors,
optionally uses .ttl semantic models, builds lexical+semantic indexes,
blocks candidates, and calls the LLM once with the full context.
"""

import os
import re
import json
import hashlib
import math
import argparse
from collections import defaultdict, Counter
#import pdb

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.neighbors import NearestNeighbors

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()  # load environment variables from .env if present

# rdflib is optional (for .ttl semantic models)
try:
    from rdflib import Graph
    RDFlIB_AVAILABLE = True
except Exception:
    RDFlIB_AVAILABLE = False

np.random.seed(0)

client = None


def norm_name(s):
    s = str(s)
    s = re.sub(r"\s+", " ", s.strip())
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = s.replace("_", " ").replace("-", " ").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens_12(s):
    w = s.split()
    b = [" ".join(w[i:i+2]) for i in range(len(w)-1)]
    return w + b

def char_ngrams(s, low=3, high=5):
    s = "_" + re.sub(r"[^\w]", "_", s) + "_"
    out = []
    for n in range(low, high+1):
        out += ["".join(s[i:i+n]) for i in range(len(s)-n+1)]
    return out

def coarse_type_from_dtype(dtype):
    if pd.api.types.is_integer_dtype(dtype): return "int"
    if pd.api.types.is_float_dtype(dtype): return "float"
    if pd.api.types.is_bool_dtype(dtype): return "bool"
    if pd.api.types.is_datetime64_any_dtype(dtype): return "datetime"
    return "string"

def sample_values(col, n=5):
    s = col.dropna().astype(str)
    if s.empty:
        return []
    vc = s.value_counts()
    k = min(max(1, n//2), len(vc))
    top = vc.index.tolist()[:k]
    rest = [v for v in s.unique().tolist() if v not in set(top)]
    vals = (top + rest)[:n]
    return vals

def length_stats(vals):
    if not vals:
        return {"min": None, "max": None, "mean": None, "stdev": None}
    L = [len(str(v)) for v in vals]
    return {"min": int(np.min(L)), "max": int(np.max(L)), "mean": float(np.mean(L)), "stdev": float(np.std(L))}

def char_stats(vals):
    if not vals:
        return {"digit_ratio": None, "alpha_ratio": None, "charset": []}
    vals = [str(v) for v in vals]
    tot = sum(len(v) for v in vals)
    d = sum(sum(c.isdigit() for c in v) for v in vals)
    a = sum(sum(c.isalpha() for c in v) for v in vals)
    charset = sorted(list({c for v in vals for c in v}))[:64]
    return {"digit_ratio": d/tot if tot else None, "alpha_ratio": a/tot if tot else None, "charset": charset}

def pattern_tags(vals):
    tags = set()
    for s in vals:
        if re.fullmatch(r"\d{4,6}", s): tags.add("POSTAL")
        if re.fullmatch(r"[A-Z]{2}", s): tags.add("COUNTRY_ALPHA2")
        if "@" in s and "." in s:
            if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", s): tags.add("EMAIL")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s): tags.add("ISO_DATE")
    return sorted(list(tags))

def distinct_ratio(series):
    n = len(series)
    if n == 0: return 0.0
    return series.dropna().nunique() / n

def context_string(table, col_norm, neighbors, desc=None):
    parts = [table, col_norm] + neighbors
    if desc: parts.append(desc)
    s = " ".join(parts)
    return norm_name(s)

def build_metadata(df_dict, neighbors_map, descriptions=None):
    """
    This retains your original rich metadata structure. neighbors_map expected to be keyed by (table, raw_col)
    and values are lists of neighbor strings (qualified or unqualified).
    """
    records = []
    for table, df in df_dict.items():
        for col in df.columns:
            raw = col
            norm = norm_name(raw)
            # neighbors_map expects tuple keys (table, raw)
            neighbors = [norm_name(x) for x in neighbors_map.get((table, raw), [])]
            vals = sample_values(df[col], 5)
            m = {
                "table": table,
                "field": raw,
                "raw_name": raw,
                "norm_name": norm,
                "tokens": tokens_12(norm),
                "char_ngrams": char_ngrams(norm)[:64],
                "aliases": [],
                "neighbors": neighbors,
                "native_type": str(df[col].dtype),
                "coarse_type": coarse_type_from_dtype(df[col].dtype),
                "constraints": {"is_pk": False, "is_fk": False, "is_unique": False, "nullable": bool(df[col].isna().any())},
                "length": length_stats(vals),
                "char_stats": char_stats(vals),
                "locale_tags": [],
                "samples": vals,
                "sample_hashes": [hashlib.sha256(v.encode()).hexdigest()[:8] for v in vals],
                "categorical_topk": df[col].astype(str).value_counts().head(5).index.tolist(),
                "distinct_ratio": distinct_ratio(df[col]),
                "tfidf_row": None,
                "notes": []
            }
            records.append(m)
    return records

def build_tfidf_contexts(metadata, descriptions=None):
    ctx = []
    for m in metadata:
        desc = None
        if descriptions: desc = descriptions.get((m["table"], m["field"]))
        ctx.append(context_string(m["table"], m["norm_name"], m["neighbors"], desc))
    return ctx

def build_lexical_index(contexts):
    v_word = TfidfVectorizer(ngram_range=(1,2), analyzer="word", min_df=1)
    v_char = TfidfVectorizer(ngram_range=(3,5), analyzer="char_wb", min_df=1)
    Xw = v_word.fit_transform(contexts)
    Xc = v_char.fit_transform(contexts)
    X = hstack([Xw, Xc]).tocsr()
    return {"X": X, "v_word": v_word, "v_char": v_char}

def lexical_topk(lex_index, query, k=200):
    Xw = lex_index["v_word"].transform([query])
    Xc = lex_index["v_char"].transform([query])
    q = hstack([Xw, Xc]).tocsr()
    sims = (q @ lex_index["X"].T).toarray().ravel()
    order = np.argsort(-sims)[:k]
    return order, sims

def embed_texts(texts, model="text-embedding-3-small"):
    # wrapper to call the OpenAI embeddings endpoint via client
    r = client.embeddings.create(model=model, input=texts)
    V = np.array([x.embedding for x in r.data], dtype=np.float32)
    norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    return V / norms

def build_semantic_index(contexts):
    V = embed_texts(contexts)
    nn = NearestNeighbors(metric="cosine", algorithm="brute").fit(V)
    return {"V": V, "nn": nn}

def semantic_topk(sem_index, query, k=300):
    v = embed_texts([query])[0].reshape(1, -1)
    dist, idx = sem_index["nn"].kneighbors(v, n_neighbors=min(k, sem_index["V"].shape[0]))
    idx = idx.ravel(); dist = dist.ravel()
    sims = 1.0 - dist
    return idx, sims

def jaccard(a, b):
    A = set(a); B = set(b)
    if not A and not B: return 0.0
    return len(A & B) / max(1, len(A | B))

def blocking_candidates(source_meta, target_meta, contexts, lex_index, sem_index, k_lex=150, k_sem=300, k_final=50):
    """
    source_meta: a single metadata dict (source column)
    target_meta: full list of metadata entries (global)
    contexts: list of context strings aligned with target_meta
    returns: filtered list of tuples (index,score,tfidf_sim,embed_sim,jaccard)
    """
    q = context_string(source_meta["table"], source_meta["norm_name"], source_meta["neighbors"])
    idx_lex, sims_lex = lexical_topk(lex_index, q, k=k_lex)
    idx_sem, sims_sem = semantic_topk(sem_index, q, k=k_sem)
    cand = set(idx_lex.tolist()) | set(idx_sem.tolist())

    want_postal = re.fullmatch(r"\d{4,6}", source_meta["samples"][0]) if source_meta["samples"] else False
    if want_postal:
        for i, m in enumerate(target_meta):
            if "POSTAL" in pattern_tags(m["samples"]):
                cand.add(i)

    L = []
    qtoks = set(tokens_12(q))
    for i in cand:
        m = target_meta[i]
        name_tokens = set(m["tokens"])
        jl = jaccard(source_meta["tokens"], m["tokens"])
        lx = float(sims_lex[i]) if i < len(sims_lex) else 0.0
        se = float(sims_sem[i]) if i < len(sims_sem) else 0.0
        base = max(lx, se)
        bonus = 0.0
        if any(t in name_tokens for t in ["postal", "post code", "postcode", "zip", "zipcode"]): bonus += 0.03
        if any(t in qtoks for t in ["ship", "delivery", "orders"]):
            if any(t in contexts[i] for t in ["ship", "delivery", "orders"]): bonus += 0.02
        score = base + bonus
        L.append((i, score, lx, se, jl))
    L.sort(key=lambda x: -x[1])

    # Filter by type, with a small "escape" allowance for others
    filtered = []
    escape = int(max(1, math.ceil(0.1 * len(L))))
    kept_escape = 0
    for i, sc, lx, se, jl in L:
        m = target_meta[i]
        t = m["coarse_type"]
        if t in ["string", "int", "float", "datetime", "bool"]:
            filtered.append((i, sc, lx, se, jl))
        else:
            if kept_escape < escape:
                filtered.append((i, sc, lx, se, jl))
                kept_escape += 1
    filtered = filtered[:k_final]
    return filtered, q

def edge_score(equivalent, s_subset, t_subset, incompatible, flags, weq=1.0, wsub=0.5, lam=0.2):
    pen = 0.0
    if flags.get("type_conflict"): pen += 0.15
    if flags.get("context_mismatch"): pen += 0.10
    if flags.get("locale_mismatch"): pen += 0.10
    s = weq * equivalent + wsub * max(s_subset, t_subset) - lam * incompatible - pen
    return float(np.clip(s, 0.0, 1.0))

def decide_greedy(source_key, llm_obj, tau=0.6):
    # kept for compatibility when llm_obj is single-source format
    edges = []
    for sc in llm_obj["scores"]:
        s = edge_score(sc["equivalent"], sc["source_subset_of_target"], sc["target_subset_of_source"], sc["incompatible"], sc.get("flags", {}))
        edges.append((source_key, (sc["table"], sc["field"]), s))
    edges = [e for e in edges if e[2] >= tau]
    edges.sort(key=lambda x: -x[2])
    if not edges:
        return {source_key: None}, []
    chosen = edges[0]
    return {source_key: chosen[1]}, edges



def build_global_llm_payload(sources_with_cands, target_meta, K=30):
    """
    sources_with_cands: list of dicts {"source_meta": m, "candidates": [(idx, score, lx, se, jl), ...], "query": q}
    target_meta: full metadata list (indexed by idx)
    Returns a payload that contains all sources and their candidate lists (compact).
    """
    sources_payload = []
    for entry in sources_with_cands:
        sm = entry["source_meta"]
        cand_list = entry["candidates"][:K]
        c = []
        for i, sc, lx, se, jl in cand_list:
            m = target_meta[i]
            c.append({
                "table": m["table"],
                "field": m["field"],
                "type": m["coarse_type"],
                "neighbors": m["neighbors"],
                "primary_key": False,
                "samples": m["samples"][:3],
                "locale_tags": m["locale_tags"],
                "signals": {"tfidf": round(lx, 3), "jaccard": round(jl, 3), "embed_cos": round(se, 3)}
            })
        sources_payload.append({
            "source": {"table": sm["table"], "field": sm["field"], "type": sm["coarse_type"], "neighbors": sm["neighbors"], "samples": sm["samples"][:3], "locale_tags": sm["locale_tags"]},
            "candidates": c
        })
    return {"sources": sources_payload}

def compact_payload(full_payload):
    def normalize_name(name):
        return name.lower().replace('_', ' ').replace('-', ' ').strip()

    compact = {
        "schema_context": {"sources_summary": {}, "targets_summary": {}},
        "matches": []
    }

    # --- Build summaries (unchanged) ---
    for src in full_payload["sources"]:
        # 1. Populate Sources Summary
        src_table = src["source"]["table"]
        compact["schema_context"]["sources_summary"].setdefault(src_table, set()).add(src["source"]["field"])
        
        # 2. Populate Targets Summary (FIXED)
        for cand in src["candidates"]:
            tgt_table = cand["table"]
            tgt_field = cand["field"]
            # This line was missing the .add(tgt_field) part
            compact["schema_context"]["targets_summary"].setdefault(tgt_table, set()).add(tgt_field)

    # Convert sets to lists for JSON
    compact["schema_context"]["sources_summary"] = {
        k: sorted(list(v)) for k, v in compact["schema_context"]["sources_summary"].items()
    }
    compact["schema_context"]["targets_summary"] = {
        k: sorted(list(v)) for k, v in compact["schema_context"]["targets_summary"].items()
    }
    # -----------------------------------

    for src in full_payload["sources"]:
        src_info = src["source"]
        
        # 1. NEIGHBORS: Trim and normalize
        neighbors = [normalize_name(n.split()[-1]) for n in src_info.get("neighbors", [])[:5]]
        
        # 2. SAMPLES: Get up to 3 non-empty samples
        src_samples_raw = src_info.get("samples", [])
        src_samples = [s for s in src_samples_raw if s][:3]

        # 3. LOCALE TAGS: Fetch if available
        src_locale_tags = src_info.get("locale_tags", [])
        #print("src_locale_tags:", src_locale_tags)
        #breakpoint()

        src_entry = {
            "table": src_info["table"],
            "field": src_info["field"],
            "type": src_info.get("type", ""),
            "neighbors": neighbors,
            "samples": src_samples,          # Now a list of up to 3
            "locale_tags": src_locale_tags   # Added field
        }

        candidates = []
        for c in src["candidates"]:
            # 4. SIMILARITIES: Don't average, keep the specific signals
            raw_signals = c.get("signals", {})
            # Round values for cleaner JSON, keep the keys (e.g., name, value, context)
            similarities = {k: round(v, 3) for k, v in raw_signals.items()}

            # 5. CANDIDATE SAMPLES: Get up to 3
            cand_samples_raw = c.get("samples", [])
            cand_samples = [s for s in cand_samples_raw if s][:3]
            
            # 6. CANDIDATE LOCALE TAGS
            cand_locale_tags = c.get("locale_tags", [])

            candidates.append({
                "table": c["table"],
                "field": c["field"],
                "type": c.get("type", ""),
                "samples": cand_samples,       # Now a list of up to 3
                "similarities": similarities,  # Now a dict of individual scores
                "locale_tags": cand_locale_tags # Added field
            })

        compact["matches"].append({"source": src_entry, "candidates": candidates})

    return compact


def llm_score_global(payload, model , temperature=0.1, max_retries=1):
    """
    Sends a single LLM request with all sources and candidate lists.
    Expects JSON response:
    {
      "results": [
         {"source": {"table":"...", "field":"..."},
          "scores": [ { "table":..., "field":..., "equivalent":0.1, ... }, ... ]
         }, ...
      ]
    }
    Falls back to a heuristic if parsing fails.
    """
    sys = "You are a schema matching assistant. Score relationships between each source field and its shortlisted candidate target fields. Return only valid JSON matching the Output schema."
    guide = """
[Input]
payload: { "sources": [ { "source": {table,field,type,neighbors,samples}, "candidates": [ {table,field,type,neighbors,samples,signals}, ... ] }, ... ] }

[Task]
For each source, assign calibrated probabilities in [0,1] for: equivalent, source_subset_of_target, target_subset_of_source, incompatible.
Return a JSON object:
{
  "results": [
    {
      "source": {"table": str, "field": str},
      "scores": [
        {"table": str, "field": str, "equivalent": float, "source_subset_of_target": float, "target_subset_of_source": float, "incompatible": float, "rationale_tokens": [...], "reason": [...], "flags": {"type_conflict": bool, "context_mismatch": bool, "locale_mismatch": bool}}
      ]
    }, ...
  ]
}
Make sensible calibration; probabilities per candidate should sum to <= 1.0.
"""
    user = json.dumps({"instructions": guide, "payload": payload})

    for _ in range(max_retries + 1):
        r = client.chat.completions.create(model=model, temperature=temperature,
                                           messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                                           response_format={"type": "json_object"})
        try:
            obj = json.loads(r.choices[0].message.content)
            if "results" in obj:
                return obj
        except Exception:
            continue

    # Fallback: heuristic per candidate
    results = []
    for s in payload["sources"]:
        src = s["source"]
        scores = []
        for c in s["candidates"]:
            base = max(c["signals"].get("tfidf", 0.0), c["signals"].get("embed_cos", 0.0))
            eq = float(np.clip(base, 0, 0.95))
            inc = float(np.clip(1.0 - eq, 0, 0.95))
            scores.append({"table": c["table"], "field": c["field"], "equivalent": eq, "source_subset_of_target": 0.0, "target_subset_of_source": 0.0, "incompatible": inc, "rationale_tokens": ["fallback"], "flags": {"type_conflict": False, "context_mismatch": False, "locale_mismatch": False}})
        results.append({"source": {"table": src["table"], "field": src["field"]}, "scores": scores})
    return {"results": results}

def decide_greedy_for_source(source_key, scores_list, tau=0.6):
    edges = []
    for sc in scores_list:
        s = edge_score(sc["equivalent"], sc["source_subset_of_target"], sc["target_subset_of_source"], sc["incompatible"], sc.get("flags", {}))
        edges.append((source_key, (sc["table"], sc["field"]), s))
    edges = [e for e in edges if e[2] >= tau]
    edges.sort(key=lambda x: -x[2])
    if not edges:
        return {source_key: None}, []
    chosen = edges[0]
    return {source_key: chosen[1]}, edges


def generate_neighbors_auto(df_dict, threshold=0.5, sample_limit=200):
    """
    Produces neighbors_map keyed by (table, col) -> list of neighbor strings (qualified "table.col").
    Strategies:
      - include other columns in the same table,
      - columns with identical (case-insensitive) names across tables,
      - columns with value overlap above `threshold` (Jaccard on sampled unique values).
    """
    neighbors_map = defaultdict(set)
    # same-table neighbors:
    for tname, df in df_dict.items():
        cols = list(df.columns)
        for col in cols:
            for other in cols:
                if other == col: continue
                neighbors_map[(tname, col)].add(f"{tname}.{other}")

    # identical-name neighbors across tables
    name_buckets = defaultdict(list)
    for tname, df in df_dict.items():
        for col in df.columns:
            name_buckets[col.lower()].append((tname, col))
    for _, items in name_buckets.items():
        if len(items) > 1:
            for (t, c) in items:
                for (t2, c2) in items:
                    if (t, c) == (t2, c2): continue
                    neighbors_map[(t, c)].add(f"{t2}.{c2}")

    # value-overlap neighbors across all columns (sample)
    all_items = [(tname, col) for tname, df in df_dict.items() for col in df.columns]
    for i, (t1, c1) in enumerate(all_items):
        vals1 = set(map(str, df_dict[t1][c1].dropna().unique()[:sample_limit]))
        if not vals1: continue
        for (t2, c2) in all_items[i+1:]:
            vals2 = set(map(str, df_dict[t2][c2].dropna().unique()[:sample_limit]))
            if not vals2: continue
            j = len(vals1 & vals2) / len(vals1 | vals2)
            if j >= threshold:
                neighbors_map[(t1, c1)].add(f"{t2}.{c2}")
                neighbors_map[(t2, c2)].add(f"{t1}.{c1}")

    # Convert sets to sorted lists
    neighbors_map_list = {k: sorted(list(v)) for k, v in neighbors_map.items()}
    return neighbors_map_list


def parse_ttl_triples(ttl_path):
    g = Graph()
    g.parse(ttl_path, format="ttl")
    triples = []
    for s, p, o in g:
        triples.append((str(s), str(p), str(o)))
    return triples

def map_semantics_to_metadata(triples, metadata):
    """
    Heuristically attach semantic triples to metadata rows whose table/field names
    appear in any triple string or match labels. This is intentionally fuzzy to be
    broadly useful; users can refine later.
    """
    for m in metadata:
        m_norm_field = norm_name(m["field"])
        m_norm_table = norm_name(m["table"])
        for s, p, o in triples:
            s_, p_, o_ = norm_name(s), norm_name(p), norm_name(o)
            # if field or table name appears in any part of the triple, attach
            if (m_norm_field in s_) or (m_norm_field in p_) or (m_norm_field in o_) or (m_norm_table in s_) or (m_norm_table in p_) or (m_norm_table in o_):
                tag = f"{p}->{o}"
                if tag not in m["locale_tags"]:
                    m["locale_tags"].append(tag)
                if "semantic_model_attached" not in m["notes"]:
                    m["notes"].append("semantic_model_attached")
    return metadata


def load_dataset(path, name=None):
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".csv":
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin1")
    elif ext == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported format: {ext}")
    if not name:
        name = os.path.splitext(os.path.basename(path))[0]
    return name, df

def ensure_unique_table_names(paths):
    # base name collision handling: append numeric suffix
    names = []
    counts = {}
    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]
        if base in counts:
            counts[base] += 1
            base = f"{base}_{counts[base]}"
        else:
            counts[base] = 0
        names.append(base)
    return names

def main():
    parser = argparse.ArgumentParser(description="Schema matching CLI (supports CSV/JSON and optional .ttl semantics)")
    parser.add_argument("--sources", nargs="+", required=True, help="Source dataset file(s) (CSV or JSON)")
    parser.add_argument("--targets", nargs="+", required=True, help="Target dataset file(s) (CSV or JSON)")
    parser.add_argument("--semantic", nargs="*", default=[], help="Optional semantic model files (.ttl)")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (optional; prefer env OPENAI_API_KEY)")
    parser.add_argument("--jaccard-threshold", type=float, default=0.4, help="Value-overlap Jaccard threshold for neighbors")
    parser.add_argument("--k-lex", type=int, default=150)
    parser.add_argument("--k-sem", type=int, default=300)
    parser.add_argument("--k-final", type=int, default=50)
    parser.add_argument("--top-k-llm", type=int, default=30)
    parser.add_argument("--tau", type=float, default=0.6)
    parser.add_argument("--out", default=None, help="Optional output JSON file to write final mappings")
    args = parser.parse_args()

    # API key handling
    if args.api_key:
        os.getenv["OPENAI_API_KEY"] = args.api_key
        print("Warning: using API key from CLI arg (less secure than environment variable).")
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY not set. Set it in the environment or pass --api-key.")

    global client
    client = OpenAI()

    # Load datasets
    all_paths = list(args.sources) + list(args.targets)
    table_names = ensure_unique_table_names(all_paths)
    df_dict = {}
    # Map path->table name (unique)
    for p, name in zip(all_paths, table_names):
        _, ext = os.path.splitext(p)
        # If path appears multiple times, ensure correct mapping by position
        tab_name, df = load_dataset(p, name=name)
        df_dict[tab_name] = df
        print(f"Loaded {p} -> table `{tab_name}` ({df.shape[0]} rows, {df.shape[1]} cols)")

    # Derive explicit source and target table names as we stored them (preserve ordering)
    source_tables = table_names[:len(args.sources)]
    target_tables = table_names[len(args.sources):]

    # Generate neighbors automatically
    neighbors_map = generate_neighbors_auto(df_dict, threshold=args.jaccard_threshold)
    print(f"Generated neighbors for {len(neighbors_map)} columns (threshold={args.jaccard_threshold})")

    # Build metadata (keeps original rich fields)
    metadata = build_metadata(df_dict, neighbors_map)

    # Optional semantic models
    if args.semantic:
        if not RDFlIB_AVAILABLE:
            raise RuntimeError("rdflib not installed but --semantic passed. Install rdflib or omit --semantic.")
        all_triples = []
        for ttl in args.semantic:
            print(f"Parsing semantic TTL: {ttl}")
            triples = parse_ttl_triples(ttl)
            all_triples.extend(triples)
        metadata = map_semantics_to_metadata(all_triples, metadata)
        print("Attached semantic tags to metadata (heuristic mapping).")

    # Build contexts and indexes
    contexts = build_tfidf_contexts(metadata)
    lex_index = build_lexical_index(contexts)
    sem_index = build_semantic_index(contexts)
    print("Built lexical and semantic indexes for all columns.")

    # index_map: (table, field) -> idx into metadata / contexts
    index_map = {(m["table"], m["field"]): i for i, m in enumerate(metadata)}

    # For each source column (all columns in source_tables), produce blocking shortlist
    sources_with_cands = []
    for i, m in enumerate(metadata):
        if m["table"] not in source_tables:
            continue
        cand, q = blocking_candidates(m, metadata, contexts, lex_index, sem_index, k_lex=args.k_lex, k_sem=args.k_sem, k_final=args.k_final)
        # Filter shortlist to target_tables only (if any remain)
        cand_filtered = [c for c in cand if metadata[c[0]]["table"] in target_tables]
        if not cand_filtered:
            cand_filtered = cand
        sources_with_cands.append({"source_meta": m, "candidates": cand_filtered, "query": q})

    print(f"Prepared blocking shortlists for {len(sources_with_cands)} source columns (across sources).")

    # Build single global payload and call LLM once
    payload = build_global_llm_payload(sources_with_cands, metadata, K=args.top_k_llm)
    
    print("Calling LLM with a single payload containing all sources and their shortlists...")

    compact_data = compact_payload(payload)

    llm_obj = llm_score_global(compact_data, model="gpt-4.1", temperature=0.1, max_retries=1)

    # llm_obj is {"results": [ { "source": {...}, "scores": [...] }, ... ] } (fallback ensures same shape)
    results = {}
    decisions = {}
    edges_store = {}
    for res in llm_obj.get("results", []):
        src = res["source"]
        src_key = (src["table"], src["field"])
        scores = res["scores"]
        mapping, edges = decide_greedy_for_source(src_key, scores, tau=args.tau)
        decisions[src_key] = mapping[src_key]
        edges_store[src_key] = edges
        results[src_key] = {"scores": scores, "mapping": mapping[src_key]}

    # Pretty-print results
    final_out = []
    for src_entry in sources_with_cands:
        sm = src_entry["source_meta"]
        sk = (sm["table"], sm["field"])
        print("\nSOURCE")
        print(json.dumps({"table": sm["table"], "field": sm["field"], "neighbors": sm["neighbors"], "samples": sm["samples"]}, ensure_ascii=False, indent=2))
        # blocking shortlist (top 10)
        out_block = []
        for i, sc, lx, se, jl in src_entry["candidates"][:10]:
            m = metadata[i]
            out_block.append({"table": m["table"], "field": m["field"], "blocking_score": round(sc, 3), "tfidf": round(lx, 3), "embed_cos": round(se, 3), "jaccard": round(jl, 3)})
        print("BLOCKING_SHORTLIST (top 10)")
        print(json.dumps(out_block, ensure_ascii=False, indent=2))

        # LLM scores for this source
        if sk in results:
            print("LLM_SCORES (top candidates)")
            print(json.dumps(results[sk]["scores"][:10], ensure_ascii=False, indent=2))
        else:
            print("LLM did not return scores for this source (should be rare).")

        # Decision
        chosen = decisions.get(sk)
        final = {"source": f"{sk[0]}.{sk[1]}", "match": None}
        if chosen is not None:
            final["match"] = f"{chosen[0]}.{chosen[1]}"
        print("DECISION")
        print(json.dumps(final, ensure_ascii=False, indent=2))
        final_out.append(final)

    # Optionally write final mappings to file
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"mappings": final_out, "details": {f"{k[0]}.{k[1]}": {"mapping": (v[0]+"."+v[1]) if v else None} for k,v in decisions.items()}}, f, indent=2, ensure_ascii=False)
        print(f"Wrote final mappings to {args.out}")

if __name__ == "__main__":
    main()

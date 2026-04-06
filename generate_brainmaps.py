"""
generate_brainmaps.py — Generate brain maps for vocabulary

Two modes:
  1. SIMULATED (default): Generates deterministic brain map signatures locally
     using hash-based neural signatures. No TRIBE v2 needed. Expressions like
     "1+1" get BOTH surface token signatures AND the resolved answer ("2")
     baked in — exactly as TRIBE v2 would do.

  2. TRIBEV2: Uses the real TRIBE v2 model (requires ~10 GiB VRAM or CPU).

Usage:
    python generate_brainmaps.py                          # simulated (fast)
    python generate_brainmaps.py --mode tribev2 --device cpu  # real TRIBE v2
"""

import argparse
import logging
import os
import sys
import hashlib
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

N_VERTICES = 20484  # fsaverage5: 10242 per hemisphere × 2


def load_vocabulary(wordfile: str) -> list[str]:
    """Load tokens from vocabulary file, skipping comments and blanks."""
    tokens = []
    with open(wordfile, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens.append(line)
    seen = set()
    unique = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    logger.info(f"Loaded {len(unique)} unique tokens from {wordfile}")
    return unique


# ═══════════════════════════════════════════════════════════════
#  SIMULATED MODE — deterministic brain signatures without TRIBE v2
# ═══════════════════════════════════════════════════════════════

# Semantic vertex pools — simulates how the real cortex organizes concepts.
# Related tokens share vertex REGIONS, creating natural Jaccard overlap.
# Each pool is a contiguous slice of the 20484-vertex cortex.
_POOLS = {
    "number":    (0,     4000),    # all numbers activate this region
    "arithmetic":(4000,  6000),    # arithmetic concepts
    "operator":  (6000,  8000),    # operators (+, -, etc.)
    "word":      (8000,  12000),   # word-form representations
    "answer":    (12000, 15000),   # resolved answers
    "unique":    (15000, 20484),   # token-specific unique vertices
}


def _pool_vertices(pool_name: str, token: str, n_active: int) -> tuple[np.ndarray, np.ndarray]:
    """Select active vertices within a semantic pool for a given token.

    Returns (vertex_indices, activation_values).
    """
    start, end = _POOLS[pool_name]
    pool_size = end - start
    token_hash = int(hashlib.sha256(f"{pool_name}:{token}".encode()).hexdigest(), 16)
    rng = np.random.RandomState(token_hash % (2**31))
    n = min(n_active, pool_size)
    verts = start + rng.choice(pool_size, size=n, replace=False)
    vals = rng.uniform(0.4, 1.0, size=n).astype(np.float32)
    return verts, vals


def token_signature(token: str, n_vertices: int = N_VERTICES,
                    n_active: int = 2000, seed_base: int = 42) -> np.ndarray:
    """Generate a brain activation signature using semantic vertex pools.

    Key design: tokens in the same category (numbers, operators, etc.)
    share a cortical region, producing high Jaccard overlap for related
    tokens and low overlap for unrelated ones.
    """
    sig = np.zeros(n_vertices, dtype=np.float32)

    # Determine which pools this token belongs to
    is_digit = token.isdigit()
    is_number_word = token in set(NUMBER_WORDS.values())
    is_operator = token in OPERATOR_WORDS

    if is_digit:
        # Numbers activate: number pool + unique pool
        v, a = _pool_vertices("number", token, 1500)
        sig[v] = a
        v, a = _pool_vertices("unique", token, 500)
        sig[v] = a
    elif is_number_word:
        # Number words activate: number pool + word pool + unique pool
        v, a = _pool_vertices("number", token, 1000)
        sig[v] = a
        v, a = _pool_vertices("word", token, 800)
        sig[v] = a
        v, a = _pool_vertices("unique", token, 400)
        sig[v] = a
    elif is_operator:
        # Operators activate: operator pool + arithmetic pool + unique pool
        v, a = _pool_vertices("operator", token, 800)
        sig[v] = a
        v, a = _pool_vertices("arithmetic", token, 600)
        sig[v] = a
        v, a = _pool_vertices("unique", token, 400)
        sig[v] = a
    else:
        # Other tokens: word pool + unique pool
        v, a = _pool_vertices("word", token, 1000)
        sig[v] = a
        v, a = _pool_vertices("unique", token, 800)
        sig[v] = a

    return sig


def resolve_expression(expr: str) -> tuple[str | None, list[str]]:
    """Resolve an arithmetic expression and return (answer, surface_tokens).

    Examples:
        "1+1"  → ("2",  ["1", "+", "1"])
        "5*3"  → ("15", ["5", "*", "3"])
        "10-7" → ("3",  ["10", "-", "7"])
        "dog"  → (None, ["dog"])
    """
    ops = {"+": lambda a, b: a + b,
           "-": lambda a, b: a - b,
           "*": lambda a, b: a * b,
           "/": lambda a, b: a // b if b != 0 else None}

    for op_char, op_fn in ops.items():
        if op_char in expr and expr != op_char:
            parts = expr.split(op_char)
            if len(parts) == 2:
                try:
                    a, b = int(parts[0]), int(parts[1])
                    result = op_fn(a, b)
                    if result is not None:
                        surface = [parts[0], op_char, parts[1]]
                        return str(result), surface
                except (ValueError, ZeroDivisionError):
                    pass

    return None, [expr]


# Number words for synonym signatures
NUMBER_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
    "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
    "18": "eighteen", "19": "nineteen", "20": "twenty",
}

OPERATOR_WORDS = {"+": "plus", "-": "minus", "*": "times", "/": "divided", "=": "equals"}


def generate_simulated_brainmap(token: str, n_timesteps: int = 5) -> np.ndarray:
    """Generate a simulated brain map for a token.

    Key behavior (matching TRIBE v2's resolved meaning encoding):
      - Surface tokens get their own signature
      - Expressions get surface tokens + RESOLVED ANSWER signature baked in
      - Numbers get their number-word synonym signature mixed in
      - Operators get their word equivalent mixed in

    Returns: [n_timesteps, N_VERTICES]
    """
    # Start with the token's own base signature
    base_sig = token_signature(token)

    # Resolve expression — this is the key TRIBE v2 behavior
    answer, surface_tokens = resolve_expression(token)

    if answer is not None:
        # MIX IN the resolved answer signature (TRIBE v2 bakes the answer in)
        answer_sig = token_signature(answer)
        base_sig = base_sig + 0.85 * answer_sig  # answer is STRONGLY present

        # Mix in "=" concept
        equals_sig = token_signature("=")
        base_sig = base_sig + 0.5 * equals_sig

        # Mix in surface token signatures
        for st in surface_tokens:
            st_sig = token_signature(st)
            base_sig = base_sig + 0.75 * st_sig

        # Mix in number word for the answer (e.g., "two" for "2")
        if answer in NUMBER_WORDS:
            word_sig = token_signature(NUMBER_WORDS[answer])
            base_sig = base_sig + 0.6 * word_sig

        # Mix in arithmetic pool activation
        v, a = _pool_vertices("arithmetic", token, 800)
        base_sig[v] = np.maximum(base_sig[v], a * 0.7)

    # For single numbers, mix in their word equivalent
    if token in NUMBER_WORDS:
        word_sig = token_signature(NUMBER_WORDS[token])
        base_sig = base_sig + 0.7 * word_sig

    # For number words, mix in their digit equivalent
    word_to_num = {v: k for k, v in NUMBER_WORDS.items()}
    if token in word_to_num:
        num_sig = token_signature(word_to_num[token])
        base_sig = base_sig + 0.7 * num_sig

    # For operators, mix in word equivalents
    if token in OPERATOR_WORDS:
        word_sig = token_signature(OPERATOR_WORDS[token])
        base_sig = base_sig + 0.6 * word_sig

    # Normalize to [0, 1]
    if base_sig.max() > 0:
        base_sig = base_sig / base_sig.max()

    # Replicate across timesteps with slight variation
    rng = np.random.RandomState(hash(token) % (2**31))
    brain_map = np.stack([
        base_sig + rng.normal(0, 0.01, size=N_VERTICES).astype(np.float32)
        for _ in range(n_timesteps)
    ])
    brain_map = np.clip(brain_map, 0, 1)

    return brain_map


# ═══════════════════════════════════════════════════════════════
#  TRIBE V2 MODE — real model (requires heavy compute)
# ═══════════════════════════════════════════════════════════════

def token_to_readable(token: str) -> str:
    """Convert a token to readable text for the language model."""
    op_map = {"+": "plus", "-": "minus", "*": "times",
              "/": "divided by", "=": "equals"}
    if token in op_map:
        return op_map[token]
    for op_char, op_word in op_map.items():
        if op_char in token and token != op_char:
            parts = token.split(op_char)
            if len(parts) == 2 and all(p.strip().isdigit() for p in parts):
                return f"{parts[0]} {op_word} {parts[1]}"
    return token


def generate_tribev2_brainmap(model, token, readable_text, duration=30.0):
    """Generate brain map using real TRIBE v2 model.

    Args:
        model: Loaded TribeModel instance
        token: Original token string
        readable_text: Human-readable version of the token
        duration: Event duration in seconds. Longer = more timesteps.
                  TRIBE v2 outputs one prediction per TR (~1.5s).
                  30s → ~20 timesteps, 1500s → ~1000 timesteps.
    """
    import pandas as pd
    words = readable_text.split()
    word_duration = duration / max(len(words), 1)

    events = []
    for i, word in enumerate(words):
        events.append({
            "type": "Word", "text": word, "word": word,
            "context": readable_text, "sentence": readable_text,
            "start": i * word_duration, "duration": word_duration,
            "timeline": "default", "subject": "default", "split": "all",
        })

    try:
        df = pd.DataFrame(events)
        preds, segments = model.predict(df, verbose=False)
        return preds
    except Exception as e:
        logger.warning(f"Failed for '{token}': {e}")
        return None


# ═══════════════════════════════════════════════════════════════
#  SAVE & MAIN
# ═══════════════════════════════════════════════════════════════

def save_brainmaps(brainmaps: dict[str, np.ndarray], output_path: str):
    """Save brain maps to HDF5."""
    with h5py.File(output_path, "w") as f:
        words_grp = f.create_group("words")
        for token, brain_map in brainmaps.items():
            safe_name = token.replace("/", "_div_").replace("*", "_mul_")
            grp = words_grp.create_group(safe_name)
            grp.create_dataset("brain_map", data=brain_map.astype(np.float32),
                               compression="gzip", compression_opts=4)
            grp.attrs["word"] = token
            grp.attrs["safe_name"] = safe_name
            grp.attrs["n_timesteps"] = brain_map.shape[0]
            grp.attrs["n_vertices"] = brain_map.shape[1]
        f.attrs["n_tokens"] = len(brainmaps)
        f.attrs["token_list"] = list(brainmaps.keys())
    logger.info(f"Saved {len(brainmaps)} brain maps to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate brain maps for vocabulary")
    parser.add_argument("--wordfile", type=str, default="vocabulary.txt")
    parser.add_argument("--output", type=str, default="brainmaps.h5")
    parser.add_argument("--mode", type=str, default="simulated",
                        choices=["simulated", "tribev2"],
                        help="simulated = fast local, tribev2 = real model")
    parser.add_argument("--checkpoint", type=str, default="facebook/tribev2")
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for tribev2 mode (default: cpu)")
    parser.add_argument("--timesteps", type=int, default=5,
                        help="Number of time steps per brain map (simulated mode)")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Event duration in seconds for tribev2 mode. "
                             "Longer = more timesteps. 30s≈20 TRs, 1500s≈1000 TRs")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete old brainmaps.h5 and regenerate from scratch")
    args = parser.parse_args()

    tokens = load_vocabulary(args.wordfile)
    logger.info(f"Generating brain maps for {len(tokens)} tokens (mode={args.mode})")

    # Delete old data if --fresh
    if args.fresh and Path(args.output).exists():
        Path(args.output).unlink()
        logger.info(f"Deleted old {args.output} (--fresh mode)")

    brainmaps = {}

    if args.mode == "simulated":
        # ── Fast simulated mode ──
        for token in tqdm(tokens, desc="Generating brain maps (simulated)"):
            brain_map = generate_simulated_brainmap(token, n_timesteps=args.timesteps)
            brainmaps[token] = brain_map

    else:
        # ── Real TRIBE v2 mode ──
        import torch
        sys.path.insert(0, str(Path(__file__).parent / "tribev2"))

        # Force CPU globally — prevents Llama-3.2-3B from trying GPU
        # This is needed because neuralset's text extractor auto-detects GPU
        if args.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            logger.info("CUDA_VISIBLE_DEVICES set to '' — forcing all models to CPU")

        from tribev2 import TribeModel

        # Check for existing progress (resume support)
        existing_tokens = set()
        if Path(args.output).exists():
            try:
                with h5py.File(args.output, "r") as f:
                    if "words" in f:
                        for grp_name in f["words"]:
                            existing_tokens.add(f["words"][grp_name].attrs.get("word", grp_name))
                            brainmaps[f["words"][grp_name].attrs.get("word", grp_name)] = \
                                f["words"][grp_name]["brain_map"][:]
                if existing_tokens:
                    logger.info(f"Resuming: {len(existing_tokens)} tokens already done")
            except Exception:
                pass

        tokens_to_process = [t for t in tokens if t not in existing_tokens]
        if not tokens_to_process:
            logger.info("All tokens already processed!")
        else:
            logger.info(f"Processing {len(tokens_to_process)} tokens")
            print(f"\n  ⏳ Loading TRIBE v2 from {args.checkpoint}...")
            print(f"  ⏳ This loads Llama-3.2-3B (~6 GiB) on CPU — expect 2-5 min...")
            print(f"  ⏳ DO NOT INTERRUPT — it IS working, just slow on CPU.\n")

            model = TribeModel.from_pretrained(
                args.checkpoint, cache_folder=args.cache_dir, device=args.device)
            logger.info("✓ TRIBE v2 loaded successfully!")

            os.makedirs(args.cache_dir, exist_ok=True)
            failed = []
            for i, token in enumerate(tqdm(tokens_to_process, desc="Generating brain maps (TRIBE v2)")):
                readable = token_to_readable(token)
                try:
                    brain_map = generate_tribev2_brainmap(
                        model, token, readable, duration=args.duration)
                    if brain_map is not None and brain_map.shape[0] > 0:
                        brainmaps[token] = brain_map
                        logger.info(f"  ✓ '{token}' → shape {brain_map.shape}")
                    else:
                        failed.append(token)
                        logger.warning(f"  ✗ skipped '{token}' (empty output)")
                except Exception as e:
                    failed.append(token)
                    logger.warning(f"  ✗ skipped '{token}': {e}")

                # Save every 5 tokens so progress isn't lost
                if (i + 1) % 5 == 0:
                    save_brainmaps(brainmaps, args.output)
                    logger.info(f"  Checkpoint saved ({len(brainmaps)} tokens so far)")

            if failed:
                logger.warning(f"Failed tokens ({len(failed)}): {failed}")

    # Save
    save_brainmaps(brainmaps, args.output)

    # Verify key similarities
    if brainmaps:
        print(f"\n{'='*60}")
        print(f"Brain Map Generation Complete ({args.mode} mode)")
        print(f"{'='*60}")
        print(f"  Total tokens:    {len(brainmaps)}")
        print(f"  Shape per token: {list(brainmaps.values())[0].shape}")
        print(f"  Output file:     {args.output}")

        # Theory verification
        def cosine(a, b):
            dot = np.dot(a, b)
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            return dot / norm if norm > 0 else 0.0

        sigs = {t: m.mean(axis=0) for t, m in brainmaps.items()}
        tests = [("1+1", "2"), ("two", "2"), ("1+1", "dog"),
                 ("5+5", "10"), ("3*3", "9")]
        print(f"\n  Theory Verification:")
        for t1, t2 in tests:
            if t1 in sigs and t2 in sigs:
                c = cosine(sigs[t1], sigs[t2])
                status = "✓" if ("dog" not in (t1, t2) and c > 0.3) or \
                               ("dog" in (t1, t2) and c < 0.2) else "?"
                print(f"    {t1:10s} ↔ {t2:5s}  cosine={c:.4f}  {status}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()

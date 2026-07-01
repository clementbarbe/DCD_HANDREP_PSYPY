# tasks/sequence_generator.py
"""
Sequence Generator for Hand Representation Task
================================================
Generates a constrained sequence of 100 trials (5 fingers × 2 zones × 10 reps).

Constraints per finger
----------------------
    - 3 imposed pairs Z1 → Z2  (consecutive in the sequence)
    - 3 imposed pairs Z2 → Z1  (consecutive in the sequence)
    - 4 isolated Z1 + 4 isolated Z2, placed randomly

Adjacency rules
---------------
    - The same finger may appear consecutively ONLY inside an imposed pair.
    - Outside pairs, the same finger must NEVER appear twice in a row
      (regardless of zone).

Totals
------
    Each (finger, zone) appears exactly 10 times.
    30 pairs (60 trials) + 40 isolated trials = 100 trials.

Reference
---------
    Longo, M. R., & Haggard, P. (2010). PNAS, 107(26), 11727–11732.
"""

import random
from typing import Optional


FINGERS = ["thumb", "index", "middle", "ring", "little"]

N_PAIRS_Z1Z2 = 3
N_PAIRS_Z2Z1 = 3
N_ISOLATED_PER_ZONE = 4
MAX_ATTEMPTS = 10000


def _build_elements():
    """Build atomic elements: 30 pairs + 40 isolated singletons.

    Returns
    -------
    pairs : list[tuple[dict, dict]]
        30 pairs (each a tuple of 2 trial dicts).
    isolated : list[dict]
        40 single trial dicts.
    """
    pairs = []
    isolated = []

    for finger in FINGERS:
        for _ in range(N_PAIRS_Z1Z2):
            pairs.append((
                {"finger": finger, "zone": 1, "pair_type": "Z1Z2"},
                {"finger": finger, "zone": 2, "pair_type": "Z1Z2"},
            ))
        for _ in range(N_PAIRS_Z2Z1):
            pairs.append((
                {"finger": finger, "zone": 2, "pair_type": "Z2Z1"},
                {"finger": finger, "zone": 1, "pair_type": "Z2Z1"},
            ))
        for _ in range(N_ISOLATED_PER_ZONE):
            isolated.append({"finger": finger, "zone": 1, "pair_type": "isolated"})
            isolated.append({"finger": finger, "zone": 2, "pair_type": "isolated"})

    return pairs, isolated


def _finger_of_block(block):
    """Return the finger of a block (pair or singleton)."""
    if isinstance(block, tuple):
        return block[0]["finger"]
    return block["finger"]


def _last_finger(sequence):
    """Return the finger of the last trial in the sequence, or None."""
    if not sequence:
        return None
    return sequence[-1]["finger"]


def _flatten_block(block):
    """Convert a block (pair tuple or single dict) into a list of dicts."""
    if isinstance(block, tuple):
        return list(block)
    return [block]


def _try_generate(rng: random.Random) -> Optional[list]:
    """One attempt at building a valid 100-trial sequence.

    Strategy
    --------
    1. Mix all blocks (30 pairs + 40 singletons) into a pool.
    2. At each step, collect all blocks whose finger differs from the
       last placed finger (or, for pairs, whose first element differs).
    3. Pick one at random and append it.
    4. Special case: after placing the first element of a pair, the second
       element is forced (same finger is allowed here — it is the pair).
    5. If no valid block can be placed, the attempt fails.
    """
    pairs, isolated = _build_elements()

    pool = list(pairs) + isolated
    rng.shuffle(pool)

    sequence = []

    while pool:
        last_f = _last_finger(sequence)

        # Collect valid candidates (finger ≠ last finger)
        valid_indices = []
        for i, block in enumerate(pool):
            block_finger = _finger_of_block(block)
            if block_finger != last_f:
                valid_indices.append(i)

        if not valid_indices:
            return None  # dead end

        # Pick a random valid block
        idx = rng.choice(valid_indices)
        block = pool.pop(idx)

        # Append to sequence
        flat = _flatten_block(block)
        sequence.extend(flat)

        # After a pair, the last finger is the pair's finger.
        # We must also check that the NEXT block doesn't start
        # with the same finger (this is handled in the next iteration).

    return sequence


def _validate_sequence(sequence):
    """Full validation of all constraints.

    Checks
    ------
    1. Exactly 100 trials.
    2. Each (finger, zone) appears exactly 10 times.
    3. For each finger: exactly 3 consecutive Z1→Z2 and 3 consecutive Z2→Z1.
    4. Same finger never appears consecutively outside imposed pairs.
    """
    if len(sequence) != 100:
        return False

    # --- Check counts ---
    counts = {}
    for trial in sequence:
        key = (trial["finger"], trial["zone"])
        counts[key] = counts.get(key, 0) + 1

    for finger in FINGERS:
        for zone in (1, 2):
            if counts.get((finger, zone), 0) != 10:
                return False

    # --- Check adjacency + pair counts ---
    pairs_z1z2 = {f: 0 for f in FINGERS}
    pairs_z2z1 = {f: 0 for f in FINGERS}

    i = 0
    while i < len(sequence):
        curr = sequence[i]

        if i < len(sequence) - 1:
            nxt = sequence[i + 1]

            if curr["finger"] == nxt["finger"]:
                # Same finger consecutive: must be an imposed pair
                is_z1z2 = (
                    curr["pair_type"] == "Z1Z2"
                    and nxt["pair_type"] == "Z1Z2"
                    and curr["zone"] == 1
                    and nxt["zone"] == 2
                )
                is_z2z1 = (
                    curr["pair_type"] == "Z2Z1"
                    and nxt["pair_type"] == "Z2Z1"
                    and curr["zone"] == 2
                    and nxt["zone"] == 1
                )

                if is_z1z2:
                    pairs_z1z2[curr["finger"]] += 1
                    i += 2
                    continue
                elif is_z2z1:
                    pairs_z2z1[curr["finger"]] += 1
                    i += 2
                    continue
                else:
                    # Same finger consecutive but not a valid pair
                    return False

        i += 1

    for finger in FINGERS:
        if pairs_z1z2[finger] != N_PAIRS_Z1Z2:
            return False
        if pairs_z2z1[finger] != N_PAIRS_Z2Z1:
            return False

    return True


def generate_sequence(seed: Optional[int] = None) -> list:
    """Generate a valid 100-trial sequence.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    list[dict]
        100 trial dicts with keys: finger, zone, pair_type, trial_index.

    Raises
    ------
    RuntimeError
        If no valid sequence is found after MAX_ATTEMPTS tries.
    """
    rng = random.Random(seed)

    for attempt in range(MAX_ATTEMPTS):
        result = _try_generate(rng)
        if result is not None and len(result) == 100:
            if _validate_sequence(result):
                for i, trial in enumerate(result):
                    trial["trial_index"] = i
                return result

    raise RuntimeError(
        f"Could not generate a valid sequence after {MAX_ATTEMPTS} attempts."
    )


def print_sequence(sequence):
    """Pretty-print a sequence for debugging."""
    print(f"{'#':>3}  {'Finger':<10}  Zone  Type      Pair")
    print("-" * 48)

    i = 0
    while i < len(sequence):
        t = sequence[i]
        marker = ""

        # Detect if this trial is the start of a pair
        if i < len(sequence) - 1:
            nxt = sequence[i + 1]
            if (t["finger"] == nxt["finger"]
                    and t["pair_type"] != "isolated"
                    and nxt["pair_type"] != "isolated"):
                marker = " ┐"

        # Detect if this trial is the end of a pair
        if i > 0:
            prv = sequence[i - 1]
            if (t["finger"] == prv["finger"]
                    and t["pair_type"] != "isolated"
                    and prv["pair_type"] != "isolated"):
                marker = " ┘"

        print(
            f"{i + 1:>3}  {t['finger']:<10}  Z{t['zone']}    "
            f"{t['pair_type']:<10}{marker}"
        )
        i += 1


def print_stats(sequence):
    """Print validation statistics."""
    print(f"\nTotal: {len(sequence)} trials")
    print(f"\nCounts per (finger, zone):")
    for finger in FINGERS:
        z1 = sum(1 for t in sequence if t["finger"] == finger and t["zone"] == 1)
        z2 = sum(1 for t in sequence if t["finger"] == finger and t["zone"] == 2)
        print(f"  {finger:<10}: Z1={z1:>2}, Z2={z2:>2}")

    # Count consecutive pairs
    print(f"\nImposed pairs:")
    i = 0
    z1z2 = {f: 0 for f in FINGERS}
    z2z1 = {f: 0 for f in FINGERS}
    while i < len(sequence) - 1:
        c, n = sequence[i], sequence[i + 1]
        if c["finger"] == n["finger"]:
            if c["zone"] == 1 and n["zone"] == 2:
                z1z2[c["finger"]] += 1
            elif c["zone"] == 2 and n["zone"] == 1:
                z2z1[c["finger"]] += 1
            i += 2
            continue
        i += 1
    for f in FINGERS:
        print(f"  {f:<10}: Z1→Z2={z1z2[f]}, Z2→Z1={z2z1[f]}")

    # Check no same-finger repetition outside pairs
    violations = 0
    i = 0
    while i < len(sequence) - 1:
        c, n = sequence[i], sequence[i + 1]
        if c["finger"] == n["finger"]:
            is_pair = (c["pair_type"] != "isolated" and n["pair_type"] != "isolated")
            if is_pair:
                i += 2
                continue
            else:
                violations += 1
                print(f"  VIOLATION at {i + 1}-{i + 2}: {c['finger']} (isolated)")
        i += 1
    if violations == 0:
        print("\nNo adjacency violations.")
    else:
        print(f"\n{violations} adjacency violation(s)!")


# ── CLI for testing ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
    seq = generate_sequence(seed=seed)
    print_sequence(seq)
    print_stats(seq)
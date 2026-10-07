"""
Pure-function unit tests (no network, no models). Safe to run in CI.

    python eval/test_units.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, HERE)

from module_5_retrieval import _fix_mojibake, _diversify  # noqa: E402
from eval_retrieval import hit_rate_at_k, mrr_at_k, ndcg_at_k  # noqa: E402

FAILS = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILS.append(name)


# --- mojibake repair -------------------------------------------------------
check("mojibake: em-dash repaired", _fix_mojibake("Color\u00e2\u20ac\u201durine") == "Color\u2014urine")
check("mojibake: clean text untouched", _fix_mojibake("plain text") == "plain text")
check("mojibake: empty safe", _fix_mojibake("") == "")

# --- retrieval metrics -----------------------------------------------------
check("hit@k: hit", hit_rate_at_k(["a", "b"], {"b"}, 2) == 1.0)
check("hit@k: miss", hit_rate_at_k(["a"], {"b"}, 1) == 0.0)
check("mrr@k: rank 2", mrr_at_k(["a", "b"], {"b"}, 10) == 0.5)
check("ndcg@k: perfect", ndcg_at_k(["a"], {"a"}, 10) == 1.0)

# --- diversity -------------------------------------------------------------
docs = [("1", "alpha beta gamma", 1.0, {}),
        ("2", "alpha beta gamma", 0.9, {}),
        ("3", "delta epsilon zeta", 0.8, {})]
out = _diversify(docs, 2)
check("diversify drops near-duplicate", len(out) == 2 and {d[0] for d in out} == {"1", "3"})

print("\n" + ("ALL UNIT TESTS PASSED" if not FAILS else f"FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)

"""
DEPRECATED entry point, kept only for backward compatibility.

The interactive/auto review CLI now lives in review.py. This file is a
thin redirect so `python demo_human_review.py` keeps working; it does not
duplicate any decision logic - see review.py's run_auto_demo() for the
actual implementation.

Prefer: `python review.py --auto` (or `python review.py` for the
interactive terminal workflow) going forward.
"""

from review import run_auto_demo

if __name__ == "__main__":
    run_auto_demo()

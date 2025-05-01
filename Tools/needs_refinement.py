import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Tools.State_dict import StateDict

MAX_ROUNDS = 3


def needs_refinement(state: StateDict) -> str:
    current_round = state.get("refine_round", 0)
    if current_round >= MAX_ROUNDS:
        return "end"
    if any(h.get("status") == "needs_refine" for h in state.get("hypotheses", [])):
        return "refine"
    return "end"
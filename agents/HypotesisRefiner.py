import sys
import os
#promt template
from langchain.prompts import PromptTemplate
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Tools.State_dict import StateDict
from langchain_google_genai import GoogleGenerativeAI as LLM
import pandas as pd
LLM_TEMP = 0.7
GEMINI_API_KEY=os.getenv('GEMINI_API_KEY')

MAX_ROUNDS = 3

llm_gen = LLM(
    model="gemini-1.5-flash",
    temperature=LLM_TEMP,
    google_api_key=GEMINI_API_KEY
)

REF_PROMPT = PromptTemplate(
    input_variables=["hypothesis"],
    template="""
Refine the biomedical research hypothesis below to maximize coherence, novelty, and falsifiability.
Output only the improved hypothesis text.

ORIGINAL HYPOTHESIS:
{hypothesis}

IMPROVED HYPOTHESIS:
"""
)

def agent_refine_hypotheses(state: StateDict) -> StateDict:
    print(f"♻️ [HypRefine] Refinement round {state.get('refine_round', 0) + 1}/{MAX_ROUNDS}...")
    improved_hypotheses = []
    for h in state["hypotheses"]:
        if h.get("status") == "needs_refine":
            try:
                llm_response_text = (REF_PROMPT | llm_gen).invoke({"hypothesis": h["text"]})
                # Upewnijmy się, że wynik jest stringiem przed strip()
                new_text = str(llm_response_text).strip() if llm_response_text else ""
                if new_text and new_text != h["text"]:
                    improved_hypotheses.append({"text": new_text, "score": None, "status": "raw"})
                    h["status"] = "refined"
                else:
                    h["status"] = "refine_failed"
            except Exception as e:
                print(f"❌ Error refining hypothesis: {e}")
                h["status"] = "refine_error"

    state["hypotheses"].extend(improved_hypotheses)
    state["refine_round"] += 1
    print(f"♻️ [HypRefine] Refinement round {state['refine_round']} complete.")
    return state
import sys
import os
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Tools.State_dict import StateDict
from langchain.prompts import PromptTemplate
import json
from langchain_google_genai import ChatGoogleGenerativeAI as LLM
import pandas as pd
LLM_TEMP = 0.7

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

GEN_PROMPT = PromptTemplate(
    input_variables=["analysis"],
    template="""
You are a biomedical scientist. Using the domain description below,
propose up to **5 testable research hypotheses**. Return pure JSON list:

[{{"hyp":"..."}}]

DOMAIN:
{analysis}
"""
)

llm_gen = LLM(
    model="gemini-1.5-flash",
    temperature=LLM_TEMP,
    google_api_key=GEMINI_API_KEY
)

def agent_generate_hypotheses(state: StateDict) -> StateDict:
    if "analysis" not in state or not state["analysis"]:
        print("⚠️ [HypGen] Analysis field is empty. Cannot generate hypotheses.")
        state["hypotheses"] = []
        state["refine_round"] = 0
        return state

    print("🧠 [HypGen] Generating initial hypotheses...")
    raw = (GEN_PROMPT | llm_gen).invoke({"analysis": state["analysis"]})
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, list) or not all(isinstance(item, dict) and "hyp" in item for item in parsed):
            raise ValueError("Parsed JSON is not in expected format list[{'hyp': ...}]")
    except Exception as e:
        print(f"⚠️ [HypGen] Failed to parse LLM output: {e}. Using fallback parsing.")
        parsed = [{"hyp": line.strip("-•* ")} for line in raw.splitlines() if line.strip()]

    state["hypotheses"] = [{"text": h["hyp"], "score": None, "status": "raw"} for h in parsed if h.get("hyp")]
    state["refine_round"] = 0
    print(f"🧠 [HypGen] Generated {len(state['hypotheses'])} initial hypotheses.")
    return state

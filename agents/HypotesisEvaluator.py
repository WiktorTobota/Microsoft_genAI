from langchain.evaluation import CriteriaEvalChain
import sys
import os
from dotenv import load_dotenv
load_dotenv()
import numpy as np
from langchain.chains import RetrievalQA
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import client, COLLECTION_NAME, embed_fn
from Tools.State_dict import StateDict
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAI as LLM
import pandas as pd
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


vector_store = Chroma(client = client
                ,collection_name = COLLECTION_NAME
    ,embedding_function=embed_fn)
LLM_TEMP = 0.7


llm_gen = LLM(
    model="gemini-1.5-flash",
    temperature=LLM_TEMP,
    google_api_key=GEMINI_API_KEY
)

RUBRIC = {
    "coherence": "Is the hypothesis logically connected to known biological principles or data?",
    "novelty": "Does the hypothesis introduce a genuinely new insight or question?",
    "falsifiable": "Can the hypothesis be experimentally tested or potentially proven false?"
}

eval_chain = CriteriaEvalChain.from_llm(llm=llm_gen, criteria=RUBRIC)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
qa_chain = RetrievalQA.from_chain_type(llm=llm_gen, retriever=retriever, return_source_documents=False)

def agent_evaluate_hypotheses(state: StateDict) -> StateDict:
    print("🔍 [HypEval] Evaluating hypotheses...")
    if "hypotheses" not in state or not state["hypotheses"]:
        print("⚠️ [HypEval] No hypotheses to evaluate.")
        return state

    for h in state["hypotheses"]:
        if h.get("status") != "raw":
            continue
        hypothesis_text = h.get("text")
        if not hypothesis_text:
            print(f"⚠️ [HypEval] Skipping empty hypothesis.")
            h["status"] = "skipped_empty"
            continue

        try:
            evidence_result = qa_chain.invoke({"query": hypothesis_text})
            evidence_text = evidence_result.get("result", "") or "No specific supporting documents found."
            eval_result = eval_chain.evaluate_strings(
                prediction=hypothesis_text,
                reference=evidence_text,
                input="Evaluate this hypothesis."
            )
            scores = [v for v in eval_result.get("results", {}).values() if isinstance(v, (int, float))]
            h["score"] = float(np.mean(scores)) if scores else 0.0
            h["status"] = "accepted" if h["score"] >= 0.6 else "needs_refine"
        except Exception as e:
            print(f"❌ Error evaluating hypothesis: {e}")
            h["status"] = "eval_error"
            h["score"] = None

    return state

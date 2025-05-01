from Tools.State_dict import StateDict
import json

def save_hypotheses(state: StateDict, path="hypotheses_output.json"):
    if "hypotheses" not in state:
        print("ℹ️ No hypotheses found in state to save.")
        return
    try:
        with open(path, "w", encoding="utf-8") as f:
            accepted_hypotheses = [h for h in state["hypotheses"] if h.get("status") == "accepted"]
            json.dump(accepted_hypotheses, f, indent=2, ensure_ascii=False)
        print(f"💾 Hypotheses saved to {path}")
    except Exception as e:
        print(f"❌ Error saving hypotheses: {e}")
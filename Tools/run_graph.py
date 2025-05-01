from langgraph.graph import StateGraph, END
from Tools.State_dict import StateDict
from agents.agent_pubmed import agent_pubmed
from agents.QuestionGenAgent import agent_generate_questions
from agents.FinetuneEmbed import agent_embedding
from agents.UpdateEmbedding import agent_update_embeddings

def build_data_prep_graph() -> StateGraph:
    builder = StateGraph(StateDict)
    builder.add_node("PubMedFetch", agent_pubmed)
    builder.add_node("QuestionGen", agent_generate_questions)
    builder.add_node("FineTuneEmbed", agent_embedding)
    builder.add_node("UpdateEmbeddings", agent_update_embeddings)
    builder.set_entry_point("PubMedFetch")
    builder.add_edge("PubMedFetch", "QuestionGen")
    builder.add_edge("QuestionGen", "FineTuneEmbed")
    builder.add_edge("FineTuneEmbed", "UpdateEmbeddings")
    builder.add_edge("UpdateEmbeddings", END)
    return builder.compile()
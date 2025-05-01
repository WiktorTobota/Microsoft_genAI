from pathlib import Path
import os, json, datetime

def main():
    # Unified imports and config
    from dotenv import load_dotenv
    load_dotenv()
    from Tools.State_dict import StateDict
    from agents.OntologistAgent import OntologistAgent
    from agents.PubMedExpanderAgent import PubMedExpanderAgent
    from agents.PaperAnalsytAgent import PaperAnalystAgent
    from agents.agent_pubmed import agent_pubmed
    from agents.QuestionGenAgent import agent_generate_questions
    from agents.FinetuneEmbed import agent_embedding
    from agents.UpdateEmbedding import agent_update_embeddings
    from Tools.run_graph import build_data_prep_graph
    from agents.HypotesisGenerator import agent_generate_hypotheses
    from agents.HypotesisEvaluator import agent_evaluate_hypotheses
    from agents.HypotesisRefiner import agent_refine_hypotheses
    from Tools.needs_refinement import needs_refinement
    from Tools.save_hypotheses import save_hypotheses

    state: StateDict = {}
    # Part1: Ontology + PubMed expansion + summarization
    input_dir = Path(state.get('input_dir', 'Inputs/Subgraphs - rheumatology'))
    ont = OntologistAgent(model_name='gemini-1.5-flash-002')
    expander = PubMedExpanderAgent(model_name='gemini-1.5-flash-002', num_queries=3)
    analyst = PaperAnalystAgent()
    state['analyses'] = {}
    state['articles'] = {}
    state['summaries'] = {}
    for file in sorted(input_dir.glob('*.json')):
        key = file.stem
        with file.open('r', encoding='utf-8') as f:
            sg = json.load(f)
        analysis = ont.run(sg)
        state['analyses'][key] = analysis
        arts = []
        if not analysis.startswith('ERROR:'):
            arts = expander.search_with_minimum(analysis, max_per_query=5, minimum=10, max_rounds=3)
        state['articles'][key] = arts
        state['summaries'][key] = [analyst.summarize(a) for a in arts]
    # Part2: data prep graph (PubMed fetch, QG, fine-tune, update)
    data_prep_graph = build_data_prep_graph()
    state_init = {'csv_path': state.get('csv_path', 'Inputs/Articles/full.csv'),
                  'pubmed_data': [], 'questions': {}, 'model': None}
    # seed from existing chroma if any
    processed = data_prep_graph.invoke(state_init)
    state['pubmed_data'] = processed['pubmed_data']
    state['questions'] = processed['questions']
    state['model'] = processed['model']
    # Part3: hypothesis graph
    from langgraph.graph import StateGraph, END
    hypothesis_builder = StateGraph(StateDict)
    hypothesis_builder.add_node('HypothesisGeneration', agent_generate_hypotheses)
    hypothesis_builder.add_node('HypothesisEvaluation', agent_evaluate_hypotheses)
    hypothesis_builder.add_node('HypothesisRefinement', agent_refine_hypotheses)
    hypothesis_builder.set_entry_point('HypothesisGeneration')
    hypothesis_builder.add_edge('HypothesisGeneration','HypothesisEvaluation')
    hypothesis_builder.add_conditional_edges('HypothesisEvaluation', needs_refinement, {'refine':'HypothesisRefinement','end':END})
    hypothesis_builder.add_edge('HypothesisRefinement','HypothesisEvaluation')
    hypothesis_graph = hypothesis_builder.compile()
    # Run hypotheses for each analysis
    for key, analysis in state['analyses'].items():
        run_state = {'pubmed_data': state['pubmed_data'], 'questions': state['questions'], 'model': state['model'], 'analysis': analysis, 'hypotheses': [], 'refine_round': 0}
        final = hypothesis_graph.invoke(run_state)
        save_hypotheses(final, path=f'hypotheses_{key}.json')
    print('✅ Unified workflow completed')

if __name__ == '__main__':
    main()
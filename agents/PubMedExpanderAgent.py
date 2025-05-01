from agents.PubMedAgent import PubMedSearchAgent
from Tools.search_pubmed import search_pubmed
from dotenv import load_dotenv
import os
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI as LLM
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

from langchain import LLMChain
from typing import List, Dict, Any
from langchain.prompts import PromptTemplate as _PT
import pandas as pd
class PubMedExpanderAgent(PubMedSearchAgent):
    def __init__(self,
                    model_name: str = 'gemini-1.5-flash-002',
                    num_queries: int = 3):
        super().__init__(model_name=model_name, num_queries=num_queries)
        # Initialize LLM for query expansion
        self.llm = LLM(model=model_name, google_api_key=GEMINI_API_KEY, temperature=0.0)

        #here we initialize PromptTemplate
        self.expander_template = _PT(
            input_variables=['analysis', 'existing_queries', 'needed', 'num_queries'],
            template="""
            You are a biomedical research librarian.
            We have this ontologist analysis text:
            {analysis}
            Current PubMed search queries:
            {existing_queries}
            We need to expand the search with {needed} more queries.
            Generate up to {num_queries} new, broader PubMed queries (use fewer MeSH tags, wider keywords) to reach ≥10.
            Output each on a new line without numbering.
            """
        )
        self.expander_chain = LLMChain(llm=self.llm, prompt=self.expander_template)

    def expand_queries(self,
                        analysis: str,
                        existing_queries: List[str],
                        needed: int) -> List[str]:
        raw = self.expander_chain.run({
            'analysis': analysis,
            'existing_queries': "\n".join(existing_queries),
            'needed': needed,
            'num_queries': self.num_queries
        })
        return [q.strip() for q in raw.split('\n') if q.strip()] #returning a joined list of queries

    def search_with_minimum(self,
                            analysis: str,
                            max_per_query: int = 5,
                            minimum: int = 10,
                            max_rounds: int = 3) -> List[Dict[str, Any]]:
        articles = self.search(analysis, max_results_per_query=max_per_query)
        queries  = self.generate_queries(analysis)
        rnd = 0
        while len(articles) < minimum and rnd < max_rounds:
            needed = minimum - len(articles)
            print(f"\n⚠️ Only {len(articles)} articles; need {needed}. Expanding…")
            new_qs = self.expand_queries(analysis, queries, needed)
            print("🔄 Expanded queries:")
            for q in new_qs: print(f"  - {q}")
            queries += new_qs
            for q in new_qs:
                arts = search_pubmed(q, max_results=max_per_query)
                print(f"    • '{q[:60]}...' → {len(arts)}")
                articles.extend(arts)
            #checking for duplicates
            seen, unique = set(), []
            for art in articles:
                pmid = art.get('pmid')
                if pmid and pmid not in seen:
                    seen.add(pmid)
                    unique.append(art)
            articles = unique
            rnd += 1
        print(f"\n✅ Final articles count: {len(articles)} after {rnd} rounds.")
        return articles

if __name__ == "__main__":
    # Local testing
    agent = PubMedExpanderAgent(model_name='gemini-1.5-flash-002', num_queries=3)
    analysis = "Bone regeneration mechanisms in osteoporosis"
    articles = agent.search_with_minimum(analysis)
    for article in articles:
        print(f"PMID: {article['pmid']}, Title: {article['title']}")
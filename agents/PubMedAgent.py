import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_google_genai import ChatGoogleGenerativeAI as LLM
from langchain.prompts import PromptTemplate as _PT
from Tools.search_pubmed import search_pubmed
from typing import List, Dict, Any
from langchain import LLMChain
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

class PubMedSearchAgent:
    """
    Base class for PubMed search agents.
    """
    
    def __init__(self, model_name: str, num_queries: int):
        self.model_name = model_name
        self.num_queries = num_queries

    def search(self, analysis: str, max_results_per_query: int) -> List[Dict[str, Any]]:
        """Basic PubMed search using Entrez esearch and efetch"""
        return search_pubmed(analysis, max_results=max_results_per_query)

    def generate_queries(self, analysis: str) -> List[str]:
        """Initial set of search queries (can be overridden)"""
        return [analysis]

class PubMedExpanderAgent(PubMedSearchAgent):
    """
    Agent that generates PubMed search queries based on a ontologist analysis text.
    """
    
    def __init__(self,
                 model_name: str = 'gemini-1.5-flash-002',
                 num_queries: int = 3):
        super().__init__(model_name, num_queries)
        self.llm = LLM(model=model_name, google_api_key=GEMINI_API_KEY, temperature=0)
        prompt = f"""
        You are an expert biomedical research librarian.
        Based on the following ontologists analysis text, generate up to {num_queries} optimized PubMed search queries.
        Use boolean operators (AND, OR, NOT) and PubMed-specific field tags like [tiab], [MeSH Terms].
        Output each query on a new line, without numbering or commentary

        Ontologist analysis:
        {{analysis}}
        """
        # here we initialize the LLMChain with the prompt and the LLM
        self.chain = LLMChain(llm=self.llm,
                              prompt = _PT(input_variables=['analysis'], template=prompt))
    
    #After this, we use the LLMChain in function to generate the queries
    def generate_queries(self, analysis: str) -> List[str]:
        raw = self.chain.run({'analysis': analysis})
        return [q.strip() for q in raw.split('\n') if q.strip()]

    #Here we defining the search function that will use generate_queries and search_pubmed functions to get the articles
    def search(self, analysis: str, max_results_per_query: int = 5) -> List[Dict[str, Any]]:
        queries = self.generate_queries(analysis)
        
        print("🔍 Generated PubMed queries:")
        for q in queries: print(f"  - {q}")
        all_articles, seen = [], set()
        for q in queries:
            arts = search_pubmed(q, max_results=max_results_per_query)
            print(f"  • '{q[:60]}...' → {len(arts)} articles")
            for art in arts:
                pmid = art.get('pmid')
                if pmid and pmid  not in seen:
                    seen.add(pmid)
                    all_articles.append(art)
        return all_articles

if __name__ == "__main__":
    # Local testing
    
    agent = PubMedExpanderAgent(model_name='gemini-1.5-flash-002', num_queries=3)
    analysis = "Bone regeneration mechanisms in osteoporosis"
    articles = agent.search(analysis, max_results_per_query=5)
    for article in articles:
        print(f"PMID: {article['pmid']}, Title: {article['title']}")

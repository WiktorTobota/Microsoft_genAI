"""
OntologistAgent.py
This script defines the OntologistAgent class, which uses the Google Gemini API to analyze knowledge graphs in JSON format. It includes methods for loading environment variables, configuring the API key, and processing subgraphs.
"""
import os
from dotenv import load_dotenv
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI as LLM
import pandas as pd

from langchain.prompts import PromptTemplate
import json
from langchain.chains import LLMChain 

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


class OntologistAgent:
    def __init__(self,
                 model_name: str = 'gemini-1.5-flash',
                 temperature: float = 0.0):
        self.llm = LLM(model=model_name,
                       google_api_key=GEMINI_API_KEY, 
                       temperature=temperature)
        self.template = PromptTemplate(
            input_variables=['subgraph_json'],
            template="""
You are a sophisticated ontologist.

Your input is a knowledge graph in JSON format, containing:
- "nodes": array of objects with fields including "id" and "name"
- "edges": array of objects with fields "source", "relation", "target"

Tasks:
1. Define each term (use the "name" field).
2. Discuss each relationship between terms.

Output:

### Definitions:
- Term1: …
- Term2: …
…

### Relationships:
- TermA —[relation]→ TermB: explanation…
- TermC —[relation2]→ TermD: explanation…
…

Graph JSON:
{subgraph_json}
"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.template) 
    def run(self, subgraph: Dict[str, Any]) -> str:
        subgraph_json = json.dumps(subgraph, indent=2, ensure_ascii=False)

        return self.chain.run({'subgraph_json': subgraph_json})

if __name__ == "__main__":
    agent = OntologistAgent()
    try:
        with open('Subgraphs - rheumatology/Autoimmunity.json', 'r', encoding='utf-8') as f:
            sample_subgraph = json.load(f)
        output = agent.run(sample_subgraph)
        print(output)
    except FileNotFoundError:
        print("Error: Subgraphs - rheumatology/Autoimmunity.json not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
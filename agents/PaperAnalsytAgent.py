from langchain import LLMChain
from langchain.prompts import PromptTemplate as _PT
import os 
from typing import Dict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI as LLM
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
import pandas as pd
class PaperAnalystAgent:
    def __init__(self,
                model_name: str = 'gemini-1.5-flash-002',
                temperature: float = 0.0):
        self.llm = LLM(model=model_name,
                        google_api_key=GEMINI_API_KEY,
                        temperature=temperature)
        template = _PT(
            input_variables=['title', 'abstract'], 
            template="""
            You are a biomedical scientist.
            Summarize the paper titled "{title}".
            Under two headings, list bullet points"
            
            - Strong Evidence:
            - Weak Evidence:
            """)
        self.chain = LLMChain(llm=self.llm, prompt=template)
    def summarize(self, article: Dict[str, str]) -> str:
        return self.chain.run({
            'title':    article.get('title', ''),
            'abstract': article.get('abstract', '')
        })

if __name__ == "__main__":
    # Local testing
    
    agent = PaperAnalystAgent(model_name='gemini-1.5-flash-002', temperature=0.0)
    article = {
        'title': 'Ultrasound-Responsive Piezoelectric Membrane Promotes Osteoporotic Bone Regeneration via the "Two-Way Regulation" Bone Homeostasis Strategy.',
        'abstract': 'The repair of osteoporotic bone defects remains inadequately addressed, primarily due to a disruption in bone homeostasis, characterized by insufficient bone formation and excessive bone resorption. Current research either focuses on promoting bone formation or inhibiting bone resorption, however, the bone repair efficacy of these single-target therapeutic strategies is limited. Herein, a "two-way regulation" bone homeostasis strategy is proposed utilizing piezoelectric composite membranes (DAT/KS), capable of simultaneously regulating osteogenesis and osteoclastogenesis, with high piezoelectric performance, good biocompatibility, and excellent degradability, to promote bone regeneration under osteoporotic conditions. The DAT/KS membrane under ultrasound (US) treatment enables the controlled modulation of piezoelectric stimulation and the release of saikosaponin D (SSD), which promotes osteogenic differentiation while simultaneously inhibiting osteoclast differentiation and function, thereby effectively restoring bone homeostasis and enhancing osteoporotic bone repair. Mechanistic insights reveal the promotion of both canonical and non-canonical Wnt signaling in bone marrow mesenchymal stem cells (BMSCs), which determines their osteogenic differentiation fate, and the downregulation of the NF-κB signaling in bone marrow mononuclear macrophages (BMMs). This study presents optimized sono-piezoelectric biomaterials capable of bidirectionally regulating both osteogenic and osteoclastic differentiation, providing a new potential therapeutic approach for pathological bone injuries.'
    }
    print(agent.summarize(article))
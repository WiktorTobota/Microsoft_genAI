from typing import TypedDict, Dict, List, Optional, Any
from sentence_transformers import SentenceTransformer

class StateDict(TypedDict, total=False):
    # Inputs
    input_dir: str
    csv_path: str
    # Part1 outputs
    analyses: Dict[str, str]
    articles: Dict[str, List[Dict[str, Any]]]
    summaries: Dict[str, List[str]]
    # Part2 state
    pubmed_data: List[Dict[str, Any]]
    questions: Dict[str, List[str]]
    model: Optional[SentenceTransformer]
    # Part3 state
    analysis: str
    hypotheses: List[Dict[str, Any]]
    refine_round: int
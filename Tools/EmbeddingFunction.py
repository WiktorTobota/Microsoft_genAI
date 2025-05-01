from typing import List

class EmbeddingFunction:
    def __init__(self, model):
        self.model=model
    def __call__(self, input: List[str]) -> List[List[float]]:
        valid_input = [str(text) if text is not None else "" for text in input]

        if not valid_input:
            return []
        return self.model.encode(valid_input, batch_size=64, show_progress_bar=False).tolist()
    

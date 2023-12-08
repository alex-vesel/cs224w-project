from sentence_transformers import SentenceTransformer
from typing import List, Union
from numpy import ndarray
from torch import Tensor

class LMEncoder():
    def __init__(self, model_name):

        # Initialize encoder from HuggingFace sentence-transformers
        self.model = SentenceTransformer(model_name)

    def encode(self, sentences: Union[str, List[str]],
               batch_size: int = 32,
               output_value: str = 'sentence_embedding',
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               normalize_embeddings: bool = False) -> Union[List[Tensor], ndarray, Tensor]:
        
        output = self.model.encode(
            sentences= sentences,
            batch_size= batch_size,
            output_value= output_value,
            convert_to_numpy= convert_to_numpy,
            convert_to_tensor= convert_to_tensor,
            normalize_embeddings= normalize_embeddings
        )
        output = torch.tensor(output)

        return output
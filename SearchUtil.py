import numpy as np
from numba import jit
from transformers import AutoTokenizer, AutoModel
from Preprocessor import Preprocessor
import torch


@jit(nopython=True, cache=True)
def cosine_similarity_numba(u: np.ndarray, v: np.ndarray) -> np.float64:
    if np.linalg.norm(u) == 0 or np.linalg.norm(v) == 0:
        return 1.
    return np.dot(u, v)


class SearchUtil:
    def __init__(self, matrix: np.ndarray, model: AutoModel, tokenizer: AutoTokenizer):
        """
        matrix: np.ndarray
            shape: (n_lines, 313)
        """
        self.matrix = matrix
        self.model = model
        self.tokenizer = tokenizer
        self.preprocessor = Preprocessor()

    def embed_bert_cls(self, text: str) -> torch.FloatTensor:
        t = self.tokenizer(text, padding=True,
                           truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**{k: v.to(self.model.device)
                                         for k, v in t.items()})
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings[0].cpu()

    # @jit(nopython=True, cache=True)
    def sort(self, text: str) -> np.ndarray:
        query_text = self.preprocessor.clean_text(
            text=text,
            remove_duplicates=True
        )
        query_vec = np.array(self.embed_bert_cls(query_text), dtype=np.float32)
        for i in range(self.matrix.shape[0]):
            self.matrix[i][1] = cosine_similarity_numba(
                query_vec,
                self.matrix[i][3:]
            )
        return self.matrix[self.matrix[:, 1].argsort()][::-1]

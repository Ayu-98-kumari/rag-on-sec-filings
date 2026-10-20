"""The embedding model: text in, 384 numbers out.

bge-small-en-v1.5 runs locally on the M4 GPU through PyTorch's MPS backend.
Local and free matters here: the index is rebuilt every time the chunking
strategy changes, and a rebuild that costs nothing is a rebuild we are willing
to run. With a paid API we would run fewer experiments.

Note the asymmetry: bge models expect a short instruction prefix on the QUERY
but not on the stored passages. Embedding both sides the same way measurably
degrades retrieval, so the two paths are separate functions on purpose.
"""

import warnings
from functools import lru_cache

warnings.filterwarnings("ignore")

import torch
from sentence_transformers import SentenceTransformer

from src.config import EMBED_MODEL, QUERY_PREFIX


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL, device=device())


def embed_passages(texts: list[str], batch_size: int = 64,
                   show_progress: bool = False):
    """Embed stored chunks. No prefix."""
    return model().encode(
        texts, batch_size=batch_size, normalize_embeddings=True,
        show_progress_bar=show_progress, convert_to_numpy=True,
    )


def embed_query(text: str):
    """Embed a user question. Prefixed, as bge expects."""
    return model().encode(
        [QUERY_PREFIX + text], normalize_embeddings=True, convert_to_numpy=True,
    )[0]

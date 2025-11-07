"""vLLM embedding client using msgspec for fast serialization.

OpenAI-compatible /v1/embeddings endpoint with batching support.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import msgspec
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel_rev.config.settings import VLLMConfig


class EmbeddingRequest(msgspec.Struct):
    """OpenAI-compatible embedding request payload.

    Request structure for the vLLM /v1/embeddings endpoint, matching the OpenAI
    API format. This allows vLLM to be used as a drop-in replacement for OpenAI's
    embedding API.

    The request can embed a single text (str) or a batch of texts (list[str]).
    Batch requests are more efficient as they're processed in parallel by the
    model.

    Attributes
    ----------
    input : str | list[str]
        Text(s) to embed. Can be a single string for one embedding, or a list
        of strings for batch embedding. Batch processing is more efficient.
        Each text should be a code chunk or query string.
    model : str
        Model identifier string. Must match a model loaded by the vLLM server.
        Defaults to "nomic-ai/nomic-embed-code" which is a code-specific
        embedding model with 2560 dimensions.
    """

    input: str | list[str]
    model: str


class EmbeddingData(msgspec.Struct):
    """Single embedding result from a batch request.

    Represents one embedding vector from a batch embedding request. Each text
    in the input batch produces one EmbeddingData object. The index field
    indicates the position in the original batch, which is important because
    vLLM may return results in a different order than the input.

    Attributes
    ----------
    embedding : list[float]
        Embedding vector as a list of floats. The length matches the model's
        dimension (e.g., 2560 for nomic-embed-code). Values are typically
        normalized for cosine similarity.
    index : int
        Zero-based index indicating the position of this embedding in the
        original input batch. Used to match embeddings back to their input
        texts when results may be reordered.
    """

    embedding: list[float]
    index: int


class EmbeddingResponse(msgspec.Struct):
    """OpenAI-compatible embedding response payload.

    Response structure from the vLLM /v1/embeddings endpoint, matching the OpenAI
    API format. Contains the embedding vectors along with metadata about the
    request and model usage.

    The data field contains one EmbeddingData per input text. Results may be
    returned in a different order than the input, so use the index field to
    match embeddings to their original texts.

    Attributes
    ----------
    data : list[EmbeddingData]
        List of embedding results, one per input text. The list length matches
        the number of texts in the request. Each EmbeddingData contains the
        vector and its index in the original batch.
    model : str
        Model identifier that was used to generate the embeddings. Should match
        the model field from the request. Useful for verification and logging.
    usage : dict
        Token usage statistics dictionary. Typically contains keys like "prompt_tokens"
        and "total_tokens" indicating how many tokens were processed. Useful for
        monitoring and cost tracking.
    """

    data: list[EmbeddingData]
    model: str
    usage: dict


class VLLMClient:
    """vLLM embedding client.

    Parameters
    ----------
    config : VLLMConfig
        vLLM configuration.
    """

    def __init__(self, config: VLLMConfig) -> None:
        self.config = config
        self._encoder = msgspec.json.Encoder()
        self._decoder = msgspec.json.Decoder(EmbeddingResponse)

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of texts using the vLLM service.

        Sends a batch of texts to the vLLM embedding service and returns their
        embedding vectors as a NumPy array. The function uses msgspec for fast
        JSON encoding/decoding, which is significantly faster than standard
        json module.

        The function handles reordering - vLLM may return embeddings in a
        different order than the input, so results are sorted by index to
        match the input sequence.

        Parameters
        ----------
        texts : Sequence[str]
            Sequence of texts to embed. Can be any sequence type (list, tuple, etc.).
            Empty sequence returns empty array. Each text should be a code chunk
            or query string suitable for the embedding model.

        Returns
        -------
        np.ndarray
            Embedding vectors as a 2D NumPy array of shape (len(texts), vec_dim)
            where vec_dim is the model's embedding dimension (e.g., 2560).
            Dtype is float32 for memory efficiency. Returns empty array (shape
            (0, vec_dim)) if texts is empty. The order matches the input texts.

        Notes
        -----
        The function uses httpx.Client for HTTP requests with the configured
        timeout. If the request fails (network error, timeout, HTTP error),
        httpx will raise an exception (HTTPStatusError, TimeoutException, etc.).
        These exceptions propagate to the caller - handle them appropriately.

        The function does not retry failed requests. For production use, consider
        adding retry logic with exponential backoff for transient failures.
        """
        if not texts:
            return np.array([], dtype=np.float32)

        request = EmbeddingRequest(input=list(texts), model=self.config.model)
        payload = self._encoder.encode(request)

        with httpx.Client(timeout=self.config.timeout_s) as client:
            response = client.post(
                f"{self.config.base_url}/embeddings",
                content=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

        result = self._decoder.decode(response.content)

        # Sort by index and extract vectors
        sorted_data = sorted(result.data, key=lambda d: d.index)
        vectors = np.array([d.embedding for d in sorted_data], dtype=np.float32)

        return vectors

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single string and return its vector."""
        vectors = self.embed_batch([text])
        if vectors.size == 0:
            msg = "vLLM returned no vectors for single embed request"
            raise RuntimeError(msg)
        return vectors[0]

    def embed_chunks(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Embed texts in batches.

        Parameters
        ----------
        texts : Sequence[str]
            Texts to embed.
        batch_size : int | None
            Batch size; defaults to config.batch_size.

        Returns
        -------
        np.ndarray
            Embeddings array of shape (len(texts), vec_dim).
        """
        if not texts:
            return np.array([], dtype=np.float32)

        bs = batch_size or self.config.batch_size
        all_vectors = []

        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            vectors = self.embed_batch(batch)
            all_vectors.append(vectors)

        return np.vstack(all_vectors)


__all__ = [
    "EmbeddingData",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "VLLMClient",
]

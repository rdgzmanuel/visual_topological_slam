"""Language-driven place retrieval (reworked NLP module).

What changed relative to the thesis code and why:

1. **No stitched panoramas.** Node semantics were previously computed from
   SIFT-homography panoramas, which frequently degenerated into nonsense and
   poisoned the embeddings — your observed failure mode. Each node now keeps
   up to three *raw* representative views; a query is scored against every
   view and the node score is the max (multi-view max pooling), a standard
   retrieval technique that is robust to one bad view.
2. **Prompt ensembling on by default.** Text queries are embedded through a
   small ensemble of CLIP-style templates and averaged, which is the
   documented way to close the modality gap of contrastive VLMs. The
   hand-crafted ``_SCENE_MAPPINGS`` dictionary is removed: it was a hidden,
   environment-specific vocabulary that contradicts dataset-agnosticism.
3. **Model-agnostic backbone.** Any Hugging Face CLIP- or SigLIP-family
   checkpoint can be selected by name. ViT-B/32 remains the default; for the
   paper, evaluate at least one stronger checkpoint (e.g.
   ``openai/clip-vit-large-patch14`` or ``google/siglip-base-patch16-224``).
4. **Calibrated rejection instead of a similarity threshold.** Raw CLIP
   cosine similarities are not comparable across queries, so a fixed
   threshold cannot work. We instead softmax the scores over nodes with the
   model's own learned logit scale and reject when the distribution is too
   flat — measured by the margin between the top-2 posterior masses. The
   rejection rule is relative (per-query) rather than an absolute cutoff.
5. **Translation removed.** Per your instruction, the module is
   English-only; queries are used as given.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from vts_core.topo_graph import TopoGraph, TopoNode

_PROMPT_TEMPLATES: tuple[str, ...] = (
    "a photo of a {}",
    "a photo of a {} in a building",
    "an indoor scene showing a {}",
    "a picture of the {} of an office building",
)

# Minimum top-2 posterior margin for an unambiguous answer. With softmax at
# CLIP's logit scale, 0.1 means "the best node holds at least 10 percentage
# points more probability mass than the runner-up" — a relative,
# query-independent criterion, not a similarity cutoff.
_MIN_POSTERIOR_MARGIN: float = 0.1

# Imperative command phrasings stripped to recover the bare place description,
# so "take me to the printer area" embeds as "printer area" (the templates
# expect a noun phrase, not a full sentence).
_COMMAND_PREFIXES: tuple[str, ...] = (
    "take me to", "bring me to", "lead me to", "guide me to", "navigate to",
    "go to", "drive to", "move to", "head to", "walk to", "get to",
    "show me", "find", "locate", "where is", "where's", "where are",
    "i want to go to", "can you take me to",
)


def place_phrase(query: str) -> str:
    """Reduce a natural-language command to its place noun phrase."""
    phrase: str = query.strip().lower().rstrip(" ?.!")
    for prefix in _COMMAND_PREFIXES:
        if phrase.startswith(prefix):
            phrase = phrase[len(prefix):].strip()
            break
    for article in ("the ", "a ", "an "):
        if phrase.startswith(article):
            phrase = phrase[len(article):]
            break
    return phrase or query.strip().lower()


class SemanticEncoder:
    """Wrapper around a Hugging Face CLIP/SigLIP checkpoint."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        self._device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model = AutoModel.from_pretrained(model_name).to(self._device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(model_name)
        scale = getattr(self._model, "logit_scale", None)
        self._logit_scale: float = (
            float(scale.exp().item()) if scale is not None else 100.0
        )

    @property
    def logit_scale(self) -> float:
        """The model's learned similarity temperature."""
        return self._logit_scale

    @staticmethod
    def _as_embedding(output: object) -> torch.Tensor:
        """Extract the projected embedding tensor across transformers versions.

        transformers 5.x returns a model-output object from
        ``get_image_features`` / ``get_text_features`` whose ``pooler_output``
        is already the projected (shared-space) embedding; 4.x returns the
        tensor directly. Handle both.
        """
        if torch.is_tensor(output):
            return output
        for attr in ("image_embeds", "text_embeds", "pooler_output"):
            value = getattr(output, attr, None)
            if value is not None:
                return value
        return output.last_hidden_state[:, 0]  # type: ignore[attr-defined]

    @torch.no_grad()
    def embed_images(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        """Embed BGR images; returns (n, d) L2-normalized float32."""
        pil_images: list[Image.Image] = [
            Image.fromarray(img[:, :, ::-1].copy()) for img in images_bgr
        ]
        inputs = self._processor(images=pil_images, return_tensors="pt").to(
            self._device
        )
        features: torch.Tensor = self._as_embedding(
            self._model.get_image_features(**inputs)
        )
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def embed_text(self, query: str) -> np.ndarray:
        """Embed a text query through the prompt-template ensemble.

        The query is first reduced to its place noun phrase (so a full command
        like "take me to the kitchen" embeds as "kitchen") and then averaged
        over the prompt templates.
        """
        phrase: str = place_phrase(query)
        prompts: list[str] = [t.format(phrase) for t in _PROMPT_TEMPLATES]
        inputs = self._processor(
            text=prompts, return_tensors="pt", padding=True
        ).to(self._device)
        features: torch.Tensor = self._as_embedding(
            self._model.get_text_features(**inputs)
        )
        features = features / features.norm(dim=-1, keepdim=True)
        mean: torch.Tensor = features.mean(dim=0)
        mean = mean / mean.norm()
        return mean.cpu().numpy().astype(np.float32)


class PlaceRetriever:
    """Maps natural-language queries to nodes of a topological map."""

    def __init__(self, encoder: SemanticEncoder, graph: TopoGraph) -> None:
        self._encoder: SemanticEncoder = encoder
        self._graph: TopoGraph = graph
        self._ensure_view_embeddings()

    def _ensure_view_embeddings(self) -> None:
        """Embed all node views once (lazy fill on load or after fusion)."""
        for node in self._graph.nodes.values():
            if node.view_embeddings is not None or not node.views:
                continue
            node.view_embeddings = self._encoder.embed_images(node.views)

    def query(
        self, sentence: str, top_k: int = 3
    ) -> tuple[list[tuple[TopoNode, float, float]], bool]:
        """Retrieve the nodes best matching a sentence.

        Args:
            sentence: English natural-language place description.
            top_k: Number of candidates to return.

        Returns:
            (ranked list of (node, posterior probability, raw cosine score),
            confident_flag). The raw cosine score is the multi-view max-pooled
            image/text similarity, surfaced for debugging and inspection.
            ``confident_flag`` is False when the top-2 posterior margin is
            below the rejection bound — the caller should treat the answer
            as ambiguous (e.g. ask the user to disambiguate among top-k).
        """
        text_embedding: np.ndarray = self._encoder.embed_text(sentence)

        node_ids: list[int] = []
        scores: list[float] = []
        for node_id in sorted(self._graph.nodes):
            node: TopoNode = self._graph.nodes[node_id]
            if node.view_embeddings is None or node.view_embeddings.size == 0:
                continue
            similarities: np.ndarray = node.view_embeddings @ text_embedding
            node_ids.append(node_id)
            scores.append(float(similarities.max()))  # multi-view max pooling

        if not node_ids:
            return [], False

        score_array: np.ndarray = np.array(scores, dtype=np.float64)
        logits: np.ndarray = self._encoder.logit_scale * score_array
        logits -= logits.max()
        posterior: np.ndarray = np.exp(logits)
        posterior /= posterior.sum()

        order: np.ndarray = np.argsort(posterior)[::-1]
        ranked: list[tuple[TopoNode, float, float]] = [
            (
                self._graph.nodes[node_ids[int(i)]],
                float(posterior[int(i)]),
                float(score_array[int(i)]),
            )
            for i in order[:top_k]
        ]

        if posterior.shape[0] >= 2:
            margin: float = float(
                posterior[order[0]] - posterior[order[1]]
            )
            confident: bool = margin >= _MIN_POSTERIOR_MARGIN
        else:
            confident = True

        return ranked, confident

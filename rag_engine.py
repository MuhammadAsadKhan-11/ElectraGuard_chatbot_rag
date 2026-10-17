"""In-memory vector search over rag/index/knowledge_index.json (numpy)."""
import json
import logging
from pathlib import Path
import numpy as np

log = logging.getLogger("rag")
INDEX_PATH = Path(__file__).parent / "rag" / "index" / "knowledge_index.json"
MIN_SCORE = 0.55  # ignore weak matches, same threshold as the Node version


class KnowledgeBase:
    def __init__(self):
        self.texts: list[str] = []
        self.sources: list[str] = []
        self.matrix: np.ndarray | None = None

    def load(self):
        try:
            records = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
            m = np.asarray([r["embedding"] for r in records], dtype=np.float32)
            norms = np.linalg.norm(m, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.matrix = m / norms  # pre-normalised: cosine == dot product
            self.texts = [r["text"] for r in records]
            self.sources = [r["source"] for r in records]
            log.info("RAG: loaded %d chunks from knowledge_index.json", len(records))
        except Exception as e:
            self.matrix = None
            log.warning("RAG: index missing/unreadable, answering without retrieval (%s)", e)

    @property
    def ready(self) -> bool:
        return self.matrix is not None and len(self.texts) > 0

    def search(self, query_vec: list[float], top_k: int = 4):
        if not self.ready:
            return [], []
        q = np.asarray(query_vec, dtype=np.float32)
        n = np.linalg.norm(q)
        if n == 0:
            return [], []
        scores = self.matrix @ (q / n)
        order = np.argsort(-scores)
        top = [int(i) for i in order if scores[i] >= MIN_SCORE][:top_k]
        chunks = [self.texts[i] for i in top]
        sources = list(dict.fromkeys(self.sources[i] for i in top))  # unique, ordered
        return chunks, sources

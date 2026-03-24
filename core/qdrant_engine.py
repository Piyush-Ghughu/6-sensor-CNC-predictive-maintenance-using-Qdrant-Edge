# core/qdrant_engine.py
# Qdrant Edge — uses EdgeShard (qdrant-edge-py), fully embedded, no server

import os
import shutil
import numpy as np

from qdrant_edge import (
    Distance,
    EdgeConfig,
    EdgeVectorParams,
    EdgeShard,
    Point,
    UpdateOperation,
    Query,
    QueryRequest,
)
import config


def _shard_exists(path: str) -> bool:
    return os.path.isdir(path) and any(os.scandir(path))


class QdrantEdgeEngine:
    """
    Wraps Qdrant EdgeShard for on-device vector storage + similarity search.
    No server. No network. Runs entirely on the edge device.
    """

    def __init__(self, fresh: bool = True):
        path = config.QDRANT_SHARD_PATH
        cfg  = EdgeConfig(
            vectors={
                config.VECTOR_NAME: EdgeVectorParams(
                    size=config.VECTOR_SIZE,
                    distance=Distance.Cosine,
                )
            }
        )

        if fresh and _shard_exists(path):
            shutil.rmtree(path)

        os.makedirs(path, exist_ok=True)

        if _shard_exists(path) and not fresh:
            self._shard = EdgeShard.load(path, cfg)
        else:
            self._shard = EdgeShard.create(path, cfg)

        self._count = 0        # patterns stored so far
        self._id_counter = 0   # monotonic integer IDs (EdgeShard requires uint)


    def store(self, vector: np.ndarray) -> None:
        """Store a learned normal-pattern vector."""
        self._id_counter += 1
        point = Point(
            id=self._id_counter,
            vector={config.VECTOR_NAME: vector.tolist()},
        )
        self._shard.update(UpdateOperation.upsert_points([point]))
        self._count += 1


    def search(self, vector: np.ndarray, top_k: int = 5) -> float:
        """
        Find the most similar stored pattern.
        Returns cosine similarity in [0, 1].  Higher = more similar to normal.
        Returns 0.0 when no patterns are stored yet.
        """
        if self._count == 0:
            return 0.0

        req = QueryRequest(
            query=Query.Nearest(query=vector.tolist(), using=config.VECTOR_NAME),
            limit=min(top_k, self._count),
        )
        results = self._shard.query(req)

        if not results:
            return 0.0

        
        return float(results[0].score)


    def flush(self) -> None:
        self._shard.flush()

    def close(self) -> None:
        self._shard.close()

    @property
    def pattern_count(self) -> int:
        return self._count

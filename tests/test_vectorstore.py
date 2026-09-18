"""Tests for the vector store: pure score conversion plus real (local) Chroma
behavior in a temporary directory. No network is involved."""

from __future__ import annotations

from rag_mvp.ingest import Chunk
from rag_mvp.vectorstore import VectorStore, _distance_to_score


class TestDistanceToScore:
    def test_zero_distance_is_max_similarity(self):
        assert _distance_to_score(0.0) == 1.0

    def test_distance_one_is_half(self):
        assert _distance_to_score(1.0) == 0.0

    def test_negative_distance_clamped_to_one(self):
        assert _distance_to_score(-0.5) == 1.0

    def test_large_distance_clamped_to_zero(self):
        assert _distance_to_score(2.0) == 0.0
        assert _distance_to_score(5.0) == 0.0

    def test_monotonic_decreasing(self):
        assert _distance_to_score(0.2) > _distance_to_score(0.8)


def _chunk(cid: str, text: str, index: int = 0) -> Chunk:
    return Chunk(id=cid, text=text, source="src.txt", chunk_index=index)


class TestVectorStore:
    def _store(self, tmp_path):
        return VectorStore(str(tmp_path / "store"), "test_collection")

    def test_empty_store_count_is_zero(self, tmp_path):
        store = self._store(tmp_path)
        assert store.count() == 0

    def test_query_on_empty_store_returns_empty(self, tmp_path):
        store = self._store(tmp_path)
        assert store.query([0.1, 0.2, 0.3], top_k=5) == []

    def test_add_and_count(self, tmp_path):
        store = self._store(tmp_path)
        chunks = [_chunk("a", "alpha", 0), _chunk("b", "beta", 1)]
        embeddings = [[1.0, 0.0], [0.0, 1.0]]
        store.add(chunks, embeddings)
        assert store.count() == 2

    def test_add_empty_is_noop(self, tmp_path):
        store = self._store(tmp_path)
        store.add([], [])
        assert store.count() == 0

    def test_add_length_mismatch_raises(self, tmp_path):
        store = self._store(tmp_path)
        try:
            store.add([_chunk("a", "x")], [[1.0], [2.0]])
            raised = False
        except ValueError:
            raised = True
        assert raised

    def test_query_returns_nearest_first(self, tmp_path):
        store = self._store(tmp_path)
        chunks = [_chunk("a", "alpha", 0), _chunk("b", "beta", 1)]
        # Two orthogonal unit vectors.
        store.add(chunks, [[1.0, 0.0], [0.0, 1.0]])
        # Query aligned with the first vector.
        results = store.query([1.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0].id == "a"
        assert results[0].score > results[1].score
        # Metadata round-trips.
        assert results[0].source == "src.txt"
        assert results[0].chunk_index == 0

    def test_query_respects_top_k(self, tmp_path):
        store = self._store(tmp_path)
        chunks = [_chunk(str(i), f"t{i}", i) for i in range(5)]
        embs = [[float(i), 1.0] for i in range(5)]
        store.add(chunks, embs)
        assert len(store.query([1.0, 1.0], top_k=3)) == 3

    def test_top_k_larger_than_store_is_clamped(self, tmp_path):
        store = self._store(tmp_path)
        store.add([_chunk("a", "x")], [[1.0, 0.0]])
        # Requesting more than exist should not error.
        assert len(store.query([1.0, 0.0], top_k=100)) == 1

    def test_upsert_is_idempotent_on_id(self, tmp_path):
        store = self._store(tmp_path)
        store.add([_chunk("a", "first")], [[1.0, 0.0]])
        store.add([_chunk("a", "second")], [[0.0, 1.0]])
        assert store.count() == 1

    def test_reset_clears_store(self, tmp_path):
        store = self._store(tmp_path)
        store.add([_chunk("a", "x"), _chunk("b", "y", 1)], [[1.0, 0.0], [0.0, 1.0]])
        assert store.count() == 2
        store.reset()
        assert store.count() == 0

    def test_persistence_across_instances(self, tmp_path):
        path = str(tmp_path / "persist")
        store1 = VectorStore(path, "persist_col")
        store1.add([_chunk("a", "x")], [[1.0, 0.0]])
        # A fresh instance pointing at the same dir sees the data.
        store2 = VectorStore(path, "persist_col")
        assert store2.count() == 1

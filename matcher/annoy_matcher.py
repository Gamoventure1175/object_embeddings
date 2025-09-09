# matcher/annoy_matcher.py
from annoy import AnnoyIndex
import numpy as np
import os
import json

class AnnoyFaceMatcher:
    def __init__(
        self,
        vector_size=512,
        index_path="faces.ann",
        meta_path="faces_meta.json",
        emb_path="face_embeddings.npy",
        threshold=1.2,
        n_trees=10,
        match_ratio=0.85,   # ratio test threshold (top1/top2)
        min_margin=0.20,    # margin between top1 and top2
        search_k=None       # Annoy search depth
    ):
        """
        vector_size: dimension of embeddings (ArcFace -> 512)
        index_path: annoy index filename
        meta_path: JSON mapping of id -> name
        emb_path: numpy file storing embeddings in order
        threshold: euclidean distance threshold for match acceptance
        n_trees: Annoy trees for build
        match_ratio: ratio test for disambiguating close matches
        min_margin: minimum distance gap between top1 and top2
        search_k: optional Annoy search depth (higher = more accurate, slower)
        """
        self.vector_size = vector_size
        self.index_path = index_path
        self.meta_path = meta_path
        self.emb_path = emb_path
        self.threshold = threshold
        self.n_trees = n_trees
        self.match_ratio = match_ratio
        self.min_margin = min_margin
        self.search_k = search_k

        # meta: {"0": "Tom", "1": "Zendaya", ...}
        self.meta = {}

        # all_embeddings: list of tuples (id:int, emb:np.ndarray)
        self.all_embeddings = []
        self.next_id = 0

        # in-memory Annoy index
        self.index = AnnoyIndex(self.vector_size, "angular")

    # ---------------------------
    # Disk I/O helpers
    # ---------------------------
    def _save_all_embeddings(self):
        if not self.all_embeddings:
            np.save(self.emb_path, np.zeros((0, self.vector_size), dtype=np.float32))
            return
        emb_array = np.stack([emb for _, emb in self.all_embeddings]).astype(np.float32)
        np.save(self.emb_path, emb_array)

    def _load_all_embeddings(self):
        if os.path.exists(self.emb_path):
            emb_array = np.load(self.emb_path)
            self.all_embeddings = [(i, emb_array[i].astype(np.float32)) for i in range(len(emb_array))]
        else:
            self.all_embeddings = []

    def _save_meta(self):
        with open(self.meta_path, "w") as f:
            json.dump(self.meta, f)

    def _load_meta(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r") as f:
                self.meta = json.load(f)
            if self.meta:
                self.next_id = max(map(int, self.meta.keys())) + 1
            else:
                self.next_id = 0
        else:
            self.meta = {}
            self.next_id = 0

    # ---------------------------
    # Index (build/save/load)
    # ---------------------------
    def _rebuild_index(self):
        self.index = AnnoyIndex(self.vector_size, "euclidean")
        for idx, emb in self.all_embeddings:
            self.index.add_item(int(idx), emb.tolist())
        if self.all_embeddings:
            self.index.build(self.n_trees)
            self.index.save(self.index_path)
        else:
            if os.path.exists(self.index_path):
                try:
                    os.remove(self.index_path)
                except Exception:
                    pass

    # ---------------------------
    # Public API
    # ---------------------------
    def register(self, name, embedding):
        embedding = np.asarray(embedding, dtype=np.float32).flatten()
        if embedding.shape[0] != self.vector_size:
            raise ValueError(f"Embedding dimension mismatch: got {embedding.shape[0]}, expected {self.vector_size}")

        self._load_meta()
        self._load_all_embeddings()

        assigned_id = self.next_id
        self.all_embeddings.append((assigned_id, embedding))
        self.meta[str(assigned_id)] = name

        self._save_all_embeddings()
        self._save_meta()
        self._rebuild_index()

        print(f"[AnnoyFaceMatcher] Registered '{name}' with id {assigned_id}")
        self.next_id += 1
        return assigned_id

    def add_face(self, name, embedding):
        return self.register(name, embedding)

    def load_for_matching(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError("No index/meta found. Register at least one face first.")

        self.index = AnnoyIndex(self.vector_size, "euclidean")
        loaded = self.index.load(self.index_path)
        if not loaded:
            raise RuntimeError("Failed to load Annoy index from disk.")
        with open(self.meta_path, "r") as f:
            self.meta = json.load(f)
        self.next_id = max(map(int, self.meta.keys())) + 1 if self.meta else 0
        self._load_all_embeddings()

    def match(self, embedding, top_k=10):
        embedding = np.asarray(embedding, dtype=np.float32).flatten()
        if embedding.shape[0] != self.vector_size:
            raise ValueError(f"Embedding dimension mismatch: got {embedding.shape[0]}, expected {self.vector_size}")

        try:
            K = max(top_k, 10)
            if self.search_k is not None:
                idxs, dists = self.index.get_nns_by_vector(
                    embedding.tolist(), K, search_k=self.search_k, include_distances=True
                )
            else:
                idxs, dists = self.index.get_nns_by_vector(
                    embedding.tolist(), K, include_distances=True
                )
        except Exception:
            return []

        best_matches = {}
        for idx, dist in zip(idxs, dists):
            if dist <= self.threshold:
                person = self.meta.get(str(idx), None)
                if person is None:
                    continue
                if person not in best_matches or dist < best_matches[person]:
                    best_matches[person] = dist

        if not best_matches:
            return []

        sorted_matches = sorted(best_matches.items(), key=lambda x: x[1])

        # Apply ratio / margin test
        if len(sorted_matches) >= 2:
            (p1, d1), (p2, d2) = sorted_matches[0], sorted_matches[1]
            ratio_ok = (d1 / (d2 + 1e-6)) <= self.match_ratio
            margin_ok = (d2 - d1) >= self.min_margin
            if not (ratio_ok or margin_ok):
                return []

        return sorted_matches

    def list_registered(self):
        return {int(k): v for k, v in self.meta.items()}

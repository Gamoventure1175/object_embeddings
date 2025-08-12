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
    ):
        """
        vector_size: dimension of embeddings (ArcFace -> 512)
        index_path: annoy index filename
        meta_path: JSON mapping of id -> name
        emb_path: numpy file storing embeddings in order
        threshold: euclidean distance threshold for match acceptance
        n_trees: Annoy trees for build
        """
        self.vector_size = vector_size
        self.index_path = index_path
        self.meta_path = meta_path
        self.emb_path = emb_path
        self.threshold = threshold
        self.n_trees = n_trees

        # meta: {"0": "Tom", "1": "Zendaya", ...}
        self.meta = {}

        # all_embeddings: list of tuples (id:int, emb:np.ndarray)
        self.all_embeddings = []
        self.next_id = 0

        # in-memory Annoy index (can be rebuilt by register)
        self.index = AnnoyIndex(self.vector_size, "euclidean")

    # ---------------------------
    # Disk I/O helpers
    # ---------------------------
    def _save_all_embeddings(self):
        """Save only embedding vectors to .npy in index order (0..N-1)."""
        if not self.all_embeddings:
            # Save an empty array
            np.save(self.emb_path, np.zeros((0, self.vector_size), dtype=np.float32))
            return
        emb_array = np.stack([emb for _, emb in self.all_embeddings]).astype(np.float32)
        np.save(self.emb_path, emb_array)

    def _load_all_embeddings(self):
        """Load embeddings from .npy and reconstruct self.all_embeddings as (id, emb)."""
        if os.path.exists(self.emb_path):
            emb_array = np.load(self.emb_path)
            # Reconstruct ids as 0..N-1
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
            # Set next_id consistently (make sure integer)
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
        """Rebuild the Annoy index from self.all_embeddings and save to disk."""
        self.index = AnnoyIndex(self.vector_size, "euclidean")
        for idx, emb in self.all_embeddings:
            # Annoy requires list-like floats
            self.index.add_item(int(idx), emb.tolist())
        if self.all_embeddings:
            self.index.build(self.n_trees)
            self.index.save(self.index_path)
        else:
            # Make sure index file doesn't contain stale data
            if os.path.exists(self.index_path):
                try:
                    os.remove(self.index_path)
                except Exception:
                    pass

    # ---------------------------
    # Public API
    # ---------------------------
    def register(self, name, embedding):
        """
        Register a new embedding under `name`.
        Rebuilds Annoy index and persists embeddings + meta.
        """
        # ensure embedding is numpy float32
        embedding = np.asarray(embedding, dtype=np.float32).flatten()
        if embedding.shape[0] != self.vector_size:
            raise ValueError(f"Embedding dimension mismatch: got {embedding.shape[0]}, expected {self.vector_size}")

        # load meta & embeddings (so we append to existing)
        self._load_meta()
        self._load_all_embeddings()

        # append new embedding
        assigned_id = self.next_id
        self.all_embeddings.append((assigned_id, embedding))
        self.meta[str(assigned_id)] = name

        # save to disk
        self._save_all_embeddings()
        self._save_meta()

        # rebuild index and save
        self._rebuild_index()

        print(f"[AnnoyFaceMatcher] Registered '{name}' with id {assigned_id}")
        self.next_id += 1
        return assigned_id

    def add_face(self, name, embedding):
        """
        Convenience wrapper to register new face. Keeps behavior explicit.
        """
        return self.register(name, embedding)

    def load_for_matching(self):
        """
        Load the persisted index and meta for matching.
        Raises FileNotFoundError if no saved DB exists.
        """
        if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError("No index/meta found. Register at least one face first.")

        self.index = AnnoyIndex(self.vector_size, "euclidean")
        loaded = self.index.load(self.index_path)
        if not loaded:
            raise RuntimeError("Failed to load Annoy index from disk.")
        with open(self.meta_path, "r") as f:
            self.meta = json.load(f)
        # ensure next_id consistent
        self.next_id = max(map(int, self.meta.keys())) + 1 if self.meta else 0
        # Also load embeddings array into memory so register() can append correctly later
        self._load_all_embeddings()

    def match(self, embedding, top_k=10):
        """
        Returns list of (person_name, distance) sorted by distance ascending,
        for matches with distance <= threshold.
        If no index exists or no candidates under threshold, returns [].
        """
        embedding = np.asarray(embedding, dtype=np.float32).flatten()
        if embedding.shape[0] != self.vector_size:
            raise ValueError(f"Embedding dimension mismatch: got {embedding.shape[0]}, expected {self.vector_size}")

        # If index file doesn't exist or index has no items, nothing to match
        # Annoy index must be built or loaded before matching
        try:
            # If index is empty, get_nns_by_vector returns []
            idxs, dists = self.index.get_nns_by_vector(embedding.tolist(), top_k, include_distances=True)
        except Exception:
            # If index is not ready (not built or not loaded), return empty list
            return []

        # Collect best match per person (since multiple embeddings can belong to same name)
        best_matches = {}
        for idx, dist in zip(idxs, dists):
            if dist <= self.threshold:
                person = self.meta.get(str(idx), None)
                if person is None:
                    continue
                # pick smallest dist per person
                if person not in best_matches or dist < best_matches[person]:
                    best_matches[person] = dist

        if not best_matches:
            return []

        sorted_matches = sorted(best_matches.items(), key=lambda x: x[1])
        return sorted_matches

    # Optional: helper to list registered people
    def list_registered(self):
        return {int(k): v for k, v in self.meta.items()}

from annoy import AnnoyIndex
import numpy as np
import os
import json

class AnnoyFaceMatcher:
    def __init__(self, vector_size=512, index_path="faces.ann", meta_path="faces_meta.json", emb_path="face_embeddings.npy", threshold=1.2, n_trees=10):
        self.vector_size = vector_size
        self.index_path = index_path
        self.meta_path = meta_path
        self.emb_path = emb_path
        self.threshold = threshold
        self.n_trees = n_trees

        # meta: {int_embedding_id: person_name}
        self.meta = {}

        # Store embeddings in a list: [(embedding_id, embedding_vector)]
        self.all_embeddings = []
        self.next_id = 0
        self.index = AnnoyIndex(self.vector_size, 'euclidean')

    def _save_all_embeddings(self):
        np.save(self.emb_path, np.array([emb for _, emb in self.all_embeddings], dtype=np.float32))

    def _load_all_embeddings(self):
        if os.path.exists(self.emb_path):
            emb_array = np.load(self.emb_path)
            # We lose the IDs after saving, so we reconstruct IDs and meta from files
            self.all_embeddings = [(i, emb_array[i]) for i in range(len(emb_array))]
        else:
            self.all_embeddings = []

    def register(self, name, embedding):
        # Load previous meta and embeddings
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r") as f:
                self.meta = json.load(f)
            self.next_id = max(map(int, self.meta.keys())) + 1 if self.meta else 0
        else:
            self.meta = {}
            self.next_id = 0

        self._load_all_embeddings()

        # Add new embedding
        self.all_embeddings.append((self.next_id, embedding))
        self.meta[str(self.next_id)] = name

        # Save embeddings and meta
        self._save_all_embeddings()
        with open(self.meta_path, "w") as f:
            json.dump(self.meta, f)

        # Rebuild Annoy index
        self.index = AnnoyIndex(self.vector_size, 'euclidean')
        for idx, emb in self.all_embeddings:
            self.index.add_item(idx, emb)
        self.index.build(self.n_trees)
        self.index.save(self.index_path)

        print(f"Registered '{name}' with embedding ID {self.next_id}")
        self.next_id += 1

    def load_for_matching(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError("No index/meta found. Register at least one face first.")

        self.index = AnnoyIndex(self.vector_size, 'euclidean')
        self.index.load(self.index_path)

        with open(self.meta_path, "r") as f:
            self.meta = json.load(f)

    def match(self, embedding, top_k=10):
        idxs, dists = self.index.get_nns_by_vector(embedding, top_k, include_distances=True)

        # Collect matches per person: person -> best dist
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

        # Sort matches by distance ascending
        sorted_matches = sorted(best_matches.items(), key=lambda x: x[1])

        # Return top match or all below threshold
        return sorted_matches


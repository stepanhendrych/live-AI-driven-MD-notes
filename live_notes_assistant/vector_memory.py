import hashlib


class VectorMemory:
    """Persistent semantic vector store for notes, backed by ChromaDB.

    All methods are no-ops when ChromaDB is not installed or the database
    cannot be initialised – the application degrades gracefully.
    """

    def __init__(self, persist_dir, collection_name="notes"):
        self._available = False
        self._collection = None
        try:
            import chromadb

            self._client = chromadb.PersistentClient(path=str(persist_dir))
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._available = True
            print(f"[vector_memory] ready (dir={persist_dir}, collection={collection_name})")
        except ImportError:
            print("[vector_memory] chromadb not installed; vector memory disabled")
        except Exception as exc:
            print(f"[vector_memory] init failed: {exc}; vector memory disabled")

    @property
    def available(self):
        return self._available

    def add(self, text, doc_id=None):
        """Upsert *text* into the vector store.

        *doc_id* is derived from the content hash when not supplied, so
        re-inserting identical text is idempotent.
        """
        if not self._available or not text or not text.strip():
            return
        if doc_id is None:
            doc_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
        try:
            self._collection.upsert(documents=[text.strip()], ids=[doc_id])
        except Exception as exc:
            print(f"[vector_memory] add failed: {exc}")

    def query(self, text, n_results=3):
        """Return the *n_results* most similar stored documents.

        Returns a list of ``(document_text, distance)`` tuples where
        *distance* is the cosine distance in ``[0, 1]``:

        - ``0.0``  – identical
        - ``1.0``  – completely dissimilar (orthogonal)

        An empty list is returned when the store is unavailable or empty.
        """
        if not self._available or not text or not text.strip():
            return []
        try:
            count = self._collection.count()
            if count == 0:
                return []
            n = min(n_results, count)
            results = self._collection.query(
                query_texts=[text.strip()],
                n_results=n,
            )
            docs = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]
            return list(zip(docs, distances))
        except Exception as exc:
            print(f"[vector_memory] query failed: {exc}")
            return []

    def is_semantic_duplicate(self, text, threshold=0.05):
        """Return ``True`` when *text* is nearly identical to a stored note.

        *threshold* is the maximum cosine distance (lower = stricter).
        The default of ``0.05`` corresponds to roughly ≥95 % cosine similarity.
        """
        results = self.query(text, n_results=1)
        if not results:
            return False
        _doc, distance = results[0]
        return distance <= threshold

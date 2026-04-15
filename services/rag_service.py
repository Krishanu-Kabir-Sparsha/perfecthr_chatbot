# -*- coding: utf-8 -*-
"""
RAG (Retrieval-Augmented Generation) Service for Perfect HR Chatbot.

Uses FAISS for vector similarity search and Ollama for embeddings.
Falls back to sentence-transformers if Ollama embeddings are unavailable.
"""
import json
import logging
import os
import pickle
import threading
import base64

import numpy as np

_logger = logging.getLogger(__name__)

# Thread lock for FAISS index operations
_index_lock = threading.Lock()


class RAGService:
    """Manages FAISS vector index and RAG pipeline."""

    def __init__(self, ollama_url='http://localhost:11434',
                 embedding_model='nomic-embed-text',
                 data_dir=None):
        self.ollama_url = ollama_url.rstrip('/')
        self.embedding_model = embedding_model
        self.data_dir = data_dir or '/tmp/chatbot_faiss'
        self._index = None
        self._metadata = []  # Vector-indexed metadata (1:1 with FAISS vectors)
        self._keyword_metadata = []  # Full article corpus for keyword fallback
        self._dimension = None
        self._faiss = None
        self._st_model = None  # Sentence-transformers fallback

        # Try importing FAISS
        try:
            import faiss
            self._faiss = faiss
            _logger.info('FAISS loaded successfully')
        except ImportError:
            _logger.info(
                'faiss-cpu not installed. Using keyword fallback + Ollama generation only.'
            )

    # â”€â”€ Embedding â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def embed_text(self, text):
        """
        Generate embedding vector for text.
        Primary: Ollama /api/embeddings
        Fallback: sentence-transformers
        """
        # Try Ollama first
        embedding = self._embed_via_ollama(text)
        if embedding is not None:
            return np.array(embedding, dtype=np.float32)

        # Fallback to sentence-transformers
        embedding = self._embed_via_st(text)
        if embedding is not None:
            return embedding

        _logger.error('No embedding method available')
        return None

    def _embed_via_ollama(self, text):
        """Generate embedding using Ollama API."""
        try:
            import requests
            # Ollama current API: /api/embed with "input"
            resp = requests.post(
                f'{self.ollama_url}/api/embed',
                json={
                    'model': self.embedding_model,
                    'input': text,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get('embeddings') or []
            if embeddings and isinstance(embeddings, list):
                first = embeddings[0]
                if first:
                    return first
        except Exception as e:
            _logger.warning('Ollama /api/embed failed, trying legacy endpoint: %s', e)

        try:
            import requests
            # Legacy Ollama endpoint retained for backward compatibility
            resp = requests.post(
                f'{self.ollama_url}/api/embeddings',
                json={
                    'model': self.embedding_model,
                    'prompt': text,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            embedding = data.get('embedding')
            if embedding:
                return embedding
        except Exception as e:
            _logger.warning('Ollama embedding failed, trying fallback: %s', e)
        return None

    def set_keyword_corpus(self, articles):
        """Update full article corpus used by keyword fallback search."""
        self._keyword_metadata = [
            {
                'id': article['id'],
                'name': article['name'],
                'category': article.get('category', 'general'),
                'summary': article.get('content_summary', ''),
                'version': int(article.get('version', 1) or 1),
                'content': article.get('content', ''),
            }
            for article in (articles or [])
            if article.get('id')
        ]

    def _deserialize_embedding(self, raw_embedding):
        """Deserialize embedding payload from Odoo Binary field safely."""
        if not raw_embedding:
            return None

        candidates = []
        if isinstance(raw_embedding, str):
            candidates.append(raw_embedding.encode('utf-8'))
        elif isinstance(raw_embedding, (bytes, bytearray)):
            candidates.append(bytes(raw_embedding))
        else:
            return None

        # Try both direct pickle bytes and base64-encoded pickle.
        expanded = []
        for item in candidates:
            expanded.append(item)
            try:
                expanded.append(base64.b64decode(item, validate=False))
            except Exception:
                pass

        for payload in expanded:
            try:
                embedding = pickle.loads(payload)
                if embedding is None:
                    continue
                return np.array(embedding, dtype=np.float32)
            except Exception:
                continue

        return None

    def _embed_via_st(self, text):
        """Fallback: Generate embedding using sentence-transformers."""
        try:
            if self._st_model is None:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer('all-MiniLM-L6-v2')
                _logger.info('sentence-transformers loaded as fallback')
            embedding = self._st_model.encode(text)
            return np.array(embedding, dtype=np.float32)
        except ImportError:
            _logger.warning('sentence-transformers not installed')
        except Exception as e:
            _logger.error('sentence-transformers embedding failed: %s', e)
        return None

    # â”€â”€ Index Management â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def build_index(self, articles):
        """
        Build FAISS index from knowledge base articles.

        Args:
            articles: List of dicts with keys:
                      'id', 'name', 'content', 'content_summary', 'embedding_data'
        """
        if not self._faiss:
            _logger.info('FAISS not available — using keyword fallback for retrieval')
            return False

        embeddings = []
        vector_metadata = []

        # Keep full corpus for keyword fallback, including not-yet-embedded articles.
        self.set_keyword_corpus(articles)

        for article in articles:
            embedding = None

            # Use ONLY pre-computed embeddings to avoid blocking user requests.
            # Articles without embeddings should be embedded via cron or manual action.
            if article.get('embedding_data'):
                embedding = self._deserialize_embedding(article['embedding_data'])
                if embedding is None:
                    _logger.warning(
                        'Failed to deserialize embedding for article %s (id=%s)',
                        article.get('name', '?'), article.get('id'),
                    )

            if embedding is None:
                _logger.debug(
                    'Skipping article "%s" (id=%s) â€” no pre-computed embedding. '
                    'Run the auto-embed cron or embed manually.',
                    article.get('name', '?'), article.get('id'),
                )
                continue

            if embedding is not None:
                embeddings.append(embedding)
                vector_metadata.append({
                    'id': article['id'],
                    'name': article['name'],
                    'category': article.get('category', 'general'),
                    'summary': article.get('content_summary', ''),
                    'version': int(article.get('version', 1) or 1),
                    'content': article['content'],  # Preserve full articlesections for complete answers
                })

        if not embeddings:
            with _index_lock:
                self._index = None
                self._metadata = []
            _logger.warning('No embeddings available for vector index; using keyword fallback only')
            return False

        # Build FAISS index
        vectors = np.array(embeddings, dtype=np.float32)
        self._dimension = vectors.shape[1]

        with _index_lock:
            self._index = self._faiss.IndexFlatL2(self._dimension)
            self._index.add(vectors)
            self._metadata = vector_metadata

        _logger.info(
            'FAISS index built: %d vectors, dimension %d',
            len(embeddings), self._dimension,
        )

        # Persist index
        self._save_index()
        return True

    def search(self, query, top_k=3):
        """
        Search the FAISS index for relevant knowledge articles.

        Args:
            query: Search query text.
            top_k: Number of results to return.

        Returns:
            List of dicts: [{'id', 'name', 'summary', 'content', 'score'}]
        """
        if not self._faiss:
            return self._keyword_fallback(query, top_k)

        # Load index if not in memory
        if self._index is None:
            self._load_index()

        if self._index is None or self._index.ntotal == 0:
            _logger.warning('FAISS index is empty or not built')
            return []

        # Embed query
        query_embedding = self.embed_text(query)
        if query_embedding is None:
            return self._keyword_fallback(query, top_k)

        query_vector = np.array([query_embedding], dtype=np.float32)

        with _index_lock:
            distances, indices = self._index.search(query_vector, min(top_k, self._index.ntotal))

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = self._metadata[idx]
            results.append({
                'id': meta['id'],
                'name': meta['name'],
                'category': meta.get('category', 'general'),
                'summary': meta['summary'],
                'content': meta['content'],
                'score': float(1.0 / (1.0 + distances[0][i])),  # Convert L2 to similarity
            })

        return results

    def _keyword_fallback(self, query, top_k=3):
        """Simple keyword matching when FAISS is unavailable."""
        source = self._keyword_metadata or self._metadata
        if not source:
            return []

        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored = []
        for meta in source:
            content_lower = meta['content'].lower()
            name_lower = meta['name'].lower()
            # Calculate relevance by word overlap
            matches = sum(
                1 for word in query_words
                if word in content_lower or word in name_lower
            )
            if matches > 0:
                scored.append((matches, meta))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                'id': meta['id'],
                'name': meta['name'],
                'category': meta.get('category', 'general'),
                'summary': meta['summary'],
                'content': meta['content'],
                'score': float(score / max(len(query_words), 1)),
            }
            for score, meta in scored[:top_k]
        ]

    def is_index_synced(self, articles):
        """Check whether current in-memory metadata matches article ids/versions."""
        if not self._metadata:
            return False
        try:
            current = {
                int(m.get('id')): int(m.get('version', 1) or 1)
                for m in self._metadata
                if m.get('id')
            }
            incoming = {
                int(a.get('id')): int(a.get('version', 1) or 1)
                for a in (articles or [])
                if a.get('id')
            }
            return bool(current) and current == incoming
        except Exception:
            return False

    # â”€â”€ Persistence â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        if not self._faiss or not self._index:
            return

        os.makedirs(self.data_dir, exist_ok=True)
        index_path = os.path.join(self.data_dir, 'chatbot.index')
        meta_path = os.path.join(self.data_dir, 'chatbot_meta.pkl')

        try:
            with _index_lock:
                self._faiss.write_index(self._index, index_path)
                with open(meta_path, 'wb') as f:
                    pickle.dump(self._metadata, f)
            _logger.info('FAISS index saved to %s', self.data_dir)
        except Exception as e:
            _logger.error('Failed to save FAISS index: %s', e)

    def _load_index(self):
        """Load FAISS index and metadata from disk."""
        if not self._faiss:
            return

        index_path = os.path.join(self.data_dir, 'chatbot.index')
        meta_path = os.path.join(self.data_dir, 'chatbot_meta.pkl')

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            return

        try:
            with _index_lock:
                self._index = self._faiss.read_index(index_path)
                with open(meta_path, 'rb') as f:
                    self._metadata = pickle.load(f)
            self._dimension = self._index.d
            _logger.info(
                'FAISS index loaded: %d vectors', self._index.ntotal,
            )
        except Exception as e:
            _logger.error('Failed to load FAISS index: %s', e)


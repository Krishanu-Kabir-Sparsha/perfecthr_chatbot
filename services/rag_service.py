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
        self._metadata = []  # List of (article_id, title, content_summary)
        self._dimension = None
        self._faiss = None
        self._st_model = None  # Sentence-transformers fallback

        # Try importing FAISS
        try:
            import faiss
            self._faiss = faiss
            _logger.info('FAISS loaded successfully')
        except ImportError:
            _logger.warning(
                'faiss-cpu not installed. Vector search will be unavailable. '
                'Install with: pip install faiss-cpu'
            )

    # ── Embedding ───────────────────────────────────────────────────
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

    # ── Index Management ────────────────────────────────────────────
    def build_index(self, articles):
        """
        Build FAISS index from knowledge base articles.

        Args:
            articles: List of dicts with keys:
                      'id', 'name', 'content', 'content_summary', 'embedding_data'
        """
        if not self._faiss:
            _logger.error('Cannot build index: FAISS not available')
            return False

        embeddings = []
        metadata = []

        for article in articles:
            embedding = None

            # Try pre-computed embedding first
            if article.get('embedding_data'):
                try:
                    embedding = pickle.loads(article['embedding_data'])
                except Exception:
                    pass

            # Generate embedding if not available
            if embedding is None:
                embedding = self.embed_text(article['content'])

            if embedding is not None:
                embeddings.append(embedding)
                metadata.append({
                    'id': article['id'],
                    'name': article['name'],
                    'category': article.get('category', 'general'),
                    'summary': article.get('content_summary', ''),
                    'content': article['content'][:1800],  # Balanced context depth vs model latency
                })

        if not embeddings:
            _logger.warning('No embeddings generated for index')
            return False

        # Build FAISS index
        vectors = np.array(embeddings, dtype=np.float32)
        self._dimension = vectors.shape[1]

        with _index_lock:
            self._index = self._faiss.IndexFlatL2(self._dimension)
            self._index.add(vectors)
            self._metadata = metadata

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
        if not self._metadata:
            return []

        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored = []
        for meta in self._metadata:
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

    # ── Persistence ─────────────────────────────────────────────────
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

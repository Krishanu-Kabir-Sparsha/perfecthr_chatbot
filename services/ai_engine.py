# -*- coding: utf-8 -*-
"""
Ollama AI Engine — Exclusive LLM integration for Perfect HR Chatbot.

Communicates with a local Ollama instance to generate AI responses
using Mistral 7B, LLaMA 3, or Phi-3 models.
"""
import json
import logging
import time

import requests

_logger = logging.getLogger(__name__)

# Default fallback message when Ollama is unavailable
FALLBACK_MESSAGE = (
    "I apologize, but I'm temporarily unable to process your request. "
    "Our AI system is currently being updated. Please try again in a moment, "
    "or feel free to contact our team directly at support@perfecthr.com."
)


class OllamaEngine:
    """Dedicated Ollama LLM integration for Perfect HR Chatbot."""

    def __init__(self, ollama_url='http://localhost:11434', model='mistral',
                 max_tokens=512, temperature=0.7, system_prompt=''):
        self.ollama_url = ollama_url.rstrip('/')
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self._generation_timeout = 60
        self._embedding_timeout = 30
        self._retry_count = 3
        self._retry_delay = 2

    # ── Health Check ────────────────────────────────────────────────
    def is_available(self):
        """Check if Ollama server is reachable."""
        try:
            resp = requests.get(
                f'{self.ollama_url}/api/tags',
                timeout=5,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self):
        """List available models on the Ollama server."""
        try:
            resp = requests.get(
                f'{self.ollama_url}/api/tags',
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return [m.get('name', '') for m in data.get('models', [])]
        except Exception as e:
            _logger.error('Failed to list Ollama models: %s', e)
            return []

    def get_model_info(self):
        """Get info about the current model."""
        try:
            resp = requests.post(
                f'{self.ollama_url}/api/show',
                json={'name': self.model},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            _logger.error('Failed to get model info: %s', e)
            return {}

    # ── Chat Completion ─────────────────────────────────────────────
    def chat(self, messages, context_text=''):
        """
        Multi-turn conversation using Ollama /api/chat endpoint.

        Args:
            messages: List of dicts with 'role' and 'content' keys.
                      Roles: 'user', 'assistant', 'system'
            context_text: Additional RAG context to include in system prompt.

        Returns:
            dict: {'response': str, 'model': str, 'duration_ms': int, 'success': bool}
        """
        # Build system message with RAG context
        system_content = self.system_prompt or ''
        if context_text:
            system_content += (
                '\n\n--- KNOWLEDGE BASE CONTEXT ---\n'
                'Use the following information to answer the user\'s question. '
                'If the answer is not in the context, say you don\'t have specific '
                'information but offer general guidance.\n\n'
                f'{context_text}\n'
                '--- END CONTEXT ---'
            )

        # Prepare message list
        ollama_messages = []
        if system_content:
            ollama_messages.append({
                'role': 'system',
                'content': system_content,
            })

        for msg in messages:
            ollama_messages.append({
                'role': msg.get('role', 'user'),
                'content': msg.get('content', ''),
            })

        payload = {
            'model': self.model,
            'messages': ollama_messages,
            'stream': False,
            'options': {
                'num_predict': self.max_tokens,
                'temperature': self.temperature,
            },
        }

        return self._send_request('/api/chat', payload)

    # ── Single-Shot Generation ──────────────────────────────────────
    def generate(self, prompt, context_text=''):
        """
        Single-shot generation using Ollama /api/generate endpoint.
        Used as fallback if /api/chat is not available.

        Args:
            prompt: The user's prompt text.
            context_text: Additional RAG context.

        Returns:
            dict: {'response': str, 'model': str, 'duration_ms': int, 'success': bool}
        """
        full_prompt = ''
        if self.system_prompt:
            full_prompt += f"System: {self.system_prompt}\n\n"
        if context_text:
            full_prompt += f"Context:\n{context_text}\n\n"
        full_prompt += f"User: {prompt}\n\nAssistant:"

        payload = {
            'model': self.model,
            'prompt': full_prompt,
            'stream': False,
            'options': {
                'num_predict': self.max_tokens,
                'temperature': self.temperature,
            },
        }

        return self._send_request('/api/generate', payload)

    # ── Embeddings ──────────────────────────────────────────────────
    def embed(self, text, model=None):
        """
        Generate embeddings using Ollama /api/embeddings endpoint.

        Args:
            text: Text to embed.
            model: Embedding model name (defaults to nomic-embed-text).

        Returns:
            list: Embedding vector, or None on failure.
        """
        embed_model = model or 'nomic-embed-text'
        try:
            resp = requests.post(
                f'{self.ollama_url}/api/embeddings',
                json={
                    'model': embed_model,
                    'prompt': text,
                },
                timeout=self._embedding_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get('embedding')
        except Exception as e:
            _logger.error('Ollama embedding failed: %s', e)
            return None

    # ── Internal ────────────────────────────────────────────────────
    def _send_request(self, endpoint, payload):
        """Send request to Ollama with retry logic."""
        start_time = time.time()

        for attempt in range(self._retry_count):
            try:
                resp = requests.post(
                    f'{self.ollama_url}{endpoint}',
                    json=payload,
                    timeout=self._generation_timeout,
                )
                resp.raise_for_status()
                data = resp.json()

                elapsed_ms = int((time.time() - start_time) * 1000)

                # /api/chat returns 'message.content', /api/generate returns 'response'
                response_text = ''
                if 'message' in data:
                    response_text = data['message'].get('content', '')
                elif 'response' in data:
                    response_text = data['response']

                return {
                    'response': response_text.strip(),
                    'model': data.get('model', self.model),
                    'duration_ms': elapsed_ms,
                    'success': True,
                }

            except requests.ConnectionError:
                _logger.warning(
                    'Ollama connection attempt %d/%d failed',
                    attempt + 1, self._retry_count,
                )
                if attempt < self._retry_count - 1:
                    time.sleep(self._retry_delay)
            except requests.Timeout:
                _logger.error('Ollama request timed out after %ds', self._generation_timeout)
                break
            except requests.HTTPError as e:
                _logger.error('Ollama HTTP error: %s', e)
                # Check if model not found
                if e.response and e.response.status_code == 404:
                    return {
                        'response': (
                            f"The AI model '{self.model}' is not installed. "
                            "Please ask an administrator to pull the model in Settings."
                        ),
                        'model': self.model,
                        'duration_ms': int((time.time() - start_time) * 1000),
                        'success': False,
                    }
                break
            except Exception as e:
                _logger.error('Ollama unexpected error: %s', e)
                break

        # All retries exhausted
        elapsed_ms = int((time.time() - start_time) * 1000)
        return {
            'response': FALLBACK_MESSAGE,
            'model': self.model,
            'duration_ms': elapsed_ms,
            'success': False,
        }

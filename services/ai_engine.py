# -*- coding: utf-8 -*-
"""
Ollama AI Engine â€” Exclusive LLM integration for Perfect HR Chatbot.

Communicates with a local Ollama instance to generate AI responses
using Mistral 7B, LLaMA 3, or Phi-3 models.
"""
import json
import logging
import time
import threading

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
                 max_tokens=2048, temperature=0.7, system_prompt=''):
        self.ollama_url = ollama_url.rstrip('/')
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self._generation_timeout = 30
        self._embedding_timeout = 60
        self._retry_count = 1
        self._retry_delay = 1
        self._resolved_model = None
        self._last_model_refresh = 0
        self._model_refresh_ttl = 120
        self._model_lock = threading.Lock()
        self._session = requests.Session()

    def _resolve_model_name(self, force=False):
        """Resolve configured model to an installed Ollama tag (e.g. mistral -> mistral:latest)."""
        now = time.time()
        if not force and self._resolved_model and (now - self._last_model_refresh) < self._model_refresh_ttl:
            return self._resolved_model

        with self._model_lock:
            now = time.time()
            if not force and self._resolved_model and (now - self._last_model_refresh) < self._model_refresh_ttl:
                return self._resolved_model
            try:
                resp = self._session.get(f'{self.ollama_url}/api/tags', timeout=8)
                resp.raise_for_status()
                names = [m.get('name', '') for m in resp.json().get('models', [])]
                exact = next((n for n in names if n == self.model), None)
                prefixed = next((n for n in names if n.startswith(f'{self.model}:')), None)
                self._resolved_model = exact or prefixed or self.model
                self._last_model_refresh = now
            except Exception as e:
                _logger.warning('Model resolution failed, using configured model %s: %s', self.model, e)
                self._resolved_model = self.model
                self._last_model_refresh = now
            return self._resolved_model

    # â”€â”€ Health Check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
            resp = self._session.get(
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
            model = self._resolve_model_name()
            resp = self._session.post(
                f'{self.ollama_url}/api/show',
                json={'name': model},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            _logger.error('Failed to get model info: %s', e)
            return {}

    # â”€â”€ Chat Completion â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def chat(self, messages, context_text='', max_tokens=None, temperature=None,
             language_hint=None, response_guidance='', force_synthesis=False):
        """
        Multi-turn conversation using Ollama /api/chat endpoint.

        Args:
            messages: List of dicts with 'role' and 'content' keys.
                      Roles: 'user', 'assistant', 'system'
            context_text: Additional RAG context to include in system prompt.
            language_hint: Language code or name for the response language.
            response_guidance: Additional instructions for response formatting.
            force_synthesis: If True, enforce synthesis requirement in system prompt.

        Returns:
            dict: {'response': str, 'model': str, 'duration_ms': int, 'success': bool}
        """
        normalized_hint = (language_hint or '').strip().lower()
        is_non_english = normalized_hint and normalized_hint not in ('auto', 'en', 'english')

        # Map ISO codes to full language names â€” Mistral 7B understands
        # "Bengali" much better than "bn" in prompts.
        _lang_names = {
            'bn': 'Bengali (à¦¬à¦¾à¦‚à¦²à¦¾)', 'hi': 'Hindi (à¤¹à¤¿à¤¨à¥à¤¦à¥€)', 'ar': 'Arabic (Ø§Ù„Ø¹Ø±Ø¨ÙŠØ©)',
            'es': 'Spanish (EspaÃ±ol)', 'fr': 'French (FranÃ§ais)', 'de': 'German (Deutsch)',
            'pt': 'Portuguese (PortuguÃªs)', 'ru': 'Russian (Ð ÑƒÑÑÐºÐ¸Ð¹)', 'zh': 'Chinese (ä¸­æ–‡)',
            'ja': 'Japanese (æ—¥æœ¬èªž)', 'ko': 'Korean (í•œêµ­ì–´)', 'tr': 'Turkish (TÃ¼rkÃ§e)',
            'ur': 'Urdu (Ø§Ø±Ø¯Ùˆ)', 'it': 'Italian (Italiano)', 'nl': 'Dutch (Nederlands)',
            'pl': 'Polish (Polski)', 'th': 'Thai (à¹„à¸—à¸¢)', 'vi': 'Vietnamese (Tiáº¿ng Viá»‡t)',
            'id': 'Indonesian (Bahasa Indonesia)', 'ms': 'Malay (Bahasa Melayu)',
            'fa': 'Persian (ÙØ§Ø±Ø³ÛŒ)', 'sw': 'Swahili',
        }
        lang_display = _lang_names.get(normalized_hint, language_hint or 'English')

        # â”€â”€ Build a CONCISE system prompt â”€â”€
        # For 7B models, shorter = better instruction following.
        # Language directive goes FIRST in system prompt for maximum compliance.
        sys_parts = []

        # 0. SYNTHESIS ENFORCER â€” if force_synthesis, make it top priority
        if force_synthesis:
            sys_parts.append(
                'SYNTHESIS REQUIREMENT: You MUST read the reference articles below, '
                'understand their content, and compose an ORIGINAL synthesized answer in your own words. '
                'NEVER copy-paste or directly repeat article text. '
                'Your job is to explain and rewrite the information clearly and naturally. '
                'Use ONLY facts from the articles â€” do not add external knowledge.'
            )

        # 1. LANGUAGE DIRECTIVE â€” top position (highest priority for small models)
        if is_non_english:
            sys_parts.append(
                f'LANGUAGE IMPERATIVE: Your ENTIRE response MUST be in {lang_display} ONLY. '
                f'Every single word must be {lang_display}. No English. No mixing. '
                f'All headings, bullets, examples, everything: {lang_display}. '
                f'This is NON-NEGOTIABLE. If you see [RESPOND IN {lang_display}], comply exactly.'
            )
        else:
            sys_parts.append(
                'LANGUAGE RULE: Your response MUST match the user input language exactly. '
                'If input is Bengali, respond ONLY in Bengali. '
                'If input is Arabic, respond ONLY in Arabic. '
                'If input is Spanish, respond ONLY in Spanish. '
                'Detect and mirror the user language with 100% adherence. CRITICAL.'
            )

        # 2. ROLE & BEHAVIOR â€” the configured system prompt
        if self.system_prompt:
            sys_parts.append(self.system_prompt)
        else:
            sys_parts.append(
                'You are Perfect HR AI Assistant â€” a knowledgeable, professional digital '
                'sales consultant. Answer questions about Perfect HR modules, pricing, '
                'and features. Be concise, professional, and helpful.'
            )

        # 3. SYNTHESIS â€” prevent copy-pasting (short form)
        if response_guidance:
            sys_parts.append(response_guidance)
        else:
            sys_parts.append(
                'When reference articles are provided, synthesize an original answer '
                'in your own words. Do NOT copy-paste article text verbatim.'
            )

        system_content = '\n\n'.join(sys_parts)

        # â”€â”€ Build message list â”€â”€
        ollama_messages = [{'role': 'system', 'content': system_content}]

        # Add RAG context as a separate user-role reference message.
        # This keeps the system prompt short and gives the model context
        # in the "conversation" where it pays more attention.
        if context_text:
            references_header = (
                '[REFERENCE ARTICLES â€” use these as your ONLY source truth]\n\n'
                'You must:\n'
                '1. Read these carefully\n'
                '2. Synthesize a NEW answer in your own words (do NOT copy-paste)\n'
                '3. Use ONLY facts from these articles\n'
                '4. Respond ONLY in the user\'s language\n\n'
                'ARTICLES:\n\n'
            )
            references_footer = (
                '\n\n[END REFERENCE]\n\n'
                '[RESPOND IN ' + lang_display + '. SYNTHESIZE NATURALLY. NO COPY-PASTE.]'
            )
            ollama_messages.append({
                'role': 'user',
                'content': references_header + context_text + references_footer,
            })
            ollama_messages.append({
                'role': 'assistant',
                'content': f'I have read and understood the reference articles. I will synthesize a clear answer in {lang_display} using only the provided information.',
            })

        # Add conversation history (filter out any stray system messages)
        for msg in messages:
            role = msg.get('role', 'user')
            if role == 'system':
                continue
            ollama_messages.append({
                'role': role,
                'content': msg.get('content', ''),
            })

        # For non-English: INJECT language + synthesis requirement into the LAST user message.
        # This is critical â€” adding a separate user message creates two consecutive
        # user turns which confuses Mistral 7B. Instead, we append the instruction
        # directly to the user's own message so the model sees ONE coherent request.
        if is_non_english:
            for i in range(len(ollama_messages) - 1, -1, -1):
                if ollama_messages[i]['role'] == 'user' and not ollama_messages[i]['content'].startswith('[REFERENCE'):
                    ollama_messages[i]['content'] += (
                        f'\n\n'
                        f'*** CRITICAL INSTRUCTIONS ***\n'
                        f'1. RESPOND ENTIRELY IN {lang_display} â€” every word, no English\n'
                        f'2. Synthesize knowledge into natural explanation â€” never copy-paste\n'
                        f'3. Answer directly and clearly in {lang_display}\n'
                        f'*** END INSTRUCTIONS ***'
                    )
                    break

        model_name = self._resolve_model_name()
        payload = {
            'model': model_name,
            'messages': ollama_messages,
            'stream': False,
            'keep_alive': '10m',
            'options': {
                'num_predict': max_tokens if max_tokens is not None else self.max_tokens,
                'temperature': temperature if temperature is not None else self.temperature,
            },
        }

        return self._send_request('/api/chat', payload)

    # â”€â”€ Single-Shot Generation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def generate(self, prompt, context_text='', language_hint=None):
        """
        Single-shot generation using Ollama /api/generate endpoint.
        Used as fallback if /api/chat is not available, or for
        translation / rewriting tasks.

        Args:
            prompt: The user's prompt text.
            context_text: Additional RAG context.
            language_hint: Language code or name for the response.

        Returns:
            dict: {'response': str, 'model': str, 'duration_ms': int, 'success': bool}
        """
        parts = []

        # Language directive first â€” strongest position
        normalized_hint = (language_hint or '').strip().lower()
        if normalized_hint and normalized_hint not in ('auto', 'en', 'english'):
            parts.append(
                f"IMPORTANT: You MUST respond entirely in {language_hint}. "
                f"Do NOT respond in English."
            )
        else:
            parts.append(
                "IMPORTANT: Detect the language of the user's prompt and respond ENTIRELY in that same language."
            )

        if self.system_prompt:
            parts.append(f"System: {self.system_prompt}")
        if context_text:
            parts.append(f"Context:\n{context_text}")

        parts.append(f"User: {prompt}")

        # Reinforce language at the end
        if language_hint:
            normalized = (language_hint or '').strip().lower()
            if normalized and normalized not in ('en', 'english'):
                parts.append(f"REMINDER: Respond in {language_hint}.")

        parts.append("Assistant:")
        full_prompt = '\n\n'.join(parts)

        model_name = self._resolve_model_name()
        payload = {
            'model': model_name,
            'prompt': full_prompt,
            'stream': False,
            'keep_alive': '10m',
            'options': {
                'num_predict': self.max_tokens,
                'temperature': self.temperature,
            },
        }

        return self._send_request('/api/generate', payload)

    # â”€â”€ Embeddings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
        # Prefer current Ollama embedding endpoint.
        try:
            resp = self._session.post(
                f'{self.ollama_url}/api/embed',
                json={
                    'model': embed_model,
                    'input': text,
                },
                timeout=self._embedding_timeout,
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

        # Legacy fallback endpoint.
        try:
            resp = self._session.post(
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

    # â”€â”€ Internal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _send_request(self, endpoint, payload):
        """Send request to Ollama with retry logic and detailed logging."""
        start_time = time.time()
        
        _logger.info(
            'Ollama Request: endpoint=%s, model=%s, timeout=%ds',
            endpoint, payload.get('model', '?'), self._generation_timeout
        )

        for attempt in range(self._retry_count):
            try:
                url = f'{self.ollama_url}{endpoint}'
                _logger.debug('Attempt %d/%d: POST %s', attempt + 1, self._retry_count, url)
                
                resp = self._session.post(
                    url,
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

                _logger.info(
                    'Ollama Success: model=%s, response_len=%d, elapsed=%dms',
                    data.get('model', self.model), len(response_text), elapsed_ms
                )

                return {
                    'response': response_text.strip(),
                    'model': data.get('model', self.model),
                    'duration_ms': elapsed_ms,
                    'success': True,
                    'error_type': None,
                }

            except requests.ConnectionError as e:
                _logger.warning(
                    'Ollama connection attempt %d/%d failed: %s',
                    attempt + 1, self._retry_count, str(e)
                )
                if attempt < self._retry_count - 1:
                    time.sleep(self._retry_delay)
            except requests.Timeout as e:
                elapsed_ms = int((time.time() - start_time) * 1000)
                _logger.error(
                    'Ollama request timed out after %ds (elapsed=%dms): %s',
                    self._generation_timeout, elapsed_ms, str(e)
                )
                return {
                    'response': FALLBACK_MESSAGE,
                    'model': self.model,
                    'duration_ms': elapsed_ms,
                    'success': False,
                    'error_type': 'timeout',
                }
            except requests.HTTPError as e:
                elapsed_ms = int((time.time() - start_time) * 1000)
                _logger.error(
                    'Ollama HTTP error %d: %s',
                    e.response.status_code if e.response else 0, str(e)
                )
                # Check if model not found
                if e.response and e.response.status_code == 404:
                    self._resolve_model_name(force=True)
                    return {
                        'response': (
                            f"The AI model '{self._resolved_model or self.model}' is not installed. "
                            "Please ask an administrator to pull the model in Settings."
                        ),
                        'model': self._resolved_model or self.model,
                        'duration_ms': elapsed_ms,
                        'success': False,
                        'error_type': 'model_not_found',
                    }
                return {
                    'response': FALLBACK_MESSAGE,
                    'model': self.model,
                    'duration_ms': int((time.time() - start_time) * 1000),
                    'success': False,
                    'error_type': 'http',
                }
            except Exception as e:
                _logger.error('Ollama unexpected error: %s', e)
                return {
                    'response': FALLBACK_MESSAGE,
                    'model': self.model,
                    'duration_ms': int((time.time() - start_time) * 1000),
                    'success': False,
                    'error_type': 'unknown',
                }

        # All retries exhausted
        elapsed_ms = int((time.time() - start_time) * 1000)
        return {
            'response': FALLBACK_MESSAGE,
            'model': self.model,
            'duration_ms': elapsed_ms,
            'success': False,
            'error_type': 'connection',
        }


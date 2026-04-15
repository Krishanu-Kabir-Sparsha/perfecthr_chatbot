# -*- coding: utf-8 -*-
"""
Ollama AI Engine — Exclusive LLM integration for Perfect HR Chatbot.

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
        self._connect_timeout = 10
        self._read_timeout = 500
        self._embedding_timeout = 60
        self._retry_count = 2
        self._retry_delay = 3
        self._resolved_model = None
        self._last_model_refresh = 0
        self._model_refresh_ttl = 120
        self._model_lock = threading.Lock()

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
                resp = requests.get(f'{self.ollama_url}/api/tags', timeout=8)
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

    # ── Warm-up / Preload ───────────────────────────────────────────
    def warm_up(self):
        """Pre-load the model into Ollama memory.

        Uses a streaming request with num_predict=1 so:
        - Model gets loaded into memory
        - Response is immediate (1 token)
        - Connection closes quickly, not blocking other requests.
        """
        model_name = self._resolve_model_name()
        try:
            _logger.info('Warming up Ollama model %s ...', model_name)
            resp = requests.post(
                f'{self.ollama_url}/api/generate',
                json={
                    'model': model_name,
                    'prompt': 'Hi',
                    'stream': True,
                    'keep_alive': '10m',
                    'options': {'num_predict': 1},
                },
                timeout=(self._connect_timeout, self._read_timeout),
                stream=True,
            )
            # Consume the stream to properly close the connection
            for line in resp.iter_lines():
                pass
            resp.close()
            _logger.info('Ollama model %s is warm and ready.', model_name)
        except Exception as e:
            _logger.warning('Ollama warm-up failed (non-fatal): %s', e)

    # ── Health Check ────────────────────────────────────────────────
    def is_available(self):
        """Check if Ollama server is reachable."""
        try:
            resp = requests.get(
                f'{self.ollama_url}/api/tags',
                timeout=(3, 5),
            )
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self):
        """List available models on the Ollama server."""
        try:
            resp = requests.get(
                f'{self.ollama_url}/api/tags',
                timeout=(5, 10),
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
            resp = requests.post(
                f'{self.ollama_url}/api/show',
                json={'name': model},
                timeout=(5, 10),
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            _logger.error('Failed to get model info: %s', e)
            return {}

    # ── Chat Completion ─────────────────────────────────────────────
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

        # Map ISO codes to full language names — Mistral 7B understands
        # "Bengali" much better than "bn" in prompts.
        _lang_names = {
            'bn': 'Bengali (বাংলা)', 'hi': 'Hindi (हिन्दी)', 'ar': 'Arabic (العربية)',
            'es': 'Spanish (Español)', 'fr': 'French (Français)', 'de': 'German (Deutsch)',
            'pt': 'Portuguese (Português)', 'ru': 'Russian (Русский)', 'zh': 'Chinese (中文)',
            'ja': 'Japanese (日本語)', 'ko': 'Korean (한국어)', 'tr': 'Turkish (Türkçe)',
            'ur': 'Urdu (اردو)', 'it': 'Italian (Italiano)', 'nl': 'Dutch (Nederlands)',
            'pl': 'Polish (Polski)', 'th': 'Thai (ไทย)', 'vi': 'Vietnamese (Tiếng Việt)',
            'id': 'Indonesian (Bahasa Indonesia)', 'ms': 'Malay (Bahasa Melayu)',
            'fa': 'Persian (فارسی)', 'sw': 'Swahili',
        }
        lang_display = _lang_names.get(normalized_hint, language_hint or 'English')

        # ── Build a CONCISE system prompt ──
        # For 7B models, shorter = better instruction following.
        # Language directive goes FIRST in system prompt for maximum compliance.
        sys_parts = []

        # 0. SYNTHESIS ENFORCER — if force_synthesis, make it top priority
        if force_synthesis:
            sys_parts.append(
                'SYNTHESIS REQUIREMENT: You MUST read the reference articles below, '
                'understand their content, and compose an ORIGINAL synthesized answer in your own words. '
                'NEVER copy-paste or directly repeat article text. '
                'Your job is to explain and rewrite the information clearly and naturally. '
                'Use ONLY facts from the articles — do not add external knowledge.'
            )

        # 1. LANGUAGE DIRECTIVE — top position (highest priority for small models)
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

        # 2. ROLE & BEHAVIOR — the configured system prompt
        if self.system_prompt:
            sys_parts.append(self.system_prompt)
        else:
            sys_parts.append(
                'You are Perfect HR AI Assistant — a knowledgeable, professional digital '
                'sales consultant. Answer questions about Perfect HR modules, pricing, '
                'and features. Be concise, professional, and helpful.'
            )

        # 3. SYNTHESIS — prevent copy-pasting (short form)
        if response_guidance:
            sys_parts.append(response_guidance)
        else:
            sys_parts.append(
                'When reference articles are provided, synthesize an original answer '
                'in your own words. Do NOT copy-paste article text verbatim.'
            )

        system_content = '\n\n'.join(sys_parts)

        # ── Build message list ──
        ollama_messages = [{'role': 'system', 'content': system_content}]

        # Add RAG context as a separate user-role reference message.
        # This keeps the system prompt short and gives the model context
        # in the "conversation" where it pays more attention.
        if context_text:
            references_header = (
                '[REFERENCE ARTICLES — use these as your ONLY source truth]\n\n'
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
        # This is critical — adding a separate user message creates two consecutive
        # user turns which confuses Mistral 7B. Instead, we append the instruction
        # directly to the user's own message so the model sees ONE coherent request.
        if is_non_english:
            for i in range(len(ollama_messages) - 1, -1, -1):
                if ollama_messages[i]['role'] == 'user' and not ollama_messages[i]['content'].startswith('[REFERENCE'):
                    ollama_messages[i]['content'] += (
                        f'\n\n'
                        f'*** CRITICAL INSTRUCTIONS ***\n'
                        f'1. RESPOND ENTIRELY IN {lang_display} — every word, no English\n'
                        f'2. Synthesize knowledge into natural explanation — never copy-paste\n'
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

    # ── Single-Shot Generation ──────────────────────────────────────
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

        # Language directive first — strongest position
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
        # Prefer current Ollama embedding endpoint.
        try:
            resp = requests.post(
                f'{self.ollama_url}/api/embed',
                json={
                    'model': embed_model,
                    'input': text,
                },
                timeout=(self._connect_timeout, self._embedding_timeout),
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
            resp = requests.post(
                f'{self.ollama_url}/api/embeddings',
                json={
                    'model': embed_model,
                    'prompt': text,
                },
                timeout=(self._connect_timeout, self._embedding_timeout),
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get('embedding')
        except Exception as e:
            _logger.error('Ollama embedding failed: %s', e)
            return None

    # ── Internal ────────────────────────────────────────────────────
    def _send_request(self, endpoint, payload):
        """Send request to Ollama using STREAMING mode.

        Streaming keeps the TCP connection alive with incremental data,
        preventing read timeouts that plague non-streaming mode when
        models take a long time to load or generate.
        """
        start_time = time.time()

        # Force streaming — this is the key fix.
        payload = dict(payload)
        payload['stream'] = True

        for attempt in range(self._retry_count):
            resp = None
            try:
                _logger.info(
                    'Ollama %s request (attempt %d/%d, model=%s)',
                    endpoint, attempt + 1, self._retry_count,
                    payload.get('model', self.model),
                )
                resp = requests.post(
                    f'{self.ollama_url}{endpoint}',
                    json=payload,
                    timeout=(self._connect_timeout, self._read_timeout),
                    stream=True,
                )
                resp.raise_for_status()

                # Collect streamed response chunks
                response_text = ''
                model_name = self.model
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # /api/chat streams {message: {content: "..."}}
                    if 'message' in chunk:
                        response_text += chunk['message'].get('content', '')
                    # /api/generate streams {response: "..."}
                    elif 'response' in chunk:
                        response_text += chunk.get('response', '')

                    if chunk.get('model'):
                        model_name = chunk['model']

                    # Done flag — final chunk
                    if chunk.get('done'):
                        break

                resp.close()
                elapsed_ms = int((time.time() - start_time) * 1000)

                _logger.info(
                    'Ollama response received in %dms (model=%s, chars=%d)',
                    elapsed_ms, model_name, len(response_text),
                )

                return {
                    'response': response_text.strip(),
                    'model': model_name,
                    'duration_ms': elapsed_ms,
                    'success': True,
                    'error_type': None,
                }

            except requests.ConnectionError as e:
                _logger.warning(
                    'Ollama connection attempt %d/%d failed: %s',
                    attempt + 1, self._retry_count, e,
                )
                if attempt < self._retry_count - 1:
                    time.sleep(self._retry_delay)
            except requests.Timeout as e:
                _logger.warning(
                    'Ollama request timed out (attempt %d/%d): %s',
                    attempt + 1, self._retry_count, e,
                )
                if attempt < self._retry_count - 1:
                    time.sleep(self._retry_delay)
                    continue
                _logger.error('Ollama timed out after all %d attempts', self._retry_count)
                return {
                    'response': FALLBACK_MESSAGE,
                    'model': self.model,
                    'duration_ms': int((time.time() - start_time) * 1000),
                    'success': False,
                    'error_type': 'timeout',
                }
            except requests.HTTPError as e:
                _logger.error('Ollama HTTP error: %s', e)
                if e.response is not None and e.response.status_code == 404:
                    self._resolve_model_name(force=True)
                    return {
                        'response': (
                            f"The AI model '{self._resolved_model or self.model}' is not installed. "
                            "Please ask an administrator to pull the model in Settings."
                        ),
                        'model': self._resolved_model or self.model,
                        'duration_ms': int((time.time() - start_time) * 1000),
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
            finally:
                # Always close the response to free the TCP connection
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:
                        pass

        # All retries exhausted
        elapsed_ms = int((time.time() - start_time) * 1000)
        return {
            'response': FALLBACK_MESSAGE,
            'model': self.model,
            'duration_ms': elapsed_ms,
            'success': False,
            'error_type': 'connection',
        }

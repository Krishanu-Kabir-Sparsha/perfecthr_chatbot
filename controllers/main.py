# -*- coding: utf-8 -*-
"""
Main controller for Perfect HR AI Chatbot.
Handles all JSON-RPC endpoints for the chat widget.
"""
import json
import logging
import re
import time
import uuid

try:
    from langdetect import detect, LangDetectException, DetectorFactory
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except Exception:
    LANGDETECT_AVAILABLE = False
    detect = None
    LangDetectException = Exception

from odoo import http, fields, _
from odoo.http import request

_logger = logging.getLogger(__name__)

_AI_ENGINE_CACHE = {}
_RAG_SERVICE_CACHE = {}

# Patterns that indicate the user is asking to switch language only
_LANG_SWITCH_PATTERNS = [
    # English explicit requests — broad to catch variations like "respond in Spanish",
    # "reply in Bengali", "say it in French", "tell me in Arabic", etc.
    re.compile(r'\b(?:respond|reply|answer|speak|say|tell|write|translate|convert)\s+(?:it\s+)?(?:me\s+)?(?:this\s+)?in\s+([a-zA-Z\- ]{2,30})', re.I),
    re.compile(r'\b(?:switch|change|use)\s+(?:to\s+)?([a-zA-Z\- ]{2,30})\s*(?:language)?', re.I),
    re.compile(r'\bin\s+(bengali|bangla|hindi|arabic|spanish|french|german|portuguese|russian|chinese|japanese|korean|urdu|turkish|italian)\b', re.I),
    # Bengali — broad variations
    re.compile(r'বাংলা', re.I),  # Contains "বাংলা" anywhere
    # Hindi — broad variations
    re.compile(r'हिन्दी|हिंदी', re.I),  # Contains "हिन्दी" or "हिंदी"
    # Arabic
    re.compile(r'بالعربي|العربية', re.I),
    # Spanish
    re.compile(r'en\s+español', re.I),
    # French
    re.compile(r'en\s+français', re.I),
    # Generic: short messages in non-Latin scripts (likely language switch commands)
    re.compile(r'^[\u0980-\u09FF\s\u0964\u0965]{3,60}$'),   # Bengali + Devanagari punctuation
    re.compile(r'^[\u0900-\u097F\s\u0964\u0965]{3,60}$'),   # Hindi/Devanagari
    re.compile(r'^[\u0600-\u06FF\s]{3,60}$'),                # Arabic
]


class PerfectHRChatbotController(http.Controller):
    """JSON-RPC endpoints for the chatbot frontend widget."""

    # ── Helpers ─────────────────────────────────────────────────────
    def _get_config(self):
        """Fetch chatbot configuration from system parameters."""
        ICP = request.env['ir.config_parameter'].sudo()
        return {
            'ollama_url': ICP.get_param(
                'perfecthr_chatbot.ollama_url', 'http://localhost:11434'
            ),
            'model_name': ICP.get_param(
                'perfecthr_chatbot.model_name', 'mistral'
            ),
            'embedding_model': ICP.get_param(
                'perfecthr_chatbot.embedding_model', 'nomic-embed-text'
            ),
            'max_tokens': int(ICP.get_param(
                'perfecthr_chatbot.max_tokens', '2048'
            )),
            'temperature': float(ICP.get_param(
                'perfecthr_chatbot.temperature', '0.7'
            )),
            'greeting_message': ICP.get_param(
                'perfecthr_chatbot.greeting_message',
                "👋 Hi there! I'm your Perfect HR AI Assistant. How can I help you today?"
            ),
            'system_prompt': ICP.get_param(
                'perfecthr_chatbot.system_prompt', ''
            ),
            'sales_team_id': int(ICP.get_param(
                'perfecthr_chatbot.sales_team_id', '0'
            ) or '0'),
            'auto_create_lead': ICP.get_param(
                'perfecthr_chatbot.auto_create_lead', 'True'
            ) == 'True',
        }

    def _get_ai_engine(self, config):
        """Initialize the Ollama AI engine."""
        from ..services.ai_engine import OllamaEngine
        key = (
            config['ollama_url'],
            config['model_name'],
            config['max_tokens'],
            config['temperature'],
            config['system_prompt'],
        )
        engine = _AI_ENGINE_CACHE.get(key)
        if engine is None:
            engine = OllamaEngine(
                ollama_url=config['ollama_url'],
                model=config['model_name'],
                max_tokens=config['max_tokens'],
                temperature=config['temperature'],
                system_prompt=config['system_prompt'],
            )
            _AI_ENGINE_CACHE.clear()
            _AI_ENGINE_CACHE[key] = engine
        return engine

    def _get_rag_service(self, config):
        """Initialize the RAG service."""
        from ..services.rag_service import RAGService
        from odoo.tools import config as odoo_config
        data_dir = odoo_config.get('data_dir', '/tmp')
        rag_dir = f'{data_dir}/perfecthr_chatbot_faiss'
        key = (config['ollama_url'], config['embedding_model'], rag_dir)
        rag_service = _RAG_SERVICE_CACHE.get(key)
        if rag_service is None:
            rag_service = RAGService(
                ollama_url=config['ollama_url'],
                embedding_model=config['embedding_model'],
                data_dir=rag_dir,
            )
            _RAG_SERVICE_CACHE.clear()
            _RAG_SERVICE_CACHE[key] = rag_service
        return rag_service

    def _get_session(self, session_token):
        """Retrieve session by token."""
        Session = request.env['perfecthr.chatbot.session'].sudo()
        return Session.search([
            ('session_token', '=', session_token),
            ('state', '=', 'active'),
        ], limit=1)

    def _language_from_context(self):
        """Return ISO language code from Odoo context (e.g. en, bn, ar)."""
        lang_code = (request.env.context.get('lang') or 'en_US')
        return lang_code.split('_')[0].lower()

    def _detect_script_language(self, text):
        """Best-effort script detection for short multilingual messages."""
        for ch in (text or ''):
            code = ord(ch)
            if 0x0980 <= code <= 0x09FF:
                return 'bn'  # Bengali
            if 0x0600 <= code <= 0x06FF:
                return 'ar'  # Arabic
            if 0x0900 <= code <= 0x097F:
                return 'hi'  # Devanagari (Hindi/Nepali)
            if 0x0400 <= code <= 0x04FF:
                return 'ru'  # Cyrillic (best-effort)
            if 0x4E00 <= code <= 0x9FFF:
                return 'zh'  # CJK Unified Ideographs
        return None

    def _infer_response_language(self, message, session=None):
        """Infer preferred response language code/name dynamically."""
        text = (message or '').strip()
        text_lower = text.lower()

        # Explicit request patterns without hardcoding fixed language list.
        explicit_patterns = [
            r'\b(?:respond|reply|answer)\s+in\s+([a-zA-Z\- ]{2,30})',
            r'\b(?:response\s+language|language)\s*[:=]\s*([a-zA-Z\- ]{2,30})',
        ]
        for pattern in explicit_patterns:
            m = re.search(pattern, text_lower)
            if m:
                requested = m.group(1).strip()
                # If ISO code is provided directly, keep it.
                if re.match(r'^[a-z]{2}(?:-[a-z]{2})?$', requested):
                    return requested.split('-')[0]
                # Otherwise pass through (e.g. "arabic", "spanish", etc.) to model hint.
                return requested

        # Script detection helps for short messages where statistical detection is weak.
        script_lang = self._detect_script_language(text)
        if script_lang:
            return script_lang

        # Dynamic language detection from input text when langdetect is available.
        if LANGDETECT_AVAILABLE and len(text) >= 4:
            try:
                detected = detect(text)
                if detected:
                    return detected
            except LangDetectException:
                pass

        if session and session.preferred_language:
            return session.preferred_language

        return self._language_from_context()

    def _is_language_switch_request(self, message):
        """Check if message is purely a language-switch command (not a new question)."""
        text = (message or '').strip()
        if not text or len(text) > 120:
            return False
        for pattern in _LANG_SWITCH_PATTERNS:
            if pattern.search(text):
                _logger.info('[Language Switch] Detected language switch: "%s" matched pattern %s', text, pattern.pattern)
                return True
        return False

    def _get_previous_assistant_response(self, session):
        """Get the last assistant response from the session."""
        Message = request.env['perfecthr.chatbot.message'].sudo()
        last_msg = Message.search([
            ('session_id', '=', session.id),
            ('role', '=', 'assistant'),
        ], order='id desc', limit=1)
        if last_msg:
            return last_msg.content
        return None

    def _trim_article_content(self, text, max_chars=8000):
        """Clip article text to keep within context window while preserving structure.

        Default increased to 8000 chars — Mistral 7B handles ~24K chars (8K tokens).
        No artificial word limitations.
        """
        if not text:
            return ''
        cleaned = (text or '').strip()
        if len(cleaned) <= max_chars:
            return cleaned
        # Try to cut at line boundary for cleaner prompt context.
        clipped = cleaned[:max_chars]
        last_break = clipped.rfind('\n')
        if last_break > int(max_chars * 0.65):
            clipped = clipped[:last_break]
        return f"{clipped}\n..."

    def _infer_target_categories(self, query, intent):
        """Infer relevant KB categories from query and detected intent."""
        query_lower = (query or '').lower()
        categories = []

        if intent == 'pricing' or any(k in query_lower for k in ['price', 'pricing', 'cost', 'package', 'plan']):
            categories.append('pricing')
        if intent in ('product_inquiry', 'support') or any(
            k in query_lower for k in ['payroll', 'attendance', 'leave', 'recruit', 'module', 'feature',
                                        'crm', 'sales', 'expense', 'analytics', 'reporting', 'task',
                                        'project', 'employee', 'management']
        ):
            categories.append('product')
        if any(k in query_lower for k in ['faq', 'question', 'how to', 'how do', 'why', 'what is']):
            categories.append('faq')
        if any(k in query_lower for k in ['case study', 'success story', 'example', 'client']):
            categories.append('case_study')

        if not categories:
            categories.append('general')

        return categories

    def _compute_keyword_hits(self, query_words, text):
        """Count keyword overlaps for explainable relevance scoring."""
        haystack = (text or '').lower()
        return sum(1 for word in query_words if len(word) > 2 and word in haystack)

    def _tokenize(self, text):
        """Normalize text into comparable lowercase tokens, supporting Unicode."""
        stopwords = {
            'the', 'is', 'at', 'which', 'on', 'for', 'a', 'an', 'and', 'or', 'to', 'in', 'of',
            'with', 'about', 'how', 'what', 'why', 'when', 'can', 'do', 'does', 'please',
            'el', 'la', 'los', 'las', 'en', 'para', 'como', 'que', 'por'  # basic multi-lingual
        }
        return [
            token for token in re.findall(r'\w+', (text or '').lower(), re.UNICODE)
            if len(token) > 2 and token not in stopwords
        ]

    def _extract_module_signals(self, query):
        """Extract domain module/topic signals from query."""
        query_lower = (query or '').lower()
        signal_map = {
            'payroll': ['payroll', 'salary', 'payslip', 'tax', 'deduction', 'bonus'],
            'attendance': ['attendance', 'biometric', 'gps', 'checkin', 'checkout', 'shift', 'overtime'],
            'leave': ['leave', 'vacation', 'timeoff', 'holiday'],
            'recruitment': ['recruitment', 'recruit', 'hiring', 'candidate', 'interview'],
            'performance': ['performance', 'kpi', 'appraisal'],
            'crm': ['crm', 'customer', 'relationship'],
            'sales': ['sales', 'sale', 'selling'],
            'expenses': ['expense', 'expenses', 'reimbursement'],
            'analytics': ['analytics', 'reporting', 'report', 'dashboard'],
            'task': ['task', 'project', 'project management'],
            'pricing': ['pricing', 'price', 'plan', 'package', 'cost', 'subscription'],
            'overview': ['overview', 'about', 'explain', 'functionalities', 'features'],
        }
        matched = []
        for signal, variants in signal_map.items():
            if any(variant in query_lower for variant in variants):
                matched.append(signal)
        return matched

    def _compute_article_rank(self, article, semantic_score, query_tokens, preferred_categories, module_signals):
        """Hybrid relevance score to avoid wrong-module answers."""
        name = (article.get('name') or '').lower()
        summary = (article.get('summary') or article.get('content_summary') or '').lower()
        content = (article.get('content') or '').lower()
        category = article.get('category', 'general')

        title_hits = sum(1 for token in query_tokens if token in name)
        summary_hits = sum(1 for token in query_tokens if token in summary)
        content_hits = sum(1 for token in query_tokens if token in content)

        keyword_score = title_hits * 0.45 + summary_hits * 0.2 + content_hits * 0.08
        category_bonus = 0.22 if category in preferred_categories else 0.0

        module_bonus = 0.0
        if module_signals:
            for signal in module_signals:
                if signal in name:
                    module_bonus += 0.7
                elif signal in summary:
                    module_bonus += 0.3
                elif signal in content:
                    module_bonus += 0.12
                if signal == 'pricing' and category == 'pricing':
                    module_bonus += 0.3
                if signal in ('payroll', 'attendance', 'leave', 'recruitment', 'performance',
                              'crm', 'sales', 'expenses', 'analytics', 'task') and category == 'product':
                    module_bonus += 0.08

        total = semantic_score * 0.45 + keyword_score + category_bonus + module_bonus
        return {
            'total': round(total, 4),
            'keyword_hits': title_hits + summary_hits + content_hits,
            'title_hits': title_hits,
            'module_bonus': round(module_bonus, 3),
        }

    def _build_rag_context(self, query, intent, rag_service):
        """Search knowledge base and build context text."""
        try:
            # Ensure index is loaded
            Knowledge = request.env['perfecthr.chatbot.knowledge'].sudo()
            articles = Knowledge.search([('is_active', '=', True)])
            article_data = []
            article_map = {}
            query_tokens = self._tokenize(query)
            preferred_categories = set(self._infer_target_categories(query, intent))
            module_signals = self._extract_module_signals(query)

            if articles:
                article_data = []
                for art in articles:
                    data = {
                        'id': art.id,
                        'name': art.name,
                        'content': art.content,
                        'content_summary': art.content_summary or '',
                        'category': art.category,
                        'version': art.version,
                    }
                    if art.embedding_data:
                        data['embedding_data'] = art.embedding_data
                    article_data.append(data)
                    article_map[art.id] = data

                # Load persisted index — do NOT rebuild inline to avoid blocking.
                if rag_service._index is None:
                    rag_service._load_index()

                # Only rebuild from pre-computed embeddings (fast operation).
                # This will NOT call Ollama for embedding — articles without
                # embeddings are skipped and included only via keyword fallback.
                if (
                    rag_service._index is None
                    or rag_service._index.ntotal == 0
                    or not rag_service.is_index_synced(article_data)
                ):
                    rag_service.build_index(article_data)

            # Search
            semantic_results = rag_service.search(query, top_k=10)
            semantic_by_id = {item.get('id'): item for item in semantic_results if item.get('id')}
            semantic_ids = set(semantic_by_id.keys())

            # Build a smaller candidate set for ranking to reduce per-request latency.
            shortlist_ids = set(semantic_ids)
            if query_tokens:
                for article in article_data:
                    name = (article.get('name') or '').lower()
                    summary = (article.get('content_summary') or '').lower()
                    content = (article.get('content') or '').lower()
                    if any(token in name or token in summary or token in content for token in query_tokens):
                        shortlist_ids.add(article['id'])
                    if len(shortlist_ids) >= 30:
                        break
            if not shortlist_ids:
                shortlist_ids = set(a['id'] for a in article_data[:30])

            candidates = []
            for article in article_data:
                if article['id'] not in shortlist_ids:
                    continue
                semantic_item = semantic_by_id.get(article['id'], {})
                candidate = {
                    'id': article['id'],
                    'name': article['name'],
                    'category': article.get('category', 'general'),
                    'summary': (semantic_item.get('summary') or article.get('content_summary') or '').strip(),
                    'content': semantic_item.get('content') or article.get('content', ''),
                    'semantic_score': float(semantic_item.get('score', 0.0)),
                }
                rank_info = self._compute_article_rank(
                    candidate,
                    candidate['semantic_score'],
                    query_tokens,
                    preferred_categories,
                    module_signals,
                )
                candidate.update(rank_info)
                candidates.append(candidate)

            if candidates:
                candidates.sort(key=lambda item: item['total'], reverse=True)
                top_results = candidates[:5]

                context_parts = []
                sources = []
                context_char_budget = 16000
                used_chars = 0
                for idx, r in enumerate(top_results):
                    summary = (r.get('summary') or '').strip()
                    if not summary:
                        first_line = next((ln.strip() for ln in (r.get('content', '') or '').splitlines() if ln.strip()), '')
                        summary = first_line or 'No summary provided.'

                    # Tiered content budget: top-1 gets full, 2-3 get 6000, 4-5 get summary only
                    if idx == 0:
                        clipped_content = self._trim_article_content(r.get('content', ''), max_chars=12000)
                    elif idx <= 2:
                        clipped_content = self._trim_article_content(r.get('content', ''), max_chars=6000)
                    else:
                        clipped_content = summary

                    category = r.get('category', 'general')
                    block = (
                        f"[Article: {r['name']}]\n"
                        f"Category: {category}\n"
                        f"Summary: {summary}\n"
                        f"Content:\n{clipped_content}"
                    )
                    if used_chars + len(block) > context_char_budget and context_parts:
                        break
                    context_parts.append(block)
                    used_chars += len(block)
                    sources.append({
                        'id': r['id'],
                        'name': r['name'],
                        'score': r['semantic_score'],
                        'category': r.get('category', 'general'),
                        'keyword_hits': r.get('keyword_hits', 0),
                        'rank': r.get('total', 0.0),
                    })

                best_score = top_results[0].get('total', 0.0)
                strong_match = (
                    best_score >= 0.55
                    or top_results[0].get('title_hits', 0) >= 1
                    or top_results[0].get('module_bonus', 0.0) >= 0.7
                )

                return '\n\n'.join(context_parts), sources, {
                    'strong_match': strong_match,
                    'best_score': round(best_score, 3),
                    'matched_categories': [r.get('category', 'general') for r in top_results],
                }

        except Exception as e:
            _logger.error('RAG context build failed: %s', e)

        return '', [], {'strong_match': False, 'best_score': 0.0, 'matched_categories': []}

    def _build_rule_based_reply(self, query, context_text, rag_sources, response_language='en'):
        """Last-resort fallback when all model generation attempts fail."""
        if not context_text:
            return (
                "I apologize, but I'm temporarily unable to process your request. "
                "Please try again in a moment, or feel free to contact our team directly at support@perfecthr.com."
            )

        # Try to generate a useful deterministic answer from KB context.
        parsed_articles = []
        blocks = context_text.split('[Article:')
        for block in blocks[1:]:
            try:
                title_part, rest = block.split(']', 1)
                title = title_part.strip()
                category_match = re.search(r'Category:\s*(.+)', rest)
                summary_match = re.search(r'Summary:\s*(.+)', rest)
                content_match = re.search(r'Content:\s*(.+)', rest, re.S)
                parsed_articles.append({
                    'title': title,
                    'category': category_match.group(1).strip() if category_match else 'general',
                    'summary': summary_match.group(1).strip() if summary_match else '',
                    'content': content_match.group(1).strip() if content_match else '',
                })
            except Exception:
                continue

        if not parsed_articles:
            return (
                "I apologize, but I'm temporarily unable to process your request. "
                "Please try again in a moment, or feel free to contact our team directly at support@perfecthr.com."
            )

        query_lower = (query or '').lower()
        employee_match = re.search(r'(\d{1,5})\s*(employees|employee|staff|people)', query_lower)
        employee_count = int(employee_match.group(1)) if employee_match else None

        # Pricing-focused synthesis with recommendation.
        if any(k in query_lower for k in ['price', 'pricing', 'plan', 'package', 'cost']) and employee_count:
            combined = '\n\n'.join(a['content'] for a in parsed_articles)
            plan_pattern = re.compile(
                r'([A-Za-z ]+PLAN)\s*\(\$?([0-9]+)[^)]*\)(.*?)(?=(?:\n[A-Z][A-Za-z ]+PLAN\s*\()|\Z)',
                re.S,
            )
            plans = []
            for m in plan_pattern.finditer(combined):
                plan_name = m.group(1).strip().title()
                per_user = m.group(2).strip()
                section = m.group(3)
                range_match = re.search(r'(\d+)\s*-\s*(\d+)\s*employees', section)
                plus_match = re.search(r'(\d+)\+\s*employees', section)
                min_emp = int(range_match.group(1)) if range_match else (int(plus_match.group(1)) if plus_match else None)
                max_emp = int(range_match.group(2)) if range_match else (999999 if plus_match else None)
                feature_lines = [
                    ln.strip().lstrip('-').strip()
                    for ln in section.splitlines()
                    if ln.strip().startswith('-')
                ][:4]
                plans.append({
                    'name': plan_name,
                    'price': per_user,
                    'min': min_emp,
                    'max': max_emp,
                    'features': feature_lines,
                })

            chosen = None
            for plan in plans:
                if plan['min'] is not None and plan['max'] is not None and plan['min'] <= employee_count <= plan['max']:
                    chosen = plan
                    break

            # Heuristic fallback when article text is partial or ranges are missing.
            if not chosen and plans:
                normalized = sorted(plans, key=lambda p: int(p['price']))
                if employee_count <= 50:
                    chosen = normalized[0]
                elif employee_count <= 500:
                    chosen = normalized[min(1, len(normalized) - 1)]
                else:
                    chosen = normalized[-1]

            if chosen:
                monthly = employee_count * int(chosen['price'])
                feature_text = '\n'.join(f"- {f}" for f in chosen['features']) if chosen['features'] else ''
                return (
                    f"For a team of about {employee_count} employees, the best fit is **{chosen['name']}** "
                    f"at approximately **${chosen['price']}/employee/month**.\n\n"
                    f"Estimated monthly license: **${monthly:,}**.\n\n"
                    f"Key inclusions:\n{feature_text}\n\n"
                    "Recommendation: start with this plan and adjust after a 30-day pilot based on adoption and support needs."
                )

        # Return full content from the best-matched article.
        primary = parsed_articles[0]
        summary_text = primary['summary'] if primary['summary'] else 'Here is what I found:'

        return (
            f"{summary_text}\n\n"
            f"{primary['content']}\n\n"
            "For more specific recommendations, please contact our support team."
        )

    def _build_handover_reply(self):
        """User-facing response when no strong knowledge match exists."""
        return (
            "Thanks for your question. I want to make sure you get a precise answer, "
            "but I could not find a strong match in our current knowledge base. "
            "If you would like, I can hand this to our support team and share your query with them."
        )

    # ── Endpoints ───────────────────────────────────────────────────

    @http.route('/perfecthr_chatbot/start', type='json', auth='public',
                website=True, csrf=False)
    def start_session(self, **kwargs):
        """Create a new chat session and return greeting."""
        config = self._get_config()
        Session = request.env['perfecthr.chatbot.session'].sudo()

        # Create session
        preferred_language = self._language_from_context()
        session = Session.create({
            'session_token': str(uuid.uuid4()),
            'visitor_ip': request.httprequest.remote_addr,
            'visitor_user_agent': request.httprequest.headers.get(
                'User-Agent', ''
            )[:255],
            'source': 'website',
            'preferred_language': preferred_language,
        })

        # Create greeting message
        greeting = config['greeting_message']
        request.env['perfecthr.chatbot.message'].sudo().create({
            'session_id': session.id,
            'role': 'assistant',
            'content': greeting,
            'intent': 'greeting',
            'confidence_score': 1.0,
        })

        return {
            'status': 'success',
            'session_token': session.session_token,
            'greeting': greeting,
        }

    @http.route('/perfecthr_chatbot/message', type='json', auth='public',
                website=True, csrf=False)
    def send_message(self, session_token, message, **kwargs):
        """Process a user message and return AI response."""
        start_time = time.time()

        # Validate session
        session = self._get_session(session_token)
        if not session:
            return {
                'status': 'error',
                'error': 'Invalid or expired session.',
            }

        config = self._get_config()
        Message = request.env['perfecthr.chatbot.message'].sudo()

        # Detect intent
        from ..services.intent_detector import IntentDetector
        detector = IntentDetector()
        intent_result = detector.detect(message)

        # Detect response language
        response_language = self._infer_response_language(message, session)
        if response_language and session.preferred_language != response_language:
            session.write({'preferred_language': response_language})

        # Save user message
        user_msg = Message.create({
            'session_id': session.id,
            'role': 'user',
            'content': message,
            'intent': intent_result['intent'],
            'confidence_score': intent_result['confidence'],
        })

        # Handle human handover intent
        if intent_result['intent'] == 'human_handover':
            session.action_escalate()
            response_text = (
                "I understand you'd like to speak with a human representative. "
                "I've notified our team, and someone will reach out to you shortly. "
                "In the meantime, feel free to share your contact information "
                "so we can get back to you faster."
            )
            assistant_msg = Message.create({
                'session_id': session.id,
                'role': 'assistant',
                'content': response_text,
                'intent': 'human_handover',
                'confidence_score': 1.0,
                'response_time_ms': int((time.time() - start_time) * 1000),
            })
            return {
                'status': 'success',
                'response': response_text,
                'intent': 'human_handover',
                'session_state': 'escalated',
            }

        # ── Handle language-switch requests ──
        # When the user just asks to switch language (e.g. "বাংলায় বলুন", "respond in Spanish"),
        # re-use the previous answer and translate it instead of doing a
        # fresh RAG search that returns a random article.
        is_lang_switch = self._is_language_switch_request(message)
        _logger.info(
            '[Language] message="%s", detected_lang="%s", is_switch=%s',
            message[:80], response_language, is_lang_switch,
        )

        if is_lang_switch:
            prev_response = self._get_previous_assistant_response(session)
            if prev_response:
                ai_engine = self._get_ai_engine(config)

                # Map ISO codes to full language names for clearer prompts
                lang_name_map = {
                    'bn': 'Bengali (বাংলা)', 'hi': 'Hindi (हिन्दी)', 'ar': 'Arabic (العربية)',
                    'es': 'Spanish (Español)', 'fr': 'French (Français)', 'de': 'German (Deutsch)',
                    'pt': 'Portuguese', 'ru': 'Russian (Русский)', 'zh': 'Chinese (中文)',
                    'ja': 'Japanese (日本語)', 'ko': 'Korean (한국어)', 'tr': 'Turkish (Türkçe)',
                    'ur': 'Urdu (اردو)', 'it': 'Italian (Italiano)',
                }
                target_lang_display = lang_name_map.get(
                    response_language, response_language
                )

                rewrite_prompt = (
                    f"Your task: Translate the following text into {target_lang_display}.\n\n"
                    f"Rules:\n"
                    f"- Translate EVERYTHING into {target_lang_display}\n"
                    f"- Keep the same structure (bullet points, paragraphs, etc.)\n"
                    f"- Keep factual information accurate\n"
                    f"- Do NOT add any commentary or explanation\n"
                    f"- Output ONLY the translated text\n\n"
                    f"Text to translate:\n"
                    f"---\n{prev_response}\n---"
                )

                _logger.info(
                    '[Language Switch] Translating previous response to %s (lang=%s)',
                    target_lang_display, response_language,
                )

                result = ai_engine.chat(
                    messages=[{'role': 'user', 'content': rewrite_prompt}],
                    language_hint=response_language,
                    max_tokens=config['max_tokens'],
                )

                if result.get('success') and (result.get('response') or '').strip():
                    response_text = result['response'].strip()
                    _logger.info('[Language Switch] Translation successful (%d chars)', len(response_text))
                else:
                    # Translation failed — give an informative message instead of
                    # silently echoing the same English text.
                    _logger.warning(
                        '[Language Switch] Translation FAILED for lang=%s, success=%s',
                        response_language, result.get('success'),
                    )
                    response_text = (
                        f"I apologize, I'm having difficulty translating to {target_lang_display} right now. "
                        f"Here is the original response:\n\n{prev_response}"
                    )

                response_time_ms = int((time.time() - start_time) * 1000)

                assistant_msg = Message.create({
                    'session_id': session.id,
                    'role': 'assistant',
                    'content': response_text,
                    'model_used': result.get('model', config['model_name']),
                    'response_time_ms': response_time_ms,
                    'intent': 'language_switch',
                    'confidence_score': 1.0,
                })

                return {
                    'status': 'success',
                    'response': response_text,
                    'intent': 'language_switch',
                    'confidence': 1.0,
                    'response_time_ms': response_time_ms,
                    'qualification_score': session.qualification_score,
                    'is_qualified': session.is_qualified,
                    'ai_available': True,
                    'assistant_timestamp': assistant_msg.timestamp.isoformat() if assistant_msg.timestamp else '',
                    'model_used': result.get('model', config['model_name']),
                }

        # Build conversation history for multi-turn
        history_msgs = Message.search([
            ('session_id', '=', session.id),
        ], order='id desc', limit=10)

        chat_messages = []
        for msg in history_msgs[::-1]:
            if msg.role in ('user', 'assistant'):
                chat_messages.append({
                    'role': msg.role,
                    'content': msg.content,
                })

        # RAG context
        rag_service = self._get_rag_service(config)
        context_text, rag_sources, match_meta = self._build_rag_context(
            message,
            intent_result['intent'],
            rag_service,
        )

        # Generate AI response
        ai_engine = self._get_ai_engine(config)

        # Build response guidance
        response_guidance = (
            "Read the provided knowledge base context carefully and synthesize a complete, natural answer to the user's question. "
            "Use your own words instead of directly copying the articles. "
            "Address all aspects of the user's query comprehensively, including relevant features, pricing, or benefits found in the context. "
            "Be helpful, professional, and conversational."
        )

        effective_context = context_text
        if not match_meta.get('strong_match'):
            effective_context = (
                (context_text + "\n\n") if context_text else ""
            ) + "[Context Confidence]\nLow confidence match."

        result = ai_engine.chat(
            chat_messages,
            context_text=effective_context,
            max_tokens=config['max_tokens'],
            language_hint=response_language,
            response_guidance=response_guidance,
        )

        if not result.get('success') or not (result.get('response') or '').strip():
            # Model failed — build a deterministic KB reply directly.
            # Do NOT make a second AI call (_rewrite_in_language) to avoid
            # doubling response time in failure scenarios.
            fallback_text = self._build_rule_based_reply(
                message,
                context_text,
                rag_sources,
                response_language=response_language,
            )
            result['response'] = fallback_text
            result['model'] = 'kb_fallback'

        response_time_ms = int((time.time() - start_time) * 1000)

        # Save assistant message
        assistant_msg = Message.create({
            'session_id': session.id,
            'role': 'assistant',
            'content': result['response'],
            'model_used': result.get('model', config['model_name']),
            'response_time_ms': response_time_ms,
            'rag_sources': json.dumps(rag_sources) if rag_sources else False,
            'intent': intent_result['intent'],
            'confidence_score': intent_result['confidence'],
        })

        # Lead qualification check
        from ..services.lead_qualifier import LeadQualifier
        qualifier = LeadQualifier()
        session_data = {
            'visitor_name': session.visitor_name or '',
            'visitor_email': session.visitor_email or '',
            'visitor_phone': session.visitor_phone or '',
            'visitor_company': session.visitor_company or '',
            'visitor_employee_size': session.visitor_employee_size or '',
        }
        all_messages = [
            {'role': m.role, 'content': m.content}
            for m in history_msgs
        ]
        qual_result = qualifier.qualify(session_data, all_messages)

        # Update session qualification
        session.write({
            'qualification_score': qual_result['score'],
            'is_qualified': qual_result['is_qualified'],
        })

        # Auto-update extracted info
        extracted = qual_result.get('extracted', {})
        update_vals = {}
        if extracted.get('name') and not session.visitor_name:
            update_vals['visitor_name'] = extracted['name']
        if extracted.get('email') and not session.visitor_email:
            update_vals['visitor_email'] = extracted['email']
        if extracted.get('phone') and not session.visitor_phone:
            update_vals['visitor_phone'] = extracted['phone']
        if extracted.get('company') and not session.visitor_company:
            update_vals['visitor_company'] = extracted['company']
        if extracted.get('employee_size') and not session.visitor_employee_size:
            update_vals['visitor_employee_size'] = extracted['employee_size']
        if update_vals:
            session.write(update_vals)

        response = {
            'status': 'success',
            'response': result['response'],
            'intent': intent_result['intent'],
            'confidence': intent_result['confidence'],
            'response_time_ms': response_time_ms,
            'qualification_score': qual_result['score'],
            'is_qualified': qual_result['is_qualified'],
            'ai_available': result.get('success', True),
            'assistant_timestamp': assistant_msg.timestamp.isoformat() if assistant_msg.timestamp else '',
            'model_used': result.get('model', config['model_name']),
        }

        # Suggest lead capture if qualified but no lead yet
        if qual_result['is_qualified'] and not session.lead_id:
            response['suggest_lead_capture'] = True

        return response

    @http.route('/perfecthr_chatbot/submit_lead', type='json', auth='public',
                website=True, csrf=False)
    def submit_lead(self, session_token, name='', email='', phone='',
                    company='', employee_size='', **kwargs):
        """Capture lead and create CRM record."""
        session = self._get_session(session_token)
        if not session:
            return {'status': 'error', 'error': 'Invalid session.'}

        if session.lead_id:
            return {
                'status': 'exists',
                'message': 'Lead already captured for this session.',
            }

        config = self._get_config()

        # Update session with visitor info
        session.write({
            'visitor_name': name or session.visitor_name,
            'visitor_email': email or session.visitor_email,
            'visitor_phone': phone or session.visitor_phone,
            'visitor_company': company or session.visitor_company,
            'visitor_employee_size': employee_size or session.visitor_employee_size,
        })

        # Create CRM lead
        lead_vals = {
            'name': f"[Chatbot] {company or name or 'Website Visitor'}",
            'contact_name': name,
            'email_from': email,
            'phone': phone,
            'partner_name': company,
            'description': session._get_chat_transcript_html(),
            'chatbot_session_id': session.id,
            'chatbot_qualified': session.is_qualified,
            'chatbot_score': session.qualification_score,
            'employee_size': employee_size,
            'is_from_chatbot': True,
            'type': 'lead',
        }

        # Assign sales team
        if config.get('sales_team_id'):
            lead_vals['team_id'] = config['sales_team_id']

        try:
            lead = request.env['crm.lead'].sudo().create(lead_vals)
            session.write({'lead_id': lead.id})

            # Log message on lead
            lead.message_post(
                body=_(
                    '🤖 Lead created by AI Chatbot<br/>'
                    'Session: %s<br/>'
                    'Qualification Score: %d/100<br/>'
                    'Messages: %d'
                ) % (
                    session.display_name,
                    session.qualification_score,
                    session.message_count,
                ),
                message_type='notification',
            )

            return {
                'status': 'success',
                'message': 'Thank you! Our team will reach out to you soon.',
                'lead_id': lead.id,
            }
        except Exception as e:
            _logger.error('Lead creation failed: %s', e)
            return {
                'status': 'error',
                'error': 'Failed to save your information. Please try again.',
            }

    @http.route('/perfecthr_chatbot/history', type='json', auth='public',
                website=True, csrf=False)
    def get_history(self, session_token, **kwargs):
        """Return message history for a session."""
        session = self._get_session(session_token)
        if not session:
            return {'status': 'error', 'error': 'Invalid session.'}

        messages = request.env['perfecthr.chatbot.message'].sudo().search([
            ('session_id', '=', session.id),
        ], order='timestamp asc')

        return {
            'status': 'success',
            'messages': [
                {
                    'role': msg.role,
                    'content': msg.content,
                    'timestamp': msg.timestamp.isoformat() if msg.timestamp else '',
                    'intent': msg.intent or '',
                }
                for msg in messages
            ],
        }

    @http.route('/perfecthr_chatbot/feedback', type='json', auth='public',
                website=True, csrf=False)
    def submit_feedback(self, session_token, message_id, feedback,
                        note='', **kwargs):
        """Store user feedback on a specific response."""
        session = self._get_session(session_token)
        if not session:
            return {'status': 'error', 'error': 'Invalid session.'}

        Message = request.env['perfecthr.chatbot.message'].sudo()
        msg = Message.browse(int(message_id))

        if not msg.exists() or msg.session_id.id != session.id:
            return {'status': 'error', 'error': 'Message not found.'}

        msg.write({
            'feedback': feedback,
            'feedback_note': note,
        })

        return {'status': 'success'}

    @http.route('/perfecthr_chatbot/config', type='json', auth='public',
                website=True, csrf=False)
    def get_widget_config(self, **kwargs):
        """Return widget configuration for the frontend."""
        config = self._get_config()
        return {
            'status': 'success',
            'greeting_message': config['greeting_message'],
            'auto_create_lead': config['auto_create_lead'],
        }

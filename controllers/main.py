# -*- coding: utf-8 -*-
"""
Main controller for Perfect HR AI Chatbot.
Handles all JSON-RPC endpoints for the chat widget.
"""
import json
import logging
import time
import uuid

from odoo import http, fields, _
from odoo.http import request

_logger = logging.getLogger(__name__)


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
                'perfecthr_chatbot.max_tokens', '512'
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
        return OllamaEngine(
            ollama_url=config['ollama_url'],
            model=config['model_name'],
            max_tokens=config['max_tokens'],
            temperature=config['temperature'],
            system_prompt=config['system_prompt'],
        )

    def _get_rag_service(self, config):
        """Initialize the RAG service."""
        from ..services.rag_service import RAGService
        from odoo.tools import config as odoo_config
        data_dir = odoo_config.get('data_dir', '/tmp')
        return RAGService(
            ollama_url=config['ollama_url'],
            embedding_model=config['embedding_model'],
            data_dir=f'{data_dir}/perfecthr_chatbot_faiss',
        )

    def _get_session(self, session_token):
        """Retrieve session by token."""
        Session = request.env['perfecthr.chatbot.session'].sudo()
        return Session.search([
            ('session_token', '=', session_token),
            ('state', '=', 'active'),
        ], limit=1)

    def _build_rag_context(self, query, rag_service):
        """Search knowledge base and build context text."""
        try:
            # Ensure index is loaded
            Knowledge = request.env['perfecthr.chatbot.knowledge'].sudo()
            articles = Knowledge.search([('is_active', '=', True)])

            if articles:
                article_data = []
                for art in articles:
                    data = {
                        'id': art.id,
                        'name': art.name,
                        'content': art.content,
                        'content_summary': art.content_summary or '',
                    }
                    if art.embedding_data:
                        data['embedding_data'] = art.embedding_data
                    article_data.append(data)

                # Build index if needed
                if rag_service._index is None or rag_service._index.ntotal == 0:
                    rag_service.build_index(article_data)

            # Search
            results = rag_service.search(query, top_k=3)

            if results:
                context_parts = []
                sources = []
                for r in results:
                    context_parts.append(
                        f"[{r['name']}]\n{r['content']}"
                    )
                    sources.append({
                        'id': r['id'],
                        'name': r['name'],
                        'score': r['score'],
                    })
                return '\n\n'.join(context_parts), sources

        except Exception as e:
            _logger.error('RAG context build failed: %s', e)

        return '', []

    # ── Endpoints ───────────────────────────────────────────────────

    @http.route('/perfecthr_chatbot/start', type='json', auth='public',
                website=True, csrf=False)
    def start_session(self, **kwargs):
        """Create a new chat session and return greeting."""
        config = self._get_config()
        Session = request.env['perfecthr.chatbot.session'].sudo()

        # Create session
        session = Session.create({
            'session_token': str(uuid.uuid4()),
            'visitor_ip': request.httprequest.remote_addr,
            'visitor_user_agent': request.httprequest.headers.get(
                'User-Agent', ''
            )[:255],
            'source': 'website',
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

        # Build conversation history for multi-turn
        history_msgs = Message.search([
            ('session_id', '=', session.id),
        ], order='timestamp asc', limit=20)

        chat_messages = []
        for msg in history_msgs:
            if msg.role in ('user', 'assistant'):
                chat_messages.append({
                    'role': msg.role,
                    'content': msg.content,
                })

        # RAG context
        rag_service = self._get_rag_service(config)
        context_text, rag_sources = self._build_rag_context(message, rag_service)

        # Generate AI response
        ai_engine = self._get_ai_engine(config)
        result = ai_engine.chat(chat_messages, context_text=context_text)

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

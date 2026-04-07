# -*- coding: utf-8 -*-
import json
import logging

import requests

from odoo import api, fields, models, _
from odoo.exceptions import UserError

_logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are the Perfect HR AI Assistant — a knowledgeable, friendly, and professional digital sales consultant for Perfect HR, a comprehensive Human Resource Management solution built on Odoo.

Your responsibilities:
1. Answer questions about Perfect HR modules (Payroll, Attendance, Leave Management, Recruitment, Performance, etc.)
2. Help visitors understand how Perfect HR can solve their HR challenges
3. Qualify potential leads by naturally gathering: name, company, email/phone, employee count
4. Recommend appropriate Perfect HR packages based on company size and needs
5. Offer to schedule demos when visitors show interest

Guidelines:
- Stay strictly within the Perfect HR domain. Politely redirect off-topic questions.
- Be concise but helpful. Use bullet points for feature lists.
- If you don't know something, say so honestly and offer to connect them with a human expert.
- Always be warm, professional, and solution-oriented.
- When the visitor seems interested, gently ask qualifying questions one at a time.
"""


class ResConfigSettings(models.TransientModel):
    _inherit = 'res.config.settings'

    # ── Ollama Connection ───────────────────────────────────────────
    chatbot_ollama_url = fields.Char(
        string='Ollama Server URL',
        config_parameter='perfecthr_chatbot.ollama_url',
        default='http://localhost:11434',
        help='URL of the Ollama server (e.g., http://localhost:11434)',
    )
    chatbot_model_name = fields.Selection([
        ('mistral', 'Mistral 7B'),
        ('llama3', 'LLaMA 3'),
        ('phi3', 'Phi-3'),
    ], string='LLM Model',
        config_parameter='perfecthr_chatbot.model_name',
        default='mistral',
        help='The Ollama model to use for chat responses.',
    )
    chatbot_embedding_model = fields.Char(
        string='Embedding Model',
        config_parameter='perfecthr_chatbot.embedding_model',
        default='nomic-embed-text',
        help='Ollama model for generating text embeddings (RAG).',
    )

    # ── Response Settings ───────────────────────────────────────────
    chatbot_max_tokens = fields.Integer(
        string='Max Response Tokens',
        config_parameter='perfecthr_chatbot.max_tokens',
        default=512,
        help='Maximum number of tokens in AI response.',
    )
    chatbot_temperature = fields.Float(
        string='Temperature',
        config_parameter='perfecthr_chatbot.temperature',
        default=0.7,
        digits=(3, 2),
        help='Controls response creativity (0.0 = focused, 1.0 = creative).',
    )

    # ── Chat UI Settings ────────────────────────────────────────────
    chatbot_greeting_message = fields.Char(
        string='Greeting Message',
        size=2048,
        config_parameter='perfecthr_chatbot.greeting_message',
        default=(
            "👋 Hi there! I'm your Perfect HR AI Assistant. "
            "I can help you with:\n\n"
            "• HR module features & capabilities\n"
            "• Pricing and packages\n"
            "• Scheduling a demo\n"
            "• Any HR management questions\n\n"
            "How can I help you today?"
        ),
    )
    chatbot_system_prompt = fields.Char(
        string='System Prompt',
        size=4096,
        config_parameter='perfecthr_chatbot.system_prompt',
        default=DEFAULT_SYSTEM_PROMPT,
        help='System prompt that controls AI behavior and personality.',
    )

    # ── CRM Settings ───────────────────────────────────────────────
    chatbot_sales_team_id = fields.Many2one(
        'crm.team',
        string='Default Sales Team',
        config_parameter='perfecthr_chatbot.sales_team_id',
        help='Sales team to assign chatbot-generated leads.',
    )
    chatbot_auto_create_lead = fields.Boolean(
        string='Auto-create Leads',
        config_parameter='perfecthr_chatbot.auto_create_lead',
        default=True,
        help='Automatically create CRM leads when visitor provides contact info.',
    )

    # ── Ollama Status (non-stored readonly computed fields) ─────────
    # Note: Removed computed fields from settings model as they cause
    # form compiler issues in Odoo 18 res.config.settings. Status display
    # can be handled by the get_config() controller method instead.

    # ── Actions ─────────────────────────────────────────────────────
    def _list_ollama_model_names(self, url):
        """Return available model names from Ollama /api/tags."""
        resp = requests.get(f'{url}/api/tags', timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [m.get('name', '') for m in data.get('models', [])]

    def _is_model_available(self, model, model_names):
        """Check model by exact name or with :latest suffix."""
        return any(name == model or name.startswith(f'{model}:') for name in model_names)

    def action_test_ollama_connection(self):
        """Test connectivity to the Ollama server."""
        self.ensure_one()
        url = self.chatbot_ollama_url or 'http://localhost:11434'

        try:
            resp = requests.get(f'{url}/api/tags', timeout=10)
            resp.raise_for_status()
            data = resp.json()
            models_list = [m.get('name', '') for m in data.get('models', [])]

            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': _('Ollama Connection'),
                    'message': _(
                        '✅ Connected successfully!\n'
                        'Available models: %s'
                    ) % ', '.join(models_list) if models_list else _('No models found'),
                    'type': 'success',
                    'sticky': True,
                },
            }
        except requests.ConnectionError:
            raise UserError(_(
                '❌ Cannot connect to Ollama at %s.\n\n'
                'Make sure Ollama is installed and running:\n'
                '  1. Install: https://ollama.ai\n'
                '  2. Start: ollama serve\n'
                '  3. Pull model: ollama pull mistral'
            ) % url)
        except Exception as e:
            raise UserError(_(
                '❌ Connection failed: %s'
            ) % str(e))

    def action_pull_model(self):
        """Pull the selected model via Ollama API."""
        self.ensure_one()
        url = self.chatbot_ollama_url or 'http://localhost:11434'
        model = self.chatbot_model_name or 'mistral'

        try:
            model_names = self._list_ollama_model_names(url)
            if self._is_model_available(model, model_names):
                return {
                    'type': 'ir.actions.client',
                    'tag': 'display_notification',
                    'params': {
                        'title': _('Model Pull'),
                        'message': _('Model "%s" is already available.') % model,
                        'type': 'info',
                        'sticky': False,
                    },
                }

            resp = requests.post(
                f'{url}/api/pull',
                json={'name': model, 'stream': False},
                timeout=180,
            )
            resp.raise_for_status()

            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': _('Model Pull'),
                    'message': _('✅ Model "%s" pulled successfully!') % model,
                    'type': 'success',
                    'sticky': False,
                },
            }
        except requests.ConnectionError:
            raise UserError(_(
                '❌ Cannot connect to Ollama. Make sure it is running.'
            ))
        except requests.Timeout:
            raise UserError(_(
                '⌛ Pull request timed out while downloading "%s".\n\n'
                'If download is still in progress, wait a bit and click List Models to verify.'
            ) % model)
        except Exception as e:
            raise UserError(_(
                '❌ Failed to pull model "%s": %s'
            ) % (model, str(e)))

    def action_pull_embedding_model(self):
        """Pull the embedding model via Ollama API."""
        self.ensure_one()
        url = self.chatbot_ollama_url or 'http://localhost:11434'
        model = self.chatbot_embedding_model or 'nomic-embed-text'

        try:
            model_names = self._list_ollama_model_names(url)
            if self._is_model_available(model, model_names):
                return {
                    'type': 'ir.actions.client',
                    'tag': 'display_notification',
                    'params': {
                        'title': _('Embedding Model Pull'),
                        'message': _('Embedding model "%s" is already available.') % model,
                        'type': 'info',
                        'sticky': False,
                    },
                }

            resp = requests.post(
                f'{url}/api/pull',
                json={'name': model, 'stream': False},
                timeout=180,
            )
            resp.raise_for_status()

            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': _('Embedding Model Pull'),
                    'message': _(
                        '✅ Embedding model "%s" pulled successfully!'
                    ) % model,
                    'type': 'success',
                    'sticky': False,
                },
            }
        except requests.ConnectionError:
            raise UserError(_(
                '❌ Cannot connect to Ollama. Make sure it is running.'
            ))
        except requests.Timeout:
            raise UserError(_(
                '⌛ Pull request timed out while downloading embedding model "%s".\n\n'
                'If download is still in progress, wait a bit and click List Models to verify.'
            ) % model)
        except Exception as e:
            raise UserError(_(
                '❌ Failed to pull embedding model: %s'
            ) % str(e))

    def action_list_models(self):
        """Fetch and display available models from Ollama."""
        self.ensure_one()
        url = self.chatbot_ollama_url or 'http://localhost:11434'

        try:
            resp = requests.get(f'{url}/api/tags', timeout=10)
            resp.raise_for_status()
            data = resp.json()
            models_list = data.get('models', [])

            if not models_list:
                msg = _('No models found. Pull a model first:\n  ollama pull mistral')
            else:
                lines = []
                for m in models_list:
                    size_gb = m.get('size', 0) / (1024**3)
                    lines.append(f"• {m.get('name', '?')} ({size_gb:.1f} GB)")
                msg = _('Available Models:\n') + '\n'.join(lines)

            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': _('Ollama Models'),
                    'message': msg,
                    'type': 'info',
                    'sticky': True,
                },
            }
        except Exception as e:
            raise UserError(_(
                '❌ Failed to list models: %s'
            ) % str(e))

# -*- coding: utf-8 -*-
import json
import logging

from odoo import api, fields, models, _
from odoo.exceptions import UserError

_logger = logging.getLogger(__name__)


class ChatbotKnowledge(models.Model):
    _name = 'perfecthr.chatbot.knowledge'
    _description = 'Chatbot Knowledge Base'
    _order = 'category, sequence, name'

    name = fields.Char(string='Title', required=True, index=True)
    sequence = fields.Integer(string='Sequence', default=10)

    category = fields.Selection([
        ('product', 'Product Module'),
        ('faq', 'FAQ'),
        ('pricing', 'Pricing'),
        ('case_study', 'Case Study'),
        ('general', 'General'),
    ], string='Category', required=True, default='general')

    content = fields.Text(
        string='Content',
        required=True,
        help='The knowledge content that will be indexed for RAG retrieval.',
    )
    content_summary = fields.Char(
        string='Summary',
        help='Brief summary for display in search results.',
    )

    # ── Embedding ───────────────────────────────────────────────────
    embedding_data = fields.Binary(
        string='Embedding Vector',
        attachment=True,
        help='Serialized numpy array of the embedding vector.',
    )
    is_embedded = fields.Boolean(
        string='Embedded',
        default=False,
        compute='_compute_is_embedded',
        store=True,
    )
    last_embedded_at = fields.Datetime(string='Last Embedded')
    embedding_model = fields.Char(
        string='Embedding Model',
        help='Model used to generate the embedding.',
    )

    # ── Versioning ──────────────────────────────────────────────────
    version = fields.Integer(string='Version', default=1)
    is_active = fields.Boolean(string='Active', default=True)

    # ── Tags ────────────────────────────────────────────────────────
    tag_ids = fields.Many2many(
        'perfecthr.chatbot.knowledge.tag',
        string='Tags',
    )

    @api.depends('embedding_data')
    def _compute_is_embedded(self):
        for rec in self:
            rec.is_embedded = bool(rec.embedding_data)

    def write(self, vals):
        """Increment version on content change and clear embedding."""
        if 'content' in vals:
            for rec in self:
                vals_copy = dict(vals)
                vals_copy['version'] = rec.version + 1
                vals_copy['embedding_data'] = False
                vals_copy['last_embedded_at'] = False
                super(ChatbotKnowledge, rec).write(vals_copy)
            return True
        return super().write(vals)

    def action_embed(self):
        """Trigger embedding generation for selected articles."""
        try:
            from ..services.rag_service import RAGService
        except ImportError:
            raise UserError(_(
                'RAG Service is not available. '
                'Ensure faiss-cpu and sentence-transformers are installed.'
            ))

        config = self.env['ir.config_parameter'].sudo()
        ollama_url = config.get_param(
            'perfecthr_chatbot.ollama_url', 'http://localhost:11434'
        )
        embedding_model = config.get_param(
            'perfecthr_chatbot.embedding_model', 'nomic-embed-text'
        )

        rag = RAGService(ollama_url=ollama_url, embedding_model=embedding_model)

        count = 0
        errors = []
        for rec in self:
            try:
                embedding = rag.embed_text(rec.content)
                if embedding is not None:
                    import pickle
                    rec.sudo().write({
                        'embedding_data': pickle.dumps(embedding),
                        'last_embedded_at': fields.Datetime.now(),
                        'embedding_model': embedding_model,
                    })
                    count += 1
                else:
                    errors.append(rec.name)
            except Exception as e:
                _logger.error('Embedding failed for %s: %s', rec.name, e)
                errors.append(f"{rec.name}: {str(e)}")

        msg = _('%d article(s) embedded successfully.') % count
        if errors:
            msg += _('\n\nFailed: %s') % ', '.join(errors)

        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': _('Embedding Complete'),
                'message': msg,
                'type': 'success' if not errors else 'warning',
                'sticky': bool(errors),
            },
        }

    @api.model
    def _cron_auto_embed(self):
        """Auto-embed articles that haven't been embedded yet."""
        articles = self.search([
            ('is_active', '=', True),
            ('is_embedded', '=', False),
        ], limit=50)
        if articles:
            articles.action_embed()
        return True


class ChatbotKnowledgeTag(models.Model):
    _name = 'perfecthr.chatbot.knowledge.tag'
    _description = 'Knowledge Base Tag'

    name = fields.Char(string='Tag Name', required=True)
    color = fields.Integer(string='Color Index')

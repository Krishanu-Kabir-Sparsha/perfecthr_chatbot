# -*- coding: utf-8 -*-
from odoo import api, fields, models


class ChatbotMessage(models.Model):
    _name = 'perfecthr.chatbot.message'
    _description = 'Chatbot Message'
    _order = 'timestamp asc'

    # ── Relations ───────────────────────────────────────────────────
    session_id = fields.Many2one(
        'perfecthr.chatbot.session',
        string='Session',
        required=True,
        ondelete='cascade',
        index=True,
    )

    # ── Content ─────────────────────────────────────────────────────
    role = fields.Selection([
        ('user', 'User'),
        ('assistant', 'AI Assistant'),
        ('system', 'System'),
    ], string='Role', required=True)
    content = fields.Text(string='Content', required=True)

    # ── AI Metadata ─────────────────────────────────────────────────
    intent = fields.Char(string='Detected Intent')
    detected_language = fields.Char(
        string='Detected Language',
        help='Language code detected for this message (e.g. en, bn, ar)',
    )
    confidence_score = fields.Float(
        string='Confidence',
        digits=(3, 2),
        help='Intent detection confidence (0.0 - 1.0)',
    )
    model_used = fields.Char(
        string='Model Used',
        help='Ollama model that generated this response',
    )
    response_time_ms = fields.Integer(
        string='Response Time (ms)',
        help='Time taken to generate response in milliseconds',
    )

    # ── Timestamps ──────────────────────────────────────────────────
    timestamp = fields.Datetime(
        string='Timestamp',
        default=fields.Datetime.now,
        required=True,
        index=True,
    )

    # ── Feedback ────────────────────────────────────────────────────
    feedback = fields.Selection([
        ('positive', 'Helpful'),
        ('negative', 'Not Helpful'),
    ], string='Feedback')
    feedback_note = fields.Text(string='Feedback Note')

    # ── RAG Context ─────────────────────────────────────────────────
    rag_sources = fields.Text(
        string='RAG Sources',
        help='Knowledge base articles used to generate this response (JSON)',
    )

    @api.model_create_multi
    def create(self, vals_list):
        """Override create to update session write_date for activity tracking."""
        messages = super().create(vals_list)
        sessions = messages.mapped('session_id')
        if sessions:
            self.env.cr.execute(
                "UPDATE perfecthr_chatbot_session SET write_date = NOW() WHERE id = ANY(%s)",
                [sessions.ids],
            )
        return messages

# -*- coding: utf-8 -*-
import json
import logging
from collections import Counter
from datetime import timedelta

from odoo import api, fields, models

_logger = logging.getLogger(__name__)


class ChatbotAnalytics(models.Model):
    _name = 'perfecthr.chatbot.analytics'
    _description = 'Chatbot Analytics'
    _order = 'date desc'
    _rec_name = 'date'

    date = fields.Date(
        string='Date',
        required=True,
        index=True,
    )

    # ── KPIs ────────────────────────────────────────────────────────
    total_conversations = fields.Integer(
        string='Total Conversations',
        default=0,
    )
    total_messages = fields.Integer(
        string='Total Messages',
        default=0,
    )
    total_leads = fields.Integer(
        string='Leads Generated',
        default=0,
    )
    conversion_rate = fields.Float(
        string='Conversion Rate (%)',
        digits=(5, 2),
        default=0.0,
    )
    avg_response_time_ms = fields.Integer(
        string='Avg Response Time (ms)',
        default=0,
    )
    drop_off_count = fields.Integer(
        string='Drop-offs',
        default=0,
        help='Sessions with only 1-2 messages (user left quickly).',
    )
    resolution_rate = fields.Float(
        string='AI Resolution Rate (%)',
        digits=(5, 2),
        default=0.0,
    )
    escalation_count = fields.Integer(
        string='Escalations',
        default=0,
    )

    # ── Intent Analysis ─────────────────────────────────────────────
    top_intents_json = fields.Text(
        string='Top Intents (JSON)',
        default='{}',
    )

    # ── Qualified Leads ─────────────────────────────────────────────
    qualified_leads = fields.Integer(
        string='Qualified Leads',
        default=0,
    )
    qualified_rate = fields.Float(
        string='Qualified Rate (%)',
        digits=(5, 2),
        default=0.0,
    )

    _sql_constraints = [
        ('date_unique', 'unique(date)', 'Analytics record already exists for this date.'),
    ]

    @api.model
    def _cron_compute_daily(self):
        """Compute analytics for yesterday."""
        yesterday = fields.Date.today() - timedelta(days=1)
        self._compute_for_date(yesterday)
        return True

    @api.model
    def _compute_for_date(self, target_date):
        """Compute or update analytics for a specific date."""
        Session = self.env['perfecthr.chatbot.session'].sudo()
        Message = self.env['perfecthr.chatbot.message'].sudo()

        date_start = fields.Datetime.to_datetime(target_date)
        date_end = date_start + timedelta(days=1)

        # Sessions started on this date
        sessions = Session.search([
            ('start_time', '>=', date_start),
            ('start_time', '<', date_end),
        ])

        total_convos = len(sessions)
        total_leads = len(sessions.filtered('lead_id'))
        qualified = len(sessions.filtered('is_qualified'))
        escalated = len(sessions.filtered(lambda s: s.state == 'escalated'))

        # Drop-offs: sessions with <= 2 messages
        drop_offs = len(sessions.filtered(lambda s: s.message_count <= 2))

        # Messages on this date
        messages = Message.search([
            ('timestamp', '>=', date_start),
            ('timestamp', '<', date_end),
        ])
        total_msgs = len(messages)

        # Average response time for assistant messages
        assistant_msgs = messages.filtered(
            lambda m: m.role == 'assistant' and m.response_time_ms > 0
        )
        avg_rt = 0
        if assistant_msgs:
            avg_rt = int(
                sum(assistant_msgs.mapped('response_time_ms'))
                / len(assistant_msgs)
            )

        # Intent analysis
        intents = [m.intent for m in messages if m.intent]
        intent_counter = Counter(intents)
        top_intents = dict(intent_counter.most_common(10))

        # Resolution rate: sessions closed (not escalated) / total
        resolved = len(sessions.filtered(
            lambda s: s.state == 'closed'
        ))
        resolution = (resolved / total_convos * 100) if total_convos else 0.0

        # Conversion rate
        conversion = (total_leads / total_convos * 100) if total_convos else 0.0
        qualified_rate = (qualified / total_leads * 100) if total_leads else 0.0

        vals = {
            'date': target_date,
            'total_conversations': total_convos,
            'total_messages': total_msgs,
            'total_leads': total_leads,
            'conversion_rate': round(conversion, 2),
            'avg_response_time_ms': avg_rt,
            'drop_off_count': drop_offs,
            'resolution_rate': round(resolution, 2),
            'escalation_count': escalated,
            'top_intents_json': json.dumps(top_intents),
            'qualified_leads': qualified,
            'qualified_rate': round(qualified_rate, 2),
        }

        existing = self.search([('date', '=', target_date)], limit=1)
        if existing:
            existing.write(vals)
        else:
            self.create(vals)

        _logger.info(
            'Chatbot analytics computed for %s: %d convos, %d leads',
            target_date, total_convos, total_leads,
        )
        return True

    def action_refresh(self):
        """Manually refresh analytics for this date."""
        for rec in self:
            self._compute_for_date(rec.date)
        return True

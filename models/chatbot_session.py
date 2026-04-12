# -*- coding: utf-8 -*-
import uuid
from odoo import api, fields, models, _
from odoo.exceptions import UserError


class ChatbotSession(models.Model):
    _name = 'perfecthr.chatbot.session'
    _description = 'AI Chatbot Session'
    _order = 'start_time desc'
    _rec_name = 'display_name'

    # ── Identification ──────────────────────────────────────────────
    session_token = fields.Char(
        string='Session Token',
        required=True,
        index=True,
        default=lambda self: str(uuid.uuid4()),
        copy=False,
        readonly=True,
    )
    display_name = fields.Char(
        string='Session',
        compute='_compute_display_name',
        store=True,
    )

    # ── Visitor Info ────────────────────────────────────────────────
    visitor_name = fields.Char(string='Visitor Name')
    visitor_email = fields.Char(string='Visitor Email')
    visitor_phone = fields.Char(string='Visitor Phone')
    visitor_company = fields.Char(string='Company')
    visitor_employee_size = fields.Selection([
        ('1-10', '1-10'),
        ('11-50', '11-50'),
        ('51-200', '51-200'),
        ('201-500', '201-500'),
        ('501-1000', '501-1000'),
        ('1000+', '1000+'),
    ], string='Employee Size')
    visitor_ip = fields.Char(string='IP Address')
    visitor_user_agent = fields.Char(string='User Agent')
    preferred_language = fields.Char(
        string='Preferred Language',
        help='Language used for AI responses in this session.',
    )

    # ── State ───────────────────────────────────────────────────────
    state = fields.Selection([
        ('active', 'Active'),
        ('closed', 'Closed'),
        ('escalated', 'Escalated'),
    ], string='Status', default='active', required=True, tracking=True)

    # ── Messages ────────────────────────────────────────────────────
    message_ids = fields.One2many(
        'perfecthr.chatbot.message', 'session_id',
        string='Messages',
    )
    message_count = fields.Integer(
        string='Message Count',
        compute='_compute_message_count',
        store=True,
    )

    # ── CRM Integration ────────────────────────────────────────────
    lead_id = fields.Many2one(
        'crm.lead', string='CRM Lead',
        ondelete='set null',
    )
    is_qualified = fields.Boolean(
        string='Qualified Lead',
        default=False,
    )
    qualification_score = fields.Integer(
        string='Qualification Score',
        default=0,
        help='AI-calculated score 0-100',
    )

    # ── Timestamps ──────────────────────────────────────────────────
    start_time = fields.Datetime(
        string='Started At',
        default=fields.Datetime.now,
        readonly=True,
    )
    end_time = fields.Datetime(string='Ended At')
    duration_minutes = fields.Float(
        string='Duration (min)',
        compute='_compute_duration',
        store=True,
    )

    # ── Source ──────────────────────────────────────────────────────
    source = fields.Selection([
        ('website', 'Website'),
        ('whatsapp', 'WhatsApp'),
    ], string='Source', default='website')

    # ── Computed ────────────────────────────────────────────────────
    @api.depends('session_token', 'visitor_name')
    def _compute_display_name(self):
        for rec in self:
            name = rec.visitor_name or 'Anonymous'
            token_short = (rec.session_token or '')[:8]
            rec.display_name = f"{name} ({token_short})"

    @api.depends('message_ids')
    def _compute_message_count(self):
        for rec in self:
            rec.message_count = len(rec.message_ids)

    @api.depends('start_time', 'end_time')
    def _compute_duration(self):
        for rec in self:
            if rec.start_time and rec.end_time:
                delta = rec.end_time - rec.start_time
                rec.duration_minutes = round(delta.total_seconds() / 60.0, 1)
            else:
                rec.duration_minutes = 0.0

    # ── Actions ─────────────────────────────────────────────────────
    def action_escalate(self):
        """Escalate session to human support."""
        self.ensure_one()
        self.write({
            'state': 'escalated',
        })
        # Create activity on the lead if exists
        if self.lead_id:
            self.lead_id.activity_schedule(
                'mail.mail_activity_data_todo',
                note=_(
                    'Chatbot session escalated. '
                    'Visitor requires human assistance.\n\n'
                    'Session: %s'
                ) % self.display_name,
                summary=_('Chatbot Escalation'),
            )
        return True

    def action_close(self):
        """Close the chat session."""
        self.ensure_one()
        self.write({
            'state': 'closed',
            'end_time': fields.Datetime.now(),
        })
        return True

    def action_view_lead(self):
        """Open the linked CRM lead."""
        self.ensure_one()
        if not self.lead_id:
            raise UserError(_('No CRM lead linked to this session.'))
        return {
            'type': 'ir.actions.act_window',
            'res_model': 'crm.lead',
            'res_id': self.lead_id.id,
            'view_mode': 'form',
            'target': 'current',
        }

    # ── Cron ────────────────────────────────────────────────────────
    @api.model
    def _cron_close_stale_sessions(self):
        """Close sessions inactive for more than 24 hours."""
        cutoff = fields.Datetime.subtract(fields.Datetime.now(), hours=24)
        stale = self.search([
            ('state', '=', 'active'),
            ('write_date', '<', cutoff),
        ])
        stale.write({
            'state': 'closed',
            'end_time': fields.Datetime.now(),
        })
        return True

    def _get_chat_transcript_html(self):
        """Generate HTML transcript of the conversation."""
        self.ensure_one()
        lines = []
        for msg in self.message_ids.sorted('timestamp'):
            role = 'User' if msg.role == 'user' else 'AI Assistant'
            lines.append(
                f'<p><strong>{role}</strong> '
                f'<small>({msg.timestamp})</small>:<br/>'
                f'{msg.content}</p>'
            )
        return ''.join(lines) or '<p><em>No messages</em></p>'

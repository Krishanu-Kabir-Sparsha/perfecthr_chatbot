# -*- coding: utf-8 -*-
"""
Lead Qualifier for Perfect HR Chatbot.

Analyzes conversation to determine if a visitor
is a qualified lead based on captured information.
"""
import re
import logging

_logger = logging.getLogger(__name__)

# Email regex
EMAIL_RE = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

# Phone regex (supports international formats)
PHONE_RE = re.compile(
    r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}'
)

# Employee size patterns
EMPLOYEE_SIZE_RE = re.compile(
    r'(\d+)\s*(?:employees?|staff|people|persons?|members?|workers?)',
    re.IGNORECASE,
)


class LeadQualifier:
    """Qualifies leads based on conversation data and visitor information."""

    # Scoring weights
    WEIGHTS = {
        'has_name': 15,
        'has_email': 20,
        'has_phone': 15,
        'has_company': 15,
        'has_employee_size': 10,
        'expressed_interest': 15,
        'demo_requested': 10,
    }

    def qualify(self, session_data, messages):
        """
        Determine lead qualification from session data and messages.

        Args:
            session_data: dict with visitor info fields
            messages: list of message dicts with 'role' and 'content'

        Returns:
            dict: {
                'score': int,           # 0-100
                'is_qualified': bool,   # True if score >= 40
                'signals': dict,        # Individual signal values
                'extracted': dict,      # Extracted info from conversation
            }
        """
        signals = {
            'has_name': False,
            'has_email': False,
            'has_phone': False,
            'has_company': False,
            'has_employee_size': False,
            'expressed_interest': False,
            'demo_requested': False,
        }

        extracted = {
            'name': session_data.get('visitor_name', ''),
            'email': session_data.get('visitor_email', ''),
            'phone': session_data.get('visitor_phone', ''),
            'company': session_data.get('visitor_company', ''),
            'employee_size': session_data.get('visitor_employee_size', ''),
        }

        # Check session data
        if extracted['name']:
            signals['has_name'] = True
        if extracted['email']:
            signals['has_email'] = True
        if extracted['phone']:
            signals['has_phone'] = True
        if extracted['company']:
            signals['has_company'] = True
        if extracted['employee_size']:
            signals['has_employee_size'] = True

        # Analyze messages for additional signals
        user_messages = [
            m['content'] for m in messages if m.get('role') == 'user'
        ]
        all_user_text = ' '.join(user_messages).lower()

        # Try to extract info from conversation if not in session
        if not signals['has_email']:
            emails = EMAIL_RE.findall(all_user_text)
            if emails:
                signals['has_email'] = True
                extracted['email'] = emails[0]

        if not signals['has_phone']:
            phones = PHONE_RE.findall(all_user_text)
            if phones:
                signals['has_phone'] = True
                extracted['phone'] = phones[0]

        if not signals['has_employee_size']:
            emp_match = EMPLOYEE_SIZE_RE.search(all_user_text)
            if emp_match:
                signals['has_employee_size'] = True
                count = int(emp_match.group(1))
                extracted['employee_size'] = self._categorize_size(count)

        if not signals['has_name']:
            name = self._extract_name(all_user_text)
            if name:
                signals['has_name'] = True
                extracted['name'] = name

        if not signals['has_company']:
            company = self._extract_company(all_user_text)
            if company:
                signals['has_company'] = True
                extracted['company'] = company

        # Interest signals
        interest_keywords = [
            'interested', 'want to buy', 'purchase', 'implement',
            'need', 'looking for', 'solution', 'we need', 'can you help',
            'want to use', 'how to get started', 'sign up',
        ]
        for kw in interest_keywords:
            if kw in all_user_text:
                signals['expressed_interest'] = True
                break

        # Demo request signals
        demo_keywords = [
            'demo', 'demonstration', 'schedule', 'book', 'show me',
            'try', 'trial', 'see it', 'walk me through',
        ]
        for kw in demo_keywords:
            if kw in all_user_text:
                signals['demo_requested'] = True
                break

        # Calculate score
        score = sum(
            self.WEIGHTS[signal]
            for signal, is_set in signals.items()
            if is_set
        )
        score = min(score, 100)

        return {
            'score': score,
            'is_qualified': score >= 40,
            'signals': signals,
            'extracted': extracted,
        }

    def _categorize_size(self, count):
        """Categorize employee count into selection values."""
        if count <= 10:
            return '1-10'
        elif count <= 50:
            return '11-50'
        elif count <= 200:
            return '51-200'
        elif count <= 500:
            return '201-500'
        elif count <= 1000:
            return '501-1000'
        else:
            return '1000+'

    def _extract_name(self, text):
        """Try to extract visitor name from text."""
        patterns = [
            r'(?:my name is|i\'m|i am|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'(?:name|Name)\s*[:=]\s*([A-Za-z]+(?:\s+[A-Za-z]+)?)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip().title()
        return ''

    def _extract_company(self, text):
        """Try to extract company name from text."""
        patterns = [
            r'(?:my company|our company|i work at|i\'m from|we are)\s+(?:is\s+)?([A-Za-z][\w\s&.-]{1,40})',
            r'(?:company|Company|organization)\s*[:=]\s*([A-Za-z][\w\s&.-]{1,40})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ''

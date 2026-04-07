# -*- coding: utf-8 -*-
"""
Intent Detector for Perfect HR Chatbot.

Rule-based + keyword intent classification (MVP).
Detects user intent from message text with a confidence score.
"""
import re
import logging

_logger = logging.getLogger(__name__)

# Intent definitions with keyword patterns
INTENT_PATTERNS = {
    'greeting': {
        'keywords': [
            'hello', 'hi', 'hey', 'good morning', 'good afternoon',
            'good evening', 'greetings', 'howdy', 'what\'s up',
            'assalamu alaikum', 'salam',
        ],
        'patterns': [
            r'^(hi|hello|hey)\b',
            r'^good\s+(morning|afternoon|evening)',
        ],
        'weight': 1.0,
    },
    'farewell': {
        'keywords': [
            'bye', 'goodbye', 'see you', 'thanks bye', 'thank you bye',
            'take care', 'later', 'have a good day',
        ],
        'patterns': [
            r'\b(bye|goodbye|see\s+you)\b',
        ],
        'weight': 1.0,
    },
    'product_inquiry': {
        'keywords': [
            'feature', 'module', 'payroll', 'attendance', 'leave',
            'recruitment', 'performance', 'employee', 'hr management',
            'human resource', 'how does it work', 'what can it do',
            'capabilities', 'functionality', 'offer', 'provide',
            'solution', 'system', 'software', 'platform',
            'perfect hr', 'tell me about', 'what is',
        ],
        'patterns': [
            r'\b(what|how|tell me|explain).*\b(feature|module|work|do)\b',
            r'\b(payroll|attendance|leave|recruit|performance)\b',
        ],
        'weight': 0.9,
    },
    'pricing': {
        'keywords': [
            'price', 'pricing', 'cost', 'how much', 'package',
            'plan', 'subscription', 'license', 'fee', 'budget',
            'affordable', 'expensive', 'cheap', 'discount',
            'free trial', 'trial', 'free',
        ],
        'patterns': [
            r'\b(how\s+much|price|cost|pric)\b',
            r'\b(free\s+trial|trial|subscription)\b',
        ],
        'weight': 0.95,
    },
    'demo_request': {
        'keywords': [
            'demo', 'demonstration', 'show me', 'walkthrough',
            'presentation', 'schedule', 'book', 'appointment',
            'meeting', 'call', 'try', 'test',
            'see it in action', 'interested',
        ],
        'patterns': [
            r'\b(demo|schedule|book|try|test)\b',
            r'(want|like|need)\s+to\s+(see|try|test)',
            r'\binterested\b',
        ],
        'weight': 0.95,
    },
    'support': {
        'keywords': [
            'help', 'support', 'issue', 'problem', 'error',
            'bug', 'fix', 'trouble', 'not working', 'broken',
            'complaint', 'assistance', 'stuck', 'confused',
        ],
        'patterns': [
            r'\b(help|support|issue|problem|error)\b',
            r'not\s+working',
        ],
        'weight': 0.85,
    },
    'lead_capture': {
        'keywords': [
            'contact', 'reach out', 'get in touch', 'call me',
            'email me', 'my name is', 'my company', 'i work at',
            'we have', 'employees', 'staff', 'team size',
            'our company', 'organization', 'we are looking',
        ],
        'patterns': [
            r'my\s+(name|company|email|phone)\s+(is|:)',
            r'\b\d+\s*(employees|staff|people)\b',
            r'(call|email|contact)\s+me',
        ],
        'weight': 0.9,
    },
    'comparison': {
        'keywords': [
            'compare', 'comparison', 'vs', 'versus', 'better than',
            'alternative', 'competitor', 'different from', 'advantage',
        ],
        'patterns': [
            r'\b(compare|vs|versus|better)\b',
        ],
        'weight': 0.85,
    },
    'human_handover': {
        'keywords': [
            'speak to human', 'talk to person', 'real person',
            'human agent', 'live chat', 'representative',
            'speak to someone', 'customer service', 'operator',
        ],
        'patterns': [
            r'\b(human|person|representative|agent|operator)\b',
            r'speak\s+to\s+(a\s+)?(human|person|someone)',
        ],
        'weight': 1.0,
    },
}


class IntentDetector:
    """Rule-based intent detection for chatbot messages."""

    def __init__(self):
        self.patterns = INTENT_PATTERNS
        # Compile regex patterns
        self._compiled = {}
        for intent, config in self.patterns.items():
            self._compiled[intent] = [
                re.compile(p, re.IGNORECASE)
                for p in config.get('patterns', [])
            ]

    def detect(self, text):
        """
        Detect the intent of a user message.

        Args:
            text: The user's message text.

        Returns:
            dict: {
                'intent': str,        # Detected intent name
                'confidence': float,  # Confidence score (0.0 - 1.0)
                'all_intents': list,  # All detected intents sorted by score
            }
        """
        if not text or not text.strip():
            return {
                'intent': 'unknown',
                'confidence': 0.0,
                'all_intents': [],
            }

        text_lower = text.lower().strip()
        text_words = set(re.findall(r'\w+', text_lower))

        scores = {}

        for intent, config in self.patterns.items():
            score = 0.0
            keywords = config.get('keywords', [])
            compiled_patterns = self._compiled.get(intent, [])
            weight = config.get('weight', 1.0)

            # Keyword matching
            keyword_hits = 0
            for kw in keywords:
                if kw in text_lower:
                    keyword_hits += 1

            if keywords:
                keyword_score = min(keyword_hits / max(len(keywords) * 0.15, 1), 1.0)
                score += keyword_score * 0.6

            # Regex pattern matching
            pattern_hits = 0
            for pattern in compiled_patterns:
                if pattern.search(text_lower):
                    pattern_hits += 1

            if compiled_patterns:
                pattern_score = min(pattern_hits / len(compiled_patterns), 1.0)
                score += pattern_score * 0.4

            # Apply weight
            score *= weight

            if score > 0.05:
                scores[intent] = round(min(score, 1.0), 3)

        if not scores:
            return {
                'intent': 'unknown',
                'confidence': 0.0,
                'all_intents': [],
            }

        # Sort by score
        sorted_intents = sorted(
            scores.items(), key=lambda x: x[1], reverse=True,
        )

        return {
            'intent': sorted_intents[0][0],
            'confidence': sorted_intents[0][1],
            'all_intents': [
                {'intent': k, 'confidence': v}
                for k, v in sorted_intents
            ],
        }

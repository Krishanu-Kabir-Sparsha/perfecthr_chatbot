# -*- coding: utf-8 -*-
{
    'name': 'Perfect HR - AI Chatbot',
    'version': '18.0.1.0.0',
    'category': 'Website/AI',
    'summary': 'AI-powered Sales & Support Chatbot for Perfect HR',
    'description': """
Perfect HR AI Chatbot
=====================
An AI-powered digital sales consultant and support assistant that provides:

* Intelligent customer interaction via floating chat widget
* RAG-based knowledge retrieval using Ollama (Mistral 7B / LLaMA 3 / Phi-3)
* Automated lead qualification and CRM integration
* Knowledge base management with auto-embedding pipeline
* Analytics dashboard with conversion tracking

**Requires**: Ollama running locally (http://localhost:11434)
    """,
    'author': 'Perfect HR',
    'website': 'https://www.perfecthr.com',
    'license': 'LGPL-3',
    'depends': [
        'base',
        'website',
        'crm',
        'mail',
    ],
    'data': [
        # Security
        'security/chatbot_security.xml',
        'security/ir.model.access.csv',
        # Data
        'data/chatbot_data.xml',
        'data/chatbot_cron.xml',
        # Views
        'views/chatbot_session_views.xml',
        'views/chatbot_knowledge_views.xml',
        'views/chatbot_analytics_views.xml',
        'views/chatbot_config_views.xml',
        'views/chatbot_menu.xml',
        'views/website_templates.xml',
    ],
    'assets': {
        'web.assets_frontend': [
            'perfecthr_chatbot/static/src/scss/chatbot.scss',
            'perfecthr_chatbot/static/src/js/chatbot_service.js',
            'perfecthr_chatbot/static/src/js/chatbot_message.js',
            'perfecthr_chatbot/static/src/js/chatbot_widget.js',
            'perfecthr_chatbot/static/src/xml/chatbot_templates.xml',
        ],
    },
    'images': ['static/description/icon.png'],
    'installable': True,
    'application': True,
    'auto_install': False,
}

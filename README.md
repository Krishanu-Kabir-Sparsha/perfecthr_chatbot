# Perfect HR — AI Chatbot

**AI-Powered Sales & Support Assistant for Odoo CE v18**

## Overview

An intelligent chatbot module that acts as a digital sales consultant for the Perfect HR product. 
It provides website visitors with instant, AI-driven answers about HR solutions, qualifies leads 
automatically, and integrates seamlessly with Odoo CRM.

## Features

### 💬 Chat Widget
- Premium floating widget with glassmorphism dark theme
- Mobile responsive (full-screen on small viewports)
- Typing indicator & smooth animations
- Session persistence across page navigations
- Lead capture form inline in chat

### 🧠 AI Engine (Ollama)
- Supports **Mistral 7B**, **LLaMA 3**, **Phi-3**
- 100% local — no cloud API keys needed
- Multi-turn conversation with context awareness
- RAG (Retrieval-Augmented Generation) with FAISS
- Intent detection for guided conversations
- Graceful fallback when Ollama is offline

### 📚 Knowledge Base
- Product modules, FAQs, Pricing, Case Studies
- Upload/edit/delete with versioning
- Auto-embedding pipeline (Ollama + sentence-transformers fallback)
- Kanban, list, and form views

### 🎯 Lead Capture & CRM
- Structured lead form (Name, Email, Phone, Company, Size)
- Auto-create CRM leads with chat transcript
- AI-based lead qualification scoring (0-100)
- Default sales team assignment
- Human handover with activity scheduling

### 📊 Analytics Dashboard
- Daily KPI computation (conversations, leads, conversion rate)
- Response time tracking
- Intent analysis & trends
- Graph, pivot, and list views
- Auto-computed via cron

### 🔐 Security
- Role-based access (User / Admin groups)
- Public access limited to session/message creation
- CSRF protection on all endpoints
- All AI processing runs locally

## Requirements

| Requirement | Details |
|-------------|---------|
| Odoo | Community Edition v18 |
| Ollama | [ollama.ai](https://ollama.ai) running on server |
| AI Model | `ollama pull mistral` (or llama3, phi3) |
| RAG (optional) | `pip install faiss-cpu sentence-transformers` |
| Embedding Model | `ollama pull nomic-embed-text` |

## Installation

1. Copy `perfecthr_chatbot` folder to your Odoo addons directory
2. Restart Odoo server
3. Go to **Apps** → Search "Perfect HR" → Install
4. Go to **Settings** → **AI Chatbot** → Configure Ollama URL
5. Click **Test Connection** → **Pull Model**
6. Add knowledge base articles
7. Visit your website — the chat widget appears!

## Configuration

Navigate to **Settings → AI Chatbot**:
- **Ollama Server URL**: Default `http://localhost:11434`
- **LLM Model**: Mistral 7B / LLaMA 3 / Phi-3
- **Embedding Model**: Default `nomic-embed-text`
- **Greeting Message**: Customize the welcome text
- **System Prompt**: Control AI personality & behavior
- **Sales Team**: Default team for chatbot leads

## Module Structure

```
perfecthr_chatbot/
├── controllers/     # JSON-RPC endpoints
├── models/          # Odoo models (Session, Message, Knowledge, Analytics)
├── services/        # AI engine, RAG, intent detection, lead qualification
├── views/           # Backend views & website template
├── security/        # Groups & ACLs
├── data/            # Default data & cron jobs
└── static/          # OWL widget, SCSS, templates
```

## License

LGPL-3

## Author

Perfect HR

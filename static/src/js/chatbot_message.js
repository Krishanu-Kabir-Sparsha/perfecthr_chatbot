/** @odoo-module **/

import { Component, markup } from "@odoo/owl";

function escapeHtml(text) {
    return text.replace(/[&<>"]|'/g, (char) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
    }[char]));
}

function formatInline(text) {
    const escaped = escapeHtml(text);
    return escaped
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/(^|[\s>])\*([^*]+?)\*(?=[\s<]|$)/g, '$1<em>$2</em>');
}

function renderMessageHtml(content) {
    let text = content || '';

    const decodedText = document.createElement('textarea');
    decodedText.innerHTML = text;
    text = decodedText.value;

    text = text
        .replace(/\r\n/g, '\n')
        .replace(/<br\s*\/?\s*>/gi, '\n')
        .replace(/<\/p>/gi, '\n\n')
        .replace(/<\/li>/gi, '\n')
        .replace(/<li[^>]*>/gi, '- ')
        .replace(/<\/?(ul|ol)[^>]*>/gi, '\n');

    const lines = text.split('\n');
    const html = [];
    let inList = false;

    const closeList = () => {
        if (inList) {
            html.push('</ul>');
            inList = false;
        }
    };

    for (const rawLine of lines) {
        const line = rawLine.trim();
        if (!line) {
            closeList();
            continue;
        }

        const bulletMatch = line.match(/^[-•*]\s+(.+)$/);
        if (bulletMatch) {
            if (!inList) {
                html.push('<ul>');
                inList = true;
            }
            html.push(`<li>${formatInline(bulletMatch[1])}</li>`);
            continue;
        }

        closeList();
        html.push(`<p>${formatInline(line)}</p>`);
    }

    closeList();
    return markup(html.join(''));
}

/**
 * ChatbotMessage — renders a single chat message bubble.
 */
export class ChatbotMessage extends Component {
    static template = "perfecthr_chatbot.ChatbotMessage";
    static props = {
        role: { type: String },
        content: { type: String },
        timestamp: { type: String, optional: true },
        intent: { type: String, optional: true },
    };

    get isUser() {
        return this.props.role === 'user';
    }

    get isAssistant() {
        return this.props.role === 'assistant';
    }

    get formattedTime() {
        if (!this.props.timestamp) return '';
        try {
            const date = new Date(this.props.timestamp);
            return date.toLocaleTimeString([], {
                hour: '2-digit',
                minute: '2-digit',
            });
        } catch {
            return '';
        }
    }

    get formattedContent() {
        return renderMessageHtml(this.props.content || '');
    }
}

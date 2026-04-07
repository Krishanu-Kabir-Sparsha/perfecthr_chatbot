/** @odoo-module **/

import { Component } from "@odoo/owl";

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
        let text = this.props.content || '';
        // Convert markdown-like formatting
        // Bold: **text** → <strong>text</strong>
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        // Italic: *text* → <em>text</em>
        text = text.replace(/(^|[^*])\*([^*]+?)\*(?!\*)/g, '$1<em>$2</em>');
        // Bullet points: • or - at start of line
        text = text.replace(/^[•\-]\s+(.+)$/gm, '<li>$1</li>');
        // Wrap consecutive <li> in <ul>
        text = text.replace(/((?:<li>.*<\/li>\s*)+)/g, '<ul>$1</ul>');
        // Line breaks
        text = text.replace(/\n/g, '<br/>');
        return text;
    }
}

/** @odoo-module **/

import { Component, useState, useRef, onMounted, onWillUnmount, whenReady } from "@odoo/owl";
import { mountComponent } from "@web/env";
import { ChatbotMessage } from "./chatbot_message";
import { ChatbotService } from "./chatbot_service";

const STORAGE_KEY = 'perfecthr_chatbot_session';

/**
 * PerfectHR Chatbot Widget — main floating chat component.
 */
export class ChatbotWidget extends Component {
    static template = "perfecthr_chatbot.ChatbotWidget";
    static components = { ChatbotMessage };

    setup() {
        this.chatService = new ChatbotService();
        this.messageListRef = useRef("messageList");

        this.state = useState({
            isOpen: false,
            isMinimized: false,
            messages: [],
            inputText: '',
            isTyping: false,
            sessionToken: null,
            leadCaptured: false,
            showLeadForm: false,
            leadForm: {
                name: '',
                email: '',
                phone: '',
                company: '',
                employee_size: '',
            },
            leadFormErrors: {},
            qualificationScore: 0,
            isQualified: false,
            suggestLeadCapture: false,
            connectionError: false,
            unreadCount: 0,
        });

        onMounted(() => {
            this._restoreSession();
        });

        onWillUnmount(() => {
            this._saveSession();
        });
    }

    // ── Session Management ──────────────────────────────────────────
    async _restoreSession() {
        try {
            const saved = localStorage.getItem(STORAGE_KEY);
            if (saved) {
                const data = JSON.parse(saved);
                if (data.sessionToken) {
                    this.state.sessionToken = data.sessionToken;
                    this.state.leadCaptured = data.leadCaptured || false;
                    // Restore history
                    const result = await this.chatService.getHistory(data.sessionToken);
                    if (result.status === 'success' && result.messages) {
                        this.state.messages = result.messages.map(m => ({
                            role: m.role,
                            content: m.content,
                            timestamp: m.timestamp,
                            intent: m.intent,
                        }));
                    }
                }
            }
        } catch (e) {
            console.warn('[PerfectHR Chatbot] Failed to restore session:', e);
        }
    }

    _saveSession() {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify({
                sessionToken: this.state.sessionToken,
                leadCaptured: this.state.leadCaptured,
            }));
        } catch (e) {
            console.warn('[PerfectHR Chatbot] Failed to save session:', e);
        }
    }

    // ── Chat Actions ────────────────────────────────────────────────
    async toggleChat() {
        this.state.isOpen = !this.state.isOpen;
        this.state.isMinimized = false;

        if (this.state.isOpen) {
            this.state.unreadCount = 0;

            if (!this.state.sessionToken) {
                await this._startNewSession();
            }

            // Auto-scroll after render
            setTimeout(() => this._scrollToBottom(), 100);
        }

        this._saveSession();
    }

    minimizeChat() {
        this.state.isMinimized = true;
        this.state.isOpen = false;
    }

    closeChat() {
        this.state.isOpen = false;
        this.state.isMinimized = false;
    }

    async _startNewSession() {
        this.state.isTyping = true;
        this.state.connectionError = false;

        const result = await this.chatService.startSession();

        this.state.isTyping = false;

        if (result.status === 'success') {
            this.state.sessionToken = result.session_token;
            this.state.messages = [{
                role: 'assistant',
                content: result.greeting,
                timestamp: new Date().toISOString(),
                intent: 'greeting',
            }];
            this._saveSession();
        } else {
            this.state.connectionError = true;
            this.state.messages = [{
                role: 'assistant',
                content: "I'm having trouble connecting. Please refresh the page and try again.",
                timestamp: new Date().toISOString(),
                intent: 'error',
            }];
        }
    }

    async sendMessage() {
        const text = this.state.inputText.trim();
        if (!text || this.state.isTyping) return;

        // Add user message
        this.state.messages.push({
            role: 'user',
            content: text,
            timestamp: new Date().toISOString(),
        });
        this.state.inputText = '';
        this.state.isTyping = true;
        this.state.connectionError = false;
        this._scrollToBottom();

        // Send to backend
        const result = await this.chatService.sendMessage(
            this.state.sessionToken,
            text,
        );

        this.state.isTyping = false;

        if (result.status === 'success') {
            this.state.messages.push({
                role: 'assistant',
                content: result.response,
                timestamp: new Date().toISOString(),
                intent: result.intent,
            });

            this.state.qualificationScore = result.qualification_score || 0;
            this.state.isQualified = result.is_qualified || false;

            if (result.suggest_lead_capture && !this.state.leadCaptured) {
                this.state.suggestLeadCapture = true;
            }

            if (!this.state.isOpen) {
                this.state.unreadCount++;
            }
        } else {
            this.state.messages.push({
                role: 'assistant',
                content: result.error || "Sorry, I couldn't process your message. Please try again.",
                timestamp: new Date().toISOString(),
                intent: 'error',
            });
        }

        this._scrollToBottom();
        this._saveSession();
    }

    onInputKeydown(ev) {
        if (ev.key === 'Enter' && !ev.shiftKey) {
            ev.preventDefault();
            this.sendMessage();
        }
    }

    onInputChange(ev) {
        this.state.inputText = ev.target.value;
    }

    // ── Lead Form ──────────────────────────────────────────────────
    showLeadForm() {
        this.state.showLeadForm = true;
        this.state.suggestLeadCapture = false;
        this._scrollToBottom();
    }

    dismissLeadSuggestion() {
        this.state.suggestLeadCapture = false;
    }

    onLeadFieldChange(ev) {
        const field = ev.target.dataset.field;
        if (field) {
            this.state.leadForm[field] = ev.target.value;
            // Clear error for this field
            if (this.state.leadFormErrors[field]) {
                delete this.state.leadFormErrors[field];
            }
        }
    }

    async submitLeadForm() {
        // Validate
        const errors = {};
        if (!this.state.leadForm.name.trim()) {
            errors.name = 'Name is required';
        }
        if (!this.state.leadForm.email.trim()) {
            errors.email = 'Email is required';
        } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(this.state.leadForm.email)) {
            errors.email = 'Invalid email address';
        }

        if (Object.keys(errors).length > 0) {
            this.state.leadFormErrors = errors;
            return;
        }

        this.state.isTyping = true;
        const result = await this.chatService.submitLead(
            this.state.sessionToken,
            this.state.leadForm,
        );
        this.state.isTyping = false;

        if (result.status === 'success' || result.status === 'exists') {
            this.state.leadCaptured = true;
            this.state.showLeadForm = false;
            this.state.messages.push({
                role: 'assistant',
                content: result.message || "Thank you! Our team will reach out to you soon. 🎉",
                timestamp: new Date().toISOString(),
                intent: 'lead_captured',
            });
        } else {
            this.state.messages.push({
                role: 'assistant',
                content: result.error || "Sorry, something went wrong. Please try again.",
                timestamp: new Date().toISOString(),
                intent: 'error',
            });
        }

        this._scrollToBottom();
        this._saveSession();
    }

    cancelLeadForm() {
        this.state.showLeadForm = false;
    }

    // ── New Session ────────────────────────────────────────────────
    async startNewConversation() {
        this.state.messages = [];
        this.state.sessionToken = null;
        this.state.leadCaptured = false;
        this.state.showLeadForm = false;
        this.state.qualificationScore = 0;
        this.state.isQualified = false;
        this.state.suggestLeadCapture = false;
        localStorage.removeItem(STORAGE_KEY);
        await this._startNewSession();
    }

    // ── Helpers ─────────────────────────────────────────────────────
    _scrollToBottom() {
        setTimeout(() => {
            const el = this.messageListRef.el;
            if (el) {
                el.scrollTop = el.scrollHeight;
            }
        }, 50);
    }
}

// ── Auto-mount on website pages ────────────────────────────────────
whenReady(() => {
    const container = document.getElementById('perfecthr-chatbot-root');
    if (container) {
        mountComponent(ChatbotWidget, container);
    }
});

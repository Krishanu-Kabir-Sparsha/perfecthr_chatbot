/** @odoo-module **/

/**
 * Chatbot RPC Service — handles all backend communication.
 */
export class ChatbotService {
    constructor() {
        this.baseUrl = '';
    }

    async _jsonRpc(endpoint, params = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    jsonrpc: '2.0',
                    method: 'call',
                    params: params,
                    id: Math.floor(Math.random() * 1000000),
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error.message || 'RPC Error');
            }

            return data.result;
        } catch (error) {
            console.error(`[PerfectHR Chatbot] RPC failed: ${endpoint}`, error);
            return {
                status: 'error',
                error: 'Connection failed. Please try again.',
            };
        }
    }

    async startSession() {
        return await this._jsonRpc('/perfecthr_chatbot/start');
    }

    async sendMessage(sessionToken, message) {
        return await this._jsonRpc('/perfecthr_chatbot/message', {
            session_token: sessionToken,
            message: message,
        });
    }

    async submitLead(sessionToken, leadData) {
        return await this._jsonRpc('/perfecthr_chatbot/submit_lead', {
            session_token: sessionToken,
            name: leadData.name || '',
            email: leadData.email || '',
            phone: leadData.phone || '',
            company: leadData.company || '',
            employee_size: leadData.employee_size || '',
        });
    }

    async getHistory(sessionToken) {
        return await this._jsonRpc('/perfecthr_chatbot/history', {
            session_token: sessionToken,
        });
    }

    async submitFeedback(sessionToken, messageId, feedback, note = '') {
        return await this._jsonRpc('/perfecthr_chatbot/feedback', {
            session_token: sessionToken,
            message_id: messageId,
            feedback: feedback,
            note: note,
        });
    }

    async getWidgetConfig() {
        return await this._jsonRpc('/perfecthr_chatbot/config');
    }
}

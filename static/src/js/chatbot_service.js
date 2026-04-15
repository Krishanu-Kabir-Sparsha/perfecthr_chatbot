/** @odoo-module **/

/**
 * Chatbot RPC Service — handles all backend communication.
 *
 * Uses async polling to handle long-running AI model responses:
 * 1. sendMessage() fires the request and returns quickly
 * 2. If status is 'processing', pollForResponse() polls until ready
 */
export class ChatbotService {
    constructor() {
        this.baseUrl = '';
        this.requestTimeoutMs = 30000; // 30s for quick endpoints
        this.pollIntervalMs = 4000;    // Poll every 4 seconds
        this.maxPollTimeMs = 1200000;  // Max 20 minutes of polling
    }

    async _jsonRpc(endpoint, params = {}, timeoutMs = null) {
        const url = `${this.baseUrl}${endpoint}`;
        const controller = new AbortController();
        const timeout = timeoutMs || this.requestTimeoutMs;
        const timeoutId = setTimeout(() => controller.abort(), timeout);
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                signal: controller.signal,
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
            if (error.name === 'AbortError') {
                return {
                    status: 'error',
                    error: 'Request timed out. The AI is still processing your message.',
                    code: 'timeout',
                };
            }
            console.error(`[PerfectHR Chatbot] RPC failed: ${endpoint}`, error);
            return {
                status: 'error',
                error: 'Connection failed. Please try again.',
            };
        } finally {
            clearTimeout(timeoutId);
        }
    }

    async startSession() {
        return await this._jsonRpc('/perfecthr_chatbot/start');
    }

    /**
     * Send a message to the chatbot.
     * Returns immediately with {status: 'processing', user_message_id: X}
     * for AI responses, or {status: 'success'} for instant responses.
     */
    async sendMessage(sessionToken, message) {
        return await this._jsonRpc('/perfecthr_chatbot/message', {
            session_token: sessionToken,
            message: message,
        });
    }

    /**
     * Poll for an AI response that is being generated asynchronously.
     * Keeps polling every pollIntervalMs until the response is ready
     * or maxPollTimeMs is exceeded.
     *
     * @param {string} sessionToken - The session token
     * @param {number} userMessageId - The user message ID to poll for
     * @param {function} onTick - Optional callback called each poll cycle with elapsed time
     * @returns {Promise<object>} The final response
     */
    async pollForResponse(sessionToken, userMessageId, onTick = null) {
        const startTime = Date.now();

        while (true) {
            const elapsed = Date.now() - startTime;

            // Safety: stop polling after maxPollTimeMs
            if (elapsed > this.maxPollTimeMs) {
                return {
                    status: 'error',
                    error: 'The AI is taking unusually long. Please try a shorter question.',
                    code: 'poll_timeout',
                };
            }

            // Call the tick callback with elapsed time
            if (onTick) {
                onTick(elapsed);
            }

            // Wait before polling
            await new Promise(resolve => setTimeout(resolve, this.pollIntervalMs));

            // Poll for result
            const result = await this._jsonRpc('/perfecthr_chatbot/poll', {
                session_token: sessionToken,
                user_message_id: userMessageId,
            }, 15000); // 15s timeout for poll requests

            // Still processing — continue polling
            if (result.status === 'processing') {
                continue;
            }

            // Got a result (success or error)
            return result;
        }
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

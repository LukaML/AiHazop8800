// api.js - API client for the HAZOP GUI

const API = {
    baseUrl: '',

    // Generic fetch wrapper
    async request(method, path, body = null) {
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json',
            },
        };

        if (body) {
            options.body = JSON.stringify(body);
        }

        const response = await fetch(this.baseUrl + path, options);

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        return response.json();
    },

    // Settings endpoints
    async setApiKey(provider, apiKey) {
        return this.request('POST', '/api/settings/api-key', {
            provider,
            api_key: apiKey,
        });
    },

    async getProviders() {
        return this.request('GET', '/api/settings/providers');
    },

    // Analysis endpoints
    async startAnalysis(params) {
        return this.request('POST', '/api/analysis/start', params);
    },

    async getAnalysisStatus(runId) {
        return this.request('GET', `/api/analysis/${runId}/status`);
    },

    async getAnalysisResults(runId) {
        return this.request('GET', `/api/analysis/${runId}/results`);
    },

    async uploadFunctions(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(this.baseUrl + '/api/analysis/upload-functions', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        return response.json();
    },

    async uploadRagFiles(files) {
        const formData = new FormData();
        for (const file of files) {
            formData.append('files', file);
        }

        const response = await fetch(this.baseUrl + '/api/analysis/upload-rag', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        return response.json();
    },

    // Row endpoints
    async editRow(runId, rowId, data) {
        return this.request('PUT', `/api/rows/${runId}/${rowId}/edit`, data);
    },

    async rateRow(runId, rowId, rating) {
        return this.request('PUT', `/api/rows/${runId}/${rowId}/rate`, { rating });
    },

    async regenerateRows(runId, rowIds, scope, suggestion) {
        return this.request('POST', `/api/rows/${runId}/regenerate`, {
            row_ids: rowIds,
            scope,
            suggestion,
        });
    },

    // Import endpoint
    async importHtml(file) {
        const formData = new FormData();
        formData.append('file', file);
        const response = await fetch(this.baseUrl + '/api/analysis/import-html', {
            method: 'POST',
            body: formData,
        });
        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Import failed' }));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        return response.json();
    },

    // Export endpoints
    getExportUrl(runId) {
        return `${this.baseUrl}/api/export/${runId}/csv`;
    },

    getExportHtmlUrl(runId) {
        return `${this.baseUrl}/api/export/${runId}/html`;
    },
};

// Make globally available
window.API = API;

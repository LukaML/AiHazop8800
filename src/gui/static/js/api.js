// api.js - API client for the AI-HAZOP-8800 GUI

const API = {
    baseUrl: '',

    async request(method, path, body = null) {
        const options = { method, headers: { 'Content-Type': 'application/json' } };
        if (body) options.body = JSON.stringify(body);
        const response = await fetch(this.baseUrl + path, options);
        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        return response.json();
    },

    async _upload(path, file, fieldName = 'file') {
        const formData = new FormData();
        formData.append(fieldName, file);
        const response = await fetch(this.baseUrl + path, { method: 'POST', body: formData });
        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        return response.json();
    },

    // Settings
    setApiKey(provider, apiKey) { return this.request('POST', '/api/settings/api-key', { provider, api_key: apiKey }); },
    getProviders() { return this.request('GET', '/api/settings/providers'); },

    // Catalogues
    getCatalogues() { return this.request('GET', '/api/catalogues'); },
    loadExample(name) { return this.request('GET', `/api/analysis/example/${encodeURIComponent(name)}`); },

    // Analysis
    startAnalysis(params) { return this.request('POST', '/api/analysis/start', params); },
    getAnalysisStatus(runId) { return this.request('GET', `/api/analysis/${runId}/status`); },
    getAnalysisResults(runId) { return this.request('GET', `/api/analysis/${runId}/results`); },
    uploadContext(file) { return this._upload('/api/analysis/upload-context', file); },
    importHtml(file) { return this._upload('/api/analysis/import-html', file); },

    // Rows
    editRow(runId, rowId, fields) { return this.request('PUT', `/api/rows/${runId}/${rowId}/edit`, { fields }); },
    rateRow(runId, rowId, rating) { return this.request('PUT', `/api/rows/${runId}/${rowId}/rate`, { rating }); },
    regenerateRows(runId, rowIds, scope, suggestion) {
        return this.request('POST', `/api/rows/${runId}/regenerate`, { row_ids: rowIds, scope, suggestion });
    },

    // Export
    getExportUrl(runId) { return `${this.baseUrl}/api/export/${runId}/csv`; },
    getExportHtmlUrl(runId) { return `${this.baseUrl}/api/export/${runId}/html`; },
};

window.API = API;

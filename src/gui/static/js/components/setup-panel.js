// setup-panel.js - Setup panel: provider/key/model, analysis contexts, guidewords, run.

const SetupPanel = {
    init() {
        this.cacheElements();
        this.bindEvents();
        this.loadProviderStatus();
        this.loadCatalogues();
        this.filterModelsForProvider(AppState.provider || 'openai');
        this.renderContexts();
    },

    cacheElements() {
        this.providerGroup = document.getElementById('provider-group');
        this.apiKeyInput = document.getElementById('api-key');
        this.setKeyBtn = document.getElementById('set-key-btn');
        this.keyStatus = document.getElementById('key-status');
        this.modelSelect = document.getElementById('model-select');

        this.contextsContainer = document.getElementById('contexts-container');
        this.contextCount = document.getElementById('context-count');
        this.addContextBtn = document.getElementById('add-context-btn');
        this.exampleSelect = document.getElementById('example-select');
        this.contextFile = document.getElementById('context-file');
        this.uploadContextBtn = document.getElementById('upload-context-btn');
        this.guidewordsContainer = document.getElementById('guidewords-container');
        this.gwSelectAll = document.getElementById('gw-select-all');

        this.notes = document.getElementById('notes');
        this.runAnalysisBtn = document.getElementById('run-analysis-btn');

        this.resultsLoading = document.getElementById('results-loading');
        this.resultsLoadingText = document.getElementById('results-loading-text');
        this.configLoading = document.getElementById('config-loading');
        this.configLoadingText = document.getElementById('config-loading-text');
        this.pageBlocker = document.getElementById('page-blocker');

        this.importHtmlBtn = document.getElementById('import-html-btn');
        this.importHtmlFile = document.getElementById('import-html-file');
    },

    bindEvents() {
        this.providerGroup.addEventListener('change', (e) => {
            if (e.target.name === 'provider') this.onProviderChange(e.target.value);
        });
        this.setKeyBtn.addEventListener('click', () => this.onSetApiKey());
        this.apiKeyInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') this.onSetApiKey(); });

        this.addContextBtn.addEventListener('click', () => {
            AppState.contexts.push(emptyContext());
            this.renderContexts();
        });
        this.uploadContextBtn.addEventListener('click', () => this.contextFile.click());
        this.contextFile.addEventListener('change', (e) => this.onContextFileChange(e));
        this.exampleSelect.addEventListener('change', (e) => this.onLoadExample(e));

        if (this.gwSelectAll) this.gwSelectAll.addEventListener('change', (e) => {
            this.guidewordsContainer.querySelectorAll('.gw-checkbox').forEach(cb => { cb.checked = e.target.checked; });
        });

        this.runAnalysisBtn.addEventListener('click', () => this.onRunAnalysis());
        this.importHtmlBtn.addEventListener('click', () => this.importHtmlFile.click());
        this.importHtmlFile.addEventListener('change', (e) => this.onImportHtml(e));
    },

    async loadProviderStatus() {
        try {
            const response = await API.getProviders();
            for (const p of response.providers) AppState.keyStatus[p.provider] = p.key_configured;
            this.updateKeyStatusDisplay();
        } catch (error) { console.error('Failed to load provider status:', error); }
    },

    async loadCatalogues() {
        try {
            const cat = await API.getCatalogues();
            AppState.catalogues = cat;
            // Examples
            this.exampleSelect.innerHTML = '<option value="">Load example...</option>' +
                cat.examples.map(n => `<option value="${this.esc(n)}">${this.esc(n)}</option>`).join('');
            // Guidewords
            this.guidewordsContainer.innerHTML = cat.guidewords.map(g => `
                <label class="flex items-center gap-1 text-xs cursor-pointer">
                    <input type="checkbox" class="accent-blue-600 gw-checkbox" value="${this.esc(g.id)}">
                    <span title="${this.esc(g.meaning || '')}">${this.esc(String(g.id).replace(/_/g, ' '))}</span>
                </label>`).join('');
            if (this.gwSelectAll && this.gwSelectAll.checked) {
                this.guidewordsContainer.querySelectorAll('.gw-checkbox').forEach(cb => { cb.checked = true; });
            }
            this.renderContexts();  // re-render now that class/aspect options exist
        } catch (error) { console.error('Failed to load catalogues:', error); }
    },

    renderContexts() {
        const classes = AppState.catalogues.component_classes || [];
        const aspects = AppState.catalogues.aspects || [];
        const opts = (arr, sel) => arr.map(v =>
            `<option value="${this.esc(v)}" ${v === sel ? 'selected' : ''}>${this.esc(v)}</option>`).join('');

        this.contextsContainer.innerHTML = AppState.contexts.map((c, i) => `
            <div class="border border-slate-200 rounded-lg p-3 flex flex-col gap-2 bg-slate-50" data-idx="${i}">
                <div class="flex items-center justify-between">
                    <span class="text-xs font-semibold text-slate-500">Component ${i + 1}</span>
                    ${AppState.contexts.length > 1 ? `<button type="button" class="remove-ctx text-xs text-red-500 hover:text-red-700">remove</button>` : ''}
                </div>
                <input type="text" data-field="component" placeholder="Component (e.g. Camera Object Detection)" value="${this.esc(c.component)}"
                    class="px-2 py-1.5 border border-slate-300 rounded text-sm bg-white">
                <select data-field="component_class" class="px-2 py-1.5 border border-slate-300 rounded text-sm bg-white">
                    <option value="">Select class...</option>${opts(classes, c.component_class)}
                </select>
                <select data-field="aspect" class="px-2 py-1.5 border border-slate-300 rounded text-sm bg-white">
                    <option value="">Select aspect...</option>${opts(aspects, c.aspect)}
                </select>
                <textarea data-field="odd" rows="2" placeholder="ODD (operational design domain)..."
                    class="px-2 py-1.5 border border-slate-300 rounded text-sm bg-white resize-y">${this.esc(c.odd)}</textarea>
                <textarea data-field="scenario" rows="2" placeholder="Scenario..."
                    class="px-2 py-1.5 border border-slate-300 rounded text-sm bg-white resize-y">${this.esc(c.scenario)}</textarea>
            </div>`).join('');

        // Wire inputs
        this.contextsContainer.querySelectorAll('[data-idx]').forEach(card => {
            const idx = parseInt(card.dataset.idx);
            card.querySelectorAll('[data-field]').forEach(el => {
                el.addEventListener('input', () => { AppState.contexts[idx][el.dataset.field] = el.value; });
                el.addEventListener('change', () => { AppState.contexts[idx][el.dataset.field] = el.value; });
            });
            const rm = card.querySelector('.remove-ctx');
            if (rm) rm.addEventListener('click', () => {
                AppState.contexts.splice(idx, 1);
                this.renderContexts();
            });
        });

        const n = AppState.contexts.length;
        this.contextCount.textContent = `${n} component${n !== 1 ? 's' : ''}`;
    },

    onProviderChange(provider) {
        AppState.provider = provider;
        this.filterModelsForProvider(provider);
        this.updateKeyStatusDisplay();
    },

    filterModelsForProvider(provider) {
        const labels = { openai: 'OpenAI', gemini: 'Gemini', groq: 'Groq' };
        const selectedLabel = labels[provider] || 'OpenAI';
        this.modelSelect.querySelectorAll('optgroup').forEach(og => {
            const match = og.label === selectedLabel;
            og.style.display = match ? '' : 'none';
            og.querySelectorAll('option').forEach(opt => opt.disabled = !match);
        });
        const cur = this.modelSelect.querySelector(`option[value="${this.modelSelect.value}"]`);
        if (cur && cur.disabled) this.modelSelect.value = '';
    },

    async onSetApiKey() {
        const key = this.apiKeyInput.value.trim();
        if (!key) { alert('Please enter an API key'); return; }
        try {
            this.setKeyBtn.disabled = true;
            await API.setApiKey(AppState.provider, key);
            AppState.keyStatus[AppState.provider] = true;
            this.apiKeyInput.value = '';
            this.updateKeyStatusDisplay();
        } catch (error) { alert('Failed to set API key: ' + error.message); }
        finally { this.setKeyBtn.disabled = false; }
    },

    updateKeyStatusDisplay() {
        const configured = AppState.isKeyConfigured();
        this.keyStatus.textContent = configured ? 'Key configured' : 'Key not set';
        this.keyStatus.className = 'key-status ' + (configured ? 'configured' : 'not-configured');
    },

    async onContextFileChange(e) {
        const file = e.target.files[0];
        if (!file) return;
        try {
            const response = await API.uploadContext(file);
            AppState.contexts = response.contexts.map(c => ({ ...c }));
            if (AppState.contexts.length === 0) AppState.contexts = [emptyContext()];
            this.renderContexts();
        } catch (error) { alert('Failed to parse context file: ' + error.message); }
        this.contextFile.value = '';
    },

    async onLoadExample(e) {
        const name = e.target.value;
        if (!name) return;
        try {
            const response = await API.loadExample(name);
            AppState.contexts = response.contexts.map(c => ({ ...c }));
            this.renderContexts();
        } catch (error) { alert('Failed to load example: ' + error.message); }
        e.target.value = '';
    },

    selectedGuidewords() {
        return [...this.guidewordsContainer.querySelectorAll('.gw-checkbox:checked')].map(cb => cb.value);
    },

    validContexts() {
        return AppState.contexts.filter(c =>
            c.component && c.component_class && c.aspect && c.odd && c.scenario);
    },

    async onRunAnalysis() {
        if (!AppState.isKeyConfigured()) { alert('Please configure API key for ' + AppState.provider); return; }
        const contexts = this.validContexts();
        if (contexts.length === 0) {
            alert('Please fill in at least one complete context (all 5 fields).');
            return;
        }

        AppState.model = this.modelSelect.value;
        AppState.notes = this.notes.value;
        const guidewords = this.selectedGuidewords();

        const params = {
            provider: AppState.provider,
            model: AppState.model || null,
            contexts,
            notes: AppState.notes,
            guidewords: guidewords.length ? guidewords : null,
        };

        try {
            this.runAnalysisBtn.disabled = true;
            this.showLoading('Running pipeline...');
            const response = await API.startAnalysis(params);
            AppState.runId = response.run_id;
            AppState.status = response.status;
            this.startPolling();
        } catch (error) {
            alert('Failed to start analysis: ' + error.message);
            this.hideLoading();
            this.runAnalysisBtn.disabled = false;
        }
    },

    showLoading(text) {
        if (this.resultsLoading) this.resultsLoading.classList.remove('hidden');
        if (this.resultsLoadingText) this.resultsLoadingText.textContent = text;
        if (this.configLoading) this.configLoading.classList.remove('hidden');
        if (this.configLoadingText) this.configLoadingText.textContent = text;
        if (this.pageBlocker) this.pageBlocker.classList.remove('hidden');
    },
    hideLoading() {
        if (this.resultsLoading) this.resultsLoading.classList.add('hidden');
        if (this.configLoading) this.configLoading.classList.add('hidden');
        if (this.pageBlocker) this.pageBlocker.classList.add('hidden');
    },

    startPolling() {
        if (AppState.pollInterval) clearInterval(AppState.pollInterval);
        AppState.pollInterval = setInterval(async () => {
            try {
                const status = await API.getAnalysisStatus(AppState.runId);
                AppState.status = status.status;
                const text = status.progress || `Status: ${status.status}`;
                if (this.resultsLoadingText) this.resultsLoadingText.textContent = text;
                if (this.configLoadingText) this.configLoadingText.textContent = text;
                if (status.status === 'completed') this.onAnalysisComplete();
                else if (status.status === 'failed') this.onAnalysisFailed(status.error);
            } catch (error) { console.error('Polling error:', error); }
        }, 2000);
    },

    async onAnalysisComplete() {
        clearInterval(AppState.pollInterval);
        AppState.pollInterval = null;
        try {
            const results = await API.getAnalysisResults(AppState.runId);
            AppState.rows = results.rows;
            this.hideLoading();
            this.runAnalysisBtn.disabled = false;
            ResultsTable.render(results.rows);
            document.getElementById('export-csv-btn').disabled = false;
            document.getElementById('export-html-btn').disabled = false;
            document.getElementById('results-info').textContent =
                `${results.rows.length} rows | ${results.components} component(s) | ${results.provider} / ${results.model}`;
        } catch (error) {
            alert('Failed to load results: ' + error.message);
            this.hideLoading();
            this.runAnalysisBtn.disabled = false;
        }
    },

    async onImportHtml(e) {
        const file = e.target.files[0];
        if (!file) return;
        try {
            const response = await API.importHtml(file);
            AppState.runId = response.run_id;
            AppState.status = 'completed';
            const results = await API.getAnalysisResults(response.run_id);
            AppState.rows = results.rows;
            ResultsTable.render(results.rows);
            document.getElementById('export-csv-btn').disabled = false;
            document.getElementById('export-html-btn').disabled = false;
            document.getElementById('results-info').textContent =
                `${results.rows.length} rows | imported from HTML`;
        } catch (error) { alert('Failed to import HTML: ' + error.message); }
        this.importHtmlFile.value = '';
    },

    onAnalysisFailed(error) {
        clearInterval(AppState.pollInterval);
        AppState.pollInterval = null;
        alert('Analysis failed: ' + (error || 'Unknown error'));
        this.hideLoading();
        this.runAnalysisBtn.disabled = false;
    },

    esc(text) {
        const div = document.createElement('div');
        div.textContent = text == null ? '' : String(text);
        return div.innerHTML;
    },
};

window.SetupPanel = SetupPanel;

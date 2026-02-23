// setup-panel.js - Setup panel component

const SetupPanel = {
    init() {
        this.cacheElements();
        this.bindEvents();
        this.loadProviderStatus();
        // Filter models for initial provider (openai by default)
        this.filterModelsForProvider(AppState.provider || 'openai');
    },

    cacheElements() {
        this.providerGroup = document.getElementById('provider-group');
        this.apiKeySection = document.getElementById('api-key-section');
        this.apiKeyInput = document.getElementById('api-key');
        this.setKeyBtn = document.getElementById('set-key-btn');
        this.keyStatus = document.getElementById('key-status');
        this.modelSelect = document.getElementById('model-select');

        this.functionsFile = document.getElementById('functions-file');
        this.uploadFunctionsBtn = document.getElementById('upload-functions-btn');
        this.functionsFilename = document.getElementById('functions-filename');
        this.functionsText = document.getElementById('functions-text');
        this.functionCount = document.getElementById('function-count');

        this.ragEnabled = document.getElementById('rag-enabled');
        this.ragOptions = document.getElementById('rag-options');
        this.ragEmbedder = document.getElementById('rag-embedder');
        this.ragFiles = document.getElementById('rag-files');
        this.uploadRagBtn = document.getElementById('upload-rag-btn');
        this.ragFilenames = document.getElementById('rag-filenames');

        this.notes = document.getElementById('notes');
        this.maxDevs = document.getElementById('max-devs');

        this.runAnalysisBtn = document.getElementById('run-analysis-btn');

        // Results loading overlay
        this.resultsLoading = document.getElementById('results-loading');
        this.resultsLoadingText = document.getElementById('results-loading-text');

        // Config loading overlay
        this.configLoading = document.getElementById('config-loading');
        this.configLoadingText = document.getElementById('config-loading-text');

        // Page blocker (blocks all interaction during pipeline)
        this.pageBlocker = document.getElementById('page-blocker');

        // Import HTML
        this.importHtmlBtn = document.getElementById('import-html-btn');
        this.importHtmlFile = document.getElementById('import-html-file');
    },

    bindEvents() {
        // Provider selection
        this.providerGroup.addEventListener('change', (e) => {
            if (e.target.name === 'provider') {
                this.onProviderChange(e.target.value);
            }
        });

        // API key
        this.setKeyBtn.addEventListener('click', () => this.onSetApiKey());
        this.apiKeyInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.onSetApiKey();
        });

        // Functions upload
        this.uploadFunctionsBtn.addEventListener('click', () => this.functionsFile.click());
        this.functionsFile.addEventListener('change', (e) => this.onFunctionsFileChange(e));
        this.functionsText.addEventListener('input', () => this.onFunctionsTextChange());

        // RAG toggle
        this.ragEnabled.addEventListener('change', () => this.onRagToggle());
        this.uploadRagBtn.addEventListener('click', () => this.ragFiles.click());
        this.ragFiles.addEventListener('change', (e) => this.onRagFilesChange(e));

        // Run analysis
        this.runAnalysisBtn.addEventListener('click', () => this.onRunAnalysis());

        // Import HTML
        this.importHtmlBtn.addEventListener('click', () => this.importHtmlFile.click());
        this.importHtmlFile.addEventListener('change', (e) => this.onImportHtml(e));
    },

    async loadProviderStatus() {
        try {
            const response = await API.getProviders();
            for (const provider of response.providers) {
                AppState.keyStatus[provider.provider] = provider.key_configured;
            }
            this.updateKeyStatusDisplay();
        } catch (error) {
            console.error('Failed to load provider status:', error);
        }
    },

    onProviderChange(provider) {
        AppState.provider = provider;

        // All providers require API key now (openai, gemini, groq)
        this.apiKeySection.classList.remove('hidden');

        // Filter models for selected provider
        this.filterModelsForProvider(provider);

        this.updateKeyStatusDisplay();
    },

    filterModelsForProvider(provider) {
        // Map provider to optgroup label
        const providerLabels = {
            'openai': 'OpenAI',
            'gemini': 'Gemini',
            'groq': 'Groq'
        };

        const selectedLabel = providerLabels[provider] || 'OpenAI';

        // Get all optgroups
        const optgroups = this.modelSelect.querySelectorAll('optgroup');

        // Show/hide optgroups based on provider
        optgroups.forEach(optgroup => {
            if (optgroup.label === selectedLabel) {
                optgroup.style.display = '';
                // Enable all options in this group
                optgroup.querySelectorAll('option').forEach(opt => opt.disabled = false);
            } else {
                optgroup.style.display = 'none';
                // Disable all options in hidden groups
                optgroup.querySelectorAll('option').forEach(opt => opt.disabled = true);
            }
        });

        // Reset to default if current selection is not available
        const currentValue = this.modelSelect.value;
        const currentOption = this.modelSelect.querySelector(`option[value="${currentValue}"]`);
        if (currentOption && currentOption.disabled) {
            this.modelSelect.value = '';  // Reset to "Default for provider"
        }
    },

    async onSetApiKey() {
        const key = this.apiKeyInput.value.trim();
        if (!key) {
            alert('Please enter an API key');
            return;
        }

        try {
            this.setKeyBtn.disabled = true;
            await API.setApiKey(AppState.provider, key);

            AppState.keyStatus[AppState.provider] = true;
            this.apiKeyInput.value = '';
            this.updateKeyStatusDisplay();
        } catch (error) {
            alert('Failed to set API key: ' + error.message);
        } finally {
            this.setKeyBtn.disabled = false;
        }
    },

    updateKeyStatusDisplay() {
        const configured = AppState.isKeyConfigured();
        this.keyStatus.textContent = configured ? 'Key configured' : 'Key not set';
        this.keyStatus.className = 'key-status ' + (configured ? 'configured' : 'not-configured');
    },

    async onFunctionsFileChange(e) {
        const file = e.target.files[0];
        if (!file) return;

        try {
            const response = await API.uploadFunctions(file);
            AppState.setFunctions(response.functions);
            this.functionsFilename.textContent = file.name;
            this.functionsText.value = response.functions.join('\n');
            this.updateFunctionCount();
        } catch (error) {
            alert('Failed to parse functions file: ' + error.message);
        }
    },

    onFunctionsTextChange() {
        const text = this.functionsText.value;
        const funcs = text.split('\n').filter(f => f.trim());
        AppState.setFunctions(funcs);
        this.functionsFilename.textContent = '';
        this.updateFunctionCount();
    },

    updateFunctionCount() {
        const count = AppState.functions.length;
        this.functionCount.textContent = `${count} function${count !== 1 ? 's' : ''}`;
    },

    onRagToggle() {
        AppState.ragEnabled = this.ragEnabled.checked;
        if (AppState.ragEnabled) {
            this.ragOptions.classList.remove('hidden');
        } else {
            this.ragOptions.classList.add('hidden');
        }
    },

    async onRagFilesChange(e) {
        const files = Array.from(e.target.files);
        if (!files.length) return;

        try {
            const response = await API.uploadRagFiles(files);
            AppState.ragPaths = response.file_paths;
            this.ragFilenames.textContent = files.map(f => f.name).join(', ');
        } catch (error) {
            alert('Failed to upload RAG files: ' + error.message);
        }
    },

    async onRunAnalysis() {
        // Validate
        if (!AppState.isKeyConfigured()) {
            alert('Please configure API key for ' + AppState.provider);
            return;
        }

        if (AppState.functions.length === 0) {
            alert('Please enter at least one function');
            return;
        }

        // Gather parameters
        AppState.model = this.modelSelect.value;
        AppState.notes = this.notes.value;
        AppState.maxDevsPerGw = parseInt(this.maxDevs.value) || 2;
        AppState.ragEmbedder = this.ragEmbedder.value;

        const params = {
            provider: AppState.provider,
            model: AppState.model || null,
            functions: AppState.functions,
            notes: AppState.notes,
            max_devs_per_gw: AppState.maxDevsPerGw,
            rag_enabled: AppState.ragEnabled,
            rag_paths: AppState.ragPaths,
            rag_embedder: AppState.ragEmbedder,
        };

        try {
            this.runAnalysisBtn.disabled = true;

            // Show loading overlay in results section
            if (this.resultsLoading) this.resultsLoading.classList.remove('hidden');
            if (this.resultsLoadingText) this.resultsLoadingText.textContent = 'Running pipeline...';

            // Show loading overlay in config panel
            if (this.configLoading) this.configLoading.classList.remove('hidden');
            if (this.configLoadingText) this.configLoadingText.textContent = 'Running pipeline...';

            // Block entire page during analysis
            if (this.pageBlocker) this.pageBlocker.classList.remove('hidden');

            const response = await API.startAnalysis(params);
            AppState.runId = response.run_id;
            AppState.status = response.status;

            // Start polling
            this.startPolling();
        } catch (error) {
            alert('Failed to start analysis: ' + error.message);
            if (this.resultsLoading) this.resultsLoading.classList.add('hidden');
            if (this.configLoading) this.configLoading.classList.add('hidden');
            if (this.pageBlocker) this.pageBlocker.classList.add('hidden');
            this.runAnalysisBtn.disabled = false;
        }
    },

    startPolling() {
        if (AppState.pollInterval) {
            clearInterval(AppState.pollInterval);
        }

        AppState.pollInterval = setInterval(async () => {
            try {
                const status = await API.getAnalysisStatus(AppState.runId);
                AppState.status = status.status;
                const progressText = status.progress || `Status: ${status.status}`;
                if (this.resultsLoadingText) this.resultsLoadingText.textContent = progressText;
                if (this.configLoadingText) this.configLoadingText.textContent = progressText;

                if (status.status === 'completed') {
                    this.onAnalysisComplete();
                } else if (status.status === 'failed') {
                    this.onAnalysisFailed(status.error);
                }
            } catch (error) {
                console.error('Polling error:', error);
            }
        }, 2000);
    },

    async onAnalysisComplete() {
        clearInterval(AppState.pollInterval);
        AppState.pollInterval = null;

        try {
            const results = await API.getAnalysisResults(AppState.runId);
            AppState.rows = results.rows;

            if (this.resultsLoading) this.resultsLoading.classList.add('hidden');
            if (this.configLoading) this.configLoading.classList.add('hidden');
            if (this.pageBlocker) this.pageBlocker.classList.add('hidden');
            this.runAnalysisBtn.disabled = false;

            // Update results table
            ResultsTable.render(results.rows);
            document.getElementById('export-csv-btn').disabled = false;
            document.getElementById('export-html-btn').disabled = false;
            document.getElementById('results-info').textContent =
                `${results.rows.length} rows | ${results.provider} / ${results.model}`;
        } catch (error) {
            alert('Failed to load results: ' + error.message);
            if (this.resultsLoading) this.resultsLoading.classList.add('hidden');
            if (this.configLoading) this.configLoading.classList.add('hidden');
            if (this.pageBlocker) this.pageBlocker.classList.add('hidden');
            this.runAnalysisBtn.disabled = false;
        }
    },

    async onImportHtml(e) {
        const file = e.target.files[0];
        if (!file) return;
        try {
            const response = await API.importHtml(file);
            // Load the imported run's results
            AppState.runId = response.run_id;
            AppState.status = 'completed';
            const results = await API.getAnalysisResults(response.run_id);
            AppState.rows = results.rows;
            ResultsTable.render(results.rows);
            document.getElementById('export-csv-btn').disabled = false;
            document.getElementById('export-html-btn').disabled = false;
            document.getElementById('results-info').textContent =
                `${results.rows.length} rows | imported from HTML`;
        } catch (error) {
            alert('Failed to import HTML: ' + error.message);
        }
        this.importHtmlFile.value = '';  // reset so same file can be re-imported
    },

    onAnalysisFailed(error) {
        clearInterval(AppState.pollInterval);
        AppState.pollInterval = null;

        alert('Analysis failed: ' + (error || 'Unknown error'));
        if (this.resultsLoading) this.resultsLoading.classList.add('hidden');
        if (this.configLoading) this.configLoading.classList.add('hidden');
        if (this.pageBlocker) this.pageBlocker.classList.add('hidden');
        this.runAnalysisBtn.disabled = false;
    },
};

// Make globally available
window.SetupPanel = SetupPanel;

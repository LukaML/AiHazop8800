// state.js - Global state management for the HAZOP GUI

const AppState = {
    // Current run
    runId: null,
    status: null,

    // Configuration
    provider: 'openai',
    model: '',
    functions: [],
    notes: '',
    maxDevsPerGw: 2,
    ragEnabled: false,
    ragEmbedder: 'local',
    ragPaths: [],

    // Results
    rows: [],

    // Key status per provider
    keyStatus: {
        openai: false,
        gemini: false,
        ollama: true, // Ollama doesn't need a key
        vllm: false,
    },

    // Currently editing row
    editingRowId: null,

    // Polling interval
    pollInterval: null,

    // Selected rows for bulk operations
    selectedRows: new Set(),

    // Reset to initial state
    reset() {
        this.runId = null;
        this.status = null;
        this.rows = [];
        this.editingRowId = null;
        this.selectedRows = new Set();
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    },

    // Toggle row selection
    toggleRowSelection(rowId, selected) {
        if (selected) {
            this.selectedRows.add(rowId);
        } else {
            this.selectedRows.delete(rowId);
        }
        this.onSelectionChange();
    },

    // Select all rows
    selectAllRows() {
        this.rows.forEach(row => this.selectedRows.add(row.row_id));
        this.onSelectionChange();
    },

    // Clear all selections
    clearSelection() {
        this.selectedRows = new Set();
        this.onSelectionChange();
    },

    // Get selected row IDs as array
    getSelectedRowIds() {
        return [...this.selectedRows];
    },

    // Check if a row is selected
    isRowSelected(rowId) {
        return this.selectedRows.has(rowId);
    },

    // Callback for selection changes (set by ResultsTable)
    onSelectionChange() {
        // Will be overridden by ResultsTable
    },

    // Set functions from parsed input
    setFunctions(funcs) {
        this.functions = funcs.filter(f => f.trim());
    },

    // Get row by ID
    getRow(rowId) {
        return this.rows.find(r => r.row_id === rowId);
    },

    // Update row in state
    updateRow(rowData) {
        const idx = this.rows.findIndex(r => r.row_id === rowData.row_id);
        if (idx >= 0) {
            this.rows[idx] = rowData;
        }
    },

    // Check if required key is configured
    isKeyConfigured() {
        if (this.provider === 'ollama') return true;
        return this.keyStatus[this.provider] || false;
    }
};

// Make globally available
window.AppState = AppState;

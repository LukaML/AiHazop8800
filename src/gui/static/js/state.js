// state.js - Global state for the AI-HAZOP-8800 GUI

const AppState = {
    runId: null,
    status: null,

    // Configuration
    provider: 'openai',
    model: '',
    notes: '',
    contexts: [emptyContext()],   // list of {component, component_class, aspect, odd, scenario}
    guidewords: [],               // selected override ids ([] = catalogue default)

    // Catalogue data for form dropdowns
    catalogues: { component_classes: [], aspects: [], guidewords: [], examples: [] },

    // Results (RowData objects with original/final maps)
    rows: [],

    keyStatus: { openai: false, gemini: false, groq: false },

    editingRowId: null,
    pollInterval: null,
    selectedRows: new Set(),

    reset() {
        this.runId = null;
        this.status = null;
        this.rows = [];
        this.editingRowId = null;
        this.selectedRows = new Set();
        if (this.pollInterval) { clearInterval(this.pollInterval); this.pollInterval = null; }
    },

    toggleRowSelection(rowId, selected) {
        if (selected) this.selectedRows.add(rowId); else this.selectedRows.delete(rowId);
        this.onSelectionChange();
    },
    selectAllRows() { this.rows.forEach(r => this.selectedRows.add(r.row_id)); this.onSelectionChange(); },
    clearSelection() { this.selectedRows = new Set(); this.onSelectionChange(); },
    getSelectedRowIds() { return [...this.selectedRows]; },
    isRowSelected(rowId) { return this.selectedRows.has(rowId); },
    onSelectionChange() {},

    getRow(rowId) { return this.rows.find(r => r.row_id === rowId); },
    updateRow(rowData) {
        const idx = this.rows.findIndex(r => r.row_id === rowData.row_id);
        if (idx >= 0) this.rows[idx] = rowData;
    },

    isKeyConfigured() { return this.keyStatus[this.provider] || false; },
};

function emptyContext() {
    return { component: '', component_class: '', aspect: '', odd: '', scenario: '' };
}

window.AppState = AppState;
window.emptyContext = emptyContext;

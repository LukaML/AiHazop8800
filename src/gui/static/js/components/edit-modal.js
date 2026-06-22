// edit-modal.js - Edit side panel + bulk regenerate + changes diff (L1-L8).

const EditModal = {
    init() {
        this.overlay = document.getElementById('edit-modal');
        this.panel = document.getElementById('edit-panel');
        this.backdrop = document.getElementById('modal-backdrop');

        this.rowIdInput = document.getElementById('modal-row-id');
        this.componentInput = document.getElementById('modal-component');
        this.guidewordInput = document.getElementById('modal-guideword');
        this.fieldsContainer = document.getElementById('modal-fields');
        this.ratingSelect = document.getElementById('modal-rating');

        this.regenScope = document.getElementById('modal-regen-scope');
        this.suggestionInput = document.getElementById('modal-suggestion');
        this.regenBtn = document.getElementById('modal-regen-btn');
        this.regenLoading = document.getElementById('regen-loading');
        this.regenLoadingText = document.getElementById('regen-loading-text');

        this.closeBtn = document.getElementById('modal-close');
        this.cancelBtn = document.getElementById('modal-cancel-btn');
        this.saveBtn = document.getElementById('modal-save-btn');

        this.bindEvents();
    },

    bindEvents() {
        this.closeBtn.addEventListener('click', () => this.close());
        this.cancelBtn.addEventListener('click', () => this.close());
        this.backdrop.addEventListener('click', () => this.close());
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !this.overlay.classList.contains('hidden')) this.close();
        });
        this.saveBtn.addEventListener('click', () => this.onSave());
        this.regenBtn.addEventListener('click', () => this.onRegenerate());
    },

    esc(text) {
        const div = document.createElement('div');
        div.textContent = text == null ? '' : String(text);
        return div.innerHTML;
    },

    fieldInputHtml(def, final) {
        const v = final[def.key];
        const id = `mf-${def.key}`;
        const base = 'px-3 py-2 border border-slate-300 rounded-lg text-sm bg-slate-50 focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20';
        if (def.type === 'readonly') {
            return `<input type="text" id="${id}" readonly value="${this.esc(Fields.formatValue(final, def))}"
                class="px-3 py-2 border border-slate-200 rounded-lg text-sm bg-slate-100 text-slate-500">`;
        }
        if (def.type === 'bool') {
            return `<label class="flex items-center gap-2"><input type="checkbox" id="${id}" ${v ? 'checked' : ''} class="w-4 h-4 accent-blue-600"><span class="text-sm text-slate-600">Yes</span></label>`;
        }
        if (def.type === 'select') {
            const opts = (def.options || []).map(o => `<option value="${this.esc(o)}" ${o === v ? 'selected' : ''}>${this.esc(o)}</option>`).join('');
            return `<select id="${id}" class="${base}">${opts}</select>`;
        }
        if (def.type === 'list') {
            const text = Array.isArray(v) ? v.join('\n') : (v || '');
            return `<textarea id="${id}" rows="3" placeholder="One item per line" class="${base} resize-y">${this.esc(text)}</textarea>`;
        }
        // text
        return `<textarea id="${id}" rows="2" class="${base} resize-y">${this.esc(v == null ? '' : v)}</textarea>`;
    },

    buildFields(row) {
        const final = row.final || {};
        const groups = Fields.editGroups.filter(g => Fields.rowHasPhase(row, g.phase));
        this.fieldsContainer.innerHTML = groups.map(g => `
            <div class="border border-slate-200 rounded-lg overflow-hidden">
                <div class="bg-slate-100 px-3 py-2 text-sm font-semibold text-slate-700">${this.esc(g.title)}</div>
                <div class="p-3 flex flex-col gap-3">
                    ${g.fields.map(f => `
                        <div class="flex flex-col gap-1">
                            <label class="text-xs font-medium text-slate-500" for="mf-${f.key}">${this.esc(f.label)}</label>
                            ${this.fieldInputHtml(f, final)}
                        </div>`).join('')}
                </div>
            </div>`).join('');
    },

    fillScopes(row) {
        const avail = Fields.scopes.filter(s => Fields.rowHasPhase(row, s.value));
        this.regenScope.innerHTML = avail.map(s => `<option value="${s.value}">${this.esc(s.label)}</option>`).join('');
    },

    open(rowId) {
        const row = AppState.getRow(rowId);
        if (!row) { alert('Row not found'); return; }
        AppState.editingRowId = rowId;

        this.rowIdInput.value = row.display_id || row.row_id;
        this.componentInput.value = row.component || (row.final && row.final.component) || '';
        this.guidewordInput.value = String(row.guideword || (row.final && row.final.guideword) || '').replace(/_/g, ' ');
        this.ratingSelect.value = row.rating || 'unrated';
        this.suggestionInput.value = '';

        this.buildFields(row);
        this.fillScopes(row);

        this.overlay.classList.remove('hidden');
        requestAnimationFrame(() => this.panel.classList.add('open'));
    },

    close() {
        this.panel.classList.remove('open');
        setTimeout(() => { this.overlay.classList.add('hidden'); AppState.editingRowId = null; }, 300);
    },

    readField(def) {
        const el = document.getElementById(`mf-${def.key}`);
        if (!el) return undefined;
        if (def.type === 'bool') return el.checked;
        if (def.type === 'list') return el.value.split('\n').map(s => s.trim()).filter(Boolean);
        return el.value;
    },

    collectChangedFields(row) {
        const final = row.final || {};
        const changed = {};
        for (const g of Fields.editGroups) {
            if (!Fields.rowHasPhase(row, g.phase)) continue;
            for (const def of g.fields) {
                if (def.type === 'readonly') continue;
                const newVal = this.readField(def);
                if (newVal === undefined) continue;
                const oldVal = final[def.key];
                if (def.type === 'list') {
                    if (JSON.stringify(newVal) !== JSON.stringify(Array.isArray(oldVal) ? oldVal : [])) changed[def.key] = newVal;
                } else if (def.type === 'bool') {
                    if (Boolean(newVal) !== Boolean(oldVal)) changed[def.key] = newVal;
                } else if ((newVal || '') !== (oldVal == null ? '' : String(oldVal))) {
                    changed[def.key] = newVal;
                }
            }
        }
        return changed;
    },

    async onSave() {
        const rowId = AppState.editingRowId;
        if (!rowId) return;
        const row = AppState.getRow(rowId);
        const fields = this.collectChangedFields(row);

        try {
            this.saveBtn.disabled = true;
            if (Object.keys(fields).length > 0) {
                const resp = await API.editRow(AppState.runId, rowId, fields);
                AppState.updateRow(resp.row);
            }
            if (this.ratingSelect.value !== (row.rating || 'unrated')) {
                await API.rateRow(AppState.runId, rowId, this.ratingSelect.value);
                const updated = AppState.getRow(rowId);
                updated.rating = this.ratingSelect.value;
                AppState.updateRow(updated);
            }
            ResultsTable.updateRow(AppState.getRow(rowId));
            this.close();
        } catch (error) {
            alert('Failed to save changes: ' + error.message);
        } finally {
            this.saveBtn.disabled = false;
        }
    },

    async onRegenerate() {
        const rowId = AppState.editingRowId;
        if (!rowId) return;
        const scope = this.regenScope.value;
        if (!scope) { alert('No regeneration scope available for this row'); return; }
        const suggestion = this.suggestionInput.value;
        const oldFinal = { ...(AppState.getRow(rowId).final || {}) };

        try {
            this.regenBtn.disabled = true;
            this.regenBtn.textContent = 'Regenerating...';
            this.regenLoadingText.textContent = `Regenerating ${scope}...`;
            this.regenLoading.classList.remove('hidden');

            const response = await API.regenerateRows(AppState.runId, [rowId], scope, suggestion);
            for (const updated of response.rows) {
                AppState.updateRow(updated);
                ResultsTable.updateRow(updated);
            }
            const newRow = AppState.getRow(rowId);
            this.buildFields(newRow);
            this.fillScopes(newRow);
            ChangesModal.show(oldFinal, newRow);
        } catch (error) {
            alert('Failed to regenerate: ' + error.message);
        } finally {
            this.regenBtn.disabled = false;
            this.regenBtn.textContent = 'Regenerate with AI';
            this.regenLoading.classList.add('hidden');
        }
    },
};

// Bulk Regeneration Modal
const BulkRegenModal = {
    init() {
        this.modal = document.getElementById('bulk-regen-modal');
        this.backdrop = document.getElementById('bulk-modal-backdrop');
        this.infoSpan = document.getElementById('bulk-modal-info');
        this.scopeSelect = document.getElementById('bulk-regen-scope');
        this.suggestionInput = document.getElementById('bulk-suggestion');
        this.closeBtn = document.getElementById('bulk-modal-close');
        this.cancelBtn = document.getElementById('bulk-modal-cancel');
        this.confirmBtn = document.getElementById('bulk-modal-confirm');
        this.loadingOverlay = document.getElementById('bulk-regen-loading');
        this.loadingText = document.getElementById('bulk-regen-loading-text');
        this.bindEvents();
    },

    bindEvents() {
        this.closeBtn.addEventListener('click', () => this.close());
        this.cancelBtn.addEventListener('click', () => this.close());
        this.backdrop.addEventListener('click', () => this.close());
        this.confirmBtn.addEventListener('click', () => this.onConfirm());
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !this.modal.classList.contains('hidden')) this.close();
        });
    },

    esc(t) { const d = document.createElement('div'); d.textContent = t == null ? '' : String(t); return d.innerHTML; },

    open() {
        const count = AppState.selectedRows.size;
        if (count === 0) { alert('No rows selected'); return; }
        this.infoSpan.textContent = `Regenerating ${count} row${count > 1 ? 's' : ''}`;
        this.suggestionInput.value = '';
        // Scopes available across all selected rows (union).
        const ids = AppState.getSelectedRowIds();
        const avail = Fields.scopes.filter(s => ids.some(id => Fields.rowHasPhase(AppState.getRow(id), s.value)));
        this.scopeSelect.innerHTML = avail.map(s => `<option value="${s.value}">${this.esc(s.label)}</option>`).join('');
        this.modal.classList.remove('hidden');
    },

    close() { this.modal.classList.add('hidden'); },

    async onConfirm() {
        const rowIds = AppState.getSelectedRowIds();
        if (rowIds.length === 0) return;
        const scope = this.scopeSelect.value;
        const suggestion = this.suggestionInput.value;
        const oldMap = {};
        rowIds.forEach(id => { oldMap[id] = { ...(AppState.getRow(id).final || {}) }; });

        try {
            this.confirmBtn.disabled = true;
            this.confirmBtn.textContent = 'Regenerating...';
            const count = rowIds.length;
            if (this.loadingOverlay) this.loadingOverlay.classList.remove('hidden');
            if (this.loadingText) this.loadingText.textContent = `Regenerating ${count} row${count > 1 ? 's' : ''}...`;

            const response = await API.regenerateRows(AppState.runId, rowIds, scope, suggestion);
            for (const updated of response.rows) {
                AppState.updateRow(updated);
                ResultsTable.updateRow(updated);
            }
            AppState.clearSelection();
            ResultsTable.updateCheckboxes();
            this.close();
            ChangesModal.showBulk(oldMap, response.rows);
        } catch (error) {
            alert('Failed to regenerate: ' + error.message);
        } finally {
            this.confirmBtn.disabled = false;
            this.confirmBtn.textContent = 'Regenerate';
            if (this.loadingOverlay) this.loadingOverlay.classList.add('hidden');
        }
    },
};

// Changes Modal — diffs the worksheet columns before/after regeneration.
const ChangesModal = {
    init() {
        this.modal = document.getElementById('changes-modal');
        this.backdrop = document.getElementById('changes-backdrop');
        this.closeBtn = document.getElementById('changes-close');
        this.content = document.getElementById('changes-content');
        this.bindEvents();
    },

    bindEvents() {
        this.closeBtn.addEventListener('click', () => this.close());
        this.backdrop.addEventListener('click', () => this.close());
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !this.modal.classList.contains('hidden')) this.close();
        });
    },

    diff(oldFinal, newFinal) {
        const changes = [];
        for (const def of Fields.columns) {
            const o = Fields.formatValue(oldFinal, def);
            const n = Fields.formatValue(newFinal, def);
            if (o !== n) changes.push({ field: def.label, old: o, new: n });
        }
        return changes;
    },

    show(oldFinal, newRow, back = false) {
        const changes = this.diff(oldFinal, newRow.final || {});
        const backButton = back ? `<button onclick="ChangesModal.showBulkSummary()" class="mb-4 text-sm text-blue-600 hover:text-blue-800">&larr; Back to Summary</button>` : '';
        if (changes.length === 0) {
            this.content.innerHTML = backButton + `<p class="text-slate-600 text-sm">No changes detected.</p>`;
        } else {
            this.content.innerHTML = backButton + changes.map(c => `
                <div class="border border-slate-200 rounded-lg overflow-hidden">
                    <div class="bg-slate-100 px-3 py-2 text-sm font-medium text-slate-700">${this.esc(c.field)}</div>
                    <div class="p-3 space-y-2">
                        <div class="flex gap-2"><span class="text-xs font-medium text-red-600 w-14 shrink-0">Before:</span><span class="text-sm text-slate-600 line-through">${this.esc(c.old) || '<em class="text-slate-400">empty</em>'}</span></div>
                        <div class="flex gap-2"><span class="text-xs font-medium text-green-600 w-14 shrink-0">After:</span><span class="text-sm text-slate-800">${this.esc(c.new) || '<em class="text-slate-400">empty</em>'}</span></div>
                    </div>
                </div>`).join('');
        }
        this.modal.classList.remove('hidden');
    },

    showBulk(oldMap, newRows) {
        this.bulkOld = oldMap;
        this.bulkNew = newRows;
        const summaries = [];
        for (const newRow of newRows) {
            const changes = this.diff(oldMap[newRow.row_id] || {}, newRow.final || {});
            if (changes.length) summaries.push({ rowId: newRow.row_id, fields: changes.map(c => c.field) });
        }
        if (summaries.length === 0) {
            this.content.innerHTML = `<p class="text-slate-600 text-sm">No changes detected across ${newRows.length} rows.</p>`;
        } else {
            this.content.innerHTML = `<p class="text-slate-600 text-sm mb-4">${summaries.length} row(s) changed:</p>` +
                summaries.map(s => `
                    <div class="border border-slate-200 rounded-lg p-3 mb-2 cursor-pointer hover:bg-slate-50" onclick="ChangesModal.showDetail('${this.esc(s.rowId)}')">
                        <div class="font-medium text-slate-700 text-sm">${this.esc(s.rowId)}</div>
                        <div class="text-xs text-slate-500 mt-1">Changed: ${s.fields.join(', ')}</div>
                    </div>`).join('');
        }
        this.modal.classList.remove('hidden');
    },

    showDetail(rowId) {
        const newRow = this.bulkNew.find(r => r.row_id === rowId);
        if (newRow) this.show(this.bulkOld[rowId] || {}, newRow, true);
    },
    showBulkSummary() { this.showBulk(this.bulkOld, this.bulkNew); },

    close() { this.modal.classList.add('hidden'); },
    esc(t) { const d = document.createElement('div'); d.textContent = t == null ? '' : String(t); return d.innerHTML; },
};

window.EditModal = EditModal;
window.BulkRegenModal = BulkRegenModal;
window.ChangesModal = ChangesModal;

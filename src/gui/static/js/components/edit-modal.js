// edit-modal.js - Edit modal component (slide-in side panel)

const EditModal = {
    init() {
        this.overlay = document.getElementById('edit-modal');
        this.panel = document.getElementById('edit-panel');
        this.backdrop = document.getElementById('modal-backdrop');

        // Fields
        this.rowIdInput = document.getElementById('modal-row-id');
        this.functionInput = document.getElementById('modal-function');
        this.guidewordInput = document.getElementById('modal-guideword');
        this.deviationInput = document.getElementById('modal-deviation');
        this.causeInput = document.getElementById('modal-cause');
        this.effectInput = document.getElementById('modal-effect');
        this.dangerousCheckbox = document.getElementById('modal-dangerous');
        this.ratingSelect = document.getElementById('modal-rating');

        // Regeneration
        this.regenScope = document.getElementById('modal-regen-scope');
        this.suggestionInput = document.getElementById('modal-suggestion');
        this.regenBtn = document.getElementById('modal-regen-btn');

        // Regeneration loading overlay
        this.regenLoading = document.getElementById('regen-loading');
        this.regenLoadingText = document.getElementById('regen-loading-text');

        // Buttons
        this.closeBtn = document.getElementById('modal-close');
        this.cancelBtn = document.getElementById('modal-cancel-btn');
        this.saveBtn = document.getElementById('modal-save-btn');

        this.bindEvents();
    },

    bindEvents() {
        // Close modal
        this.closeBtn.addEventListener('click', () => this.close());
        this.cancelBtn.addEventListener('click', () => this.close());

        // Click backdrop to close
        this.backdrop.addEventListener('click', () => this.close());

        // Escape key to close
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !this.overlay.classList.contains('hidden')) {
                this.close();
            }
        });

        // Save changes
        this.saveBtn.addEventListener('click', () => this.onSave());

        // Regenerate
        this.regenBtn.addEventListener('click', () => this.onRegenerate());
    },

    open(rowId) {
        const row = AppState.getRow(rowId);
        if (!row) {
            alert('Row not found');
            return;
        }

        AppState.editingRowId = rowId;

        // Populate fields
        this.rowIdInput.value = row.row_id;
        this.functionInput.value = row.function;
        this.guidewordInput.value = row.guideword;
        this.deviationInput.value = row.deviation_final;
        this.causeInput.value = row.cause_final;
        this.effectInput.value = row.effect_final;
        this.dangerousCheckbox.checked = row.potentially_dangerous_final;
        this.ratingSelect.value = row.rating;

        // Clear regeneration fields
        this.suggestionInput.value = '';

        // Show modal with animation
        this.overlay.classList.remove('hidden');
        // Trigger reflow then add open class for animation
        requestAnimationFrame(() => {
            this.panel.classList.add('open');
        });
    },

    close() {
        this.panel.classList.remove('open');
        // Wait for animation to complete before hiding
        setTimeout(() => {
            this.overlay.classList.add('hidden');
            AppState.editingRowId = null;
        }, 300);
    },

    async onSave() {
        const rowId = AppState.editingRowId;
        if (!rowId) return;

        const data = {};

        // Check what changed
        const row = AppState.getRow(rowId);
        if (this.deviationInput.value !== row.deviation_final) {
            data.deviation = this.deviationInput.value;
        }
        if (this.causeInput.value !== row.cause_final) {
            data.cause = this.causeInput.value;
        }
        if (this.effectInput.value !== row.effect_final) {
            data.effect = this.effectInput.value;
        }
        if (this.dangerousCheckbox.checked !== row.potentially_dangerous_final) {
            data.potentially_dangerous = this.dangerousCheckbox.checked;
        }

        try {
            this.saveBtn.disabled = true;

            // Save field changes if any
            if (Object.keys(data).length > 0) {
                const editResponse = await API.editRow(AppState.runId, rowId, data);
                AppState.updateRow(editResponse.row);
            }

            // Save rating if changed
            if (this.ratingSelect.value !== row.rating) {
                await API.rateRow(AppState.runId, rowId, this.ratingSelect.value);
                const updatedRow = AppState.getRow(rowId);
                updatedRow.rating = this.ratingSelect.value;
                AppState.updateRow(updatedRow);
            }

            // Update table
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
        const suggestion = this.suggestionInput.value;

        // Store old values before regeneration
        const row = AppState.getRow(rowId);
        const oldValues = {
            deviation: row.deviation_final,
            cause: row.cause_final,
            effect: row.effect_final,
            potentially_dangerous: row.potentially_dangerous_final
        };

        try {
            this.regenBtn.disabled = true;
            this.regenBtn.textContent = 'Regenerating...';

            // Show loading overlay
            this.regenLoadingText.textContent = `Regenerating ${scope}...`;
            this.regenLoading.classList.remove('hidden');

            const response = await API.regenerateRows(AppState.runId, [rowId], scope, suggestion);

            // Update rows in state
            for (const updatedRow of response.rows) {
                AppState.updateRow(updatedRow);
                ResultsTable.updateRow(updatedRow);
            }

            // Update modal with new values
            const newRow = AppState.getRow(rowId);
            this.deviationInput.value = newRow.deviation_final;
            this.causeInput.value = newRow.cause_final;
            this.effectInput.value = newRow.effect_final;
            this.dangerousCheckbox.checked = newRow.potentially_dangerous_final;

            // Show changes popup
            ChangesModal.show(oldValues, newRow, scope);
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

        // Loading overlay elements
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
            if (e.key === 'Escape' && !this.modal.classList.contains('hidden')) {
                this.close();
            }
        });
    },

    open() {
        const selectedCount = AppState.selectedRows.size;
        if (selectedCount === 0) {
            alert('No rows selected');
            return;
        }

        this.infoSpan.textContent = `Regenerating ${selectedCount} row${selectedCount > 1 ? 's' : ''}`;
        this.suggestionInput.value = '';
        this.modal.classList.remove('hidden');
    },

    close() {
        this.modal.classList.add('hidden');
    },

    async onConfirm() {
        const rowIds = AppState.getSelectedRowIds();
        if (rowIds.length === 0) return;

        const scope = this.scopeSelect.value;
        const suggestion = this.suggestionInput.value;

        // Store old values for all rows BEFORE regeneration
        const oldValuesMap = {};
        for (const rowId of rowIds) {
            const row = AppState.getRow(rowId);
            oldValuesMap[rowId] = {
                deviation: row.deviation_final,
                cause: row.cause_final,
                effect: row.effect_final,
                potentially_dangerous: row.potentially_dangerous_final
            };
        }

        try {
            this.confirmBtn.disabled = true;
            this.confirmBtn.textContent = 'Regenerating...';

            // Show loading overlay
            const count = rowIds.length;
            if (this.loadingOverlay) this.loadingOverlay.classList.remove('hidden');
            if (this.loadingText) this.loadingText.textContent = `Regenerating ${count} row${count > 1 ? 's' : ''}...`;

            const response = await API.regenerateRows(AppState.runId, rowIds, scope, suggestion);

            // Update rows in state
            for (const updatedRow of response.rows) {
                AppState.updateRow(updatedRow);
                ResultsTable.updateRow(updatedRow);
            }

            // Clear selection after successful regeneration
            AppState.clearSelection();
            ResultsTable.updateCheckboxes();

            this.close();

            // Show changes popup for bulk regeneration
            ChangesModal.showBulk(oldValuesMap, response.rows, scope);
        } catch (error) {
            alert('Failed to regenerate: ' + error.message);
        } finally {
            this.confirmBtn.disabled = false;
            this.confirmBtn.textContent = 'Regenerate';
            if (this.loadingOverlay) this.loadingOverlay.classList.add('hidden');
        }
    },
};

// Changes Modal (shows what changed after regeneration)
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
            if (e.key === 'Escape' && !this.modal.classList.contains('hidden')) {
                this.close();
            }
        });
    },

    show(oldValues, newRow, scope) {
        const changes = [];

        // Check what changed based on scope
        if (scope === 'L1' || scope === 'L2' || scope === 'L3') {
            if (oldValues.deviation !== newRow.deviation_final) {
                changes.push({
                    field: 'Deviation',
                    old: oldValues.deviation,
                    new: newRow.deviation_final
                });
            }
        }

        if (scope === 'L1' || scope === 'L2') {
            if (oldValues.cause !== newRow.cause_final) {
                changes.push({
                    field: 'Cause',
                    old: oldValues.cause,
                    new: newRow.cause_final
                });
            }
        }

        if (scope === 'L1' || scope === 'L2' || scope === 'L3') {
            if (oldValues.effect !== newRow.effect_final) {
                changes.push({
                    field: 'Effect',
                    old: oldValues.effect,
                    new: newRow.effect_final
                });
            }
        }

        if (oldValues.potentially_dangerous !== newRow.potentially_dangerous_final) {
            changes.push({
                field: 'Potentially Dangerous',
                old: oldValues.potentially_dangerous ? 'Yes' : 'No',
                new: newRow.potentially_dangerous_final ? 'Yes' : 'No'
            });
        }

        // Add back button if viewing from bulk
        let backButton = '';
        if (this.showingBulkDetail) {
            backButton = `
                <button onclick="ChangesModal.showBulkSummary()"
                        class="mb-4 text-sm text-blue-600 hover:text-blue-800 flex items-center gap-1">
                    ← Back to Summary
                </button>
            `;
        }

        // Build content
        if (changes.length === 0) {
            this.content.innerHTML = backButton + `
                <p class="text-slate-600 text-sm">No changes detected. The regenerated content is the same as before.</p>
            `;
        } else {
            this.content.innerHTML = backButton + changes.map(change => `
                <div class="border border-slate-200 rounded-lg overflow-hidden">
                    <div class="bg-slate-100 px-3 py-2 text-sm font-medium text-slate-700">${this.escapeHtml(change.field)}</div>
                    <div class="p-3 space-y-2">
                        <div class="flex gap-2">
                            <span class="text-xs font-medium text-red-600 w-14 shrink-0">Before:</span>
                            <span class="text-sm text-slate-600 line-through">${this.escapeHtml(change.old) || '<em class="text-slate-400">empty</em>'}</span>
                        </div>
                        <div class="flex gap-2">
                            <span class="text-xs font-medium text-green-600 w-14 shrink-0">After:</span>
                            <span class="text-sm text-slate-800">${this.escapeHtml(change.new) || '<em class="text-slate-400">empty</em>'}</span>
                        </div>
                    </div>
                </div>
            `).join('');
        }

        this.modal.classList.remove('hidden');
    },

    close() {
        this.modal.classList.add('hidden');
    },

    showBulk(oldValuesMap, newRows, scope) {
        // Store for later use when clicking individual rows
        this.bulkOldValues = oldValuesMap;
        this.bulkNewRows = newRows;
        this.bulkScope = scope;

        // Build summary of changes across all rows
        let totalChanges = 0;
        const rowSummaries = [];

        for (const newRow of newRows) {
            const oldValues = oldValuesMap[newRow.row_id];
            const rowChanges = [];

            if (oldValues.deviation !== newRow.deviation_final) {
                rowChanges.push('Deviation');
                totalChanges++;
            }
            if (oldValues.cause !== newRow.cause_final) {
                rowChanges.push('Cause');
                totalChanges++;
            }
            if (oldValues.effect !== newRow.effect_final) {
                rowChanges.push('Effect');
                totalChanges++;
            }
            if (oldValues.potentially_dangerous !== newRow.potentially_dangerous_final) {
                rowChanges.push('Dangerous');
                totalChanges++;
            }

            if (rowChanges.length > 0) {
                rowSummaries.push({
                    rowId: newRow.row_id,
                    function: newRow.function,
                    changes: rowChanges
                });
            }
        }

        // Build content
        if (totalChanges === 0) {
            this.content.innerHTML = `
                <p class="text-slate-600 text-sm">No changes detected across ${newRows.length} rows.</p>
            `;
        } else {
            this.content.innerHTML = `
                <p class="text-slate-600 text-sm mb-4">${totalChanges} field(s) changed across ${rowSummaries.length} row(s):</p>
                ${rowSummaries.map(summary => `
                    <div class="border border-slate-200 rounded-lg p-3 mb-2 cursor-pointer hover:bg-slate-50 transition-colors"
                         onclick="ChangesModal.showDetailForRow('${this.escapeHtml(summary.rowId)}')">
                        <div class="font-medium text-slate-700 text-sm">${this.escapeHtml(summary.rowId)} - ${this.escapeHtml(summary.function)}</div>
                        <div class="text-xs text-slate-500 mt-1">Changed: ${summary.changes.join(', ')}</div>
                        <div class="text-xs text-blue-500 mt-1">Click to see details</div>
                    </div>
                `).join('')}
            `;
        }

        this.modal.classList.remove('hidden');
    },

    showDetailForRow(rowId) {
        const oldValues = this.bulkOldValues[rowId];
        const newRow = this.bulkNewRows.find(r => r.row_id === rowId);
        if (oldValues && newRow) {
            this.showingBulkDetail = true;
            this.show(oldValues, newRow, this.bulkScope);
        }
    },

    showBulkSummary() {
        this.showingBulkDetail = false;
        this.showBulk(this.bulkOldValues, this.bulkNewRows, this.bulkScope);
    },

    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
};

// Make globally available
window.EditModal = EditModal;
window.BulkRegenModal = BulkRegenModal;
window.ChangesModal = ChangesModal;

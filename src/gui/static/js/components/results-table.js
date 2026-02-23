// results-table.js - Results table component with multi-row selection

const ResultsTable = {
    init() {
        this.tbody = document.getElementById('results-body');
        this.exportBtn = document.getElementById('export-csv-btn');
        this.exportHtmlBtn = document.getElementById('export-html-btn');
        this.selectAllCheckbox = document.getElementById('select-all-checkbox');
        this.bulkActionBar = document.getElementById('bulk-action-bar');
        this.selectedCountSpan = document.getElementById('selected-count');
        this.clearSelectionBtn = document.getElementById('clear-selection-btn');
        this.bulkRegenBtn = document.getElementById('bulk-regen-btn');

        this.exportBtn.addEventListener('click', () => this.onExport());
        this.exportHtmlBtn.addEventListener('click', () => this.onExportHtml());

        // Select all checkbox
        this.selectAllCheckbox.addEventListener('change', (e) => {
            if (e.target.checked) {
                AppState.selectAllRows();
            } else {
                AppState.clearSelection();
            }
            this.updateCheckboxes();
        });

        // Clear selection button
        this.clearSelectionBtn.addEventListener('click', () => {
            AppState.clearSelection();
            this.selectAllCheckbox.checked = false;
            this.updateCheckboxes();
        });

        // Bulk regenerate button
        this.bulkRegenBtn.addEventListener('click', () => {
            BulkRegenModal.open();
        });

        // Set up selection change callback
        AppState.onSelectionChange = () => this.updateBulkActionBar();
    },

    render(rows) {
        if (!rows || rows.length === 0) {
            this.tbody.innerHTML = `
                <tr class="empty-row">
                    <td colspan="9" class="px-3 py-10 text-center text-slate-400">No results yet. Configure and run an analysis.</td>
                </tr>
            `;
            this.selectAllCheckbox.checked = false;
            this.updateBulkActionBar();
            return;
        }

        // Reset function index for alternating colors
        this.currentFunctionIndex = 0;
        this.lastFunction = null;

        this.tbody.innerHTML = rows.map((row, index) => this.renderRow(row, index, rows)).join('');

        // Add click handlers for checkboxes and row clicks
        this.tbody.querySelectorAll('tr').forEach(tr => {
            const rowId = tr.dataset.rowId;
            if (!rowId) return;

            const checkbox = tr.querySelector('input[type="checkbox"]');

            // Checkbox change handler
            checkbox.addEventListener('change', (e) => {
                e.stopPropagation();
                AppState.toggleRowSelection(rowId, e.target.checked);
            });

            // Row click handler (but not on checkbox)
            tr.addEventListener('click', (e) => {
                // Don't open modal if clicking checkbox
                if (e.target.type === 'checkbox') return;
                EditModal.open(rowId);
            });
        });

        this.updateCheckboxes();
        this.updateBulkActionBar();
    },

    renderRow(row, index, rows) {
        // Determine if this is a new function group for alternating colors
        const isNewFunction = row.function !== this.lastFunction;
        if (isNewFunction) {
            this.currentFunctionIndex = (this.currentFunctionIndex || 0) + 1;
            this.lastFunction = row.function;
        }

        // Check if this is the last row of a function group (next row has different function)
        // When called from updateRow(), rows may be undefined - in that case, check against full state
        let isLastOfFunction = false;
        if (rows) {
            const nextRow = rows[index + 1];
            isLastOfFunction = !nextRow || nextRow.function !== row.function;
        } else {
            // Called from updateRow - check against AppState.rows
            const allRows = AppState.rows;
            const currentIndex = allRows.findIndex(r => r.row_id === row.row_id);
            if (currentIndex !== -1) {
                const nextRow = allRows[currentIndex + 1];
                isLastOfFunction = !nextRow || nextRow.function !== row.function;
            }
        }

        const dangerousClass = row.potentially_dangerous_final
            ? 'bg-red-50 text-red-600 font-medium'
            : 'bg-green-50 text-green-600';
        const dangerousText = row.potentially_dangerous_final ? 'Yes' : 'No';
        const ratingClass = this.getRatingClass(row.rating);
        const ratingText = this.formatRating(row.rating);
        const isSelected = AppState.isRowSelected(row.row_id);

        // Alternating background by function group (more visible colors)
        const functionBgClass = this.currentFunctionIndex % 2 === 0 ? 'bg-white' : 'bg-blue-50';

        // Add thick border-bottom for function separation
        const separatorClass = isLastOfFunction ? 'border-b-4 border-slate-400' : 'border-b border-slate-200';

        let rowClass = `cursor-pointer hover:bg-blue-50 ${functionBgClass} ${separatorClass}`;
        if (row.edited_flag && row.regenerated_flag) {
            rowClass += ' border-l-4 border-amber-400';
        } else if (row.edited_flag) {
            rowClass += ' border-l-4 border-amber-400';
        } else if (row.regenerated_flag) {
            rowClass += ' border-l-4 border-blue-500';
        }

        return `
            <tr data-row-id="${row.row_id}" class="${rowClass}">
                <td class="px-3 py-2.5" onclick="event.stopPropagation()">
                    <input type="checkbox" class="w-4 h-4 accent-blue-600 cursor-pointer" ${isSelected ? 'checked' : ''}>
                </td>
                <td class="px-3 py-2.5 text-slate-600">${this.escapeHtml(row.row_id)}</td>
                <td class="px-3 py-2.5 min-w-[120px] max-w-[200px] text-slate-700 whitespace-normal break-words">${this.escapeHtml(row.function)}</td>
                <td class="px-3 py-2.5 text-slate-600">${this.escapeHtml(row.guideword)}</td>
                <td class="px-3 py-2.5 min-w-[200px] max-w-[300px] text-slate-700 whitespace-normal break-words">${this.escapeHtml(row.deviation_final)}</td>
                <td class="px-3 py-2.5 min-w-[200px] max-w-[300px] text-slate-700 whitespace-normal break-words">${this.escapeHtml(row.cause_final)}</td>
                <td class="px-3 py-2.5 min-w-[200px] max-w-[300px] text-slate-700 whitespace-normal break-words">${this.escapeHtml(row.effect_final)}</td>
                <td class="px-3 py-2.5"><span class="px-2 py-0.5 rounded text-xs ${dangerousClass}">${dangerousText}</span></td>
                <td class="px-3 py-2.5"><span class="px-2 py-0.5 rounded text-xs font-medium ${ratingClass}">${ratingText}</span></td>
            </tr>
        `;
    },

    getRatingClass(rating) {
        const classes = {
            correct: 'bg-green-100 text-green-800',
            partially_correct: 'bg-amber-100 text-amber-800',
            incorrect: 'bg-red-100 text-red-800',
            unrated: 'bg-slate-100 text-slate-600',
        };
        return classes[rating] || classes.unrated;
    },

    formatRating(rating) {
        const labels = {
            correct: 'Correct',
            partially_correct: 'Partial',
            incorrect: 'Incorrect',
            unrated: 'Unrated',
        };
        return labels[rating] || 'Unrated';
    },

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text || '';
        return div.innerHTML;
    },

    updateRow(row) {
        const tr = this.tbody.querySelector(`tr[data-row-id="${row.row_id}"]`);
        if (tr) {
            const newHtml = this.renderRow(row);
            const temp = document.createElement('tbody');
            temp.innerHTML = newHtml;
            const newTr = temp.querySelector('tr');

            const checkbox = newTr.querySelector('input[type="checkbox"]');
            checkbox.addEventListener('change', (e) => {
                e.stopPropagation();
                AppState.toggleRowSelection(row.row_id, e.target.checked);
            });

            newTr.addEventListener('click', (e) => {
                if (e.target.type === 'checkbox') return;
                EditModal.open(row.row_id);
            });

            tr.replaceWith(newTr);
        }
    },

    updateCheckboxes() {
        this.tbody.querySelectorAll('tr[data-row-id]').forEach(tr => {
            const rowId = tr.dataset.rowId;
            const checkbox = tr.querySelector('input[type="checkbox"]');
            if (checkbox) {
                checkbox.checked = AppState.isRowSelected(rowId);
            }
        });

        // Update select all checkbox
        const allSelected = AppState.rows.length > 0 &&
            AppState.selectedRows.size === AppState.rows.length;
        this.selectAllCheckbox.checked = allSelected;
    },

    updateBulkActionBar() {
        const count = AppState.selectedRows.size;
        if (count > 0) {
            this.bulkActionBar.classList.remove('hidden');
            this.selectedCountSpan.textContent = `${count} row${count > 1 ? 's' : ''} selected`;
        } else {
            this.bulkActionBar.classList.add('hidden');
        }
    },

    onExport() {
        if (!AppState.runId) {
            alert('No analysis to export');
            return;
        }

        const url = API.getExportUrl(AppState.runId);
        window.open(url, '_blank');
    },

    onExportHtml() {
        if (!AppState.runId) {
            alert('No analysis to export');
            return;
        }

        const url = API.getExportHtmlUrl(AppState.runId);
        window.open(url, '_blank');
    },
};

// Make globally available
window.ResultsTable = ResultsTable;

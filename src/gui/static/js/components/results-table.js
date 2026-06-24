// results-table.js - Worksheet table (dynamic L1-L8 columns) with multi-row selection.

const ResultsTable = {
    init() {
        this.head = document.getElementById('results-head');
        this.tbody = document.getElementById('results-body');
        this.exportBtn = document.getElementById('export-csv-btn');
        this.exportHtmlBtn = document.getElementById('export-html-btn');
        this.bulkActionBar = document.getElementById('bulk-action-bar');
        this.selectedCountSpan = document.getElementById('selected-count');
        this.clearSelectionBtn = document.getElementById('clear-selection-btn');
        this.bulkRegenBtn = document.getElementById('bulk-regen-btn');

        this.exportBtn.addEventListener('click', () => this.onExport());
        this.exportHtmlBtn.addEventListener('click', () => this.onExportHtml());
        this.clearSelectionBtn.addEventListener('click', () => { AppState.clearSelection(); this.updateCheckboxes(); });
        this.bulkRegenBtn.addEventListener('click', () => BulkRegenModal.open());

        AppState.onSelectionChange = () => this.updateBulkActionBar();
        this.renderHead();
    },

    colCount() { return Fields.columns.length + 2; },  // checkbox + cols + rating

    renderHead() {
        const th = (label) => `<th class="px-3 py-3 text-left font-semibold text-slate-700">${this.escapeHtml(label)}</th>`;
        this.head.innerHTML = `<tr>
            <th class="px-3 py-3 w-10"><input type="checkbox" id="select-all-checkbox" class="w-4 h-4 accent-blue-600 cursor-pointer"></th>
            ${Fields.columns.map(c => th(c.label)).join('')}
            ${th('Rating')}
        </tr>`;
        const selectAll = document.getElementById('select-all-checkbox');
        selectAll.addEventListener('change', (e) => {
            if (e.target.checked) AppState.selectAllRows(); else AppState.clearSelection();
            this.updateCheckboxes();
        });
    },

    render(rows) {
        if (!rows || rows.length === 0) {
            this.tbody.innerHTML = `<tr class="empty-row"><td colspan="${this.colCount()}" class="px-3 py-10 text-center text-slate-400">No results yet. Configure contexts and run an analysis.</td></tr>`;
            this.updateBulkActionBar();
            return;
        }
        this.tbody.innerHTML = rows.map(r => this.renderRow(r)).join('');
        this.wireRows();
        this.updateCheckboxes();
        this.updateBulkActionBar();
    },

    renderRow(row) {
        const final = row.final || {};
        const dangerous = !!row.dangerous_final;
        const ratingClass = this.getRatingClass(row.rating);
        const isSelected = AppState.isRowSelected(row.row_id);
        // Dangerous rows get a red tint; otherwise alternate by component.
        const stripe = dangerous ? 'bg-red-50' : ((row.component_index % 2 === 0) ? 'bg-white' : 'bg-blue-50');

        let rowClass = `cursor-pointer hover:bg-blue-50 border-b border-slate-200 ${stripe}`;
        if (row.complete === false) rowClass += ' ring-2 ring-amber-400';  // incomplete (failed final validation)
        if (row.edited_flag) rowClass += ' border-l-4 border-amber-400';
        else if (row.regenerated_flag) rowClass += ' border-l-4 border-blue-500';

        const cells = Fields.columns.map(def => {
            if (def.measures) return this.measuresCell(final);
            if (def.list) return this.listCell(final, def);
            const val = Fields.formatValue(final, def);
            if (def.code) return `<td class="px-3 py-2.5 text-slate-600"><code class="bg-slate-100 px-1 rounded">${this.escapeHtml(val)}</code></td>`;
            const wide = ['failure_mode', 'hazardous_behavior', 'potential_harm', 'scenario', 'odd'].includes(def.key);
            const cls = wide ? 'min-w-[180px] max-w-[280px] whitespace-normal break-words' : '';
            return `<td class="px-3 py-2.5 text-slate-700 ${cls}">${this.escapeHtml(val)}</td>`;
        }).join('');

        return `
            <tr data-row-id="${this.escapeHtml(row.row_id)}" class="${rowClass}">
                <td class="px-3 py-2.5" onclick="event.stopPropagation()">
                    <input type="checkbox" class="w-4 h-4 accent-blue-600 cursor-pointer" ${isSelected ? 'checked' : ''}>
                </td>
                ${cells}
                <td class="px-3 py-2.5"><span class="px-2 py-0.5 rounded text-xs font-medium ${ratingClass}">${this.formatRating(row.rating)}</span></td>
            </tr>`;
    },

    measuresCell(final) {
        const groups = Fields.measureGroups(final).filter(g => g.items.length);
        if (!groups.length) {
            return `<td class="px-3 py-2.5 text-slate-300 align-top">—</td>`;
        }
        const inner = groups.map(g => {
            const head = `<div class="font-semibold text-slate-600">${this.escapeHtml(g.goal)}</div>`;
            const body = g.items.map(it => `<div class="pl-2 py-0.5"><span class="font-semibold text-slate-500 mr-1">${this.escapeHtml(it.label)}</span>${this.escapeHtml(it.text)}</div>`).join('');
            return `<div class="py-1 border-t border-slate-200 first:border-t-0">${head}${body}</div>`;
        }).join('');
        return `<td class="px-3 py-2 align-top min-w-[220px] max-w-[340px] whitespace-normal break-words">${inner}</td>`;
    },

    listCell(final, def) {
        const items = Fields.listItems(final, def);
        if (!items.length) {
            return `<td class="px-3 py-2.5 text-slate-300 align-top">—</td>`;
        }
        const inner = items.map((it, i) =>
            `<div class="py-1 ${i ? 'border-t border-slate-200' : ''}"><span class="font-semibold text-slate-500 mr-1">${this.escapeHtml(it.label)}</span>${this.escapeHtml(it.text)}</div>`
        ).join('');
        return `<td class="px-3 py-2 align-top min-w-[200px] max-w-[320px] whitespace-normal break-words"><div class="flex flex-col">${inner}</div></td>`;
    },

    wireRows() {
        this.tbody.querySelectorAll('tr[data-row-id]').forEach(tr => {
            const rowId = tr.dataset.rowId;
            const checkbox = tr.querySelector('input[type="checkbox"]');
            if (checkbox) checkbox.addEventListener('change', (e) => {
                e.stopPropagation();
                AppState.toggleRowSelection(rowId, e.target.checked);
            });
            tr.addEventListener('click', (e) => {
                if (e.target.type === 'checkbox') return;
                EditModal.open(rowId);
            });
        });
    },

    getRatingClass(rating) {
        return {
            correct: 'bg-green-100 text-green-800',
            partially_correct: 'bg-amber-100 text-amber-800',
            incorrect: 'bg-red-100 text-red-800',
            unrated: 'bg-slate-100 text-slate-600',
        }[rating] || 'bg-slate-100 text-slate-600';
    },

    formatRating(rating) {
        return { correct: 'Correct', partially_correct: 'Partial', incorrect: 'Incorrect', unrated: 'Unrated' }[rating] || 'Unrated';
    },

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text == null ? '' : String(text);
        return div.innerHTML;
    },

    updateRow(row) {
        const tr = this.tbody.querySelector(`tr[data-row-id="${row.row_id}"]`);
        if (!tr) return;
        const temp = document.createElement('tbody');
        temp.innerHTML = this.renderRow(row);
        const newTr = temp.querySelector('tr');
        tr.replaceWith(newTr);
        // Re-wire just this row
        const checkbox = newTr.querySelector('input[type="checkbox"]');
        if (checkbox) checkbox.addEventListener('change', (e) => {
            e.stopPropagation();
            AppState.toggleRowSelection(row.row_id, e.target.checked);
        });
        newTr.addEventListener('click', (e) => {
            if (e.target.type === 'checkbox') return;
            EditModal.open(row.row_id);
        });
    },

    updateCheckboxes() {
        this.tbody.querySelectorAll('tr[data-row-id]').forEach(tr => {
            const cb = tr.querySelector('input[type="checkbox"]');
            if (cb) cb.checked = AppState.isRowSelected(tr.dataset.rowId);
        });
        const selectAll = document.getElementById('select-all-checkbox');
        if (selectAll) selectAll.checked = AppState.rows.length > 0 && AppState.selectedRows.size === AppState.rows.length;
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
        if (!AppState.runId) { alert('No analysis to export'); return; }
        window.open(API.getExportUrl(AppState.runId), '_blank');
    },
    onExportHtml() {
        if (!AppState.runId) { alert('No analysis to export'); return; }
        window.open(API.getExportHtmlUrl(AppState.runId), '_blank');
    },
};

window.ResultsTable = ResultsTable;

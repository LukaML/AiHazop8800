// fields.js - Shared field metadata for the AI-HAZOP-8800 worksheet (L1-L8).
// Single source of truth for table columns, editable fields (grouped by phase),
// formatting, and regeneration scopes. Used by results-table.js and edit-modal.js.

const Fields = {
    SAFETY_DECISIONS: ['ACCEPT', 'IMPROVE', 'RESTRICT', 'INVESTIGATE'],

    // Results-table columns — AI-HAZOP-8800 paper §12 order. Flags: code, list, measures, risk.
    columns: [
        { key: 'hazard_id', label: 'Hazard ID' },
        { key: 'component', label: 'Component' },
        { key: 'component_class', label: 'Class' },
        { key: 'aspect', label: 'Aspect' },
        { key: 'odd', label: 'ODD' },
        { key: 'scenario', label: 'Scenario' },
        { key: 'guideword', label: 'Guideword/Question', code: true },
        { key: 'failure_mode', label: 'Failure mode' },
        { key: 'hazardous_behavior', label: 'Hazardous behavior' },
        { key: 'potential_harm', label: 'Potential harm' },
        { key: 'initial_risk', label: 'Initial risk', risk: true },
        { key: 'risk_status', label: 'Risk Status' },
        { key: 'acceptance_criterion', label: 'Acceptance criterion' },
        { key: 'safety_decision', label: 'Safety decision' },
        { key: 'ai_safety_goals', label: 'AI Safety Goal', list: true },
        { key: 'measures', label: 'Measures', measures: true },
        { key: 'residual_risk', label: 'Residual risk', risk: true },
        { key: 'residual_status', label: 'Residual Status' },
        { key: 'evidence', label: 'Evidence', list: true },
        { key: 'open_assumptions', label: 'Open assumptions', list: true },
    ],

    // Editable fields, grouped by phase. `repr` = the field whose presence in a
    // row indicates the phase applies to that row (drives panel + scope options).
    editGroups: [
        { phase: 'L1', title: 'L1 - Failure Mode', repr: 'failure_mode', fields: [
            { key: 'failure_mode', label: 'Failure Mode', type: 'text' },
        ] },
        { phase: 'L2', title: 'L2 - Hazard & Triage', repr: 'hazardous_behavior', fields: [
            { key: 'hazardous_behavior', label: 'Hazardous Behavior', type: 'text' },
            { key: 'potential_harm', label: 'Potential Harm', type: 'text' },
            { key: 'potentially_dangerous', label: 'Potentially Dangerous', type: 'bool' },
        ] },
        { phase: 'L3', title: 'L3 - Initial Risk', repr: 'S', fields: [
            { key: 'E', label: 'E (Exposure)', type: 'text' },
            { key: 'PF', label: 'PF (P[failure])', type: 'text' },
            { key: 'PND', label: 'PND (P[not detected])', type: 'text' },
            { key: 'PNM', label: 'PNM (P[no mitigation])', type: 'text' },
            { key: 'S', label: 'S (Severity)', type: 'text' },
            { key: 'risk_rationale', label: 'Risk Rationale', type: 'text' },
            { key: 'initial_risk', label: 'Initial Risk (computed)', type: 'readonly', risk: true },
            { key: 'risk_status', label: 'Risk Status (computed)', type: 'readonly' },
        ] },
        { phase: 'L4', title: 'L4 - Acceptance', repr: 'safety_decision', fields: [
            { key: 'safety_decision', label: 'Safety Decision', type: 'select', options: ['ACCEPT', 'IMPROVE', 'RESTRICT', 'INVESTIGATE'] },
            { key: 'acceptance_rationale', label: 'Acceptance Rationale', type: 'text' },
        ] },
        { phase: 'L5', title: 'L5 - Safety Goals', repr: 'ai_safety_goals', fields: [
            { key: 'ai_safety_goals', label: 'AI Safety Goals', type: 'list' },
        ] },
        { phase: 'L6', title: 'L6 - Measures', repr: 'respecifications', fields: [
            { key: 'respecifications', label: 'Respecifications (R)', type: 'list' },
            { key: 'safety_functions', label: 'Safety Functions (SF)', type: 'list' },
            { key: 'passive_operational_measures', label: 'Passive / Operational (P)', type: 'list' },
        ] },
        { phase: 'L7', title: 'L7 - Residual Risk', repr: 'residual_S', fields: [
            { key: 'residual_E', label: 'Residual E', type: 'text' },
            { key: 'residual_PF', label: 'Residual PF', type: 'text' },
            { key: 'residual_PND', label: 'Residual PND', type: 'text' },
            { key: 'residual_PNM', label: 'Residual PNM', type: 'text' },
            { key: 'residual_S', label: 'Residual S', type: 'text' },
            { key: 'residual_rationale', label: 'Residual Rationale', type: 'text' },
            { key: 'residual_risk', label: 'Residual Risk (computed)', type: 'readonly', risk: true },
            { key: 'residual_status', label: 'Residual Status (computed)', type: 'readonly' },
        ] },
        { phase: 'L8', title: 'L8 - Evidence', repr: 'evidence', fields: [
            { key: 'evidence', label: 'Evidence', type: 'list' },
            { key: 'open_assumptions', label: 'Open Assumptions', type: 'list' },
        ] },
    ],

    // Regeneration scopes (phase to patch, then cascade downstream).
    scopes: [
        { value: 'L1', label: 'L1 - Failure Mode (cascade all)' },
        { value: 'L2', label: 'L2 - Hazard (cascade L3-L8)' },
        { value: 'L3', label: 'L3 - Initial Risk (cascade L4-L8)' },
        { value: 'L4', label: 'L4 - Acceptance (cascade L5-L8)' },
        { value: 'L5', label: 'L5 - Safety Goals (cascade L6-L8)' },
        { value: 'L6', label: 'L6 - Measures (cascade L7-L8)' },
        { value: 'L7', label: 'L7 - Residual Risk (cascade L8)' },
        { value: 'L8', label: 'L8 - Evidence' },
    ],

    // Map field key -> phase (for diffing changes after regeneration).
    phaseOf(key) {
        for (const g of this.editGroups) {
            if (g.fields.some(f => f.key === key)) return g.phase;
        }
        return null;
    },

    // Is `key`'s data present on a row's final map (drives panel/scope visibility)?
    rowHasPhase(row, phase) {
        const g = this.editGroups.find(x => x.phase === phase);
        if (!g) return false;
        const final = (row && row.final) || {};
        return final[g.repr] !== undefined;
    },

    // Label prefixes for list-type fields rendered as mini-tables.
    listPrefix: {
        ai_safety_goals: 'SG',
        evidence: 'EV',
        open_assumptions: 'A',
        respecifications: 'R',
        safety_functions: 'SF',
        passive_operational_measures: 'P',
    },

    isListColumn(def) { return !!(def.list || def.measures); },

    // Return labeled items [{label, text}] for a list/measures column.
    listItems(final, def) {
        final = final || {};
        const items = [];
        const itemText = (x) => {
            if (x && typeof x === 'object' && !Array.isArray(x)) {
                return Object.values(x).map(v => String(v == null ? '' : v).trim()).filter(Boolean).join(' — ');
            }
            return String(x == null ? '' : x).trim();
        };
        const add = (arr, prefix) => {
            let n = 0;
            (Array.isArray(arr) ? arr : []).forEach(x => {
                const t = itemText(x);
                if (t) { n += 1; items.push({ label: prefix + n, text: t }); }
            });
        };
        if (def.measures) {
            add(final.respecifications, 'R');
            add(final.safety_functions, 'SF');
            add(final.passive_operational_measures, 'P');
        } else {
            add(final[def.key], this.listPrefix[def.key] || '#');
        }
        return items;
    },

    // Parse a leading goal tag like "[SG2] ..." -> {goal: 2, text: "..."}; default goal 1.
    parseGoalTag(x) {
        let s;
        if (x && typeof x === 'object' && !Array.isArray(x)) {
            s = Object.values(x).map(v => String(v == null ? '' : v).trim()).filter(Boolean).join(' — ');
        } else {
            s = String(x == null ? '' : x).trim();
        }
        const m = s.match(/^\[?\s*SG\s*(\d+)\s*\]?\s*[-.:]?\s*/i);
        if (m) return { goal: parseInt(m[1], 10) || 1, text: s.slice(m[0].length).trim() };
        return { goal: 1, text: s };
    },

    // Build per-safety-goal measure groups with hierarchical labels (R1.1, SF1.2, P1.1).
    measureGroups(final) {
        final = final || {};
        const goals = Array.isArray(final.ai_safety_goals) ? final.ai_safety_goals.filter(g => String(g).trim()) : [];
        const ng = Math.max(goals.length, 1);
        const groups = [];
        for (let k = 1; k <= ng; k++) groups.push({ goal: 'SG' + k, goalText: goals[k - 1] || '', items: [] });
        const ID_RE = /^\((SF|R|P)\s*\d+\)\s*/i;
        const addClass = (arr, cls) => {
            const counter = {};
            (Array.isArray(arr) ? arr : []).forEach(x => {
                let { goal, text } = this.parseGoalTag(x);
                const g = Math.min(Math.max(goal, 1), ng);
                let label;
                const m = text.match(ID_RE);
                if (m) {
                    label = m[0].replace(/[()\s]/g, '').toUpperCase();   // explicit id e.g. SF1
                    text = text.slice(m[0].length).trim();
                } else {
                    counter[g] = counter[g] || {};
                    counter[g][cls] = (counter[g][cls] || 0) + 1;
                    label = `${cls}${g}.${counter[g][cls]}`;             // derived fallback
                }
                if (!text) return;
                groups[g - 1].items.push({ label, text });
            });
        };
        addClass(final.respecifications, 'R');
        addClass(final.safety_functions, 'SF');
        addClass(final.passive_operational_measures, 'P');
        return groups;
    },

    // Format a value for display, given a column/field definition.
    formatValue(final, def) {
        if (!final) final = {};
        if (def.key === 'guideword') return String(final.guideword == null ? '' : final.guideword).replace(/_/g, ' ');
        if (def.measures) {
            const parts = [];
            ['respecifications', 'safety_functions', 'passive_operational_measures'].forEach(k => {
                const v = final[k];
                if (Array.isArray(v)) v.forEach(x => { if (String(x).trim()) parts.push(String(x).trim()); });
                else if (v) parts.push(String(v).trim());
            });
            return parts.join('; ');
        }
        let v = final[def.key];
        if (Array.isArray(v)) return v.filter(x => String(x).trim()).join('; ');
        if (def.risk && typeof v === 'number') return v.toExponential(2);
        if (typeof v === 'boolean') return v ? 'Yes' : 'No';
        return v == null ? '' : String(v);
    },
};

window.Fields = Fields;

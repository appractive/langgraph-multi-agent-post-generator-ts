// DOM Elements
const topicInput = document.getElementById('topic');
const sourcesInput = document.getElementById('sources');
const webSearchCheck = document.getElementById('webSearch');
const humanInLoopCheck = document.getElementById('humanInLoop');
const generateBtn = document.getElementById('generateBtn');
const generateBtnText = document.getElementById('generateBtnText');
const stopGenerationBtn = document.getElementById('stopGenerationBtn');

// UI Toggles & Configuration Elements
const toggleConfigBtn = document.getElementById('toggleConfigBtn');
const configBody = document.getElementById('configBody');
const configChevron = document.getElementById('configChevron');
const previewWebSearch = document.getElementById('previewWebSearch');
const previewHitl = document.getElementById('previewHitl');
const previewModel = document.getElementById('previewModel');

// Result Areas
const reviewBadgeContainer = document.getElementById('reviewBadgeContainer');
const errorArea = document.getElementById('errorArea');
const errorText = document.getElementById('errorText');
const resultArea = document.getElementById('resultArea');

// Tabs
const tabBtns = document.querySelectorAll('.tab-btn');
const tabPanes = document.querySelectorAll('.tab-pane');
const draftPane = document.getElementById('draftPane');
const outlinePane = document.getElementById('outlinePane');
const reviewPane = document.getElementById('reviewPane');
const researchPane = document.getElementById('researchPane');
const finalPane = document.getElementById('finalPane');

let finalArticle = '';
let activeGenerationController = null;

function beginGenerationRequest() {
    const controller = new AbortController();
    activeGenerationController = controller;
    generateBtn.disabled = true;
    generateBtnText.textContent = 'Generating...';
    stopGenerationBtn.classList.remove('hidden');
    stopGenerationBtn.disabled = false;
    stopGenerationBtn.querySelector('span').textContent = 'Stop generating';
    return controller;
}

function finishGenerationRequest(controller) {
    if (activeGenerationController !== controller) return;
    if (controller.signal.aborted) {
        const runningStep = document.querySelector('.tab-btn.step-running')?.dataset.step;
        if (runningStep) updateStepState(runningStep, 'idle');
    }
    activeGenerationController = null;
    generateBtn.disabled = false;
    generateBtnText.textContent = 'Generate';
    stopGenerationBtn.classList.add('hidden');
    stopGenerationBtn.disabled = false;
    stopGenerationBtn.querySelector('span').textContent = 'Stop generating';
}

stopGenerationBtn.addEventListener('click', () => {
    if (!activeGenerationController) return;
    stopGenerationBtn.disabled = true;
    stopGenerationBtn.querySelector('span').textContent = 'Stopping...';
    activeGenerationController.abort();
});

const TAB_MAP = {
    researcher: 'researchPane',
    planner: 'outlinePane',
    writer: 'draftPane',
    reviewer: 'reviewPane'
};

const PIPELINE_EXPLANATIONS = {
    researchPane: {
        icon: '🔬',
        title: 'Researcher Agent',
        badge: 'Stage 1 of 4',
        desc: 'Searches live web sources when enabled, summarizes supplied references, and lists the URLs used for verification.'
    },
    outlinePane: {
        icon: '📋',
        title: 'Planner Agent',
        badge: 'Stage 2 of 4',
        desc: 'Structures the gathered research into a coherent outline with section headings, core discussion points, SEO metadata, and estimated reading time.'
    },
    draftPane: {
        icon: '✍️',
        title: 'Writer Agent',
        badge: 'Stage 3 of 4',
        desc: 'Synthesizes the outline and verified facts into a comprehensive, high-quality article draft with rich markdown formatting and engaging tone.'
    },
    reviewPane: {
        icon: '🔍',
        title: 'Reviewer Agent',
        badge: 'Stage 4 of 4',
        desc: 'Critiques the drafted post for clarity, structure, and tone, assigns an objective editorial score (0–100), and provides actionable polish.'
    },
    finalPane: {
        icon: '✨',
        title: 'Final Published Article',
        badge: 'Ready to Publish',
        desc: 'The complete, polished blog post ready for distribution. Includes one-click copy to clipboard.'
    }
};

function updatePipelineExplainer(targetPaneId) {
    const explainer = PIPELINE_EXPLANATIONS[targetPaneId];
    if (!explainer) return;
    const badgeEl = document.getElementById('explainerBadge');
    const titleEl = document.getElementById('explainerTitle');
    const descEl = document.getElementById('explainerDesc');
    if (badgeEl) badgeEl.textContent = explainer.badge;
    if (titleEl) titleEl.textContent = `${explainer.icon} ${explainer.title}`;
    if (descEl) descEl.textContent = explainer.desc;
}

function activateTabForStep(stepKey) {
    const targetPane = TAB_MAP[stepKey];
    if (!targetPane) return;
    
    tabBtns.forEach(b => b.classList.remove('active'));
    tabPanes.forEach(p => p.classList.add('hidden'));
    
    const btn = document.querySelector(`.tab-btn[data-target="${targetPane}"]`);
    if (btn) btn.classList.add('active');
    document.getElementById(targetPane).classList.remove('hidden');
    updatePipelineExplainer(targetPane);
}

// Switch to the Final tab (used when pipeline completes after HITL approval)
function activateFinalTab() {
    tabBtns.forEach(b => b.classList.remove('active'));
    tabPanes.forEach(p => p.classList.add('hidden'));
    const btn = document.querySelector('.tab-btn[data-target="finalPane"]');
    if (btn) btn.classList.add('active');
    finalPane.classList.remove('hidden');
    updatePipelineExplainer('finalPane');
}

function renderCopyArticleButton() {
    return `<div style="display: flex; justify-content: flex-end; margin-bottom: 1rem;"><button id="copyArticleBtn" class="btn-text" style="font-size: 0.875rem; gap: 0.5rem;" onclick="copyArticle()"><svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg><span>Copy Article</span></button></div>`;
}

const PIPELINE_STEPS = [
    { key: "researcher", label: "Researcher Agent", desc: "Analyzing sources", icon: "🔬" },
    { key: "planner", label: "Planner Agent", desc: "Creating outline", icon: "📋" },
    { key: "writer", label: "Writer Agent", desc: "Writing article", icon: "✍️" },
    { key: "reviewer", label: "Reviewer Agent", desc: "Reviewing quality", icon: "🔍" }
];

// Populate Model Dropdowns from API
const MODEL_SELECTS = ['researcher', 'planner', 'writer', 'reviewer'];
const modelLoadingIndicator = document.getElementById('modelLoadingIndicator');
const fetchModelsBtn = document.getElementById('fetchModelsBtn');
const fetchModelsIcon = document.getElementById('fetchModelsIcon');
const fetchModelsText = document.getElementById('fetchModelsText');

// Group available models by capability.
const STRENGTH_ORDER = ['flagship', 'standard', 'fast'];
const STRENGTH_GROUP_LABELS = {
    flagship: 'Flagship Models',
    standard: 'Standard Models',
    fast: 'Fast & Cheap Models'
};

const DEFAULT_MODEL_SELECT = document.getElementById('model-default');

function populateModelSelect(selectEl, models, defaultValue) {
    if (!selectEl) return;
    const currentVal = selectEl.value || defaultValue;
    selectEl.innerHTML = '';
    const isDefaultSelect = selectEl === DEFAULT_MODEL_SELECT;
    if (isDefaultSelect) {
        const placeholder = document.createElement('option');
        placeholder.value = '';
        placeholder.textContent = 'Choose a model for all stages';
        selectEl.appendChild(placeholder);
    }
    const groups = {};
    models.forEach(m => {
        if (!groups[m.strength]) groups[m.strength] = [];
        groups[m.strength].push(m);
    });

    STRENGTH_ORDER.forEach(tier => {
        const tierModels = groups[tier];
        if (!tierModels || tierModels.length === 0) return;
        const optgroup = document.createElement('optgroup');
        optgroup.label = STRENGTH_GROUP_LABELS[tier] || tier;
        tierModels.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.id;
            opt.textContent = m.id;
            optgroup.appendChild(opt);
        });
        selectEl.appendChild(optgroup);
    });

    const targetVal = currentVal && selectEl.querySelector(`option[value="${currentVal}"]`) ? currentVal : defaultValue;
    if (targetVal && selectEl.querySelector(`option[value="${targetVal}"]`)) {
        selectEl.value = targetVal;
    } else if (isDefaultSelect) {
        selectEl.value = '';
    }
}

function syncDefaultModelSelect() {
    if (!DEFAULT_MODEL_SELECT) return;
    const selectedModels = MODEL_SELECTS.map(agent => document.getElementById(`model-${agent}`)?.value);
    const sharedModel = selectedModels[0] && selectedModels.every(model => model === selectedModels[0])
        ? selectedModels[0]
        : '';
    DEFAULT_MODEL_SELECT.value = sharedModel;
}

function populateModelControls(models) {
    MODEL_SELECTS.forEach(agent => {
        const select = document.getElementById(`model-${agent}`);
        if (!select) return;
        populateModelSelect(select, models, select.dataset.defaultModel || 'gpt-4o');
        select.onchange = () => {
            syncDefaultModelSelect();
            updateTabModelPills();
        };
    });

    populateModelSelect(DEFAULT_MODEL_SELECT, models, '');
    syncDefaultModelSelect();
    updateTabModelPills();
}

async function loadModels(force = false) {
    if (modelLoadingIndicator) modelLoadingIndicator.classList.remove('hidden');
    try {
        const url = force ? `/api/models?t=${Date.now()}` : '/api/models';
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const models = await res.json();

        populateModelControls(models);
        if (modelLoadingIndicator) modelLoadingIndicator.classList.add('hidden');
    } catch (err) {
        console.error('Failed to load models:', err);
        const fallback = [
            { id: 'gpt-4o', name: 'gpt-4o', strength: 'flagship', strengthLabel: '🟣 Flagship' },
            { id: 'gpt-4o-mini', name: 'gpt-4o-mini', strength: 'fast', strengthLabel: '🟢 Fast & Cheap' },
            { id: 'gpt-4.1', name: 'gpt-4.1', strength: 'flagship', strengthLabel: '🟣 Flagship' },
            { id: 'gpt-4.1-mini', name: 'gpt-4.1-mini', strength: 'fast', strengthLabel: '🟢 Fast & Cheap' },
            { id: 'gpt-4-turbo', name: 'gpt-4-turbo', strength: 'flagship', strengthLabel: '🟣 Flagship' },
            { id: 'gpt-4', name: 'gpt-4', strength: 'standard', strengthLabel: '🔵 Standard' },
            { id: 'gpt-3.5-turbo', name: 'gpt-3.5-turbo', strength: 'fast', strengthLabel: '🟢 Fast & Cheap' }
        ];
        populateModelControls(fallback);
        if (modelLoadingIndicator) modelLoadingIndicator.classList.add('hidden');
    }
}

// Update the collapsed configuration header summary badges
function updateConfigPreview() {
    if (previewWebSearch && webSearchCheck) {
        previewWebSearch.textContent = `Web Search: ${webSearchCheck.checked ? 'ON' : 'OFF'}`;
        previewWebSearch.classList.toggle('is-on', webSearchCheck.checked);
        previewWebSearch.classList.toggle('is-off', !webSearchCheck.checked);
    }

    if (previewHitl && humanInLoopCheck) {
        previewHitl.textContent = humanInLoopCheck.checked ? 'HITL: PLANNER' : 'HITL: OFF';
        previewHitl.classList.toggle('is-on', humanInLoopCheck.checked);
        previewHitl.classList.toggle('is-off', !humanInLoopCheck.checked);
    }

    if (previewModel) {
        const rModel = document.getElementById('model-researcher')?.value;
        const pModel = document.getElementById('model-planner')?.value;
        const wModel = document.getElementById('model-writer')?.value;
        const revModel = document.getElementById('model-reviewer')?.value;

        if (rModel && pModel && wModel && revModel) {
            if (rModel === pModel && pModel === wModel && wModel === revModel) {
                previewModel.textContent = `Models: ${rModel}`;
                previewModel.classList.add('is-set');
                previewModel.classList.remove('is-mixed');
            } else {
                previewModel.textContent = 'Models: Mixed';
                previewModel.classList.add('is-mixed');
                previewModel.classList.remove('is-set');
            }
        }
    }
}

// Synchronize each agent's current model to its result tab pill
function updateTabModelPills() {
    MODEL_SELECTS.forEach(agent => {
        const select = document.getElementById(`model-${agent}`);
        const pill = document.getElementById(`tab-model-${agent}`);
        if (select && pill && select.value) {
            pill.textContent = select.value;
        }
    });
    updateConfigPreview();
}

// Collapsible Configuration Section Toggle
if (toggleConfigBtn && configBody) {
    toggleConfigBtn.addEventListener('click', () => {
        const isHidden = configBody.classList.toggle('hidden');
        const isExpanded = !isHidden;
        toggleConfigBtn.setAttribute('aria-expanded', isExpanded ? 'true' : 'false');
        if (configChevron) {
            configChevron.classList.toggle('open', isExpanded);
        }
    });
}

// Fetch live available models button
if (fetchModelsBtn) {
    fetchModelsBtn.addEventListener('click', async () => {
        if (fetchModelsIcon) fetchModelsIcon.classList.add('spin-animation');
        if (fetchModelsText) fetchModelsText.textContent = 'Fetching…';
        fetchModelsBtn.disabled = true;

        try {
            await loadModels(true);
            if (fetchModelsText) fetchModelsText.textContent = 'Models Loaded!';
            setTimeout(() => {
                if (fetchModelsText) fetchModelsText.textContent = 'Refresh models';
                fetchModelsBtn.disabled = false;
            }, 1500);
        } catch (e) {
            if (fetchModelsText) fetchModelsText.textContent = 'Retry Fetch';
            fetchModelsBtn.disabled = false;
        } finally {
            if (fetchModelsIcon) fetchModelsIcon.classList.remove('spin-animation');
        }
    });
}

// Selecting a shared model immediately updates every pipeline stage.
if (DEFAULT_MODEL_SELECT) {
    DEFAULT_MODEL_SELECT.addEventListener('change', () => {
        if (!DEFAULT_MODEL_SELECT.value) return;
        MODEL_SELECTS.forEach(agent => {
            const select = document.getElementById(`model-${agent}`);
            if (select) select.value = DEFAULT_MODEL_SELECT.value;
        });
        updateTabModelPills();
    });
}

// Load models on page init
loadModels();

// Initialize pipeline explanation on the starting tab
updatePipelineExplainer('researchPane');

if (humanInLoopCheck) {
    humanInLoopCheck.addEventListener('change', updateConfigPreview);
}

if (webSearchCheck) {
    webSearchCheck.addEventListener('change', updateConfigPreview);
}
updateConfigPreview();

tabBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
        const targetBtn = e.currentTarget;
        tabBtns.forEach(b => b.classList.remove('active'));
        targetBtn.classList.add('active');

        const targetPaneId = targetBtn.getAttribute('data-target');
        tabPanes.forEach(p => p.classList.add('hidden'));
        const paneEl = document.getElementById(targetPaneId);
        if (paneEl) paneEl.classList.remove('hidden');

        updatePipelineExplainer(targetPaneId);
    });
});

// Rendering functions
function renderPipeline() {
    // Reset all tab step states
    document.querySelectorAll('.tab-btn[data-step]').forEach(btn => {
        btn.classList.remove('step-running', 'step-done', 'step-error');
        const stepKey = btn.dataset.step;
        const stepDef = PIPELINE_STEPS.find(s => s.key === stepKey);
        const iconEl = btn.querySelector('.tab-icon');
        if (iconEl && stepDef) iconEl.innerHTML = stepDef.icon;
    });
}

function updateStepState(stepKey, status) {
    const btn = document.querySelector(`.tab-btn[data-step="${stepKey}"]`);
    if (!btn) return;

    btn.classList.remove('step-running', 'step-done', 'step-error');
    const iconEl = btn.querySelector('.tab-icon');
    const stepDef = PIPELINE_STEPS.find(s => s.key === stepKey);

    if (status === 'running') {
        btn.classList.add('step-running');
        if (iconEl) iconEl.innerHTML = `<div class="spinner-sm"></div>`;
    } else if (status === 'done') {
        btn.classList.add('step-done');
        if (iconEl) iconEl.innerHTML = `<span style="color: var(--emerald-text);">✓</span>`;
    } else if (status === 'error') {
        btn.classList.add('step-error');
        if (iconEl) iconEl.innerHTML = `<span style="color: var(--red-text);">✗</span>`;
    } else if (status === 'idle' && iconEl && stepDef) {
        iconEl.innerHTML = stepDef.icon;
    }
}

function getParagraphDiff(beforeArticle, afterArticle) {
    const beforeBlocks = beforeArticle.split(/\n\s*\n/).filter(block => block.trim());
    const afterBlocks = afterArticle.split(/\n\s*\n/).filter(block => block.trim());
    const columns = afterBlocks.length + 1;
    if ((beforeBlocks.length + 1) * columns > 1000000) {
        return [
            beforeArticle ? `<div class="diff-block diff-block-removed">${marked.parse(beforeArticle)}</div>` : '',
            afterArticle ? `<div class="diff-block diff-block-added">${marked.parse(afterArticle)}</div>` : ''
        ];
    }
    const lengths = new Uint32Array((beforeBlocks.length + 1) * columns);

    for (let i = beforeBlocks.length - 1; i >= 0; i--) {
        for (let j = afterBlocks.length - 1; j >= 0; j--) {
            const index = i * columns + j;
            lengths[index] = beforeBlocks[i] === afterBlocks[j]
                ? lengths[(i + 1) * columns + j + 1] + 1
                : Math.max(lengths[(i + 1) * columns + j], lengths[i * columns + j + 1]);
        }
    }

    const operations = [];
    let i = 0;
    let j = 0;
    while (i < beforeBlocks.length || j < afterBlocks.length) {
        if (i < beforeBlocks.length && j < afterBlocks.length && beforeBlocks[i] === afterBlocks[j]) {
            operations.push({ type: 'same', before: beforeBlocks[i], after: afterBlocks[j] });
            i++;
            j++;
        } else if (i < beforeBlocks.length && (j === afterBlocks.length || lengths[(i + 1) * columns + j] >= lengths[i * columns + j + 1])) {
            operations.push({ type: 'removed', before: beforeBlocks[i++] });
        } else {
            operations.push({ type: 'added', after: afterBlocks[j++] });
        }
    }

    const beforeHtml = [];
    const afterHtml = [];
    for (let index = 0; index < operations.length;) {
        const operation = operations[index];
        if (operation.type === 'same') {
            beforeHtml.push(marked.parse(operation.before));
            afterHtml.push(marked.parse(operation.after));
            index++;
            continue;
        }

        const removed = [];
        const added = [];
        while (index < operations.length && operations[index].type !== 'same') {
            const edit = operations[index++];
            if (edit.type === 'removed') removed.push(edit.before);
            else added.push(edit.after);
        }

        const pairedCount = Math.min(removed.length, added.length);
        for (let pair = 0; pair < pairedCount; pair++) {
            const [oldVersion, newVersion] = renderInlineDiff(removed[pair], added[pair]);
            beforeHtml.push(oldVersion);
            afterHtml.push(newVersion);
        }
        removed.slice(pairedCount).forEach(block => beforeHtml.push(`<div class="diff-block diff-block-removed">${marked.parse(block)}</div>`));
        added.slice(pairedCount).forEach(block => afterHtml.push(`<div class="diff-block diff-block-added">${marked.parse(block)}</div>`));
    }

    return [beforeHtml.join(''), afterHtml.join('')];
}

function renderInlineDiff(beforeBlock, afterBlock) {
    const beforeTokens = beforeBlock.match(/[\p{L}\p{N}_]+|\s+|[^\s\p{L}\p{N}_]/gu) || [];
    const afterTokens = afterBlock.match(/[\p{L}\p{N}_]+|\s+|[^\s\p{L}\p{N}_]/gu) || [];
    const columns = afterTokens.length + 1;
    const cellCount = (beforeTokens.length + 1) * columns;

    if (cellCount > 250000) {
        return [
            `<div class="diff-block diff-block-removed">${marked.parse(beforeBlock)}</div>`,
            `<div class="diff-block diff-block-added">${marked.parse(afterBlock)}</div>`
        ];
    }

    const lengths = new Uint32Array(cellCount);
    for (let i = beforeTokens.length - 1; i >= 0; i--) {
        for (let j = afterTokens.length - 1; j >= 0; j--) {
            const index = i * columns + j;
            lengths[index] = beforeTokens[i] === afterTokens[j]
                ? lengths[(i + 1) * columns + j + 1] + 1
                : Math.max(lengths[(i + 1) * columns + j], lengths[i * columns + j + 1]);
        }
    }

    let oldMarkdown = '';
    let newMarkdown = '';
    let removed = '';
    let added = '';
    const flushChanges = () => {
        if (removed) oldMarkdown += `<del class="diff-removed">${removed}</del>`;
        if (added) newMarkdown += `<ins class="diff-added">${added}</ins>`;
        removed = '';
        added = '';
    };

    let i = 0;
    let j = 0;
    while (i < beforeTokens.length || j < afterTokens.length) {
        if (i < beforeTokens.length && j < afterTokens.length && beforeTokens[i] === afterTokens[j]) {
            flushChanges();
            oldMarkdown += beforeTokens[i++];
            newMarkdown += afterTokens[j++];
        } else if (i < beforeTokens.length && (j === afterTokens.length || lengths[(i + 1) * columns + j] >= lengths[i * columns + j + 1])) {
            removed += beforeTokens[i++];
        } else {
            added += afterTokens[j++];
        }
    }
    flushChanges();

    return [marked.parse(oldMarkdown), marked.parse(newMarkdown)];
}

function renderArticleComparison(beforeArticle, afterArticle, title, beforeLabel, afterLabel) {
    const [beforeHtml, afterHtml] = getParagraphDiff(beforeArticle || '', afterArticle || '');
    const hasTextChanges = (beforeArticle || '').replace(/\s+/g, ' ').trim() !== (afterArticle || '').replace(/\s+/g, ' ').trim();
    return `
        <section class="article-comparison">
            <div class="article-comparison-heading">
                <h3 class="article-comparison-title">${title}</h3>
                <div class="diff-legend"><span><i class="diff-legend-swatch removed"></i>Removed</span><span><i class="diff-legend-swatch added"></i>Added</span></div>
            </div>
            ${hasTextChanges ? '' : '<p class="comparison-no-changes">The reviewer returned no text edits, so both versions are identical.</p>'}
            <div class="article-comparison-grid">
                <section class="article-version">
                    <h4 class="article-version-label"><span class="version-dot version-before"></span>${beforeLabel}</h4>
                    <div class="prose article-version-content">${beforeHtml}</div>
                </section>
                <section class="article-version">
                    <h4 class="article-version-label"><span class="version-dot version-after"></span>${afterLabel}</h4>
                    <div class="prose article-version-content">${afterHtml}</div>
                </section>
            </div>
        </section>
    `;
}

function renderReview(reviewData, retryCount, originalArticle = '', reviewScores = []) {
    const scoreBefore = reviewData.scoreBefore ?? reviewData.score;
    const scoreAfter = reviewData.scoreAfter;
    const beforeLabel = Number.isFinite(scoreBefore) ? scoreBefore : '—';
    const afterLabel = Number.isFinite(scoreAfter) ? scoreAfter : '—';
    reviewBadgeContainer.innerHTML = `<div class="badge badge-emerald">Quality: ${beforeLabel} before → ${afterLabel} after</div>`;
    reviewBadgeContainer.classList.remove('hidden');
    const previousScores = reviewScores.slice(0, -1).map(score => typeof score === 'number'
        ? { before: score, after: null }
        : score);

    reviewPane.innerHTML = `
        <div class="review-score-pair" aria-label="Quality scores before and after suggestions">
            <div class="review-score-card before">
                <span>Before suggestions</span>
                <strong>${beforeLabel}<small>/100</small></strong>
            </div>
            <div class="review-score-card after">
                <span>After suggestions</span>
                <strong>${afterLabel}<small>/100</small></strong>
            </div>
        </div>
        ${Number.isFinite(scoreAfter) ? '' : '<p class="comparison-no-changes">After-review scoring is unavailable from the running server. Restart the app server to enable both scores.</p>'}
        ${previousScores.length ? `
            <section class="review-score-history" aria-label="Previous review scores">
                <h3>Previous quality scores</h3>
                <ol>
                    ${previousScores.map((score, index) => `<li><span>Review ${index + 1}</span><strong>${score.before ?? '—'} → ${score.after ?? '—'}</strong></li>`).join('')}
                </ol>
            </section>
        ` : ''}
        <h3 class="section-title">Suggestions Applied</h3>
        <ul class="review-suggestions">
            ${reviewData.suggestions.length
                ? reviewData.suggestions.map(s => `
                <li class="review-suggestion">
                    <span aria-hidden="true">→</span> ${escapeHtml(s)}
                </li>`).join('')
                : '<li class="comparison-no-changes">No suggestions were applied because the reviewer returned no text edits.</li>'}
        </ul>
        ${renderArticleComparison(
            originalArticle,
            reviewData.improvedArticle || originalArticle,
            'Article before and after suggestions',
            'Before suggestions',
            'After suggestions'
        )}
    `;

    // Update Draft tab to show review iteration count
    if (retryCount && retryCount > 0) {
        const draftTabBtn = document.querySelector('.tab-btn[data-target="draftPane"]');
        if (draftTabBtn) {
            let retryBadge = draftTabBtn.querySelector('.tab-retry-badge');
            if (!retryBadge) {
                retryBadge = document.createElement('span');
                retryBadge.className = 'tab-retry-badge';
                retryBadge.style.cssText = 'background: var(--primary); color: white; font-size: 0.7rem; padding: 0.1rem 0.4rem; border-radius: 999px; margin-left: 0.35rem;';
                const label = draftTabBtn.querySelector('.tab-label');
                if (label) label.after(retryBadge);
                else draftTabBtn.appendChild(retryBadge);
            }
            retryBadge.textContent = retryCount;
        }
    }
}

function resetDraftTabBadge() {
    const draftTabBtn = document.querySelector('.tab-btn[data-target="draftPane"]');
    if (draftTabBtn) {
        const retryBadge = draftTabBtn.querySelector('.tab-retry-badge');
        if (retryBadge) retryBadge.remove();
    }
}

// Human in the Loop state
let hitlActiveSteps = [];

// Generate Action
generateBtn.addEventListener('click', async () => {
    const topic = topicInput.value.trim();
    if (!topic) return alert("Please enter a topic");

    const sources = sourcesInput.value.split("\n").map(s => s.trim()).filter(s => s);

    // Get Human in the Loop settings
    const hitlEnabled = humanInLoopCheck.checked;
    
    // UI Reset
    const controller = beginGenerationRequest();
    resultArea.classList.remove('hidden');
    errorArea.classList.add('hidden');
    reviewBadgeContainer.classList.add('hidden');
    finalArticle = '';
    // Reset Draft tab badge
    resetDraftTabBadge();

    hitlActiveSteps = PIPELINE_STEPS;
    renderPipeline();
    
    hitlActiveSteps.forEach(s => updateStepState(s.key, 'idle'));
    updateStepState(hitlActiveSteps[0].key, 'running');
    activateTabForStep(hitlActiveSteps[0].key);

    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                topic,
                sources,
                enableWebSearch: webSearchCheck.checked,
                hitlEnabled,
                hitlStages: hitlEnabled ? ['planner'] : [],
                modelConfig: {
                    researcher: document.getElementById('model-researcher').value,
                    planner: document.getElementById('model-planner').value,
                    writer: document.getElementById('model-writer').value,
                    reviewer: document.getElementById('model-reviewer').value
                }
            }),
            signal: controller.signal
        });

        await consumeStream(response);
    } catch (err) {
        if (!controller.signal.aborted) showError(err.message);
    } finally {
        finishGenerationRequest(controller);
    }
});

// ─── Stream consumption ────────────────────────────────────────────────────

async function consumeStream(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const event = JSON.parse(line.slice(6));

            if (event.status === "error") {
                updateStepState(event.node, 'error');
                throw new Error(event.error);
            }

            if (event.node === "hitl_pause") {
                if (event.data?.stage === 'planner') {
                    renderHitlPane(event.data);
                } else {
                    resumePipeline(event.data?.fullState, event.data?.nextNode);
                }
                return; // stream ends here
            }

            if (event.node === "complete") {
                if (event.data?.article) finalArticle = event.data.article;
                if (event.data?.researchSummary) {
                    researchPane.innerHTML = marked.parse(event.data.researchSummary);
                }
                PIPELINE_STEPS.forEach(step => updateStepState(step.key, 'done'));
                if (finalArticle) {
                    finalPane.innerHTML = renderCopyArticleButton() + marked.parse(finalArticle);
                }
                activateFinalTab();
                return;
            }

            // Node lifecycle
            if (event.status === "running") {
                updateStepState(event.node, 'running');
                activateTabForStep(event.node);
                continue; // no data yet
            }

            if (event.status === "done") {
                updateStepState(event.node, 'done');
                applyNodeOutput(event.node, event.data);
                continue;
            }

            // Legacy format fallback (node with data)
            updateStepState(event.node, 'done');
            applyNodeOutput(event.node, event.data);
        }
    }

}

// ─── Apply node output to UI ───────────────────────────────────────────────

function applyNodeOutput(nodeName, data) {
    if (!data) return;

    if (nodeName === "researcher" && data.researchSummary) {
        researchPane.innerHTML = marked.parse(data.researchSummary);
    }
    if (nodeName === "planner" && data.outline) {
        renderOutlineInPane(outlinePane, data.outline);
        draftPane.innerHTML = `<h1>${data.outline.seoTitle}</h1><p style="color:var(--text-muted);font-style:italic;margin-bottom:1.5rem;">${data.outline.metaDescription}</p><div class="spinner"></div>`;
    }
    if (nodeName === "writer" && data.article) {
        draftPane.innerHTML = marked.parse(data.article);
    }
    if (nodeName === "reviewer" && data.review) {
        finalArticle = data.article || finalArticle;
        draftPane.innerHTML = renderArticleComparison(
            data.originalArticle,
            finalArticle,
            'Draft comparison',
            'First draft',
            'After review'
        );
        renderReview(data.review, data.retryCount, data.originalArticle, data.reviewScores);
    }
}

// ─── Human in the Loop inline pane ─────────────────────────────────────────

function getHitlPaneForStage(stage) {
    switch (stage) {
        case 'researcher': return researchPane;
        case 'planner': return outlinePane;
        case 'writer': return draftPane;
        case 'reviewer': return reviewPane;
    }
    return draftPane;
}

function renderHitlMarkdownEditor(value, rows = 18) {
    return `
        <div class="hitl-markdown-editor">
            <div class="hitl-view-switch" role="group" aria-label="Choose preview or Markdown editing">
                <button type="button" class="active" data-hitl-mode="preview" aria-pressed="true">Preview</button>
                <button type="button" data-hitl-mode="edit" aria-pressed="false">Edit Markdown</button>
            </div>
            <div id="hitl-markdown-preview" class="prose hitl-markdown-preview">${marked.parse(value || '')}</div>
            <textarea id="hitl-edit-area" class="hitl-textarea hidden" rows="${rows}">${escapeHtml(value || '')}</textarea>
        </div>
    `;
}

function setupHitlMarkdownEditor(pane) {
    const textarea = pane.querySelector('#hitl-edit-area');
    const preview = pane.querySelector('#hitl-markdown-preview');
    if (!textarea || !preview) return;

    textarea.addEventListener('input', () => {
        preview.innerHTML = marked.parse(textarea.value);
    });

    pane.querySelectorAll('[data-hitl-mode]').forEach(button => {
        button.addEventListener('click', () => {
            const previewMode = button.dataset.hitlMode === 'preview';
            preview.classList.toggle('hidden', !previewMode);
            textarea.classList.toggle('hidden', previewMode);
            pane.querySelectorAll('[data-hitl-mode]').forEach(modeButton => {
                const active = modeButton === button;
                modeButton.classList.toggle('active', active);
                modeButton.setAttribute('aria-pressed', String(active));
            });
            if (previewMode) preview.innerHTML = marked.parse(textarea.value);
        });
    });
}

function renderHitlPane(hitlData) {
    const stage = hitlData.stage;
    const fullState = hitlData.fullState;
    const stageDef = PIPELINE_STEPS.find(s => s.key === stage);
    const stageLabel = stageDef ? stageDef.label : stage;
    const pane = getHitlPaneForStage(stage);

    activateTabForStep(stage);

    // Build the editable field based on stage
    let editorHtml = '';
    if (stage === 'researcher') {
        editorHtml = renderHitlMarkdownEditor(fullState.researchSummary || '', 14);
    } else if (stage === 'planner') {
        editorHtml = `
            <textarea id="hitl-edit-area" class="hitl-textarea" rows="18">${escapeHtml(JSON.stringify(fullState.outline, null, 2))}</textarea>
        `;
    } else if (stage === 'writer') {
        editorHtml = renderHitlMarkdownEditor(fullState.article || '');
    } else if (stage === 'reviewer') {
        const scoreBefore = fullState.review?.scoreBefore ?? fullState.review?.score;
        const scoreAfter = fullState.review?.scoreAfter;
        editorHtml = `
            <div class="hitl-review-score">Quality before/after: <strong>${Number.isFinite(scoreBefore) ? scoreBefore : '—'} → ${Number.isFinite(scoreAfter) ? scoreAfter : '—'}/100</strong></div>
            ${renderHitlMarkdownEditor(fullState.article || '')}
        `;
    }

    pane.innerHTML = `
        <div class="hitl-container">
            <div class="hitl-header">
                <div>
                    <div class="hitl-title">👤 Human in the Loop: ${stageLabel}</div>
                    <div class="hitl-subtitle">Review the formatted output, switch to Edit Markdown to make changes, then approve to continue.</div>
                </div>
            </div>
            ${editorHtml}
            <div id="hitl-validation-error" class="hitl-validation-error hidden" role="alert" aria-live="assertive"></div>
            <div class="hitl-actions">
                <button id="hitl-stop" class="hitl-btn hitl-btn-stop">✕ Stop Pipeline</button>
                <button id="hitl-approve" class="hitl-btn hitl-btn-approve">✓ Approve & Continue</button>
            </div>
        </div>
    `;

    setupHitlMarkdownEditor(pane);

    const approveBtn = document.getElementById('hitl-approve');
    const stopBtn = document.getElementById('hitl-stop');

    approveBtn.addEventListener('click', () => {
        let edited;
        try {
            edited = collectEditedState(stage, fullState);
        } catch (error) {
            const validationError = document.getElementById('hitl-validation-error');
            validationError.textContent = error.message;
            validationError.classList.remove('hidden');
            return;
        }

        // Replace the HITL editor with the (edited) stage output right away
        renderStageOutputOnly(stage, edited);
        resumePipeline(edited, hitlData.nextNode);
    });

    stopBtn.addEventListener('click', () => {
        stopPipeline();
    });
}

// Merge user edits back into the full state
function collectEditedState(stage, fullState) {
    const textarea = document.getElementById('hitl-edit-area');
    if (!textarea) return fullState;

    const newState = { ...fullState };

    if (stage === 'researcher') {
        newState.researchSummary = textarea.value;
    } else if (stage === 'planner') {
        newState.outline = parseEditedOutline(textarea.value);
    } else if (stage === 'writer' || stage === 'reviewer') {
        newState.article = textarea.value;
        if (stage === 'reviewer' && newState.review) {
            newState.review = { ...newState.review, improvedArticle: textarea.value };
        }
    }

    return newState;
}

function parseEditedOutline(value) {
    let outline;
    try {
        outline = JSON.parse(value);
    } catch (error) {
        throw new Error(`Invalid outline JSON: ${error.message}`);
    }

    if (!outline || typeof outline !== 'object' || Array.isArray(outline)) {
        throw new Error('The outline must be a JSON object.');
    }

    const issues = [];
    if (typeof outline.seoTitle !== 'string') issues.push('seoTitle must be a string');
    if (typeof outline.metaDescription !== 'string') issues.push('metaDescription must be a string');
    if (!Array.isArray(outline.sections)) {
        issues.push('sections must be an array');
    } else {
        outline.sections.forEach((section, index) => {
            if (!section || typeof section !== 'object' || Array.isArray(section)) {
                issues.push(`sections[${index}] must be an object`);
                return;
            }
            if (typeof section.heading !== 'string') issues.push(`sections[${index}].heading must be a string`);
            if (!Array.isArray(section.keyPoints) || !section.keyPoints.every(point => typeof point === 'string')) {
                issues.push(`sections[${index}].keyPoints must be an array of strings`);
            }
        });
    }
    if (typeof outline.estimatedReadingTime !== 'number' || !Number.isFinite(outline.estimatedReadingTime)) {
        issues.push('estimatedReadingTime must be a number');
    }

    if (issues.length > 0) {
        throw new Error(`The outline format is invalid: ${issues.join('; ')}.`);
    }

    return outline;
}

async function resumePipeline(fullState, nextNode) {
    if (!nextNode) {
        // Pipeline finished (e.g. reviewer approved with good score)
        if (fullState.article) {
            finalArticle = fullState.article;
            finalPane.innerHTML = renderCopyArticleButton() + marked.parse(finalArticle);
        }
        // Render the reviewer's Review tab content too
        if (fullState.review) {
            draftPane.innerHTML = renderArticleComparison(
                fullState.originalArticle,
                fullState.article,
                'Draft comparison',
                'First draft',
                'After review'
            );
            renderReview(fullState.review, fullState.retryCount, fullState.originalArticle, fullState.reviewScores);
        }
        // Show the final article
        activateFinalTab();
        // When reviewer is NOT selected for HITL, move review tab to final
        generateBtn.disabled = false;
        generateBtnText.textContent = 'Generate';
        return;
    }

    // Move UI forward
    const stepIdx = hitlActiveSteps.findIndex(s => s.key === nextNode);
    if (stepIdx > -1) {
        updateStepState(nextNode, 'running');
        activateTabForStep(nextNode);
    }

    const controller = beginGenerationRequest();
    try {
        const response = await fetch('/api/generate/continue', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                fullState,
                nextNode,
                hitlEnabled: false,
                hitlStages: []
            }),
            signal: controller.signal
        });

        await consumeStream(response);
    } catch (err) {
        if (!controller.signal.aborted) showError(err.message);
    } finally {
        finishGenerationRequest(controller);
    }
}

function stopPipeline() {
    // Full reset: all stages back to idle, no spinners, no done marks
    hitlActiveSteps.forEach(s => updateStepState(s.key, 'idle'));
    if (hitlActiveSteps.length > 0) {
        updateStepState(hitlActiveSteps[0].key, 'idle');
    }

    // Reset status indicators
    if (reviewBadgeContainer) reviewBadgeContainer.classList.add('hidden');
    if (errorArea) errorArea.classList.add('hidden');

    // Restore initial pipeline placeholder guides
    renderInitialPlaceholders();
    finalArticle = '';

    // Reset Draft tab badge
    resetDraftTabBadge();

    generateBtn.disabled = false;
    generateBtnText.textContent = 'Generate';
}

function renderInitialPlaceholders() {
    if (researchPane) {
        researchPane.innerHTML = `
            <div class="stage-placeholder">
                <div class="placeholder-icon">🔬</div>
                <h4>Researcher Agent Pipeline</h4>
                <p>Uses live web sources when enabled. When web search is off, the Researcher uses the language model's general knowledge and any references you provide. Findings and source details will appear here.</p>
                <span class="placeholder-hint">Click <strong>Generate</strong> to run the editorial pipeline</span>
            </div>`;
    }
    if (outlinePane) {
        outlinePane.innerHTML = `
            <div class="stage-placeholder">
                <div class="placeholder-icon">📋</div>
                <h4>Planner Agent Pipeline</h4>
                <p>Transforms raw research into an engaging, structured blog outline with headings, key points, and SEO metadata.</p>
                <span class="placeholder-hint">Awaiting pipeline execution</span>
            </div>`;
    }
    if (draftPane) {
        draftPane.innerHTML = `
            <div class="stage-placeholder">
                <div class="placeholder-icon">✍️</div>
                <h4>Writer Agent Pipeline</h4>
                <p>Synthesizes the outline and gathered facts into a comprehensive, high-quality blog post with engaging flow.</p>
                <span class="placeholder-hint">Awaiting pipeline execution</span>
            </div>`;
    }
    if (reviewPane) {
        reviewPane.innerHTML = `
            <div class="stage-placeholder">
                <div class="placeholder-icon">🔍</div>
                <h4>Reviewer Agent Pipeline</h4>
                <p>Critiques readability, scores quality (0–100), and provides actionable editorial suggestions to polish the post.</p>
                <span class="placeholder-hint">Awaiting pipeline execution</span>
            </div>`;
    }
    if (finalPane) {
        finalPane.innerHTML = `
            <div class="stage-placeholder">
                <div class="placeholder-icon">✨</div>
                <h4>Final Published Article</h4>
                <p>The complete, polished blog post ready for distribution, featuring a one-click copy tool.</p>
                <span class="placeholder-hint">Awaiting pipeline completion</span>
            </div>`;
    }
}

// Render just the stage output (no HITL chrome) — used after Stop
function renderStageOutputOnly(stage, fullState) {
    if (!fullState) return;
    const pane = getHitlPaneForStage(stage);

    if (stage === 'researcher') {
        pane.innerHTML = marked.parse(fullState.researchSummary || '');
    } else if (stage === 'planner') {
        renderOutlineInPane(pane, fullState.outline);
    } else if (stage === 'writer') {
        pane.innerHTML = marked.parse(fullState.article || '');
    } else if (stage === 'reviewer') {
        renderStageOutputOnly('writer', fullState); // reviewer output lives in draft
        if (fullState.review) {
            const scoreBefore = fullState.review.scoreBefore ?? fullState.review.score;
            const scoreAfter = fullState.review.scoreAfter;
            reviewBadgeContainer.innerHTML = `<div class="badge badge-emerald">Quality: ${Number.isFinite(scoreBefore) ? scoreBefore : '—'} before → ${Number.isFinite(scoreAfter) ? scoreAfter : '—'} after</div>`;
            reviewBadgeContainer.classList.remove('hidden');
        }
    }
}

function renderOutlineInPane(pane, outline) {
    if (!outline) return;
    let html = `
        <div class="flex items-center gap-4 text-sm" style="color: var(--text-muted); margin-bottom: 1.5rem;">
            <span>📖 ${outline.estimatedReadingTime} min read</span>
            <span>📑 ${outline.sections.length} sections</span>
        </div>
    `;
    outline.sections.forEach(sec => {
        html += `
        <div style="background: var(--bg-page); border: 1px solid var(--border-color); border-radius: 0.75rem; padding: 1rem; margin-bottom: 1rem;">
            <h3 style="font-size: 1.125rem; font-weight: 600; margin-bottom: 0.5rem;">${sec.heading}</h3>
            <ul style="list-style: none;">
                ${sec.keyPoints.map(pt => `<li class="flex items-center gap-2" style="font-size: 0.875rem; color: var(--text-muted); margin-bottom: 0.25rem;"><span style="color: var(--primary);">•</span> ${pt}</li>`).join('')}
            </ul>
        </div>`;
    });
    pane.innerHTML = html;
}

function showError(message) {
    errorArea.classList.remove('hidden');
    errorText.textContent = message;
    generateBtn.disabled = false;
    generateBtnText.textContent = 'Generate';

    const runningStep = document.querySelector('.tab-btn.step-running')?.dataset.step;
    if (runningStep) updateStepState(runningStep, 'error');
}

function escapeHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

function copyArticle() {
    const text = finalPane.innerText || finalPane.textContent || '';
    if (!text.trim()) return;
    navigator.clipboard.writeText(text).then(() => {
        const btn = document.getElementById('copyArticleBtn');
        const original = btn.innerHTML;
        btn.innerHTML = '<svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M20 6L9 17l-5-5"/></svg> <span>Copied!</span>';
        setTimeout(() => btn.innerHTML = original, 1500);
    }).catch(() => {
        alert('Failed to copy article');
    });
}

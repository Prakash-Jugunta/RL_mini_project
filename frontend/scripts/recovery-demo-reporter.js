import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, relative } from 'node:path';

const escape = (value) => String(value ?? '').replace(/[&<>"']/g, (character) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
})[character]);

const labels = {
  dqn: 'Trained DQN', llm: 'Model tiebreaker', unresolved: 'Unresolved',
  supported_top_match: 'The top-ranked candidate matched the original name and form scope.',
  multiple_controls_share_the_original_name: 'Multiple controls shared the original accessible name.',
  top_candidate_did_not_match_the_original_fingerprint: 'The top candidate differed from the original name or form scope.',
  q_value_gap_below_threshold: 'The two highest Q-values were too close.',
  model_selected: 'The model returned a candidate from the shortlist.',
  model_unconfigured: 'The model was unavailable. Playwright left the step unresolved.',
  model_abstained: 'The model abstained.',
  model_error: 'The model request failed.',
  invalid_model_choice: 'The model returned a candidate outside the shortlist.',
  no_candidates: 'No eligible controls were found.',
};

function attachment(result, name) {
  const item = result.attachments.find((entry) => entry.name === name);
  if (!item) return null;
  return JSON.parse(item.body?.toString('utf8') ?? readFileSync(item.path, 'utf8'));
}

function scenario({ test, result }, output) {
  const recovery = attachment(result, 'search-recovery');
  const assertion = attachment(result, 'sku-assertion');
  const action = attachment(result, 'search-action');
  const regression = test.expectedStatus === 'failed' && result.status === 'failed' && assertion?.passed === false;
  const success = test.expectedStatus === result.status && (test.expectedStatus !== 'failed' || regression);
  const status = regression ? 'Regression caught' : result.status === 'skipped' ? 'Skipped' : success ? 'Passed' : 'Failed';
  const model = recovery?.model_mode === 'mock' ? 'Local Responses API mock' : recovery?.model;
  const route = recovery ? labels[recovery.source] : assertion ? 'Original locator' : 'No decision recorded';
  const rankings = recovery?.ranked ?? [];
  const evidence = result.attachments.filter((item) => item.contentType === 'image/png' && (item.body || item.path));

  return `<article class="case"><header><div><p class="route">${escape(route)} · ${(result.duration / 1000).toFixed(1)} s</p>
    <h2>${escape(test.title)}</h2></div><span class="status ${success ? 'good' : result.status === 'skipped' ? 'quiet' : 'bad'}">${status}</span></header>
    ${recovery ? `<dl class="details"><div><dt>Original locator</dt><dd><code>${escape(recovery.original_locator)}</code> · ${escape(recovery.original_matches)} matches</dd></div>
      <div><dt>Recovery decision time</dt><dd>${escape(recovery.decision_ms)} ms</dd></div>
      <div><dt>Selected candidate</dt><dd>${recovery.candidate_id == null ? 'None' : `#${escape(recovery.candidate_id)}`}</dd></div>
      <div><dt>Replacement clicked</dt><dd>${action?.clicked ? 'Yes' : 'No'}</dd></div>
      ${model ? `<div><dt>Model</dt><dd>${escape(model)}</dd></div>` : ''}</dl>
      <p>${escape(labels[recovery.review_reason] ?? '')} ${escape(labels[recovery.reason] ?? recovery.reason)}</p>` : ''}
    ${rankings.length ? `<table><caption>DQN rankings</caption><thead><tr><th>Candidate</th><th>Control</th><th>Form scope</th><th>Q-value</th><th>Selected</th></tr></thead><tbody>
      ${rankings.map((entry) => `<tr><td>#${escape(entry.candidate_id)}</td><td>${escape(entry.name)}</td><td>${escape(entry.scope)}</td>
        <td>${entry.score.toFixed(3)}</td><td>${entry.candidate_id === recovery.candidate_id ? 'Yes' : '–'}</td></tr>`).join('')}</tbody></table>` : ''}
    ${assertion ? `<p class="assertion"><strong>Original SKU assertion</strong> · Expected <code>${escape(assertion.expected)}</code> · Observed <code>${escape(assertion.actual ?? 'none')}</code> · ${assertion.passed ? 'Pass' : 'Fail'}</p>` : ''}
    ${!success && result.errors.length ? `<pre>${escape(result.errors.map((error) => error.message).join('\n'))}</pre>` : ''}
    ${evidence.length ? `<div class="evidence">${evidence.map((item, index) => {
      const filename = `${test.id.replace(/[^\w-]/g, '_')}-${index + 1}.png`;
      const imagePath = item.path || join(output, 'screenshots', filename);
      if (!item.path) writeFileSync(imagePath, item.body);
      const path = relative(output, imagePath).split(/[\\/]/).map(encodeURIComponent).join('/');
      return `<figure><a href="${path}"><img src="${path}" alt="${escape(item.name)}" loading="lazy"></a><figcaption>${escape(item.name)}</figcaption></figure>`;
    }).join('')}</div>` : ''}</article>`;
}

const workflow = `<section class="workflow" aria-labelledby="workflow-title">
  <h2 id="workflow-title">Recovery workflow</h2>
  <div class="flow-start"><div><strong>Playwright</strong><span>Run the original locator</span></div><b aria-hidden="true">→</b><div><strong>One match?</strong><span>Check the live page</span></div></div>
  <div class="branches">
    <div><p class="route">Unique locator</p><strong>Use the original control</strong></div>
    <div><p class="route">Missing or ambiguous locator</p><strong>Capture public DOM candidates → DQN ranks them</strong>
      <div class="choices"><span>Clear match → use DQN pick</span><span>Ambiguous → ask LLM for a candidate ID</span></div>
      <small>If there is no supported candidate, leave the step unresolved.</small>
    </div>
  </div>
  <div class="flow-end"><strong>Playwright validates the target and runs the original action</strong>
    <b aria-hidden="true">↓</b><strong>Original test assertion determines pass or failure</strong></div>
</section>`;

const styles = `
  :root{color:#182727;background:#edf1ef;font:16px/1.5 system-ui,sans-serif}
  body{max-width:1140px;margin:auto;padding:clamp(1rem,4vw,3rem)}
  h1,h2,p{margin:0 0 .65rem}h1{font-size:clamp(2.3rem,6vw,4.5rem);line-height:1;letter-spacing:-.06em}
  h2{font-size:1.3rem;line-height:1.2}a{color:#154ca8}a:focus-visible{outline:3px solid #235acc;outline-offset:4px}
  .hero{padding:clamp(1.5rem,5vw,3.5rem);background:#182b2a;color:white;border-radius:16px}.hero p{max-width:65ch;color:#d2e7df}
  .hero a{color:#b6dbff}.route{color:#43605d;font-size:.85rem;font-weight:700}.hero .route{color:#a7dace}
  .workflow,.case{background:white;border:1px solid #d2ded9;border-radius:12px;padding:clamp(1rem,3vw,1.75rem);margin-top:1.25rem}
  .flow-start,.flow-end,.branches{display:flex;justify-content:center;align-items:stretch;gap:1rem;margin:1rem 0;text-align:center}
  .flow-start div,.branches>div,.flow-end strong{background:#edf5f2;border:1px solid #c4d9d0;border-radius:8px;padding:.7rem 1rem}
  .flow-start b{align-self:center}.flow-start span{display:block}.branches>div{flex:1}.branches>div:last-child{flex:2}
  .choices{display:flex;justify-content:center;gap:.65rem;flex-wrap:wrap;margin:.7rem 0}.choices span{background:white;border-radius:6px;padding:.35rem .65rem;font-size:.9rem}
  .flow-end{flex-direction:column;align-items:center;gap:.35rem;margin-bottom:0}
  .case header,.details,.evidence{display:flex;gap:1rem;flex-wrap:wrap}.case header{justify-content:space-between;align-items:start}
  .status{padding:.3rem .65rem;border-radius:5px;font-size:.85rem;font-weight:700}.good{background:#dff4e8;color:#175538}.bad{background:#ffe1de;color:#842a20}.quiet{background:#e8edeb;color:#41514f}
  .details{margin:.5rem 0 1rem}.details div{padding:.5rem .75rem;background:#f3f6f5;border-radius:6px}.details dt{font-size:.8rem;color:#465654}.details dd{margin:0;font-weight:700}
  table{border-collapse:collapse;width:100%;margin:1rem 0;text-align:left;font-size:.9rem}caption{text-align:left;font-weight:700;margin-bottom:.4rem}th,td{padding:.45rem .65rem;border-bottom:1px solid #d9e3df}th{background:#f3f6f5}
  .assertion{margin-top:1rem}.evidence figure{max-width:345px;flex:1 1 230px;margin:0}.evidence img{width:100%;border:1px solid #d2ded9;border-radius:6px}.evidence figcaption{font-size:.8rem;color:#43605d}pre{overflow:auto;background:#f7f5f5;padding:.75rem}
  @media(max-width:630px){.branches{flex-direction:column}.flow-start{gap:.35rem}th,td{padding:.3rem}.case{overflow-wrap:anywhere}}
`;

export default class RecoveryDemoReporter {
  constructor({ output }) {
    this.output = output;
    this.cases = [];
    this.errors = [];
  }

  onTestEnd(test, result) { this.cases.push({ test, result }); }
  onError(error) { this.errors.push(error.message); }

  onEnd(result) {
    const exercised = this.cases.filter(({ result: testResult }) => testResult.status !== 'skipped').length;
    mkdirSync(join(this.output, 'screenshots'), { recursive: true });
    writeFileSync(join(this.output, 'index.html'), `<!doctype html><html lang="en"><meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1"><title>Fielddesk Recovery Report</title><style>${styles}</style>
      <main><header class="hero"><p class="route">Fielddesk · Playwright recovery</p><h1>Self-healing test report</h1>
      <p>Run ${escape(result.status)} · ${exercised} scenarios executed · ${(result.duration / 1000).toFixed(1)} seconds.</p>
      <p>The Q-value is the trained agent’s estimate of future reward for a candidate action.</p>
      <a href="playwright-report/index.html">Full Playwright report</a> · <a href="results.json">Raw results</a></header>
      ${workflow}${this.errors.length ? `<pre>${escape(this.errors.join('\n'))}</pre>` : ''}
      ${this.cases.map((entry) => scenario(entry, this.output)).join('')}</main></html>`);
  }
}

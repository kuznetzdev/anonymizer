import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { buildFeedbackPayload, countFeedbackReason, examplePayloadSignature } from './feedback_payload.js';
import { isSnapshotPending, requestLogout } from './session.js';

const SAMPLE = 'Иванов Иван Иванович звонил по телефону +7911232333 и не дозвонился. Звонил он в боткинскую больницу';
const REASONS = [
  ['missed_span', 'Пропущенный фрагмент'],
  ['false_positive', 'Лишний фрагмент'],
  ['wrong_label', 'Неверная категория'],
  ['wrong_boundary', 'Неверные границы'],
  ['other', 'Другая причина'],
];
const CORRECTION_REASONS = new Set(['wrong_label', 'wrong_boundary', 'other']);

function cpLength(value) { return Array.from(value).length; }
function cpToUtf16(value, offset) { return Array.from(value).slice(0, offset).join('').length; }
function utf16ToCp(value, offset) { return cpLength(value.slice(0, offset)); }
function formatMetric(value) { return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—'; }
function preventPasswordClipboard(event) { event.preventDefault(); }

const BUILT_IN_CATEGORIES = ['ADDR', 'APART', 'DISEASE', 'EMAIL', 'HOUSE', 'INN', 'INSTITUTION', 'KPP', 'OGRN', 'ORG', 'PASSPORT', 'PER', 'PHONE', 'SNILS', 'STREET'];
function categoryColor(hue) {
  return {
    '--category-bg': `hsl(${hue} 78% 94%)`,
    '--category-color': `hsl(${hue} 48% 27%)`,
    '--category-border': `hsl(${hue} 55% 76%)`,
  };
}
function categoryStylesFor(labels) {
  const styles = new Map();
  const usedHues = new Set();
  BUILT_IN_CATEGORIES.forEach((label, index) => {
    const hue = Math.round(index * 360 / BUILT_IN_CATEGORIES.length);
    usedHues.add(hue);
    styles.set(label, categoryColor(hue));
  });
  const customLabels = [...new Set(labels)].filter((label) => !styles.has(label)).sort();
  for (const label of customLabels) {
    let hash = 0;
    for (const char of label) hash = (hash * 31 + char.codePointAt(0)) >>> 0;
    let hue = hash % 360;
    for (let offset = 0; usedHues.has(hue) && offset < 360; offset += 1) hue = (hue + 137) % 360;
    usedHues.add(hue);
    styles.set(label, categoryColor(hue));
  }
  return styles;
}

function ModelMetrics({ metrics }) {
  if (!metrics) return <p className="metrics-empty">Метрики появятся после обучения модели.</p>;
  const labels = Object.entries(metrics.per_label ?? {}).sort(([left], [right]) => left.localeCompare(right));
  return <div className="model-metrics">
    <div className="metrics-summary">
      <div className="metric-primary"><span>F1 сущностей · micro</span><strong>{formatMetric(metrics.entity_micro_f1)}</strong></div>
      <div><span>Точность · micro</span><strong>{formatMetric(metrics.entity_micro_precision)}</strong></div>
      <div><span>Полнота · micro</span><strong>{formatMetric(metrics.entity_micro_recall)}</strong></div>
      <div><span>F1 сущностей · macro</span><strong>{formatMetric(metrics.entity_macro_f1)}</strong></div>
    </div>
    <details className="per-label-details">
      <summary>Метрики по категориям <span>({labels.length})</span></summary>
      {labels.length ? <div className="metrics-table-wrap"><table><thead><tr><th>Категория</th><th>Точность</th><th>Полнота</th><th>F1</th><th>Примеров</th></tr></thead><tbody>{labels.map(([label, values]) => <tr key={label}><th scope="row">{label}</th><td>{formatMetric(values.precision)}</td><td>{formatMetric(values.recall)}</td><td>{formatMetric(values.f1)}</td><td>{values.support ?? '—'}</td></tr>)}</tbody></table></div> : <p className="metrics-empty">Нет данных по категориям.</p>}
    </details>
    <p className="token-diagnostic">Диагностика токенов · Token micro F1: {formatMetric(metrics.token_micro_f1)}</p>
  </div>;
}
function idFor() { return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`; }

async function request(path, options = {}) {
  const response = await fetch(path, {
    credentials: 'same-origin',
    ...options,
    headers: { ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...options.headers },
  });
  if (response.status === 401) throw Object.assign(new Error('Сессия завершена. Войдите снова.'), { status: 401 });
  if (!response.ok) {
    let detail = 'Не удалось выполнить запрос';
    try { detail = (await response.json()).detail || detail; } catch { /* empty error body */ }
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

function App() {
  const [auth, setAuth] = useState('checking');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [authError, setAuthError] = useState('');
  const [loginPending, setLoginPending] = useState(false);
  const [status, setStatus] = useState(null);
  const [categories, setCategories] = useState([]);
  const [corpusReleases, setCorpusReleases] = useState([]);
  const [trainingReleaseIds, setTrainingReleaseIds] = useState([]);
  const [modelVersions, setModelVersions] = useState([]);
  const [corpusArchive, setCorpusArchive] = useState(null);
  const [text, setText] = useState('');
  const [annotationText, setAnnotationText] = useState(null);
  const [entities, setEntities] = useState([]);
  const [feedback, setFeedback] = useState({});
  const [emptyReviewConfirmed, setEmptyReviewConfirmed] = useState(false);
  const [savedPayloadKey, setSavedPayloadKey] = useState(null);
  const [selection, setSelection] = useState(null);
  const [selectionCategory, setSelectionCategory] = useState('');
  const [newLabel, setNewLabel] = useState('');
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [training, setTraining] = useState(null);
  const [trainingJobId, setTrainingJobId] = useState(null);
  const [uploadingCorpus, setUploadingCorpus] = useState(false);
  const [activeTab, setActiveTab] = useState('processing');
  const [advancedModelView, setAdvancedModelView] = useState(false);
  const [examplePages, setExamplePages] = useState({});
  const [expandedCategories, setExpandedCategories] = useState({});
  const [editingExample, setEditingExample] = useState(null);
  const loadingExampleKeys = useRef(new Set());
  const textareaRef = useRef(null);
  const highlightRef = useRef(null);
  const entityRefs = useRef({});
  const saveInFlightRef = useRef(false);
  const apiRequest = useCallback(async (path, options) => {
    try { return await request(path, options); }
    catch (reason) {
      if (reason.status === 401) { setError(''); setAuth('signed-out'); setAuthError(reason.message); }
      throw reason;
    }
  }, []);

  useEffect(() => {
    if (!notice) return undefined;
    const timer = window.setTimeout(() => setNotice(''), 5000);
    return () => window.clearTimeout(timer);
  }, [notice]);
  const loadExamples = useCallback(async (category = null, append = false) => {
    const key = category ?? '*';
    if (loadingExampleKeys.current.has(key)) return;
    loadingExampleKeys.current.add(key);
    const cursor = append ? examplePages[key]?.nextCursor : null;
    setExamplePages((pages) => ({ ...pages, [key]: { ...(pages[key] ?? { items: [], nextCursor: null }), loading: true, error: '' } }));
    const params = new URLSearchParams({ limit: '50' });
    if (category) params.set('category', category);
    if (cursor !== null && cursor !== undefined) params.set('before_id', String(cursor));
    try {
      const result = await apiRequest(`/api/examples?${params.toString()}`);
      setExamplePages((pages) => {
        const previous = append ? pages[key]?.items ?? [] : [];
        const seen = new Set(previous.map((item) => String(item.id)));
        const items = [...previous, ...(result.items ?? []).filter((item) => !seen.has(String(item.id)))];
        return { ...pages, [key]: { items, nextCursor: result.next_cursor ?? null, loading: false, error: '' } };
      });
    } catch (reason) {
      setExamplePages((pages) => ({ ...pages, [key]: { ...(pages[key] ?? { items: [], nextCursor: null }), loading: false, error: reason.message } }));
    } finally { loadingExampleKeys.current.delete(key); }
  }, [apiRequest, examplePages]);

  useEffect(() => {
    if (activeTab === 'examples' && !examplePages['*']) loadExamples();
  }, [activeTab, examplePages, loadExamples]);

  const activeCategories = useMemo(() => categories.filter((item) => item.active).map((item) => item.label), [categories]);
  const categoryStyles = useMemo(() => categoryStylesFor([
    ...categories.map((category) => category.label), ...entities.map((entity) => entity.label),
  ]), [categories, entities]);
  const annotationsStale = annotationText !== null && annotationText !== text;
  const spanProblems = useMemo(() => {
    const ordered = entities.filter((item) => feedback[item.id]?.reason !== 'false_positive').sort((a, b) => a.start - b.start || a.end - b.end);
    const problems = new Set();
    let furthest = null;
    for (const entity of ordered) {
      if (!Number.isInteger(entity.start) || !Number.isInteger(entity.end) || entity.start < 0 || entity.end > cpLength(text) || entity.end <= entity.start) {
        problems.add(entity.id);
      }
      if (furthest && Number.isInteger(entity.start) && entity.start < furthest.end) {
        problems.add(entity.id); problems.add(furthest.id);
      }
      if (!furthest || entity.end > furthest.end) furthest = entity;
    }
    return problems;
  }, [entities, feedback, text]);
  const pendingEntities = entities.filter((entity) => entity.origin === 'model'
    && feedback[entity.id]?.reason !== 'false_positive'
    && !entity.confirmed
    && !(entity.edited && CORRECTION_REASONS.has(feedback[entity.id]?.reason)));
  const reviewedEntities = entities.filter((entity) => feedback[entity.id]?.reason !== 'false_positive');
  const falsePositiveCount = countFeedbackReason(feedback, 'false_positive');
  const reviewableEntities = entities.filter((entity) => feedback[entity.id]?.reason !== 'false_positive');
  const savePayload = {
    text,
    entities: reviewedEntities.map(({ start, end, label }) => ({ start, end, label })),
    empty_review_confirmed: !reviewedEntities.length && emptyReviewConfirmed,
    feedback: buildFeedbackPayload(feedback, entities),
  };
  const payloadKey = examplePayloadSignature(savePayload);
  const alreadySaved = savedPayloadKey === payloadKey;
  const canSave = !annotationsStale && !spanProblems.size && !pendingEntities.length && !alreadySaved
    && (reviewedEntities.length > 0 || falsePositiveCount > 0 || emptyReviewConfirmed);
  const reviewTotal = reviewableEntities.length + falsePositiveCount || (annotationText === text && text.trim() ? 1 : 0);
  const reviewedCount = annotationsStale ? 0 : falsePositiveCount + reviewableEntities.filter((entity) => !spanProblems.has(entity.id) && (entity.origin === 'manual' || entity.confirmed
    || (entity.edited && CORRECTION_REASONS.has(feedback[entity.id]?.reason)))).length
    || (reviewTotal && emptyReviewConfirmed ? 1 : 0);
  const snapshotPending = isSnapshotPending(busy);
  const displayedMetrics = training?.status === 'succeeded' && training.metrics
    ? training.metrics
    : status?.model?.metrics ?? status?.training_job?.metrics ?? null;
  const selectedCorpusReleases = corpusReleases.filter((release) => trainingReleaseIds.includes(release.release_id));
  const selectedCorpusLabel = selectedCorpusReleases.map((release) => `${release.corpus_id} ${release.corpus_version}`).join(' · ');
  const selectionOverlaps = selection && entities.some((entity) => feedback[entity.id]?.reason !== 'false_positive' && selection.start < entity.end && entity.start < selection.end);
  const maskedText = useMemo(() => {
    const ordered = [...entities].filter((item) => feedback[item.id]?.reason !== 'false_positive' && Number.isInteger(item.start) && Number.isInteger(item.end) && item.start >= 0 && item.end > item.start && item.end <= cpLength(text)).sort((a, b) => a.start - b.start || a.end - b.end);
    if (spanProblems.size) return text;
    let result = '';
    let cursor = 0;
    for (const entity of ordered) {
      const chars = Array.from(text);
      result += chars.slice(cursor, entity.start).join('') + `<${entity.label}>`;
      cursor = entity.end;
    }
    return result + Array.from(text).slice(cursor).join('');
  }, [entities, feedback, spanProblems, text]);
  const sourceMarkup = useMemo(() => {
    const ordered = entities.filter((entity) => feedback[entity.id]?.reason !== 'false_positive'
      && Number.isInteger(entity.start) && Number.isInteger(entity.end)
      && entity.start >= 0 && entity.end > entity.start && entity.end <= cpLength(text))
      .sort((a, b) => a.start - b.start || a.end - b.end);
    const chars = Array.from(text);
    const parts = [];
    let cursor = 0;
    for (const entity of ordered) {
      if (entity.start < cursor) continue;
      if (entity.start > cursor) parts.push(chars.slice(cursor, entity.start).join(''));
      parts.push(<mark key={entity.id} className="source-highlight" style={categoryStyles.get(entity.label)}>{chars.slice(entity.start, entity.end).join('')}</mark>);
      cursor = entity.end;
    }
    parts.push(chars.slice(cursor).join(''));
    return parts;
  }, [categoryStyles, entities, feedback, text]);
  const maskedMarkup = useMemo(() => {
    if (spanProblems.size) return maskedText;
    const ordered = [...entities].filter((entity) => feedback[entity.id]?.reason !== 'false_positive'
      && Number.isInteger(entity.start) && Number.isInteger(entity.end)
      && entity.start >= 0 && entity.end > entity.start && entity.end <= cpLength(text))
      .sort((a, b) => a.start - b.start || a.end - b.end);
    const chars = Array.from(text);
    const parts = [];
    let cursor = 0;
    for (const entity of ordered) {
      if (entity.start < cursor) continue;
      parts.push(chars.slice(cursor, entity.start).join(''));
      parts.push(<span key={entity.id} className="masked-replacement" style={categoryStyles.get(entity.label)}>{`<${entity.label}>`}</span>);
      cursor = entity.end;
    }
    parts.push(chars.slice(cursor).join(''));
    return parts;
  }, [categoryStyles, entities, feedback, maskedText, spanProblems, text]);

  const refresh = useCallback(async () => {
    const session = await request('/api/auth/me');
    if (!session.authenticated) { setAuth('signed-out'); setStatus(null); return; }
    const [currentStatus, categoryList, corpusList, modelList] = await Promise.all([
      request('/api/status'), request('/api/categories'), request('/api/corpora'), request('/api/models'),
    ]);
    setAuth('signed-in'); setStatus(currentStatus); setCategories(categoryList.categories ?? categoryList);
    setCorpusReleases(corpusList.releases ?? []);
    setTrainingReleaseIds((corpusList.training_releases ?? []).map((release) => release.release_id));
    setModelVersions(modelList.versions ?? []);
    setTraining(currentStatus.training_job ?? null);
    setTrainingJobId(currentStatus.training_job?.id ?? null);
  }, []);

  useEffect(() => {
    refresh().catch((reason) => {
      if (reason.status === 401) setAuth('signed-out');
      else { setAuth('signed-out'); setAuthError('Сервер пока недоступен. Проверьте подключение и попробуйте ещё раз.'); }
    });
  }, [refresh]);

  useEffect(() => {
    if (auth !== 'signed-in') return undefined;
    let stopped = false;
    const pollStatus = async () => {
      try {
        const next = await apiRequest('/api/status');
        if (stopped) return;
        setStatus(next);
        if (next.training_job) {
          setTraining(next.training_job);
          if (['queued', 'running'].includes(next.training_job.status)) setTrainingJobId(next.training_job.id);
        }
      } catch (reason) { if (!stopped && reason.status !== 401) setError(reason.message); }
    };
    const timer = window.setInterval(pollStatus, 10000);
    return () => { stopped = true; window.clearInterval(timer); };
  }, [apiRequest, auth]);

  useEffect(() => {
    if (!trainingJobId) return undefined;
    let stopped = false;
    const poll = async () => {
      try {
        const result = await apiRequest(`/api/training/${encodeURIComponent(trainingJobId)}`);
        if (stopped) return;
        setTraining(result);
        if (['completed', 'failed', 'succeeded', 'error'].includes(result.status)) {
          setTrainingJobId(null);
          const next = await apiRequest('/api/status');
          if (!stopped) {
            setStatus(next);
            const models = await apiRequest('/api/models');
            if (!stopped) setModelVersions(models.versions ?? []);
          }
        }
      } catch (reason) { if (!stopped) setError(reason.message); }
    };
    poll();
    const timer = window.setInterval(poll, 1800);
    return () => { stopped = true; window.clearInterval(timer); };
  }, [apiRequest, trainingJobId]);

  const runAnalyze = async (value = text) => {
    setEditingExample(null);
    setSelection(null); setSelectionCategory('');
    setEmptyReviewConfirmed(false);
    setError(''); setNotice(''); setBusy('analyze');
    try {
      const result = await apiRequest('/api/analyze', { method: 'POST', body: JSON.stringify({ text: value }) });
      setEntities(result.entities.map((entity) => ({ ...entity, id: entity.id ?? idFor(), origin: 'model', confirmed: false, edited: false })));
      setFeedback({}); setText(value); setAnnotationText(value); setNotice('Проверка завершена. Просмотрите найденные фрагменты.');
    } catch (reason) { setError(reason.message); }
    finally { setBusy(''); }
  };

  const loadSample = () => { if (snapshotPending) return; if (editingExample && !window.confirm('Загрузить демонстрационный текст? Несохранённые изменения примера будут потеряны.')) return; setEditingExample(null); setText(SAMPLE); setEntities([]); setFeedback({}); setEmptyReviewConfirmed(false); setSelection(null); setSelectionCategory(''); setAnnotationText(null); setNotice('Пример загружен. Нажмите «Проверить текст».'); setError(''); };
  const editEntity = (id, patch) => {
    const current = entities.find((item) => item.id === id);
    if (!current) return;
    const changedLabel = patch.label !== undefined && patch.label !== current.label;
    const changedRange = (patch.start !== undefined && patch.start !== current.start) || (patch.end !== undefined && patch.end !== current.end);
    const changed = changedLabel || changedRange;
    if (changed) setEmptyReviewConfirmed(false);
    setEntities((items) => items.map((item) => item.id === id ? { ...item, ...patch, ...(item.origin === 'model' && changed ? { edited: true, confirmed: false } : {}) } : item));
    if (['model', 'saved'].includes(current.origin) && changed) {
      const reason = changedLabel ? 'wrong_label' : 'wrong_boundary';
      setFeedback((items) => ({ ...items, [id]: { ...items[id], reason, start: patch.start ?? current.start, end: patch.end ?? current.end, label: patch.label ?? current.label, note: items[id]?.note ?? '' } }));
    }
  };
  const removeEntity = (id) => {
    const entity = entities.find((item) => item.id === id);
    setEmptyReviewConfirmed(false);
    setEntities((items) => items.filter((item) => item.id !== id));
    if (entity && ['model', 'saved'].includes(entity.origin)) setFeedback((items) => ({ ...items, [id]: { reason: 'false_positive', note: '', start: entity.start, end: entity.end, label: entity.label } }));
    else setFeedback((items) => { const next = { ...items }; delete next[id]; return next; });
  };

  const captureSelection = () => {
    if (snapshotPending) { setSelection(null); return; }
    const input = textareaRef.current;
    if (!input || input.selectionStart === input.selectionEnd) { setSelection(null); return; }
    const selectedText = input.value.slice(input.selectionStart, input.selectionEnd);
    const trimmedText = selectedText.trim();
    if (!trimmedText) { setSelection(null); return; }
    const leading = selectedText.length - selectedText.trimStart().length;
    const trailing = selectedText.length - selectedText.trimEnd().length;
    setSelection({ start: utf16ToCp(text, input.selectionStart) + cpLength(selectedText.slice(0, leading)), end: utf16ToCp(text, input.selectionEnd) - cpLength(selectedText.slice(selectedText.length - trailing)), text: trimmedText });
    setSelectionCategory('');
  };
  const addSelection = () => {
    if (!selection) return;
    if (!activeCategories.length) { setError('Нет активных категорий. Восстановите или добавьте категорию на вкладке «Категории».'); return; }
    if (!selectionCategory || !activeCategories.includes(selectionCategory)) { setError('Выберите категорию для фрагмента.'); return; }
    const overlaps = entities.some((entity) => feedback[entity.id]?.reason !== 'false_positive' && selection.start < entity.end && entity.start < selection.end);
    if (overlaps) { setError('Выделение пересекается с размеченным фрагментом. Измените категорию в его строке ниже.'); return; }
    setError('');
    setEmptyReviewConfirmed(false);
    const id = idFor();
    setEntities((items) => [...items, { id, start: selection.start, end: selection.end, text: selection.text, label: selectionCategory, origin: 'manual', confirmed: true, edited: false }]);
    setFeedback((items) => ({ ...items, [id]: { reason: 'missed_span', note: '', start: selection.start, end: selection.end, label: selectionCategory } }));
    setSelection(null);
    setSelectionCategory('');
  };
  const confirmEntity = (entity) => {
    setEntities((items) => items.map((item) => item.id === entity.id ? { ...item, confirmed: true } : item));
    setFeedback((items) => ({ ...items, [entity.id]: { reason: 'correct', note: '', start: entity.start, end: entity.end, label: entity.label } }));
  };
  const toggleFeedback = (entity) => {
    setEmptyReviewConfirmed(false);
    const current = feedback[entity.id];
    if (current?.reason === 'correct') {
      setEntities((items) => items.map((item) => item.id === entity.id ? { ...item, confirmed: false } : item));
      setFeedback((items) => ({ ...items, [entity.id]: { reason: 'wrong_label', note: '' } }));
    } else setFeedback((items) => {
      const next = { ...items };
      if (current) delete next[entity.id];
      else next[entity.id] = { reason: 'wrong_label', note: '' };
      return next;
    });
  };
  const focusRange = (entity) => {
    const input = textareaRef.current;
    if (!input) return;
    input.focus(); input.setSelectionRange(cpToUtf16(text, entity.start), cpToUtf16(text, entity.end));
  };
  const toggleCategoryExamples = (label) => {
    const expand = !expandedCategories[label];
    setExpandedCategories((items) => ({ ...items, [label]: expand }));
    if (expand && !examplePages[label]) loadExamples(label);
  };
  const openExample = (example) => {
    if (snapshotPending) return;
    if (editingExample) {
      if (!window.confirm(`Заменить открытую разметку примером ${example.id}? Несохранённые изменения будут потеряны.`)) return;
    } else if (text.trim() && text !== example.text
      && !window.confirm('Заменить текст в обработке выбранным сохранённым примером?')) return;
    const restoredEntities = (example.entities ?? []).map((entity) => ({
      ...entity, id: idFor(), origin: 'saved', confirmed: true, edited: false,
    }));
    const restoredFeedback = {};
    for (const item of example.feedback ?? []) {
      const start = item.start ?? item.span_start;
      const end = item.end ?? item.span_end;
      let entity = restoredEntities.find((candidate) => candidate.start === start && candidate.end === end && candidate.label === item.label);
      if (!entity) {
        entity = { id: idFor(), start, end, label: item.label, origin: 'saved', confirmed: true, edited: false };
        restoredEntities.push(entity);
      }
      restoredFeedback[entity.id] = { reason: item.reason, note: item.note ?? '', start, end, label: item.label };
    }
    setSavedPayloadKey(examplePayloadSignature({
      text: example.text,
      entities: restoredEntities.filter((entity) => restoredFeedback[entity.id]?.reason !== 'false_positive')
        .map(({ start, end, label }) => ({ start, end, label })),
      empty_review_confirmed: !(example.entities ?? []).length && !!example.training_eligible,
      feedback: buildFeedbackPayload(restoredFeedback, restoredEntities),
    }));
    setText(example.text);
    setEntities(restoredEntities);
    setFeedback(restoredFeedback);
    setEmptyReviewConfirmed(!(example.entities ?? []).length && !!example.training_eligible);
    setAnnotationText(example.text);
    setSelection(null);
    setEditingExample({ id: example.id });
    setActiveTab('processing');
    setError('');
    setNotice(`Пример ${example.id} загружен. Изменения сохранятся только после подтверждения.`);
  };
  const saveExample = async () => {
    if (saveInFlightRef.current) return;
    if (!canSave || !text.trim()) {
      setError(spanProblems.size ? 'Исправьте некорректные или пересекающиеся границы перед сохранением.' : pendingEntities.length ? 'Подтвердите каждый найденный фрагмент кнопкой «Верно» или исправьте/удалите его с указанием причины.' : !reviewedEntities.length && !emptyReviewConfirmed ? 'Подтвердите, что проверили весь текст и в нём нет фрагментов для маскировки.' : 'Сначала проверьте актуальный текст.');
      return;
    }
    saveInFlightRef.current = true;
    if (editingExample && !window.confirm(`Сохранить изменения примера ${editingExample.id}? Текст останется связан с этой записью.`)) {
      saveInFlightRef.current = false;
      return;
    }
    setError(''); setNotice(''); setBusy('save');
    try {
      const result = await apiRequest('/api/examples', { method: 'POST', body: JSON.stringify(savePayload) });
      setSavedPayloadKey(payloadKey);
      const next = await apiRequest('/api/status'); setStatus(next);
      void loadExamples();
      Object.entries(expandedCategories).filter(([, expanded]) => expanded).forEach(([label]) => { void loadExamples(label); });
      const savedMessage = result.status === 'created'
        ? `Разметка сохранена как пример №${result.id}.`
        : result.status === 'updated'
          ? `Разметка примера №${result.id} обновлена.`
          : result.status === 'unchanged'
            ? `Разметка примера №${result.id} не изменилась.`
            : result.message || 'Проверенный пример сохранён.';
      const threshold = next.auto_train_min_new_examples;
      const progress = Number.isInteger(threshold) && threshold > 0
        ? ` Новых примеров после обучения: ${next.new_examples_since_training}/${threshold}.`
        : '';
      setNotice(`${savedMessage}${progress} Новая версия появится после порога или запуска обучения во вкладке «Модель».`);
    } catch (reason) { setError(reason.message); }
    finally { saveInFlightRef.current = false; setBusy(''); }
  };
  const copyMaskedResult = async () => {
    if (!navigator.clipboard?.writeText) { setError('Копирование недоступно в этом браузере.'); return; }
    try {
      await navigator.clipboard.writeText(maskedText);
      setError(''); setNotice('Результат скопирован.');
    } catch { setError('Не удалось скопировать результат.'); }
  };
  const trainModel = async () => {
    setError(''); setNotice(''); setBusy('train');
    try {
      const result = await apiRequest('/api/train', { method: 'POST', body: '{}' });
      setTrainingJobId(result.job_id); setTraining({ status: result.status ?? 'queued' }); setNotice('Обучение поставлено в очередь. Можно продолжать разметку.');
    } catch (reason) { setError(reason.message); }
    finally { setBusy(''); }
  };

  const uploadCorpus = async () => {
    if (!corpusArchive) { setError('Выберите ZIP-архив корпуса.'); return; }
    setError(''); setNotice(''); setUploadingCorpus(true);
    try {
      const result = await apiRequest('/api/corpora', {
        method: 'POST', body: corpusArchive, headers: { 'Content-Type': 'application/zip' },
      });
      setCorpusArchive(null);
      await refresh();
      if (result.training_job) {
        setTrainingJobId(result.training_job.id);
        setTraining({ status: result.training_job.status, trigger: 'initial' });
      }
      setNotice(`Корпус ${result.release.corpus_id} ${result.release.corpus_version} сохранён для следующих запусков обучения.${result.training_job ? ' Статус текущего обучения показан ниже.' : ' Для следующей версии модели нажмите «Обучить сейчас». '}`);
    } catch (reason) { setError(reason.message); }
    finally { setUploadingCorpus(false); }
  };

  const activateModelVersion = async (modelId, versionNo) => {
    setError(''); setNotice('');
    try {
      await apiRequest(`/api/models/${encodeURIComponent(modelId)}/activate`, { method: 'POST', body: '{}' });
      await refresh();
      setNotice(`Активирована версия модели v${versionNo}.`);
    } catch (reason) { setError(reason.message); }
  };

  const login = async (event) => {
    event.preventDefault();
    if (loginPending) return;
    setAuthError(''); setLoginPending(true);
    try {
      try { await request('/api/auth/login', { method: 'POST', body: JSON.stringify({ username, password }) }); }
      catch (reason) { setAuthError(reason.status === 401 ? 'Неверный логин или пароль.' : reason.message); return; }
      setPassword('');
      try { await refresh(); } catch (reason) { setAuthError(reason.message); }
    }
    finally { setLoginPending(false); }
  };
  const logout = async () => {
    setError('');
    try {
      await requestLogout(apiRequest);
      setAuth('signed-out'); setStatus(null);
    } catch (reason) { setError(reason.message); }
  };

  const categoryAction = async (label, active) => {
    const action = active ? 'восстановить' : 'архивировать';
    if (!window.confirm(`${action[0].toUpperCase()}${action.slice(1)} категорию ${label}?`)) return;
    setError('');
    try {
      const result = await apiRequest(`/api/categories/${encodeURIComponent(label)}`, { method: 'PATCH', body: JSON.stringify({ active }) });
      setCategories(result.categories ?? (await apiRequest('/api/categories')).categories);
      if (!active && entities.some((entity) => entity.label === label)) setNotice(`Категория ${label} скрыта; существующие отметки сохранены.`);
    } catch (reason) { setError(reason.message); }
  };
  const addCategory = async (event) => {
    event.preventDefault(); setError('');
    try {
      await apiRequest('/api/categories', { method: 'POST', body: JSON.stringify({ label: newLabel.trim().toUpperCase() }) });
      const result = await apiRequest('/api/categories'); setCategories(result.categories ?? result); setNewLabel(''); setNotice('Категория добавлена.');
    } catch (reason) { setError(reason.message); }
  };

  const renderExampleList = (key) => {
    const page = examplePages[key];
    if (!page) return <p className="muted">Список ещё не загружен.</p>;
    return <>
      {page.error && <div className="alert error" role="alert">{page.error}<button className="text-button" onClick={() => loadExamples(key === '*' ? null : key, !!page.items.length && page.nextCursor !== null)}>Повторить</button></div>}
      {!page.items.length && !page.loading && !page.error && <p className="muted">Сохранённых примеров пока нет.</p>}
      <div className="examples-list">{page.items.map((example) => <article className="example-card" key={example.id}>
        <div className="example-meta"><strong>Пример {example.id}</strong><span>{example.created_at ? new Date(example.created_at).toLocaleString() : ''}</span></div>
        <p className="example-text">{example.text}</p>
        <div className="example-footer"><span>{example.entities?.length ?? 0} фрагментов · {example.training_eligible ? 'участвует в обучении' : 'не участвует в обучении'}</span><button className="secondary" onClick={() => openExample(example)}>Открыть в обработке</button></div>
      </article>)}</div>
      {page.loading && <p className="muted" role="status">Загружаем примеры…</p>}
      {page.nextCursor !== null && !page.loading && <button className="secondary load-more" onClick={() => loadExamples(key === '*' ? null : key, true)}>Показать ещё</button>}
    </>;
  };

  if (auth === 'checking') return <main className="center-screen"><span role="status"><span className="spinner" aria-hidden="true" /><span className="sr-only">Проверяем сессию…</span></span></main>;
  if (auth !== 'signed-in') return <main className="login-screen"><form className="login-card" onSubmit={login}>
    <Brand /><p className="eyebrow">ЛОКАЛЬНЫЙ ИНСТРУМЕНТ</p><h1>Вход в приложение</h1>
    <p className="muted">Введите учётные данные, настроенные для этого сервера.</p>
    <label>Логин<input id="login-username" name="username" autoComplete="username" value={username} onChange={(event) => setUsername(event.target.value)} required /></label>
    <label>Пароль<input id="login-password" name="password" type="password" autoComplete="current-password" value={password} onChange={(event) => setPassword(event.target.value)} onCopy={preventPasswordClipboard} onCut={preventPasswordClipboard} required /></label>
    {authError && <p className="alert error" role="alert">{authError}</p>}
    <button className="primary full" type="submit" disabled={loginPending}>{loginPending ? 'Входим…' : 'Войти'}</button>
  </form></main>;

  return <div className="app-shell">
    <header className="topbar"><Brand /><div className="top-actions"><button className="quiet-button" onClick={logout} disabled={snapshotPending}>Выйти</button></div></header>
    <main className="workspace">
      <nav className="tab-nav" aria-label="Разделы приложения">
        {[['processing', 'Обработка'], ['examples', 'Примеры'], ['categories', 'Категории'], ['model', 'Модель']].map(([id, label]) => <button key={id} type="button" disabled={snapshotPending} aria-current={activeTab === id ? 'page' : undefined} className={activeTab === id ? 'active' : ''} onClick={() => setActiveTab(id)}>{label}</button>)}
      </nav>
      {error && <div className="alert error" role="alert"><span>{error}</span><button className="alert-close" aria-label="Закрыть сообщение" onClick={() => setError('')}>×</button></div>}
      {notice && <div className="alert success" role="status"><span>{notice}</span><button className="alert-close" aria-label="Закрыть сообщение" onClick={() => setNotice('')}>×</button></div>}
      {activeTab === 'processing' && <>
      <section className="intro-row"><div><p className="eyebrow">ОБРАБОТКА ТЕКСТА</p><h1>Обезличить текст</h1><p className="muted">Введите текст и запустите проверку. Сверьте фрагменты и скопируйте результат. Сохраните разметку, если хотите использовать её для обучения.</p></div></section>
      {annotationsStale && <div className="alert stale" role="status">Исходный текст изменён после разметки. Проверьте его заново, чтобы обновить фрагменты.</div>}
      {!!spanProblems.size && <div className="alert error" role="alert">Есть некорректные или пересекающиеся интервалы. Исправьте границы перед сохранением; пока они конфликтуют, результат показан без маскирования.</div>}
      <section className="work-grid" aria-label="Рабочая область">
        <article className="panel source-panel"><div className="panel-heading"><div><h2>Исходный текст</h2>{editingExample && <p>Редактируется пример №{editingExample.id}</p>}</div><button className="text-button" onClick={loadSample} disabled={snapshotPending}>Загрузить пример</button></div>
          <div className="source-editor">
            <div ref={highlightRef} className={`source-highlights ${text ? '' : 'is-empty'}`} aria-hidden="true">{text ? sourceMarkup : 'Вставьте текст для проверки…'}</div>
            <textarea id="source-text" name="sourceText" ref={textareaRef} className="source-input" value={text} readOnly={!!editingExample || snapshotPending} onChange={(event) => { setText(event.target.value); setSelection(null); setEmptyReviewConfirmed(false); }} onScroll={(event) => { if (highlightRef.current) highlightRef.current.scrollTop = event.currentTarget.scrollTop; }} onClick={(event) => { const input = event.currentTarget; if (input.selectionStart !== input.selectionEnd) return; const offset = utf16ToCp(text, input.selectionStart); const entity = entities.find((item) => feedback[item.id]?.reason !== 'false_positive' && item.start <= offset && offset < item.end); if (entity) { const card = entityRefs.current[entity.id]; card?.scrollIntoView({ block: 'nearest' }); card?.focus({ preventScroll: true }); } }} onMouseUp={captureSelection} onKeyUp={captureSelection} placeholder=" " aria-label="Исходный текст" />
          </div>
          {selection && <div className="selection-bar"><span>Выбрано: «{selection.text}»</span><label htmlFor="selection-category">Категория<select id="selection-category" name="selectionCategory" value={selectionCategory} onChange={(event) => setSelectionCategory(event.target.value)} disabled={snapshotPending || !activeCategories.length}><option value="">Выберите категорию</option>{activeCategories.map((label) => <option key={label} value={label}>{label}</option>)}</select></label><button className="small-primary" onClick={addSelection} disabled={snapshotPending || !activeCategories.length || !selectionCategory || !!selectionOverlaps}>Добавить фрагмент</button>{!activeCategories.length ? <span role="status">Нет активных категорий. Добавьте или восстановите категорию.</span> : selectionOverlaps ? <span role="status">Уже размечено. Измените категорию в строке этого фрагмента ниже.</span> : null}</div>}
          <div className="source-footer"><span>{cpLength(text)} символов</span><button className="primary" onClick={() => runAnalyze()} disabled={!text.trim() || !!busy || !!editingExample} title={editingExample ? 'В сохранённом примере редактируйте только разметку' : ''}>{busy === 'analyze' ? 'Проверяем…' : 'Проверить текст'}</button></div>
          <div className="entities-heading"><h3>Проверьте фрагменты</h3><span className="count-pill">{entities.length}</span></div>
          {!entities.length ? <div className="empty-state">После проверки здесь появятся фрагменты для разметки.</div> : <div className="entity-list">{entities.map((entity, index) => {
            const item = feedback[entity.id];
            const archivedLabel = !activeCategories.includes(entity.label);
            const excluded = item?.reason === 'false_positive';
            const corrected = entity.origin === 'model' && entity.edited && CORRECTION_REASONS.has(item?.reason);
            const reviewLabel = excluded ? 'Исключено: ложное срабатывание' : entity.origin === 'manual' ? 'Ручная отметка' : entity.confirmed ? 'Подтверждено' : corrected ? 'Исправлено' : 'Нужно проверить';
            const confidence = Number.isFinite(entity.confidence) ? `${Math.round(entity.confidence * 100)}%` : null;
            return <div ref={(node) => { if (node) entityRefs.current[entity.id] = node; else delete entityRefs.current[entity.id]; }} tabIndex={-1} className={`entity-row ${item && item.reason !== 'correct' ? 'has-feedback' : ''} ${excluded ? 'excluded' : ''}`} key={entity.id} style={categoryStyles.get(entity.label)}>
              <div className="entity-main"><button className="entity-snippet" onClick={() => focusRange(entity)} title="Показать фрагмент в исходном тексте">{Array.from(text).slice(entity.start, entity.end).join('')}</button>{confidence && <span className="entity-confidence" title="Оценка выбора модели по токенам, не вероятность правильной категории. Модель учитывает слово, его форму и до двух соседних токенов.">Оценка выбора: {confidence}</span>}{entity.source === 'rule' && <span className="entity-confidence" title="Фрагмент найден встроенным правилом">Правило</span>}<label className="range-control"><span className="sr-only">Начало фрагмента</span><input id={`entity-start-${index}`} name={`entityStart${index + 1}`} aria-label={`Начало фрагмента ${index + 1}`} type="number" min="0" max={cpLength(text)} value={entity.start} disabled={snapshotPending} onChange={(event) => editEntity(entity.id, { start: Number(event.target.value) })} /></label><span className="range-separator">–</span><label className="range-control"><span className="sr-only">Конец фрагмента</span><input id={`entity-end-${index}`} name={`entityEnd${index + 1}`} aria-label={`Конец фрагмента ${index + 1}`} type="number" min="1" max={cpLength(text)} value={entity.end} disabled={snapshotPending} onChange={(event) => editEntity(entity.id, { end: Number(event.target.value) })} /></label>
                <select id={`entity-category-${index}`} name={`entityCategory${index + 1}`} style={categoryStyles.get(entity.label)} aria-label={`Категория фрагмента ${index + 1}`} value={entity.label} disabled={snapshotPending} onChange={(event) => editEntity(entity.id, { label: event.target.value })}>{archivedLabel && <option value={entity.label} disabled>{entity.label} · в архиве</option>}{activeCategories.map((label) => <option key={label} value={label}>{label}</option>)}</select>
                {entity.origin === 'model' && !entity.edited && !excluded && <button className={`confirm-button ${entity.confirmed ? 'confirmed' : ''}`} disabled={snapshotPending} onClick={() => confirmEntity(entity)}>{entity.confirmed ? 'Верно ✓' : 'Верно'}</button>}
                <span className={`review-state ${pendingEntities.some((pending) => pending.id === entity.id) || spanProblems.has(entity.id) ? 'pending' : 'settled'}`}>{spanProblems.has(entity.id) ? 'Проверьте границы' : reviewLabel}</span>
                <button className="feedback-toggle" disabled={snapshotPending} onClick={() => toggleFeedback(entity)} aria-label={`${item && item.reason !== 'correct' ? 'Скрыть исправление' : 'Исправить фрагмент'}: ${Array.from(text).slice(entity.start, entity.end).join('')}`} title={item && item.reason !== 'correct' ? 'Скрыть поля исправления' : 'Указать ошибку в разметке'}>{item && item.reason !== 'correct' ? 'Скрыть' : 'Исправить'}</button>
                <button className="remove-button" disabled={snapshotPending} onClick={() => removeEntity(entity.id)} aria-label={`Удалить фрагмент: ${Array.from(text).slice(entity.start, entity.end).join('')}`} title="Удалить фрагмент">×</button>
              </div>
              {item && item.reason !== 'correct' && <div className="feedback-fields"><label>Что исправить?<select id={`entity-reason-${index}`} name={`entityReason${index + 1}`} value={item.reason} disabled={snapshotPending} onChange={(event) => setFeedback((items) => ({ ...items, [entity.id]: { ...items[entity.id], reason: event.target.value, start: entity.start, end: entity.end, label: entity.label } }))}>{REASONS.map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select></label><label className="note-label">Комментарий<input id={`entity-note-${index}`} name={`entityNote${index + 1}`} value={item.note} disabled={snapshotPending} onChange={(event) => setFeedback((items) => ({ ...items, [entity.id]: { ...items[entity.id], note: event.target.value } }))} placeholder="Что именно не так?" /></label></div>}
            </div>;
          })}</div>}
          {annotationText === text && !!text.trim() && !reviewedEntities.length && !falsePositiveCount && <label className="empty-review-confirmation"><input id="empty-review-confirmed" name="emptyReviewConfirmed" type="checkbox" checked={emptyReviewConfirmed} disabled={snapshotPending} onChange={(event) => setEmptyReviewConfirmed(event.target.checked)} />Я проверил весь текст: фрагментов для маскировки нет</label>}
        </article>
        <article className="panel result-panel"><div className="panel-heading"><div><h2>Результат</h2></div><button className="text-button" onClick={copyMaskedResult} disabled={annotationText !== text || !text || spanProblems.size > 0}>Копировать</button></div>
          <pre className="masked-output" aria-live="polite">{annotationText !== text ? <span className="placeholder">{text.trim() ? 'Проверьте текст, чтобы увидеть результат маскирования' : 'Здесь появится текст с заменёнными фрагментами'}</span> : text ? maskedMarkup : <span className="placeholder">Здесь появится текст с заменёнными фрагментами</span>}</pre>
          <div className="result-note"><span className="shield-icon">✓</span><span>Проверьте фрагменты вручную: результат может содержать пропуски.</span></div>
        </article>
      </section>
      {annotationText !== null && <section className="action-bar" aria-label="Проверка и сохранение разметки"><div className="review-progress"><div className="review-progress-heading"><h2>Проверка разметки</h2><span aria-live="polite">{annotationsStale ? 'Нужна повторная проверка' : `${reviewedCount}/${reviewTotal} проверено`}</span></div><progress max={Math.max(reviewTotal, 1)} value={annotationsStale ? 0 : reviewedCount} aria-label="Прогресс проверки разметки" /><p aria-live="polite">{busy === 'save' ? 'Сохраняем проверенную разметку…' : annotationsStale ? 'Текст изменён — нажмите «Проверить текст», чтобы обновить найденные фрагменты.' : spanProblems.size ? 'Исправьте границы фрагментов.' : alreadySaved ? 'Разметка сохранена. Измените её, чтобы сохранить исправления.' : canSave ? 'Все фрагменты проверены. Можно сохранить разметку для обучения.' : !reviewedEntities.length && !falsePositiveCount ? 'Проверьте весь текст и подтвердите, что в нём нет фрагментов.' : 'Подтвердите или исправьте каждый найденный фрагмент.'}</p></div><div className="action-buttons">{canSave ? <button className="primary" onClick={saveExample} disabled={busy === 'save'}>{busy === 'save' ? 'Сохраняем…' : editingExample ? 'Сохранить изменения' : 'Сохранить разметку'}</button> : alreadySaved ? <span className="save-state" role="status">Сохранено ✓</span> : null}</div></section>}
      <details className="learning-help"><summary>Что попадёт в следующее обучение?</summary><p>После сохранения пример появится в разделе «Примеры» и попадёт в следующее обучение, если его текст не входит в проверочную выборку. Модель учится на исправленных фрагментах и до двух соседних слов с каждой стороны. Причина и комментарий сохраняются отдельно, но пока не влияют на обучение.</p></details>
      </>}
      {activeTab === 'examples' && <section className="management-page">
        <div className="intro-row"><div><p className="eyebrow">БИБЛИОТЕКА РАЗМЕТКИ</p><h1>Примеры</h1><p className="muted">Открывайте сохранённые примеры, чтобы проверить или исправить разметку.</p></div></div>
        {renderExampleList('*')}
      </section>}
      {activeTab === 'categories' && <section className="management-page">
        <div className="intro-row"><div><p className="eyebrow">НАСТРОЙКИ РАЗМЕТКИ</p><h1>Категории</h1><p className="muted">Раскройте категорию, чтобы просмотреть примеры. Архивирование и восстановление требуют подтверждения.</p></div></div>
        <form className="add-category category-add-inline" onSubmit={addCategory}><label htmlFor="new-category">Новая категория</label><div><input id="new-category" name="newCategory" value={newLabel} onChange={(event) => setNewLabel(event.target.value)} placeholder="Например, BANK_ACCOUNT" pattern="[A-Za-z][A-Za-z0-9_]{1,31}" title="Латинская буква, затем 1–31 латинская буква, цифра или подчёркивание" required /><button className="primary" type="submit">Добавить</button></div></form>
        <div className="category-list category-list-page">{categories.map((category) => <section className={`category-section ${!category.active ? 'archived' : ''}`} key={category.label}>
          <div className="category-section-heading"><button className="category-expand" aria-expanded={!!expandedCategories[category.label]} onClick={() => toggleCategoryExamples(category.label)}><span className="category-badge" style={categoryStyles.get(category.label)}>{category.label}</span><span>{category.deprecated ? 'Устаревшая · совместимость' : `${category.built_in ? 'Встроенная' : 'Пользовательская'} · ${category.active ? 'Активна' : 'В архиве'}`}</span><span className="category-chevron" aria-hidden="true">{expandedCategories[category.label] ? '⌄' : '›'}</span></button>{!category.deprecated && <button className={category.active ? 'archive-button' : 'restore-button'} onClick={() => categoryAction(category.label, !category.active)}>{category.active ? 'В архив' : 'Восстановить'}</button>}</div>
          {category.description && <p className="category-description">{category.description}</p>}
          {expandedCategories[category.label] && <div className="category-examples">{renderExampleList(category.label)}</div>}
        </section>)}</div>
      </section>}
      {activeTab === 'model' && <section className="management-page">
        <div className="intro-row model-intro"><div><p className="eyebrow">ОБУЧЕНИЕ</p><h1>Модель</h1><p className="muted">Добавьте корпус и проверенные разметки, чтобы создать новую версию.</p></div><button className="secondary mode-toggle" aria-expanded={advancedModelView} onClick={() => setAdvancedModelView((value) => !value)}>{advancedModelView ? 'Скрыть подробности' : 'Расширенный режим'}</button></div>
        <div className="model-page-grid">
          <article className="panel model-page-card">
            <h2>1. Базовый корпус</h2>
            <p className="muted">Загрузите ZIP с размеченными текстами.</p>
            <div className="corpus-upload">
              <label htmlFor="corpus-archive">Файл корпуса</label>
              <input key={corpusArchive?.name ?? 'empty'} id="corpus-archive" name="corpusArchive" type="file" accept=".zip,application/zip" onChange={(event) => setCorpusArchive(event.target.files?.[0] ?? null)} disabled={uploadingCorpus} />
              <button className="primary" onClick={uploadCorpus} disabled={!corpusArchive || uploadingCorpus}>{uploadingCorpus ? 'Загружаем…' : 'Загрузить корпус'}</button>
            </div>
            <div className="selected-corpus"><span>Будет использован при следующем обучении</span><strong>{selectedCorpusLabel || 'Корпус ещё не загружен'}</strong></div>
            {advancedModelView && <div className="advanced-model-details">
              <p className="muted">Архив должен содержать manifest.json и файлы train.jsonl, validation.jsonl и test.jsonl. Максимальный размер — 64 МиБ.</p>
              <div className="model-summary"><div><span>Train</span><strong>{status?.corpus?.train ?? 0}</strong></div><div><span>Validation</span><strong>{status?.corpus?.validation ?? 0}</strong></div><div><span>Test</span><strong>{status?.corpus?.test ?? 0}</strong></div></div>
              {corpusReleases.length > 0 && <div className="release-list" aria-label="Загруженные версии корпуса">{corpusReleases.map((release) => <div className="release-row" key={release.release_id}><strong>{release.corpus_id} · {release.corpus_version}</strong><span>{release.record_counts.train} train · {release.record_counts.validation} validation · {release.record_counts.test} test</span><span>{trainingReleaseIds.includes(release.release_id) ? 'В следующем обучении' : 'Не выбран'}</span><code>{release.release_id.slice(0, 12)}</code></div>)}</div>}
            </div>}
          </article>
          <article className="panel model-page-card">
            <h2>2. Новая версия</h2>
            <p className="active-model"><span className={`status-dot ${status?.model?.ready ? 'ready' : ''}`} />{status?.model?.ready ? `Сейчас работает v${status.model.version ?? '—'}` : 'Модель ещё не обучена'}</p>
            <div className="model-summary"><div><span title="F1 учитывает и неверные отметки, и пропуски">Оценка на проверочных примерах</span><strong>{formatMetric(displayedMetrics?.entity_micro_f1)}</strong></div><div><span>До автообучения</span><strong>{status?.new_examples_since_training ?? 0} / {status?.auto_train_min_new_examples ?? 100}</strong></div></div>
            <progress className="training-progress" max={Math.max(status?.auto_train_min_new_examples ?? 100, 1)} value={Math.min(status?.new_examples_since_training ?? 0, status?.auto_train_min_new_examples ?? 100)} aria-label="Разметки до автоматического обучения" />
            <button className="primary" onClick={trainModel} disabled={!selectedCorpusReleases.length || !!busy || !!trainingJobId}>{busy === 'train' ? 'Ставим в очередь…' : trainingJobId ? 'Обучение выполняется…' : 'Создать версию сейчас'}</button>
            {!corpusReleases.length && <p className="muted">Сначала загрузите корпус.</p>}
            <p className="muted training-hint">Новая версия создаётся автоматически при достижении порога или вручную этой кнопкой.</p>
            {training && <div className={`training-banner ${['failed', 'error'].includes(training.status) ? 'failed' : ''}`} role="status">{['queued', 'running'].includes(training.status) && <span className="spinner small" />}{!['queued', 'running'].includes(training.status) && <span className="training-result-icon" aria-hidden="true">{['failed', 'error'].includes(training.status) ? '×' : '✓'}</span>}{training.status === 'queued' || training.status === 'running' ? `Модель обучается${training.trigger === 'automatic' ? ' автоматически' : training.trigger === 'initial' ? ' после загрузки первого корпуса' : ''}…` : training.status === 'completed' || training.status === 'succeeded' ? `Обучение завершено${training.model_version ? `: создана v${training.model_version}` : ''}.` : ['failed', 'error'].includes(training.status) ? `Обучение не удалось.${training.error ? ` ${training.error}` : ''}` : `Обучение: ${training.status}`}</div>}
            {advancedModelView && <div className="advanced-model-details">
              <p className="muted">При обучении используются базовый корпус и сохранённые разметки. Архивные категории обучаются, но выключены при маскировании.</p>
              {displayedMetrics && <ModelMetrics metrics={displayedMetrics} />}
              {status?.model?.trained_at && <p className="muted">Обучена: {new Date(status.model.trained_at).toLocaleString()}</p>}
              <p className="muted">Категорий для маскирования: {activeCategories.length}. Порог автообучения задаётся настройкой сервера.</p>
            </div>}
          </article>
        </div>
        <article className="panel model-page-card model-history-card"><h2>Версии модели</h2>
          {!modelVersions.length ? <p className="muted">После первого обучения появится v1.</p> : <div className="version-list">{modelVersions.map((version) => {
            const versionCorpus = (version.corpus_snapshot ?? []).map((release) => `${release.corpus_id} ${release.corpus_version}`).join(' · ');
            return <div className="version-row" key={version.id}><div><strong>{version.model_series === 'ru-pii' ? `v${version.version_no}` : `Старая модель #${version.id}`}</strong><span>{version.active ? 'Активна сейчас' : `Создана ${new Date(version.trained_at).toLocaleString()}`}</span><small>Корпус: {versionCorpus || 'источник не записан'} · размеченных примеров в обучении: {version.metrics?.reviewed_examples ?? '—'}</small>{advancedModelView && <small>F1: {formatMetric(version.metrics?.entity_micro_f1)} · аннотация до записи {version.annotation_watermark ?? 0}</small>}</div>{!version.active && <button className="secondary" onClick={() => activateModelVersion(version.id, version.version_no ?? version.id)}>Сделать активной</button>}</div>;
          })}</div>}
        </article>
      </section>}
      <footer className="page-footer">
        <span>Сохранённые примеры и исправления разметки хранятся на сервере приложения.</span>
      </footer>
    </main>
  </div>;
}

function Brand() { return <div className="brand"><img className="brand-mark" src="/brand-mark.png" alt="" /><span>Морок</span></div>; }

export default App;

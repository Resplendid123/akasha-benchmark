import { useState } from 'react'
import type { EvalSampleDetail } from '../types'

export function MetricInterpretations({
  items,
  response,
}: {
  items: EvalSampleDetail['metric_interpretations']
  response: Record<string, unknown>
}) {
  const [family, setFamily] = useState('全部')
  const families = ['全部', ...Array.from(new Set(items.map((item) => item.family_label)))]
  const visible = family === '全部' ? items : items.filter((item) => item.family_label === family)
  return (
    <section className="sample-metrics">
      <div className="spread sample-metrics-title">
        <h4>逐样本指标（{visible.length}/{items.length}）</h4>
        <label className="metric-family-select">
          <span className="small muted">分组</span>
          <select value={family} onChange={(event) => setFamily(event.target.value)}>
            {families.map((name) => <option key={name} value={name}>{name}</option>)}
          </select>
        </label>
      </div>
      {items.length === 0 ? (
        <p className="small muted">这次评测没有选择指标。</p>
      ) : (
        <div className="metric-interpretation-grid">
          {visible.map((item) => renderMetric(item, response))}
        </div>
      )}
    </section>
  )
}

type MetricItem = EvalSampleDetail['metric_interpretations'][number]
type MetricProps = { item: MetricItem; response: Record<string, unknown> }

function renderMetric(item: MetricItem, response: Record<string, unknown>) {
  const base = item.name.split('@', 1)[0] ?? item.name
  const props = { key: item.name, item, response }
  switch (base) {
    case 'recall': return <RecallMetric {...props} />
    case 'ndcg': return <NdcgMetric {...props} />
    case 'hit': return <HitMetric {...props} />
    case 'full_coverage': return <FullCoverageMetric {...props} />
    case 'mrr': return <MrrMetric {...props} />
    case 'em': return <ExactMatchMetric {...props} />
    case 'f1': return <TokenF1Metric {...props} />
    case 'citation_precision': return <CitationPrecisionMetric {...props} />
    case 'citation_recall': return <CitationRecallMetric {...props} />
    case 'truncation_loss': return <TruncationLossMetric {...props} />
    case 'truncated_gold': return <TruncatedGoldMetric {...props} />
    case 'graph_neighbor_precision': return <GraphNeighborPrecisionMetric {...props} />
    case 'graph_exclusive_gold_share': return <GraphExclusiveGoldShareMetric {...props} />
    case 'graph_neighbor_gold_snippets': return <GraphNeighborGoldSnippetsMetric {...props} />
    case 'faithfulness': return <FaithfulnessMetric {...props} />
    case 'answer_relevancy': return <AnswerRelevancyMetric {...props} />
    case 'context_relevancy': return <ContextRelevancyMetric {...props} />
    case 'answer_correctness': return <AnswerCorrectnessMetric {...props} />
    default: return <UnknownMetric {...props} />
  }
}

function MetricHeading({ item }: { item: MetricItem }) {
  return (
    <div className="row tight metric-heading">
      <strong className="mono">{item.name}</strong>
      <span className="small muted">{item.family_label}</span>
      <strong className="mono metric-value metric-score">
        得分 {item.value === null ? '未计算' : item.value.toFixed(4)}
      </strong>
    </div>
  )
}

function RecallMetric({ item }: MetricProps) {
  const retrieved = item.evidence?.documents ?? []
  const snippets = item.evidence?.snippets ?? []
  return (
    <article className={`metric-interpretation-card ${item.status}`}>
      <MetricHeading item={item} />
      <details className="metric-details">
        <summary className="action small">查看</summary>
        <div className="metric-details-body">
          <dl className="kv compact-kv">
            <dt>指标计算方式</dt>
            <dd>{item.evidence?.formula ?? item.description}</dd>
          </dl>
          <h5>参与计算的检索文档片段（{retrieved.length}）</h5>
          {retrieved.length === 0 && <p className="small muted">没有检索到文档。</p>}
          {retrieved.map((document) => {
            const documentSnippets = snippets.filter((snippet) =>
              snippet.page_ids.includes(document.page_id),
            )
            return (
              <div key={`${document.rank}-${document.page_id}`} className="metric-evidence-document">
                <div>
                  <span className="mono small muted">#{document.rank}</span>{' '}
                  <strong>{document.title}</strong>{' '}
                  <span className="mono small muted">{document.doc_id ?? document.page_id}</span>{' '}
                  {document.is_gold && <span className="tag ok">gold</span>}
                  {!document.mapped && <span className="tag warn">未映射</span>}
                </div>
                {documentSnippets.length === 0 ? (
                  <p className="small muted">没有对应的 snippet 正文。</p>
                ) : documentSnippets.map((snippet) => (
                  <blockquote key={snippet.id ?? snippet.rank}>
                    <div className="readable-text">{snippet.text || '（无正文）'}</div>
                  </blockquote>
                ))}
              </div>
            )
          })}
        </div>
      </details>
    </article>
  )
}
function NdcgMetric({ item }: MetricProps) {
  const documents = item.evidence?.documents ?? []
  const gains = new Map((item.evidence?.contributions ?? []).map((row) => [row.rank, row.gain]))
  return (
    <article className={`metric-interpretation-card ${item.status}`}>
      <MetricHeading item={item} />
      <details className="metric-details"><summary className="action small">查看</summary>
        <div className="metric-details-body">
          <dl className="kv compact-kv"><dt>指标计算方式</dt><dd>{item.evidence?.formula ?? item.description}</dd></dl>
          <h5>参与排名计算的文档（{documents.length}）</h5>
          <div className="metric-document-list">
            {documents.length === 0 && <span className="small muted">没有检索到文档。</span>}
            {documents.map((document) => (
              <span key={`${document.rank}-${document.page_id}`} className="metric-document-item">
                <span className="mono small muted">#{document.rank}</span>{' '}
                <strong>{document.title}</strong>{' '}
                <span className="mono small muted">{document.doc_id ?? document.page_id}</span>{' '}
                {document.is_gold && <span className="tag ok">gold</span>}{' '}
                <span className="small muted">排名贡献 {(gains.get(document.rank) ?? 0).toFixed(4)}</span>
              </span>
            ))}
          </div>
        </div>
      </details>
    </article>
  )
}
function HitMetric({ item }: MetricProps) {
  const retrieved = item.evidence?.documents ?? []
  return (
    <article className={`metric-interpretation-card ${item.status}`}>
      <MetricHeading item={item} />
      <details className="metric-details">
        <summary className="action small">查看</summary>
        <div className="metric-details-body">
          <dl className="kv compact-kv">
            <dt>指标计算方式</dt>
            <dd>{item.evidence?.formula ?? item.description}</dd>
          </dl>
          <h5>参与计算的检索文档（{retrieved.length}）</h5>
          <div className="metric-document-list">
            {retrieved.length === 0 && <span className="small muted">没有检索到文档。</span>}
            {retrieved.map((document) => (
              <span key={`${document.rank}-${document.page_id}`} className="metric-document-item">
                <span className="mono small muted">#{document.rank}</span>{' '}
                <strong>{document.title}</strong>{' '}
                <span className="mono small muted">{document.doc_id ?? document.page_id}</span>{' '}
                {document.is_gold && <span className="tag ok">gold</span>}
                {!document.mapped && <span className="tag warn">未映射</span>}
              </span>
            ))}
          </div>
        </div>
      </details>
    </article>
  )
}
function FullCoverageMetric({ item }: MetricProps) {
  const retrieved = item.evidence?.documents ?? []
  return (
    <article className={`metric-interpretation-card ${item.status}`}>
      <MetricHeading item={item} />
      <details className="metric-details">
        <summary className="action small">查看</summary>
        <div className="metric-details-body">
          <dl className="kv compact-kv">
            <dt>指标计算方式</dt>
            <dd>{item.evidence?.formula ?? item.description}</dd>
          </dl>
          <h5>参与计算的检索文档（{retrieved.length}）</h5>
          <div className="metric-document-list">
            {retrieved.length === 0 && <span className="small muted">没有检索到文档。</span>}
            {retrieved.map((document) => (
              <span key={`${document.rank}-${document.page_id}`} className="metric-document-item">
                <span className="mono small muted">#{document.rank}</span>{' '}
                <strong>{document.title}</strong>{' '}
                <span className="mono small muted">{document.doc_id ?? document.page_id}</span>{' '}
                {document.is_gold && <span className="tag ok">gold</span>}
                {!document.mapped && <span className="tag warn">未映射</span>}
              </span>
            ))}
          </div>
        </div>
      </details>
    </article>
  )
}
function MrrMetric({ item }: MetricProps) {
  const documents = item.evidence?.documents ?? []
  const firstGoldRank = documents.find((document) => document.is_gold)?.rank
  return (
    <article className={`metric-interpretation-card ${item.status}`}>
      <MetricHeading item={item} />
      <details className="metric-details"><summary className="action small">查看</summary>
        <div className="metric-details-body">
          <dl className="kv compact-kv"><dt>指标计算方式</dt><dd>{item.evidence?.formula ?? item.description}</dd></dl>
          <h5>检索文档排名（{documents.length}）</h5>
          <div className="metric-document-list">
            {documents.length === 0 && <span className="small muted">没有检索到文档。</span>}
            {documents.map((document) => (
              <span key={`${document.rank}-${document.page_id}`} className="metric-document-item">
                <span className="mono small muted">#{document.rank}</span>{' '}
                <strong>{document.title}</strong>{' '}
                <span className="mono small muted">{document.doc_id ?? document.page_id}</span>{' '}
                {document.is_gold && <span className="tag ok">gold</span>}{' '}
                {document.rank === firstGoldRank && <span className="tag accent">首个 Gold</span>}
              </span>
            ))}
          </div>
        </div>
      </details>
    </article>
  )
}
function ExactMatchMetric({ item, response }: MetricProps) {
  const references = item.evidence?.answer_comparison?.references ?? []
  const referenceTexts = references.map((reference) =>
    typeof reference === 'string' ? reference : reference.text,
  )
  return (
    <article className={`metric-interpretation-card ${item.status}`}>
      <MetricHeading item={item} />
      <details className="metric-details">
        <summary className="action small">查看</summary>
        <div className="metric-details-body">
          <dl className="kv compact-kv">
            <dt>指标计算方式</dt>
            <dd>{item.evidence?.formula ?? item.description}</dd>
          </dl>
          <h5>系统答案</h5>
          <div className="metric-document-item readable-text">
            {String(response.answer ?? '') || '（空）'}
          </div>
          <h5>参考答案（{referenceTexts.length}）</h5>
          <div className="metric-document-list">
            {referenceTexts.length === 0 && <span className="small muted">无标答</span>}
            {referenceTexts.map((reference, index) => (
              <div key={index} className="metric-document-item readable-text">{reference}</div>
            ))}
          </div>
        </div>
      </details>
    </article>
  )
}
function TokenF1Metric({ item, response }: MetricProps) {
  const references = item.evidence?.answer_comparison?.references ?? []
  const referenceTexts = references.map((reference) =>
    typeof reference === 'string' ? reference : reference.text,
  )
  return (
    <article className={`metric-interpretation-card ${item.status}`}>
      <MetricHeading item={item} />
      <details className="metric-details">
        <summary className="action small">查看</summary>
        <div className="metric-details-body">
          <dl className="kv compact-kv">
            <dt>指标计算方式</dt>
            <dd>{item.evidence?.formula ?? item.description}</dd>
          </dl>
          <h5>系统答案</h5>
          <div className="metric-document-item readable-text">
            {String(response.answer ?? '') || '（空）'}
          </div>
          <h5>参考答案（{referenceTexts.length}）</h5>
          <div className="metric-document-list">
            {referenceTexts.length === 0 && <span className="small muted">无参考答案</span>}
            {referenceTexts.map((reference, index) => (
              <div key={index} className="metric-document-item readable-text">{reference}</div>
            ))}
          </div>
        </div>
      </details>
    </article>
  )
}
function CitationPrecisionMetric({ item }: MetricProps) {
  const citations = item.evidence?.documents ?? []
  const excerptsByPage = new Map(
    (item.evidence?.citation_excerpts ?? []).map((citation) => [
      citation.page_id,
      citation.excerpts,
    ]),
  )
  return (
    <article className={`metric-interpretation-card ${item.status}`}>
      <MetricHeading item={item} />
      <details className="metric-details">
        <summary className="action small">查看</summary>
        <div className="metric-details-body">
          <dl className="kv compact-kv">
            <dt>指标计算方式</dt>
            <dd>{item.evidence?.formula ?? item.description}</dd>
          </dl>
          <h5>Citation 文档片段（{citations.length}）</h5>
          {citations.length === 0 && <p className="small muted">没有 citation 文档片段。</p>}
          {citations.map((citation, index) => {
            const excerpts = excerptsByPage.get(citation.page_id) ?? []
            return (
              <div key={`${citation.page_id}-${index}`} className="metric-evidence-document">
                <div>
                  <strong>{citation.title}</strong>{' '}
                  <span className="mono small muted">{citation.doc_id ?? citation.page_id}</span>{' '}
                  {citation.is_gold && <span className="tag ok">gold</span>}
                </div>
                {excerpts.length === 0 ? (
                  <p className="small muted">没有证据片段。</p>
                ) : excerpts.map((text, excerptIndex) => (
                  <blockquote key={excerptIndex}>
                    <div className="readable-text">{text}</div>
                  </blockquote>
                ))}
              </div>
            )
          })}
        </div>
      </details>
    </article>
  )
}
function CitationRecallMetric({ item }: MetricProps) {
  const citations = item.evidence?.documents ?? []
  const excerptsByPage = new Map(
    (item.evidence?.citation_excerpts ?? []).map((citation) => [
      citation.page_id,
      citation.excerpts,
    ]),
  )
  return (
    <article className={`metric-interpretation-card ${item.status}`}>
      <MetricHeading item={item} />
      <details className="metric-details">
        <summary className="action small">查看</summary>
        <div className="metric-details-body">
          <dl className="kv compact-kv">
            <dt>指标计算方式</dt>
            <dd>{item.evidence?.formula ?? item.description}</dd>
          </dl>
          <h5>实际引用的所有文档（{citations.length}）</h5>
          {citations.length === 0 && <p className="small muted">没有引用文档。</p>}
          {citations.map((citation) => {
            const excerpts = excerptsByPage.get(citation.page_id) ?? []
            return (
              <div key={`${citation.rank}-${citation.page_id}`} className="metric-evidence-document">
                <div>
                  <strong>{citation.title}</strong>{' '}
                  <span className="mono small muted">{citation.doc_id ?? citation.page_id}</span>{' '}
                  {citation.is_gold && <span className="tag ok">gold</span>}
                  {!citation.mapped && <span className="tag warn">未映射</span>}
                </div>
                {excerpts.length === 0 ? (
                  <p className="small muted">没有 citationEvidence 片段。</p>
                ) : excerpts.map((text, excerptIndex) => (
                  <blockquote key={excerptIndex}>
                    <div className="readable-text">{text}</div>
                  </blockquote>
                ))}
              </div>
            )
          })}
        </div>
      </details>
    </article>
  )
}
function TruncationLossMetric({ item }: MetricProps) {
  return <DocumentMetric item={item} title="已检索但未进入引用的文档" documents={item.evidence?.difference_documents ?? []} />
}
function TruncatedGoldMetric({ item }: MetricProps) {
  return <DocumentMetric item={item} title="已检索但未进入引用的 Gold 文档" documents={item.evidence?.difference_documents ?? []} />
}
function DocumentMetric({ item, title, documents }: { item: MetricItem; title: string; documents: NonNullable<NonNullable<MetricItem['evidence']>['documents']> }) {
  return <article className={`metric-interpretation-card ${item.status}`}><MetricHeading item={item} /><details className="metric-details"><summary className="action small">查看</summary><div className="metric-details-body"><dl className="kv compact-kv"><dt>指标计算方式</dt><dd>{item.evidence?.formula ?? item.description}</dd></dl><h5>{title}（{documents.length}）</h5><div className="metric-document-list">{documents.length === 0 && <span className="small muted">没有文档。</span>}{documents.map((document) => <span key={`${document.rank}-${document.page_id}`} className="metric-document-item"><strong>{document.title}</strong>{' '}<span className="mono small muted">{document.doc_id ?? document.page_id}</span>{' '}{document.is_gold && <span className="tag ok">gold</span>}</span>)}</div></div></details></article>
}
function GraphExclusiveGoldShareMetric({ item }: MetricProps) {
  const documents = graphHitDocuments(item)
  return <article className={`metric-interpretation-card ${item.status}`}><MetricHeading item={item} /><details className="metric-details"><summary className="action small">查看</summary><div className="metric-details-body"><dl className="kv compact-kv"><dt>指标计算方式</dt><dd>{item.evidence?.formula ?? item.description}</dd></dl><GraphDocumentList documents={documents} /></div></details></article>
}
function GraphNeighborPrecisionMetric({ item }: MetricProps) {
  const documents = graphHitDocuments(item)
  return <article className={`metric-interpretation-card ${item.status}`}><MetricHeading item={item} /><details className="metric-details"><summary className="action small">查看</summary><div className="metric-details-body"><dl className="kv compact-kv"><dt>指标计算方式</dt><dd>{item.evidence?.formula ?? item.description}</dd></dl><GraphDocumentList documents={documents} /></div></details></article>
}
function GraphNeighborGoldSnippetsMetric({ item }: MetricProps) {
  const documents = graphHitDocuments(item)
  return <article className={`metric-interpretation-card ${item.status}`}><MetricHeading item={item} /><details className="metric-details"><summary className="action small">查看</summary><div className="metric-details-body"><dl className="kv compact-kv"><dt>指标计算方式</dt><dd>{item.evidence?.formula ?? item.description}</dd></dl><GraphDocumentList documents={documents} /></div></details></article>
}
type GraphDocument = { docId: string; title: string; isGold: boolean; rank: number }

function graphHitDocuments(item: MetricItem): GraphDocument[] {
  const documents = new Map<string, GraphDocument>()
  for (const snippet of item.evidence?.snippets ?? []) {
    if (!snippet.is_graph) continue
    for (const docId of snippet.doc_ids) {
      const previous = documents.get(docId)
      documents.set(docId, {
        docId,
        title: previous?.title || snippet.title || docId,
        isGold: previous?.isGold || snippet.gold_doc_ids.includes(docId),
        rank: Math.min(previous?.rank ?? snippet.rank, snippet.rank),
      })
    }
  }
  return [...documents.values()].sort((left, right) => left.rank - right.rank)
}

function GraphDocumentList({ documents }: { documents: GraphDocument[] }) {
  return (
    <>
      <h5>图扩展命中的文档（{documents.length}）</h5>
      <div className="metric-document-list">
        {documents.length === 0 && <span className="small muted">没有图扩展命中的文档。</span>}
        {documents.map((document) => (
          <span key={document.docId} className="metric-document-item">
            <strong>{document.title}</strong>{' '}
            <span className="mono small muted">{document.docId}</span>{' '}
            {document.isGold && <span className="tag ok">gold</span>}
          </span>
        ))}
      </div>
    </>
  )
}
function FaithfulnessMetric({ item }: MetricProps) {
  return <JudgeMetric item={item} label="上下文片段" mode="claims" snippets={item.evidence?.snippets ?? []} />
}
function AnswerRelevancyMetric({ item }: MetricProps) {
  return <JudgeMetric item={item} label="答案句子" mode="sentences" />
}
function ContextRelevancyMetric({ item }: MetricProps) {
  return <JudgeMetric item={item} label="上下文段落" mode="passages" snippets={item.evidence?.snippets ?? []} />
}
function AnswerCorrectnessMetric({ item, response }: MetricProps) {
  return <article className={`metric-interpretation-card ${item.status}`}><MetricHeading item={item} /><details className="metric-details"><summary className="action small">查看</summary><div className="metric-details-body"><dl className="kv compact-kv"><dt>指标计算方式</dt><dd>{item.evidence?.formula ?? item.description}</dd></dl><h5>系统答案</h5><div className="metric-document-item readable-text">{String(response.answer ?? '') || '（空）'}</div><ReferenceAnswers item={item} /><JudgeVerdict item={item} /></div></details></article>
}
function UnknownMetric({ item }: MetricProps) {
  return <article className={`metric-interpretation-card ${item.status}`}><MetricHeading item={item} /><details className="metric-details"><summary className="action small">查看</summary><div className="metric-details-body"><dl className="kv compact-kv"><dt>指标计算方式</dt><dd>{item.evidence?.formula ?? item.description}</dd></dl></div></details></article>
}

function ReferenceAnswers({ item }: { item: MetricItem }) {
  const references = item.evidence?.answer_comparison?.references ?? []
  return <><h5>参考答案（{references.length}）</h5><div className="metric-document-list">{references.length === 0 && <span className="small muted">无参考答案</span>}{references.map((reference, index) => <div key={index} className="metric-document-item readable-text">{typeof reference === 'string' ? reference : reference.text}</div>)}</div></>
}

function JudgeMetric({ item, label, mode, snippets = [] }: { item: MetricItem; label: string; mode: 'claims' | 'sentences' | 'passages'; snippets?: NonNullable<NonNullable<MetricItem['evidence']>['snippets']> }) {
  const detail = item.evidence?.judge_detail ?? {}
  const entries = Array.isArray(detail[mode]) ? detail[mode] as Record<string, unknown>[] : []
  return <article className={`metric-interpretation-card ${item.status}`}><MetricHeading item={item} /><details className="metric-details"><summary className="action small">查看</summary><div className="metric-details-body"><dl className="kv compact-kv"><dt>指标计算方式</dt><dd>{item.evidence?.formula ?? item.description}</dd></dl>{snippets.length > 0 && <><h5>{label}（{snippets.length}）</h5>{snippets.map((snippet) => <div key={snippet.id ?? snippet.rank} className="metric-evidence-document"><strong>{snippet.title || `片段 #${snippet.rank}`}</strong><div className="readable-text">{snippet.text || '（无正文）'}</div></div>)}</>}<h5>Judge 明细（{entries.length}）</h5>{entries.length === 0 && <p className="small muted">无判定明细。</p>}{entries.map((entry, index) => <div key={index} className="metric-document-item"><div className="readable-text">{String(entry.claim ?? entry.sentence ?? `段落 ${String(entry.index ?? index + 1)}`)}</div><span className="tag">{String(entry.verdict ?? '')}</span></div>)}</div></details></article>
}

function JudgeVerdict({ item }: { item: MetricItem }) {
  const detail = item.evidence?.judge_detail ?? {}
  return <><h5>Judge 判定</h5><div className="metric-document-item"><strong>{String(detail.verdict ?? '无')}</strong>{detail.reason ? `：${String(detail.reason)}` : ''}</div></>
}



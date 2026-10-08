import { useState, type ReactNode } from 'react'
import type { EvalSampleDetail } from '../types'
import { Collapsible } from '../ui'

export function MetricInterpretations({
  items,
  response,
}: {
  items: EvalSampleDetail['metric_interpretations']
  response: Record<string, unknown>
}) {
  const [family, setFamily] = useState('全部')
  const [shown, setShown] = useState(false)
  const families = ['全部', ...Array.from(new Set(items.map((item) => item.family_label)))]
  const visible = family === '全部' ? items : items.filter((item) => item.family_label === family)
  return (
    <section className="sample-metrics">
      <div className="spread sample-metrics-title">
        <button className="action small metrics-toggle" onClick={() => setShown((s) => !s)}>
          {shown ? '▾' : '▸'} 逐样本指标（{visible.length}/{items.length}）
        </button>
        {shown && (
          <label className="metric-family-select">
            <span className="small muted">分组</span>
            <select value={family} onChange={(event) => setFamily(event.target.value)}>
              {families.map((name) => <option key={name} value={name}>{name}</option>)}
            </select>
          </label>
        )}
      </div>
      {!shown ? null : items.length === 0 ? (
        <p className="small muted">这次评测没有选择指标。</p>
      ) : (
        <div className="metric-interpretation-grid">
          {visible.map((item) => (
            <Collapsible
              key={item.name}
              title={
                <div className="metric-collapsible-title">
                  <strong className="mono">{item.name}</strong>
                  <span className="small muted">{item.family_label}</span>
                  <span className="mono metric-value">
                    得分 {item.value === null ? '未计算' : item.value.toFixed(4)}
                  </span>
                </div>
              }
            >
              {renderMetric(item, response)}
            </Collapsible>
          ))}
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
    case 'recall': return <UnknownMetric {...props} />
    case 'precision': return <UnknownMetric {...props} />
    case 'retrieval_f1': return <UnknownMetric {...props} />
    case 'ndcg': return <NdcgMetric {...props} />
    case 'hit': return <HitMetric {...props} />
    case 'full_coverage': return <HitMetric {...props} />
    case 'mrr': return <MrrMetric {...props} />
    case 'em': return <AnswerMetric {...props} />
    case 'f1': return <AnswerMetric {...props} />
    case 'citation_precision': return <CitationPrecisionMetric {...props} />
    case 'citation_recall': return <CitationRecallMetric {...props} />
    case 'uncited_count': return <UncitedCountMetric {...props} />
    case 'uncited_gold_count': return <UncitedGoldMetric {...props} />
    case 'graph_neighbor_precision': return <GraphMetric {...props} />
    case 'graph_exclusive_gold_share': return <GraphMetric {...props} />
    case 'graph_neighbor_gold_snippets': return <GraphMetric {...props} />
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

function MetricCard({ item, children }: { item: MetricItem; children: ReactNode }) {
  return (
    <article className={`metric-interpretation-card ${item.status}`}>
      <MetricHeading item={item} />
      <details className="metric-details" open>
        <summary className="action small">查看</summary>
        <div className="metric-details-body">
          <MetricFormula item={item} />
          {children}
        </div>
      </details>
    </article>
  )
}

type EvidenceDocument = NonNullable<NonNullable<MetricItem['evidence']>['documents']>[number]

function DocumentList({
  title,
  documents,
  snippets = [],
  empty = '没有检索到文档。',
  badge,
}: {
  title: string
  documents: EvidenceDocument[]
  snippets?: NonNullable<NonNullable<MetricItem['evidence']>['snippets']>
  empty?: string
  badge?: (document: EvidenceDocument) => ReactNode
}) {
  return (
    <>
      <h5>{title}（{documents.length}）</h5>
      {documents.length === 0 && <p className="small muted">{empty}</p>}
      {documents.map((document) => {
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
              {badge?.(document)}
            </div>
            {documentSnippets.length > 0 && <MetricSnippetList snippets={documentSnippets} />}
          </div>
        )
      })}
    </>
  )
}

const RETRIEVAL_REASON_LABELS: Record<string, { label: string; kind?: string }> = {
  semantic: { label: '语义' },
  lexical: { label: '词面' },
  'exact-title': { label: '标题精确' },
}

const ORIGIN_LABELS: Record<string, { label: string; kind?: string }> = {
  direct: { label: '直接召回' },
  graph: { label: '图扩展', kind: 'accent' },
}

function MetricSnippetList({ snippets }: { snippets: NonNullable<NonNullable<MetricItem['evidence']>['snippets']> }) {
  return (
    <div className="metric-snippet-list">
      {snippets.map((snippet) => (
        <Collapsible
          key={snippet.id ?? snippet.rank}
          compact
          title={
            <span className="metric-snippet-meta">
            <span className="mono small muted">#{snippet.rank}</span>
            <strong>{snippet.title || '（无标题）'}</strong>
            {snippet.is_gold && <span className="tag ok">gold</span>}
            {snippet.origin && (
              <span className={`tag ${ORIGIN_LABELS[snippet.origin]?.kind ?? ''}`}>
                {ORIGIN_LABELS[snippet.origin]?.label ?? snippet.origin}
              </span>
            )}
            {snippet.retrieval_reasons.map((reason) => {
              const entry = RETRIEVAL_REASON_LABELS[reason]
              return <span key={reason} className={`tag ${entry?.kind ?? ''}`}>{entry?.label ?? reason}</span>
            })}
            </span>
          }
        >
          <div className="metric-text-block">{snippet.text || '（无正文）'}</div>
        </Collapsible>
      ))}
    </div>
  )
}
function NdcgMetric({ item }: MetricProps) {
  const gains = new Map((item.evidence?.contributions ?? []).map((row) => [row.rank, row.gain]))
  return (
    <MetricCard item={item}>
      <DocumentList
        title="参与排名计算的文档"
        documents={item.evidence?.documents ?? []}
        snippets={item.evidence?.snippets ?? []}
        badge={(document) => (
          <span className="small muted"> 排名贡献 {(gains.get(document.rank) ?? 0).toFixed(4)}</span>
        )}
      />
    </MetricCard>
  )
}
function HitMetric({ item }: MetricProps) {
  return (
    <MetricCard item={item}>
      <DocumentList
        title="参与计算的检索文档"
        documents={item.evidence?.documents ?? []}
        snippets={item.evidence?.snippets ?? []}
      />
    </MetricCard>
  )
}
function MrrMetric({ item }: MetricProps) {
  const documents = item.evidence?.documents ?? []
  const firstGoldRank = documents.find((document) => document.is_gold)?.rank
  return (
    <MetricCard item={item}>
      <DocumentList
        title="检索文档排名"
        documents={documents}
        snippets={item.evidence?.snippets ?? []}
        badge={(document) =>
          document.rank === firstGoldRank ? <span className="tag accent">首个 Gold</span> : null
        }
      />
    </MetricCard>
  )
}
function AnswerMetric({ item, response }: MetricProps) {
  return (
    <MetricCard item={item}>
      <h5>系统答案</h5>
      <div className="metric-text-block">{String(response.answer ?? '') || '（空）'}</div>
      <ReferenceAnswers item={item} />
    </MetricCard>
  )
}
function CitationPrecisionMetric({ item }: MetricProps) {
  return <CitationMetric item={item} title="Citation 文档片段" empty="没有 citation 文档片段。" />
}
function CitationRecallMetric({ item }: MetricProps) {
  return <CitationMetric item={item} title="实际引用的所有文档" empty="没有引用文档。" />
}

function CitationMetric({ item, title, empty }: { item: MetricItem; title: string; empty: string }) {
  const citations = item.evidence?.documents ?? []
  const excerptsByPage = new Map(
    (item.evidence?.citation_excerpts ?? []).map((citation) => [
      citation.page_id,
      citation.excerpts,
    ]),
  )
  return (
    <MetricCard item={item}>
      <h5>{title}（{citations.length}）</h5>
      {citations.length === 0 && <p className="small muted">{empty}</p>}
      {citations.map((citation, index) => (
        <div key={`${citation.page_id}-${index}`} className="metric-evidence-document">
          <div>
            <span className="mono small muted">#{citation.rank}</span>{' '}
            <strong>{citation.title}</strong>{' '}
            <span className="mono small muted">{citation.doc_id ?? citation.page_id}</span>{' '}
            {citation.is_gold && <span className="tag ok">gold</span>}
            {!citation.mapped && <span className="tag warn">未映射</span>}
          </div>
          {(excerptsByPage.get(citation.page_id) ?? []).map((text, excerptIndex) => (
            <div key={excerptIndex} className="metric-text-block">{text}</div>
          ))}
        </div>
      ))}
    </MetricCard>
  )
}
function UncitedCountMetric({ item }: MetricProps) {
  return (
    <MetricCard item={item}>
      <DocumentList
        title="已检索但未进入引用的文档"
        documents={item.evidence?.difference_documents ?? []}
        snippets={item.evidence?.snippets ?? []}
        empty="没有文档。"
      />
    </MetricCard>
  )
}
function UncitedGoldMetric({ item }: MetricProps) {
  return (
    <MetricCard item={item}>
      <DocumentList
        title="已检索但未进入引用的 Gold 文档"
        documents={item.evidence?.difference_documents ?? []}
        snippets={item.evidence?.snippets ?? []}
        empty="没有文档。"
      />
    </MetricCard>
  )
}
function GraphMetric({ item }: MetricProps) {
  return (
    <MetricCard item={item}>
      <GraphDocumentList item={item} documents={graphHitDocuments(item)} />
    </MetricCard>
  )
}
type GraphDocument = { docId: string; title: string; isGold: boolean; rank: number; pageIds: string[] }

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
        pageIds: [...(previous?.pageIds ?? []), ...snippet.page_ids],
      })
    }
  }
  return [...documents.values()].sort((left, right) => left.rank - right.rank)
}

function GraphDocumentList({ item, documents }: { item: MetricItem; documents: GraphDocument[] }) {
  const snippets = item.evidence?.snippets ?? []
  return (
    <>
      <h5>图扩展命中的文档（{documents.length}）</h5>
      {documents.length === 0 && <p className="small muted">没有图扩展命中的文档。</p>}
      {documents.map((document) => (
        <div key={document.docId} className="metric-evidence-document">
          <div>
            <span className="mono small muted">#{document.rank}</span>{' '}
            <strong>{document.title}</strong>{' '}
            <span className="mono small muted">{document.docId}</span>{' '}
            {document.isGold && <span className="tag ok">gold</span>}
          </div>
          <MetricSnippetList
            snippets={snippets.filter((snippet) =>
              snippet.doc_ids.includes(document.docId) && snippet.is_graph,
            )}
          />
        </div>
      ))}
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
  return (
    <MetricCard item={item}>
      <h5>系统答案</h5>
      <div className="metric-text-block">{String(response.answer ?? '') || '（空）'}</div>
      <ReferenceAnswers item={item} />
      <JudgeVerdict item={item} />
    </MetricCard>
  )
}
function UnknownMetric({ item }: MetricProps) {
  return <MetricCard item={item}>{null}</MetricCard>
}

function MetricFormula({ item }: { item: MetricItem }) {
  return (
    <>
      <h5>指标计算方式</h5>
      <p className="metric-formula">{item.evidence?.formula ?? item.description}</p>
    </>
  )
}

function ReferenceAnswers({ item }: { item: MetricItem }) {
  const references = item.evidence?.answer_comparison?.references ?? []
  return <><h5>参考答案（{references.length}）</h5><div className="metric-document-list">{references.length === 0 && <span className="small muted">无参考答案</span>}{references.map((reference, index) => <div key={index} className="metric-text-block">{typeof reference === 'string' ? reference : reference.text}</div>)}</div></>
}

function JudgeMetric({ item, label, mode, snippets = [] }: { item: MetricItem; label: string; mode: 'claims' | 'sentences' | 'passages'; snippets?: NonNullable<NonNullable<MetricItem['evidence']>['snippets']> }) {
  const detail = item.evidence?.judge_detail ?? {}
  const entries = Array.isArray(detail[mode]) ? detail[mode] as Record<string, unknown>[] : []
  return (
    <MetricCard item={item}>
      {snippets.length > 0 && (
        <>
          <h5>{label}（{snippets.length}）</h5>
          <MetricSnippetList snippets={snippets} />
        </>
      )}
      <h5>Judge 明细（{entries.length}）</h5>
      {entries.length === 0 && <p className="small muted">无判定明细。</p>}
      {entries.map((entry, index) => (
        <div key={index} className="metric-evidence-document">
          <div><span className="tag">{String(entry.verdict ?? '')}</span></div>
          <div className="metric-text-block">
            {String(entry.claim ?? entry.sentence ?? `段落 ${String(entry.index ?? index + 1)}`)}
          </div>
        </div>
      ))}
    </MetricCard>
  )
}

function JudgeVerdict({ item }: { item: MetricItem }) {
  const detail = item.evidence?.judge_detail ?? {}
  return <><h5>Judge 判定</h5><div className="metric-document-item"><strong>{String(detail.verdict ?? '无')}</strong>{detail.reason ? `：${String(detail.reason)}` : ''}</div></>
}

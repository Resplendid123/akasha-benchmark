import { useState } from 'react'
import { api } from '../api'
import type { IndexLayer } from '../types'
import { DatasetPicker, Empty, Failed, Loading, Pager, Pass, useAction, useAsync } from '../ui'
import { DocDiff } from './DocDiff'

/** 编译层：编译模型配置 + 任意已编译文档的变化。
 *
 * 分层依据是重编译语义（§12.3）：compiler / embedding 改了必须重编译，
 * 所以它们属于这一层；answer 改了不用，那属于评测层。
 */
export function Compile({
  activeLayer,
  onSelectLayer,
  onOpenTasks,
  onOpenSettings,
}: {
  activeLayer: number | null
  onSelectLayer: (id: number) => void
  onOpenTasks: () => void
  onOpenSettings: () => void
}) {
  const { data, error, loading, reload } = useAsync(() => api.layers(), [])

  if (loading) return <Loading what="编译层" />
  if (error) return <Failed error={error} />

  const layers = data?.index_layers ?? []

  return (
    <>
      <h2>编译层</h2>
      <p className="lede">
        一层 = 一批文档 + 一组 compiler/embedding 配置。Akasha 的召回跑在
        <strong>编译产物</strong>上而不是原文，所以「这篇文档编译成什么样了」
        决定了它能不能被检索到 —— 那是下面 diff 视图要回答的问题。
      </p>
      <p className="small muted">
        编译模型本身在
        <button className="action small" onClick={onOpenSettings} style={{ margin: '0 4px' }}>
          配置层
        </button>
        改。这里显示的是「这些层是用什么编出来的」，以及它们与当前配置是否还一致。
      </p>

      <NewLayer existing={layers} onDone={reload} onOpenTasks={onOpenTasks} />

      {layers.length === 0 ? (
        <Empty>用上面的表单建第一层。</Empty>
      ) : (
        layers.map((layer) => (
          <LayerCard
            key={layer.id}
            layer={layer}
            active={activeLayer === layer.id}
            onSelect={() => onSelectLayer(layer.id)}
            onOpenTasks={onOpenTasks}
          />
        ))
      )}

      {activeLayer !== null && <DocBrowser layerId={activeLayer} />}
    </>
  )
}

/** 建一层：抽样参数在这里编辑。
 *
 * 抽样顺序是**先 QA 后 corpus**，不是随机抽文档 —— 随机抽 100 篇的话大部分 gold
 * 会落在子集外，Recall 会因为跟检索器毫无关系的原因被钉在 0 附近。所以这里的
 * 「条数」指的是 QA 条数，语料规模由它的 gold 全集加负样本推出来。
 *
 * **label 不能复用已入库的层**：重抽会让那一层的 page_map 指向已经不在子集里的
 * 文档，而那个错配不报错，只让每条检索指标变成 0。要改参数就换个 label。
 */
function NewLayer({
  existing,
  onDone,
  onOpenTasks,
}: {
  existing: IndexLayer[]
  onDone: () => void
  onOpenTasks: () => void
}) {
  const datasets = useAsync(() => api.datasets(), [])
  const [open, setOpen] = useState(existing.length === 0)
  const [label, setLabel] = useState('')
  const [picked, setPicked] = useState<string[]>([])
  const [seed, setSeed] = useState(20260908)
  const [qaLimit, setQaLimit] = useState(100)
  const [negatives, setNegatives] = useState(1.0)
  const [narrativeDocs, setNarrativeDocs] = useState(2)
  const start = useAction<{ id: number }>()

  const available = (datasets.data ?? []).map((entry) => entry.name)
  const chosen = picked.length > 0 ? picked : available
  const ingestedLabels = new Set(
    existing.filter((l) => l.ingest_identity.ingested_at).map((l) => l.label),
  )
  const clash = ingestedLabels.has(label.trim())
  const reused = existing.some((l) => l.label === label.trim()) && !clash

  if (!open) {
    return (
      <div className="row" style={{ marginBottom: 14 }}>
        <button className="action primary" onClick={() => setOpen(true)}>
          建一层
        </button>
        <span className="small muted">抽样条数、种子、负样本比例都在那里配</span>
      </div>
    )
  }

  return (
    <div className="panel">
      <div className="spread">
        <h3 style={{ margin: 0 }}>建一层</h3>
        <button className="action small" onClick={() => setOpen(false)}>
          收起
        </button>
      </div>

      <div className="row" style={{ marginTop: 10 }}>
        <label className="field">
          标签
          <input
            value={label}
            placeholder="run003"
            onChange={(event) => setLabel(event.target.value)}
          />
        </label>
        <label className="field">
          抽多少条 QA
          <input
            type="number"
            min={1}
            value={qaLimit}
            onChange={(event) => setQaLimit(Number(event.target.value))}
          />
        </label>
        <label className="field">
          随机种子
          <input
            type="number"
            value={seed}
            onChange={(event) => setSeed(Number(event.target.value))}
          />
        </label>
        <label className="field">
          每篇 gold 配几篇负样本
          <input
            type="number"
            step="0.1"
            min={0}
            value={negatives}
            onChange={(event) => setNegatives(Number(event.target.value))}
          />
        </label>
        <label className="field">
          narrativeqa 取几篇文档
          <input
            type="number"
            min={1}
            value={narrativeDocs}
            onChange={(event) => setNarrativeDocs(Number(event.target.value))}
          />
        </label>
      </div>

      <div className="row" style={{ marginTop: 8 }}>
        <label className="field">
          数据集（不选则全部）
          <DatasetPicker all={available} selected={picked} onChange={setPicked} />
        </label>
      </div>

      <div className="note plain small">
        抽样顺序是<strong>先 QA 后 corpus</strong>：先按种子抽 {qaLimit} 条问题，
        它们的 gold 文档全集必选，再按比例补负样本。反过来随机抽文档的话，
        大部分 gold 会落在子集外，Recall 会因为跟检索器无关的原因被钉在 0 附近。
        <div style={{ marginTop: 4 }}>
          narrativeqa 走另一条路（整篇整篇取文档）—— 它没有 gold 标注，
          而 293 个问题只覆盖 10 篇文档，按问题抽会把绝大多数 chunk 都牵进来。
        </div>
      </div>

      {clash && (
        <div className="note bad small">
          <strong>{label} 已经入库过了。</strong>
          重抽会让它的 page_map 指向已经不在子集里的文档，而那个错配不报错 ——
          只让每条检索指标变成 0。换个标签，或者先在那一层上「清掉入库产物」。
        </div>
      )}
      {reused && (
        <div className="note warn small">
          {label} 这一层已存在但还没入库，会**重抽**它而不是新建。
        </div>
      )}

      <div className="row" style={{ marginTop: 10 }}>
        <button
          className="action primary"
          disabled={start.busy || !label.trim() || clash}
          onClick={() =>
            start.run(async () => {
              const task = await api.startTask('subset', {
                label: label.trim(),
                datasets: chosen,
                seed,
                qa_limit: qaLimit,
                negatives_ratio: negatives,
                narrativeqa_docs: narrativeDocs,
              })
              onDone()
              return task
            })
          }
        >
          {start.busy ? '启动中…' : '抽子集'}
        </button>
        <span className="small muted">
          离线，秒级。抽完再在层卡片上「入库编译」—— 那一步约 40 秒/篇。
        </span>
      </div>

      {start.error && <div className="note bad">{start.error}</div>}
      {start.result && (
        <div className="note">
          已启动任务 #{start.result.id}。
          <button className="action small" style={{ marginLeft: 8 }} onClick={onOpenTasks}>
            看进度
          </button>
          <button className="action small" style={{ marginLeft: 4 }} onClick={onDone}>
            刷新层列表
          </button>
        </div>
      )}
    </div>
  )
}

function LayerCard({
  layer,
  active,
  onSelect,
  onOpenTasks,
}: {
  layer: IndexLayer
  active: boolean
  onSelect: () => void
  onOpenTasks: () => void
}) {
  const imported = Object.values(layer.page_map_counts).reduce((a, b) => a + b, 0)

  return (
    <div className="panel">
      <div className="spread">
        <div>
          <strong>#{layer.id}</strong> <span className="tag accent">{layer.label}</span>{' '}
          <Pass ok={layer.ready_for_query} yes="可跑查询" no="未就绪" />
        </div>
        <div className="row tight">
          <span className="small muted mono">
            subset {layer.subset_hash}
            {layer.config_hash ? ` · config ${layer.config_hash}` : ' · config 待入库补齐'}
          </span>
          <button className={`action small${active ? ' primary' : ''}`} onClick={onSelect}>
            {active ? '已选中' : '看文档变化'}
          </button>
        </div>
      </div>

      <div className="row small muted" style={{ marginTop: 6 }}>
        <span>seed {layer.seed}</span>
        <span>qa-limit {layer.qa_limit}</span>
        <span>negatives {layer.negatives_ratio}</span>
        <span>已编译入库 {imported} 篇</span>
        {layer.same_subset_layers.length > 0 && (
          <span title="同抽样配置的其他层，可做「同子集换 embedding」的对照">
            同子集层 #{layer.same_subset_layers.join(' #')}
          </span>
        )}
      </div>

      <IngestIdentityRow layer={layer} onOpenTasks={onOpenTasks} />

      {!layer.ready_for_query && (
        <div className="note warn">
          <strong>这一层还不能跑查询。</strong>
          <ul>
            {layer.not_ready_reasons.map((reason) => (
              <li key={reason} className="small">
                {reason}
              </li>
            ))}
          </ul>
          <div className="small" style={{ marginTop: 6 }}>
            半成品索引会产出一份「recall 低、拒答率高」的报告 —— 那看起来像配置差，
            实际是索引没建好。
          </div>
        </div>
      )}

      <table style={{ marginTop: 10 }}>
        <thead>
          <tr>
            <th>数据集</th>
            <th>抽样策略</th>
            <th className="num">QA</th>
            <th className="num">语料</th>
            <th className="num">gold</th>
            <th className="num">已导入</th>
            <th>Space</th>
          </tr>
        </thead>
        <tbody>
          {layer.datasets.map((entry) => {
            const count = layer.page_map_counts[entry.dataset] ?? 0
            const complete = count === entry.corpus_count
            return (
              <tr key={entry.dataset}>
                <td>{entry.dataset}</td>
                <td className="small muted">{entry.strategy}</td>
                <td className="num">{entry.qa_count}</td>
                <td className="num">{entry.corpus_count}</td>
                <td className="num">{entry.gold_doc_count}</td>
                <td className="num">
                  {count}
                  {!complete && (
                    <span className="tag bad" style={{ marginLeft: 6 }}>
                      缺
                    </span>
                  )}
                </td>
                <td className="small mono muted">{entry.space_slug ?? '—'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

/** 这一层**入库时**跑在什么上，以及清掉入库产物。
 *
 * 显示的是历史真相而不是当前配置：配置改过之后这两者会不同，而这一层的
 * page_map 属于前者。显示当前配置会让人以为那一层跑在新配置上。
 */
function IngestIdentityRow({
  layer,
  onOpenTasks,
}: {
  layer: IndexLayer
  onOpenTasks: () => void
}) {
  const identity = layer.ingest_identity
  const discard = useAction<{ discarded: Record<string, number>; note: string }>()
  const ingest = useAction<{ id: number }>()

  if (!identity.ingested_at) {
    const docs = layer.datasets.reduce((sum, entry) => sum + entry.corpus_count, 0)
    const hours = (docs * 40) / 3600
    return (
      <div className="row small" style={{ marginTop: 6 }}>
        <span className="tag warn">未入库</span>
        <span className="muted">
          {docs} 篇待编译，约 {hours.toFixed(1)} 小时
        </span>
        <button
          className="action small primary"
          disabled={ingest.busy || docs === 0}
          onClick={() =>
            ingest.run(() =>
              api.startTask('ingest', {
                label: layer.label,
                datasets: layer.datasets.map((entry) => entry.dataset),
              }),
            )
          }
        >
          {ingest.busy ? '启动中…' : '入库编译'}
        </button>
        {ingest.error && (
          <div className="note bad small" style={{ width: '100%' }}>
            {ingest.error}
          </div>
        )}
        {ingest.result && (
          <div className="note small" style={{ width: '100%' }}>
            已启动任务 #{ingest.result.id}。约 40 秒/篇是 Akasha 的 BullMQ worker 吞吐，
            客户端调不动。
            <button className="action small" style={{ marginLeft: 8 }} onClick={onOpenTasks}>
              看进度
            </button>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="row small" style={{ marginTop: 6 }}>
      <span className="muted">入库于</span>
      <span className="mono muted">{identity.ingested_at}</span>
      {identity.base_url && <span className="mono muted">{identity.base_url}</span>}
      {identity.workspace_id && (
        <span className="mono muted" title="从 users/me 解析，不是配置项">
          ws {identity.workspace_id}
        </span>
      )}
      {identity.akasha_user_role && identity.akasha_user_role !== 'owner' && (
        <span className="tag bad" title="非 owner 会在授权闸门静默丢弃 chunk">
          {identity.akasha_user_role}
        </span>
      )}
      <button
        className="action small danger"
        disabled={discard.busy}
        onClick={() => discard.run(() => api.discardIngest(layer.id, false))}
        title="配置改到了另一个 workspace 时才需要这个"
      >
        清掉入库产物
      </button>

      {discard.error && (
        <div className="note bad small" style={{ width: '100%' }}>
          {discard.error}
          {discard.error.includes('confirm=true') && (
            <div style={{ marginTop: 6 }}>
              <button
                className="action small danger"
                onClick={() => discard.run(() => api.discardIngest(layer.id, true))}
              >
                我确认：清掉并重新入库
              </button>
              <span className="muted" style={{ marginLeft: 8 }}>
                子集保留；远端的 space 不删。
              </span>
            </div>
          )}
        </div>
      )}
      {discard.result && (
        <div className="note small" style={{ width: '100%' }}>
          {discard.result.note}
          <span className="mono">
            {' '}
            清掉：
            {Object.entries(discard.result.discarded)
              .filter(([, n]) => n > 0)
              .map(([table, n]) => `${table}=${n}`)
              .join(' ')}
          </span>
        </div>
      )}
    </div>
  )
}

/** 文档浏览器：选一篇看它的原文 vs 编译产物。 */
function DocBrowser({ layerId }: { layerId: number }) {
  const [offset, setOffset] = useState(0)
  const [term, setTerm] = useState('')
  const [q, setQ] = useState('')
  const [goldOnly, setGoldOnly] = useState(false)
  const [open, setOpen] = useState<{ dataset: string; docId: string } | null>(null)
  const limit = 20

  const { data, error, loading } = useAsync(
    () => api.layerDocs(layerId, { q, gold_only: goldOnly, limit, offset }),
    [layerId, q, goldOnly, offset],
  )

  return (
    <div className="panel">
      <h3 style={{ marginTop: 0 }}>编译层 #{layerId} 的文档</h3>

      <div className="row">
        <input
          placeholder="按 doc_id 搜索"
          value={term}
          onChange={(event) => setTerm(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              setQ(term)
              setOffset(0)
            }
          }}
        />
        <button
          className="action"
          onClick={() => {
            setQ(term)
            setOffset(0)
          }}
        >
          搜索
        </button>
        <label className="check">
          <input
            type="checkbox"
            checked={goldOnly}
            onChange={(event) => {
              setGoldOnly(event.target.checked)
              setOffset(0)
            }}
          />
          只看 gold
        </label>
      </div>

      {loading && <Loading what="文档" />}
      {error && <Failed error={error} />}
      {data && (
        <>
          <p className="small muted" style={{ marginTop: 8 }}>
            共 {data.total} 篇，已导入 {data.imported} 篇。
            {data.imported < data.total && (
              <span className="tag bad" style={{ marginLeft: 6 }}>
                {data.total - data.imported} 篇没有 page_id，看不到编译产物
              </span>
            )}
          </p>
          <table>
            <thead>
              <tr>
                <th>数据集</th>
                <th>doc_id</th>
                <th>gold</th>
                <th>page_id</th>
              </tr>
            </thead>
            <tbody>
              {data.docs.map((doc) => {
                const selected =
                  open?.dataset === doc.dataset && open?.docId === doc.doc_id
                return (
                  <tr
                    key={`${doc.dataset}/${doc.doc_id}`}
                    className={`clickable${selected ? ' selected' : ''}`}
                    onClick={() =>
                      setOpen(selected ? null : { dataset: doc.dataset, docId: doc.doc_id })
                    }
                  >
                    <td className="small">{doc.dataset}</td>
                    <td className="small mono">{doc.doc_id}</td>
                    <td>{doc.is_gold && <span className="tag ok">gold</span>}</td>
                    <td className="small mono muted">
                      {doc.page_id ? (
                        doc.page_id.slice(0, 8)
                      ) : (
                        <span className="tag bad">未导入</span>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
          <Pager total={data.total} offset={offset} limit={limit} onChange={setOffset} />
        </>
      )}

      {open && <DocView layerId={layerId} dataset={open.dataset} docId={open.docId} />}
    </div>
  )
}

function DocView({
  layerId,
  dataset,
  docId,
}: {
  layerId: number
  dataset: string
  docId: string
}) {
  const { data, error, loading } = useAsync(
    () => api.layerDoc(layerId, dataset, docId),
    [layerId, dataset, docId],
  )

  if (loading) return <Loading what="文档" />
  if (error) return <Failed error={error} />
  if (!data) return null

  return (
    <div style={{ marginTop: 12 }}>
      <h4>
        {dataset} / {docId} {data.is_gold && <span className="tag ok">gold</span>}
      </h4>
      {data.note && <div className="note warn small">{data.note}</div>}
      {data.page_id ? (
        <DocDiff pageId={data.page_id} uploaded={data.md_text} />
      ) : (
        <pre className="block">{data.md_text}</pre>
      )}
    </div>
  )
}

import { useState } from 'react'
import { api } from '../api'
import type { CompileRun, DatasetEntry } from '../types'
import {
  CleanupButton,
  DatasetPicker,
  Failed,
  Field,
  Loading,
  Pager,
  Pass,
  StatusTag,
  Timing,
  num,
  useAction,
  useAsync,
  usePoll,
} from '../ui'

/** 当天日期作为种子初值，与后端 default_seed() 一致。 */
function todaySeed(): number {
  const now = new Date()
  const month = `${now.getMonth() + 1}`.padStart(2, '0')
  const day = `${now.getDate()}`.padStart(2, '0')
  return Number(`${now.getFullYear()}${month}${day}`)
}

/** 编译层：抽子集 + 入 Akasha 库。一次编译一个随机创建的空间。 */
export function Compile({
  activeCompile,
  onSelect,
  onOpenQuery,
  onOpenTasks,
}: {
  activeCompile: number | null
  onSelect: (id: number) => void
  onOpenQuery: (id: number) => void
  onOpenTasks: () => void
}) {
  const compiles = useAsync(() => api.compiles(), [])
  const datasets = useAsync(() => api.datasets(), [])
  const cleanup = useAction<unknown>()

  usePoll(
    (compiles.data?.compiles ?? []).some((c) => c.status === 'running'),
    compiles.reload,
  )

  if (compiles.loading || datasets.loading) return <Loading what="编译记录" />
  if (compiles.error) return <Failed error={compiles.error} />
  if (datasets.error) return <Failed error={datasets.error} />
  if (!compiles.data || !datasets.data) return null

  const normalized = datasets.data.datasets.filter((d) => d.normalized)
  const selected = compiles.data.compiles.find((c) => c.id === activeCompile) ?? null

  return (
    <>
      <h2>编译层</h2>

      {cleanup.error && <Failed error={cleanup.error} />}

      <NewCompile
        datasets={normalized}
        onStarted={() => {
          compiles.reload()
          onOpenTasks()
        }}
      />

      <h3>编译记录</h3>
      {compiles.data.compiles.length === 0 && <p className="muted">还没有编译记录。</p>}
      {compiles.data.compiles.length > 0 && (
        <table className="records-table">
          <thead>
            <tr>
              <th>run_id</th>
              <th>数据集</th>
              <th>配置组</th>
              <th>状态</th>
              <th>质量闸门</th>
              <th className="num">语料</th>
              <th className="num">已导入</th>
              <th>耗时</th>
              <th>可用于查询</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {compiles.data.compiles.map((run) => {
              const stats = Object.values(run.stats)
              const docs = stats.reduce((sum, s) => sum + (s.docs ?? 0), 0)
              const imported = stats.reduce((sum, s) => sum + (s.imported ?? 0), 0)
              return (
                <tr key={run.id} className={run.id === activeCompile ? 'selected' : ''}>
                  <td className="mono small">
                    {run.run_id} <span className="muted">#{run.id}</span>
                  </td>
                  <td className="small">{run.datasets.join(', ')}</td>
                  <td className="small">{run.config_group ?? '—'}</td>
                  <td>
                    <StatusTag status={run.status} />
                  </td>
                  <td>{run.quality ? <Pass ok={run.quality.passed} /> : <span className="tag">未执行</span>}</td>
                  <td className="num">{docs}</td>
                  <td className="num">{imported}</td>
                  <td>
                    {/* 每篇耗时是估算，不是实测。 */}
                    <Timing
                      startedAt={run.created_at}
                      finishedAt={run.finished_at}
                      latencyMs={run.pace?.per_page_ms}
                      perLabel="篇"
                    />
                  </td>
                  <td>
                    <Pass ok={run.readiness.ready} yes="就绪" no="未就绪" />
                  </td>
                  <td className="table-actions-cell">
                    <div className="table-actions">
                      <button className="action small" onClick={() => onSelect(run.id)}>
                        {run.id === activeCompile ? '已选中' : '查看'}
                      </button>
                      <button
                        className="action small"
                        disabled={!run.readiness.ready}
                        onClick={() => onOpenQuery(run.id)}
                      >
                        去查询
                      </button>
                      <CleanupButton
                        what={`编译 ${run.run_id}`}
                        detail="子集、语料映射，以及它下面的查询、评测、归因都会从数据库删除。Akasha 那边的空间不会被删。"
                        busy={cleanup.busy}
                        onConfirm={() =>
                          cleanup.run(async () => {
                            const result = await api.deleteCompile(run.id)
                            compiles.reload()
                            return result
                          })
                        }
                      />
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}

      {selected && <CompileDetail run={selected} />}
    </>
  )
}

function NewCompile({
  datasets,
  onStarted,
}: {
  datasets: DatasetEntry[]
  onStarted: () => void
}) {
  const [selected, setSelected] = useState<string[]>([])
  const [runId, setRunId] = useState('')
  const [fullQa, setFullQa] = useState(true)
  const [qaLimit, setQaLimit] = useState(20)
  const [seed, setSeed] = useState(todaySeed)
  const [fullCorpus, setFullCorpus] = useState(true)
  const [ratio, setRatio] = useState(1)
  const [importConcurrency, setImportConcurrency] = useState(10)
  const start = useAction<unknown>()
  const selectedQaMax = Math.max(
    0,
    ...selected.map((name) => datasets.find((dataset) => dataset.name === name)?.qa_rows ?? 0),
  )
  const effectiveQaLimit = fullQa ? selectedQaMax : qaLimit

  return (
    <div className="panel">
      <h3 style={{ marginTop: 0 }}>新建编译</h3>
      {datasets.length === 0 && (
        <div className="note warn">还没有归一化过的数据集，请先在「归一化」页处理。</div>
      )}
      {start.error && <Failed error={start.error} />}

      <DatasetPicker
        all={datasets.map((dataset) => dataset.name)}
        selected={selected}
        onChange={setSelected}
      />

      <div className="row" style={{ marginTop: 10 }}>
        <Field label="run_id" hint="留空自动生成；填已有的则续跑">
          <input value={runId} onChange={(e) => setRunId(e.target.value)} placeholder="自动" />
        </Field>
        <Field label="QA 范围">
          <select
            value={fullQa ? 'all' : 'sampled'}
            onChange={(e) => setFullQa(e.target.value === 'all')}
          >
            <option value="all">全量</option>
            <option value="sampled">抽样</option>
          </select>
        </Field>
        {!fullQa && (
          <Field label="每组抽取数">
            <input
              type="number"
              min={1}
              value={qaLimit}
              onChange={(e) => setQaLimit(Number(e.target.value))}
            />
          </Field>
        )}
        <Field label="随机种子" hint="同种子抽同一批">
          <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
        </Field>
        <Field label="语料范围">
          <select
            value={fullCorpus ? 'all' : 'sampled'}
            onChange={(e) => setFullCorpus(e.target.value === 'all')}
          >
            <option value="all">全量</option>
            <option value="sampled">Gold + 负样本</option>
          </select>
        </Field>
        {!fullCorpus && (
          <Field label="负样本比例" hint="每篇 gold 配几篇">
            <input
              type="number"
              step={0.5}
              min={0}
              value={ratio}
              onChange={(e) => setRatio(Number(e.target.value))}
            />
          </Field>
        )}
        <Field label="导入并发" hint="同时上传的文档数">
          <input
            type="number"
            min={1}
            max={16}
            value={importConcurrency}
            onChange={(e) => setImportConcurrency(Number(e.target.value))}
          />
        </Field>
      </div>

      <div className="panel-actions">
        <button
          className="action primary"
          disabled={
            start.busy ||
            selected.length === 0 ||
            effectiveQaLimit < 1 ||
            importConcurrency < 1 ||
            importConcurrency > 16
          }
          onClick={() =>
            start.run(async () => {
              const task = await api.startTask('compile', {
                datasets: selected,
                qa_limit: effectiveQaLimit,
                seed,
                full_corpus: fullCorpus,
                import_concurrency: importConcurrency,
                ...(fullCorpus ? {} : { negatives_ratio: ratio }),
                ...(runId.trim() ? { run_id: runId.trim() } : {}),
              })
              onStarted()
              return task
            })
          }
        >
          {start.busy ? '启动中…' : '开始编译'}
        </button>
      </div>
    </div>
  )
}

function CompileDetail({ run }: { run: CompileRun }) {
  const [dataset, setDataset] = useState<string>('')
  const [goldOnly, setGoldOnly] = useState(false)
  const [offset, setOffset] = useState(0)
  const [pageId, setPageId] = useState<string | null>(null)
  const limit = 10

  const docs = useAsync(
    () =>
      api.compileDocs(run.id, {
        dataset: dataset || undefined,
        gold_only: goldOnly,
        limit,
        offset,
      }),
    [run.id, dataset, goldOnly, offset],
  )

  return (
    <div className="panel" style={{ marginTop: 14 }}>
      <div className="spread">
        <h3 style={{ margin: 0 }}>
          {run.run_id} <span className="muted small">#{run.id}</span>
        </h3>
        <span className="small mono muted">space {run.space_id ?? '—'}</span>
      </div>

      {!run.readiness.ready && (
        <div className="note warn">
          <strong>这次编译还不能用于查询。</strong>
          <ul>
            {run.readiness.reasons.map((reason) => (
              <li key={reason}>{reason}</li>
            ))}
          </ul>
        </div>
      )}

      {run.quality && (
        <p className="small muted">
          质量闸门：
          {Object.entries(run.quality.gates).map(([name, value]) => (
            <span key={name} className={`tag ${value === 0 ? 'ok' : 'bad'}`} style={{ marginLeft: 4 }}>
              {name}={value ?? 'n/a'}
            </span>
          ))}
        </p>
      )}

      <table>
        <thead>
          <tr>
            <th>数据集</th>
            <th className="num">样本</th>
            <th className="num">语料</th>
            <th className="num">gold</th>
            <th className="num">已导入</th>
          </tr>
        </thead>
        <tbody>
          {Object.values(run.stats).map((entry) => (
            <tr key={entry.dataset}>
              <td>{entry.dataset}</td>
              <td className="num">{num(entry.samples)}</td>
              <td className="num">{num(entry.docs)}</td>
              <td className="num">{num(entry.gold)}</td>
              <td className="num">{num(entry.imported)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h4>已导入的文档</h4>
      <div className="row tight" style={{ marginBottom: 8 }}>
        <select
          value={dataset}
          onChange={(e) => {
            setDataset(e.target.value)
            setOffset(0)
          }}
        >
          <option value="">全部数据集</option>
          {run.datasets.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
        <label className="check">
          <input
            type="checkbox"
            checked={goldOnly}
            onChange={() => {
              setGoldOnly(!goldOnly)
              setOffset(0)
            }}
          />
          只看 gold
        </label>
      </div>

      {docs.loading && <Loading what="文档" />}
      {docs.error && <Failed error={docs.error} />}
      {docs.data && (
        <>
          <table>
            <thead>
              <tr>
                <th>doc_id</th>
                <th>数据集</th>
                <th>文档标题</th>
                <th>gold</th>
                <th>page_id</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {docs.data.docs.map((doc) => (
                <tr key={`${doc.dataset}/${doc.doc_id}`}>
                  <td className="mono small">{doc.doc_id}</td>
                  <td className="small muted">{doc.dataset}</td>
                  <td className="small">{doc.title || '—'}</td>
                  <td>{doc.is_gold ? <span className="tag ok">gold</span> : '—'}</td>
                  <td className="mono small muted truncate">
                    {doc.page_id ?? <span className="tag bad">未导入</span>}
                  </td>
                  <td>
                    {doc.page_id && (
                      <button
                        className="action small"
                        onClick={() => setPageId(pageId === doc.page_id ? null : doc.page_id)}
                      >
                        {pageId === doc.page_id ? '收起' : '编译变化'}
                      </button>
                    )}
                    {doc.error && (
                      <span className="small" style={{ color: 'var(--bad)' }} title={doc.error}>
                        导入失败
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <Pager total={docs.data.total} offset={offset} limit={limit} onChange={setOffset} />
        </>
      )}

      {pageId && <LineageView key={pageId} pageId={pageId} />}
    </div>
  )
}

/** 原文 vs 编译产物并排。编译产物才是被检索的文本。 */
export function LineageView({ pageId, question = '' }: { pageId: string; question?: string }) {
  const { data, error, loading } = useAsync(
    () => api.lineage(pageId, question),
    [pageId, question],
  )

  if (loading) return <Loading what="编译链路" />
  if (error) return <Failed error={error} />
  if (!data) return null

  return (
    <div className="panel flat" style={{ marginTop: 12 }}>
      <div className="note plain small">
        编译扩写比 {data.diff.expansion_ratio?.toFixed(2) ?? '—'}，
        实词留存 {data.diff.retention ? `${(data.diff.retention * 100).toFixed(1)}%` : '—'}。
      </div>

      {data.question_terms_lost.length > 0 && (
        <div className="note bad">
          <strong>{data.verdict}</strong>
          <div style={{ marginTop: 4 }}>
            {data.question_terms_lost.map((term) => (
              <span key={term} className="chip lost">
                {term}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="side-by-side">
        <div>
          <h4>原文（{data.source.chunk_count} 块，不参与召回）</h4>
          <ChunkPager
            items={data.source_chunks.map((c) => ({ text: c.text }))}
            empty="（无原文块）"
          />
        </div>
        <div>
          <h4>
            编译产物（{data.compiled.chunk_count} 块 / {data.compiled.artifact_count} 个 artifact，被检索的文本）
          </h4>
          {data.artifacts.length > 0 && (
            <div className="row tight" style={{ marginBottom: 6 }}>
              {data.artifacts.map((a, i) => (
                <span key={i} className="tag accent">
                  {a.title || '（无标题）'}
                  {a.page_type && <span className="muted"> · {a.page_type}</span>}
                </span>
              ))}
            </div>
          )}
          <ChunkPager
            items={data.chunks.map((c) => ({
              text: c.text,
              label: [c.title, c.chunk_role].filter(Boolean).join(' · '),
            }))}
            empty="（无编译块）"
          />
        </div>
      </div>

      <h4>编译丢掉的实词（{data.diff.dropped_total}）</h4>
      <div>
        {data.diff.dropped.map((term) => (
          <span key={term} className="chip dropped">
            {term}
          </span>
        ))}
        {data.diff.dropped.length === 0 && <span className="small muted">无</span>}
      </div>
    </div>
  )
}

/** 逐块翻页展示。一页一块，块上方标注它的 artifact / 角色。 */
function ChunkPager({ items, empty }: { items: { text: string; label?: string }[]; empty: string }) {
  const [i, setI] = useState(0)
  if (items.length === 0) return <p className="small muted">{empty}</p>
  const idx = Math.min(i, items.length - 1)
  const item = items[idx]!

  return (
    <>
      {item.label && <div className="small muted mono" style={{ marginBottom: 4 }}>{item.label}</div>}
      <pre className="block tall">{item.text || '（空）'}</pre>
      {items.length > 1 && (
        <div className="row tight" style={{ marginTop: 6 }}>
          <button className="action small" disabled={idx === 0} onClick={() => setI(idx - 1)}>
            上一块
          </button>
          <span className="small muted">
            {idx + 1} / {items.length}
          </span>
          <button
            className="action small"
            disabled={idx >= items.length - 1}
            onClick={() => setI(idx + 1)}
          >
            下一块
          </button>
        </div>
      )}
    </>
  )
}

import { useState } from 'react'
import { api } from '../api'
import type { NormalizedSampleDetail } from '../types'
import { Failed, Loading, Pager, useAction, useAsync } from '../ui'

/** 归一化层：适配器状态 + 归一化后的任意样本。
 *
 * **没有适配器的源无法入库。** 这不是「暂时缺个功能」：原始数据的身份规则
 * （哪个字段当 doc_id、gold 怎么解析）必须逐组核对，猜不出来 —— 按字段存在性
 * 去猜的话，数据换个版本就会静默走错分支，而症状是一个看着挺合理的指标。
 */
export function Normalize({ onOpenTasks }: { onOpenTasks: () => void }) {
  const { data, error, loading, reload } = useAsync(() => api.adapters(), [])
  const [dataset, setDataset] = useState<string | null>(null)
  const deletion = useAction<void>()
  const normalization = useAction<void>()

  if (loading) return <Loading what="适配器状态" />
  if (error) return <Failed error={error} />
  if (!data) return null

  const ready = data.adapters.filter((a) => a.normalized)
  const pending = data.adapters.filter((a) => a.implemented && a.files_present && !a.normalized)
  const normalize = (datasets: string[]) => normalization.run(async () => {
    await api.startTask('normalize', { datasets })
    onOpenTasks()
  })

  return (
    <>
      <h2>归一化</h2>
      {deletion.error && <Failed error={deletion.error} />}
      {normalization.error && <Failed error={normalization.error} />}

      <table>
        <thead>
          <tr>
            <th>数据集</th>
            <th>适配器</th>
            <th>拥有的标注</th>
            <th>原始文件</th>
            <th>已归一化</th>
            <th className="num">QA</th>
            <th className="num">语料</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {data.adapters.map((entry) => (
            <tr key={entry.name} className={dataset === entry.name ? 'selected' : ''}>
              <td>{entry.name}</td>
              <td className="small mono muted">
                {entry.implemented ? (
                  <>
                    {entry.adapter}
                    <span className="muted"> v{entry.adapter_version}</span>
                  </>
                ) : (
                  <span className="tag">预留</span>
                )}
              </td>
              <td>
                {entry.provides.map((d) => (
                  <span key={d} className="tag ok">
                    {d}
                  </span>
                ))}
              </td>
              <td>
                {!entry.implemented ? (
                  <span className="tag" title={entry.blocked_reason ?? ''}>
                    待接入
                  </span>
                ) : entry.files_present ? (
                  <span className="tag ok">在位</span>
                ) : (
                  <span className="tag bad" title={entry.blocked_reason ?? ''}>
                    缺文件
                  </span>
                )}
              </td>
              <td>
                {entry.normalized ? (
                  <span className="tag ok" title={entry.normalized_at ?? ''}>
                    已入库
                  </span>
                ) : (
                  <span className="tag">未入库</span>
                )}
              </td>
              <td className="num">{entry.qa_rows ?? '—'}</td>
              <td className="num">{entry.corpus_rows ?? '—'}</td>
              <td>
                {!entry.normalized && entry.implemented && (
                  <button
                    className="action small"
                    disabled={!entry.files_present || normalization.busy}
                    onClick={() => normalize([entry.name])}
                  >
                    归一化
                  </button>
                )}
                <button
                  className="action small"
                  disabled={!entry.normalized}
                  onClick={() => setDataset(dataset === entry.name ? null : entry.name)}
                >
                  {dataset === entry.name ? '收起' : '查看'}
                </button>
                {entry.normalized && (
                  <button
                    className="action small"
                    disabled={deletion.busy}
                    onClick={() => {
                      if (!window.confirm(`删除 ${entry.name} 的归一化样本和语料？原始文件会保留。`)) return
                      deletion.run(async () => {
                        await api.deleteDataset(entry.name)
                        setDataset((current) => current === entry.name ? null : current)
                        reload()
                      })
                    }}
                  >
                    删除
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {data.unclaimed_files.length > 0 && (
        <div className="note warn">
          <strong>{data.unclaimed_files.length} 个文件没有适配器认领，无法入库。</strong>
          <div className="small mono" style={{ marginTop: 4 }}>
            {data.unclaimed_files.join('、')}
          </div>
          <div className="small" style={{ marginTop: 6 }}>
            {data.note}
          </div>
        </div>
      )}

      {ready.length === 0 && (
        <div style={{ marginTop: 14 }}>
          <button
            className="action primary"
            disabled={pending.length === 0 || normalization.busy}
            onClick={() => normalize(pending.map((entry) => entry.name))}
          >
            {normalization.busy ? '启动中…' : '开始归一化'}
          </button>
        </div>
      )}

      {dataset && <SampleBrowser key={dataset} dataset={dataset} />}
    </>
  )
}

/** 归一化后的样本浏览器。可搜索，可展开看单条与它的 gold 正文。 */
function SampleBrowser({ dataset }: { dataset: string }) {
  const [offset, setOffset] = useState(0)
  const [term, setTerm] = useState('')
  const [q, setQ] = useState('')
  const [openSample, setOpenSample] = useState<string | null>(null)
  const limit = 5

  const { data, error, loading } = useAsync(
    () => api.normalizedSamples(dataset, { q, limit, offset }),
    [dataset, q, offset],
  )

  const search = () => {
    setQ(term)
    setOffset(0)
  }

  if (openSample) {
    return (
      <div className="panel" style={{ marginTop: 14 }}>
        <div className="spread">
          <h3 style={{ margin: 0 }}>{dataset} 样本详情</h3>
          <button className="action small" onClick={() => setOpenSample(null)}>
            ← 返回样本列表
          </button>
        </div>
        <SampleDetail dataset={dataset} sampleId={openSample} />
      </div>
    )
  }

  return (
    <div className="panel" style={{ marginTop: 14 }}>
      <div className="spread">
        <h3 style={{ margin: 0 }}>{dataset} 归一化后的样本</h3>
        {data && (
          <span className="small muted mono">
            {data.adapter} v{data.adapter_version}
          </span>
        )}
      </div>

      <div className="row" style={{ margin: '10px 0' }}>
        <input
          placeholder="按问题文本或 sample_id 搜索"
          value={term}
          onChange={(event) => setTerm(event.target.value)}
          onKeyDown={(event) => event.key === 'Enter' && search()}
          style={{ minWidth: 320 }}
        />
        <button className="action" onClick={search}>
          搜索
        </button>
        {q && (
          <button
            className="action small"
            onClick={() => {
              setTerm('')
              setQ('')
              setOffset(0)
            }}
          >
            清除
          </button>
        )}
      </div>

      {data && (
        <p className="small muted">
          身份规则：
          {Object.entries(data.identity_rules).map(([key, rule]) => (
            <span key={key} className="tag" style={{ marginLeft: 4 }}>
              {key}={rule}
            </span>
          ))}
        </p>
      )}

      {loading && <Loading what="样本" />}
      {error && <Failed error={error} />}
      {data && (
        <>
          <table>
            <thead>
              <tr>
                <th>sample_id</th>
                <th>问题</th>
                <th className="num">gold</th>
                <th>参考答案</th>
              </tr>
            </thead>
            <tbody>
              {data.samples.map((sample) => (
                <tr
                  key={sample.sample_id}
                  className="clickable"
                  onClick={() => setOpenSample(sample.sample_id)}
                >
                  <td className="small mono">{sample.dataset_sample_id}</td>
                  <td className="small">{sample.question}</td>
                  <td className="num">{sample.gold_doc_ids.length}</td>
                  <td className="small muted truncate">{sample.answers.join(' / ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <Pager total={data.total} offset={offset} limit={limit} onChange={setOffset} />
        </>
      )}

    </div>
  )
}

function SampleDetail({ dataset, sampleId }: { dataset: string; sampleId: string }) {
  const { data, error, loading } = useAsync(
    () => api.normalizedSample(dataset, sampleId),
    [dataset, sampleId],
  )

  if (loading) return <Loading what="样本明细" />
  if (error) return <Failed error={error} />
  if (!data) return null

  return (
    <div className="panel" style={{ marginTop: 12 }}>
      <h4 style={{ marginTop: 0 }}>{data.sample_id}</h4>
      <dl className="kv">
        <dt>问题</dt>
        <dd>{data.question}</dd>
        <dt>参考答案</dt>
        <dd>{data.answers.join(' / ')}</dd>
        <dt>metadata</dt>
        <dd className="small mono muted">{JSON.stringify(data.metadata)}</dd>
      </dl>

      {data.missing_gold_doc_ids.length > 0 && (
        <div className="note bad">
          <strong>{data.missing_gold_doc_ids.length} 个 gold doc_id 不在语料里。</strong>
          这是身份规则出错的信号：标注指向了一篇找不到的文档。
          <div className="small mono">{data.missing_gold_doc_ids.join('、')}</div>
        </div>
      )}

      <GoldDocs docs={data.gold_docs} />
    </div>
  )
}

function GoldDocs({ docs }: { docs: NormalizedSampleDetail['gold_docs'] }) {
  if (docs.length === 0) return <p className="small muted">这个数据集没有 gold 文档标注。</p>
  return (
    <>
      <h4>gold 文档（{docs.length} 篇）</h4>
      {docs.map((doc) => (
        <div key={doc.doc_id} style={{ marginBottom: 10 }}>
          <div className="small">
            <span className="tag accent">{doc.doc_id}</span> {doc.title}
          </div>
          <pre className="block" style={{ marginTop: 4 }}>
            {doc.text}
            {doc.text_truncated && '\n\n…（已截断）'}
          </pre>
        </div>
      ))}
    </>
  )
}

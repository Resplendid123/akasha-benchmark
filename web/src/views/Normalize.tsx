import { useState } from 'react'
import { api } from '../api'
import type { NormalizedSampleDetail } from '../types'
import { Failed, Loading, Pager, useAsync } from '../ui'

/** 归一化层：适配器状态 + 归一化后的任意样本。
 *
 * **没有适配器的源无法入库。** 这不是「暂时缺个功能」：原始数据的身份规则
 * （哪个字段当 doc_id、gold 怎么解析）必须逐组核对，猜不出来 —— 按字段存在性
 * 去猜的话，数据换个版本就会静默走错分支，而症状是一个看着挺合理的指标。
 */
export function Normalize() {
  const { data, error, loading } = useAsync(() => api.adapters(), [])
  const [dataset, setDataset] = useState<string | null>(null)

  if (loading) return <Loading what="适配器状态" />
  if (error) return <Failed error={error} />
  if (!data) return null

  const ready = data.adapters.filter((a) => a.normalized)

  return (
    <>
      <h2>归一化</h2>
      <p className="lede">
        原始数据经适配器整成库里的 <code>sample</code> + <code>corpus_doc</code>。
        适配器同时声明这个数据集<strong>拥有</strong>哪些标注，那决定了它能算哪些指标。
      </p>

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
                {entry.adapter}
                <span className="muted"> v{entry.adapter_version}</span>
              </td>
              <td>
                {entry.provides.map((d) => (
                  <span key={d} className="tag ok">
                    {d}
                  </span>
                ))}
              </td>
              <td>
                {entry.files_present ? (
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
                <button
                  className="action small"
                  disabled={!entry.normalized}
                  onClick={() => setDataset(dataset === entry.name ? null : entry.name)}
                >
                  {dataset === entry.name ? '收起' : '看样本'}
                </button>
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
        <div className="note warn">
          还没有任何数据集归一化过。从「任务」层起一次 normalize。
        </div>
      )}

      {dataset && <SampleBrowser dataset={dataset} />}
    </>
  )
}

/** 归一化后的样本浏览器。可搜索，可展开看单条与它的 gold 正文。 */
function SampleBrowser({ dataset }: { dataset: string }) {
  const [offset, setOffset] = useState(0)
  const [term, setTerm] = useState('')
  const [q, setQ] = useState('')
  const [openSample, setOpenSample] = useState<string | null>(null)
  const limit = 20

  const { data, error, loading } = useAsync(
    () => api.normalizedSamples(dataset, { q, limit, offset }),
    [dataset, q, offset],
  )

  const search = () => {
    setQ(term)
    setOffset(0)
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
                  className={`clickable${openSample === sample.sample_id ? ' selected' : ''}`}
                  onClick={() =>
                    setOpenSample(openSample === sample.sample_id ? null : sample.sample_id)
                  }
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

      {openSample && <SampleDetail dataset={dataset} sampleId={openSample} />}
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

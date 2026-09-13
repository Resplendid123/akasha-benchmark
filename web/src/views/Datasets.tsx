import { useState } from 'react'
import { api } from '../api'
import { Empty, Failed, Loading, Pager, useAction, useAsync } from '../ui'

export function Datasets({ onOpenTasks }: { onOpenTasks: () => void }) {
  const { data, error, loading } = useAsync(() => api.adapters(), [])
  const [open, setOpen] = useState<string | null>(null)
  const download = useAction<void>()

  if (loading) return <Loading what="数据集" />
  if (error) return <Failed error={error} />
  const entries = data?.adapters.filter((entry) => entry.implemented && entry.files_present) ?? []
  const missing = data?.adapters.some((entry) => entry.implemented && !entry.files_present)
  const downloadButton = (
    <button
      className="action primary"
      disabled={download.busy}
      onClick={() => download.run(async () => {
        await api.startTask('download', {})
        onOpenTasks()
      })}
    >
      {download.busy ? '启动中…' : '下载数据集'}
    </button>
  )

  return (
    <>
      <h2>数据集</h2>
      {download.error && <Failed error={download.error} />}
      {entries.length === 0 ? (
        <Empty>
          <p>暂时还无数据集。</p>
          {downloadButton}
        </Empty>
      ) : (
        <>
          {missing && <div style={{ margin: '12px 0' }}>{downloadButton}</div>}
          <table>
            <thead>
              <tr>
                <th>数据集</th>
                <th>标注</th>
                <th>样本文件</th>
                <th>语料文件</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {entries.map((entry) => (
                <tr key={entry.name} className={open === entry.name ? 'selected' : ''}>
                  <td>{entry.name}</td>
                  <td>{entry.provides.map((dependency) => (
                    <span key={dependency} className="tag ok">{dependency}</span>
                  ))}</td>
                  <td className="small mono">{entry.qa_file}</td>
                  <td className="small mono">{entry.corpus_file}</td>
                  <td>
                    <button
                      className="action small"
                      onClick={() => setOpen(open === entry.name ? null : entry.name)}
                    >
                      {open === entry.name ? '收起' : '原始样例'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
      {open && <RawSamples key={open} dataset={open} />}
    </>
  )
}

/** 原始样例查看器。整行原样给出，不裁字段 —— 裁了就看不到上游还有哪些字段没用上。 */
function RawSamples({ dataset }: { dataset: string }) {
  const [offset, setOffset] = useState(0)
  const limit = 1
  const { data, error, loading } = useAsync(
    () => api.rawSamples(dataset, { limit, offset }),
    [dataset, offset],
  )

  return (
    <div className="panel" style={{ marginTop: 14 }}>
      <div className="spread">
        <h3 style={{ margin: 0 }}>{dataset} 的原始样例</h3>
        {data && <span className="small muted mono">{data.source_file}</span>}
      </div>
      {loading && <Loading what="样例" />}
      {error && <Failed error={error} />}
      {data && (
        <>
          {data.rows.map((row, index) => (
            <pre key={offset + index} className="block" style={{ marginBottom: 8 }}>
              {JSON.stringify(row, null, 2)}
            </pre>
          ))}
          <Pager total={data.total} offset={offset} limit={limit} onChange={setOffset} />
        </>
      )}
    </div>
  )
}

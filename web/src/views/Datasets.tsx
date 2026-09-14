import { useState } from 'react'
import { api } from '../api'
import type { DatasetEntry } from '../types'
import { DatasetPicker, Failed, Loading, Pager, num, useAction, useAsync } from '../ui'

/** 数据集层：原始文件的下载与校验，以及原始样例。 */
export function Datasets({ onOpenTasks }: { onOpenTasks: () => void }) {
  const { data, error, loading, reload } = useAsync(() => api.datasets(), [])
  const [selected, setSelected] = useState<string[]>([])
  const [preview, setPreview] = useState<{ dataset: string; kind: 'qa' | 'corpus' } | null>(null)
  const [reopened, setReopened] = useState(false)
  const start = useAction<unknown>()

  if (loading) return <Loading what="数据集状态" />
  if (error) return <Failed error={error} />
  if (!data) return null

  const names = data.datasets.map((d) => d.name)
  const targets = selected.length ? selected : names
  // 全部就绪就不必再看下载那一栏；想重下由「重新下载」显式打开。
  const allReady = data.datasets.every((d) => d.files_ready)

  return (
    <>
      <div className="spread">
        <h2>数据集</h2>
        <div className="row tight">
          {allReady && !reopened && (
            <button className="action small" onClick={() => setReopened(true)}>
              重新下载
            </button>
          )}
          <button className="action small" onClick={reload}>
            刷新
          </button>
        </div>
      </div>

      {start.error && <Failed error={start.error} />}

      {(!allReady || reopened) && (
        <div className="panel">
          <div className="row">
            <DatasetPicker all={names} selected={selected} onChange={setSelected} />
            <button
              className="action primary"
              disabled={start.busy}
              onClick={() =>
                start.run(async () => {
                  const task = await api.startTask('download', { datasets: targets })
                  onOpenTasks()
                  return task
                })
              }
            >
              {start.busy ? '启动中…' : `下载并校验（${targets.length} 组）`}
            </button>
            {reopened && (
              <button className="action small" onClick={() => setReopened(false)}>
                收起
              </button>
            )}
          </div>
        </div>
      )}

      <table>
        <thead>
          <tr>
            <th>数据集</th>
            <th>文件</th>
            <th>状态</th>
            <th className="num">体积</th>
            <th className="num">行数</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {data.datasets.flatMap((entry) =>
            entry.files.map((file, index) => (
              <tr key={file.file}>
                {index === 0 && <td rowSpan={entry.files.length}>{entry.name}</td>}
                <td className="small mono">{file.file}</td>
                <td>
                  {!file.present ? (
                    <span className="tag bad">缺失</span>
                  ) : file.error ? (
                    <span className="tag bad" title={file.error}>
                      解析失败
                    </span>
                  ) : (
                    <span className="tag ok">就绪</span>
                  )}
                </td>
                <td className="num">
                  {file.size_bytes ? `${(file.size_bytes / 1e6).toFixed(2)} MB` : '—'}
                </td>
                <td className="num">{num(file.rows)}</td>
                {index === 0 && (
                  <td rowSpan={entry.files.length}>
                    <div className="row tight">
                      {(['qa', 'corpus'] as const).map((kind) => {
                        const open = preview?.dataset === entry.name && preview.kind === kind
                        return (
                          <button
                            key={kind}
                            className={`action small${open ? ' primary' : ''}`}
                            disabled={!entry.files_ready}
                            onClick={() =>
                              setPreview(open ? null : { dataset: entry.name, kind })
                            }
                          >
                            {kind === 'qa' ? '样本' : '语料'}
                          </button>
                        )
                      })}
                    </div>
                  </td>
                )}
              </tr>
            )),
          )}
        </tbody>
      </table>

      {preview && (
        <RawPreview
          key={`${preview.dataset}:${preview.kind}`}
          dataset={preview.dataset}
          kind={preview.kind}
        />
      )}
    </>
  )
}

/** 原始样例，一页一条。刻意不走适配器：这里要回答的是「上游给的是什么」。 */
function RawPreview({ dataset, kind }: { dataset: string; kind: 'qa' | 'corpus' }) {
  const [offset, setOffset] = useState(0)
  const limit = 1
  const { data, error, loading } = useAsync(
    () => api.rawSamples(dataset, { kind, limit, offset }),
    [dataset, kind, offset],
  )

  if (loading) return <Loading what="原始样例" />
  if (error) return <Failed error={error} />
  if (!data) return null

  const row = data.rows[0]

  return (
    <div className="panel" style={{ marginTop: 14 }}>
      <div className="spread">
        <h3 style={{ margin: 0 }}>
          {dataset} 原始{kind === 'qa' ? '样本' : '语料'}
        </h3>
        <span className="small muted mono">{data.source_file}</span>
      </div>
      <pre className="block tall">{row ? JSON.stringify(row, null, 2) : '（没有内容）'}</pre>
      <Pager total={data.total} offset={offset} limit={limit} onChange={setOffset} />
    </div>
  )
}

export type { DatasetEntry }

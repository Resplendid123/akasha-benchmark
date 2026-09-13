import { useState } from 'react'
import { api } from '../api'
import { Failed, Loading, ModeTag, Pager, useAsync } from '../ui'

export function Responses({ queryLayerId }: { queryLayerId: number }) {
  const [offset, setOffset] = useState(0)
  const [mode, setMode] = useState('')
  const [openSample, setOpenSample] = useState<string | null>(null)
  const limit = 5

  const { data, error, loading } = useAsync(
    () => api.responses(queryLayerId, { answer_mode: mode || undefined, limit, offset }),
    [queryLayerId, mode, offset],
  )

  return (
    <div style={{ marginTop: 12 }}>
      {loading && <Loading what="响应" />}
      {error && <Failed error={error} />}
      {data && (
        <>
          <div className="row tight small">
            <span className="muted">按 answerMode：</span>
            <button
              className={`action small${mode === '' ? ' primary' : ''}`}
              onClick={() => {
                setMode('')
                setOffset(0)
              }}
            >
              全部 {data.total}
            </button>
            {Object.entries(data.count_by_answer_mode).map(([name, count]) => (
              <button
                key={name}
                className={`action small${mode === name ? ' primary' : ''}`}
                onClick={() => {
                  setMode(name)
                  setOffset(0)
                }}
              >
                {name} {count}
              </button>
            ))}
          </div>

          <table>
            <thead>
              <tr>
                <th>sample_id</th>
                <th>问题</th>
                <th>模式</th>
                <th className="num">召回</th>
                <th className="num">引用</th>
                <th className="num">延迟</th>
              </tr>
            </thead>
            <tbody>
              {data.responses.map((row) => (
                <tr
                  key={row.sample_id}
                  className={`clickable${openSample === row.sample_id ? ' selected' : ''}`}
                  onClick={() =>
                    setOpenSample(openSample === row.sample_id ? null : row.sample_id)
                  }
                >
                  <td className="small mono truncate">{row.sample_id}</td>
                  <td className="small truncate">{row.question}</td>
                  <td>
                    <ModeTag mode={row.answer_mode} />
                    {row.http_status >= 300 && (
                      <span className="tag bad" style={{ marginLeft: 4 }}>
                        HTTP {row.http_status}
                      </span>
                    )}
                  </td>
                  <td className="num">{row.retrieved_count}</td>
                  <td className="num">{row.citation_count}</td>
                  <td className="num">{row.latency_ms ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <Pager total={data.total} offset={offset} limit={limit} onChange={setOffset} />
        </>
      )}

      {openSample && <ResponseBody queryLayerId={queryLayerId} sampleId={openSample} />}
    </div>
  )
}

function ResponseBody({
  queryLayerId,
  sampleId,
}: {
  queryLayerId: number
  sampleId: string
}) {
  const { data, error, loading } = useAsync(
    () => api.response(queryLayerId, sampleId),
    [queryLayerId, sampleId],
  )

  if (loading) return <Loading what="响应体" />
  if (error) return <Failed error={error} />
  if (!data) return null

  const body = data.response as Record<string, unknown> | null
  const answer = (body?.answer as string) ?? ''
  const sources = Array.isArray(body?.retrievedSources) ? body.retrievedSources : []
  const citations = Array.isArray(body?.citations) ? body.citations : []

  return (
    <div className="panel" style={{ marginTop: 12 }}>
      <h4 style={{ marginTop: 0 }}>{sampleId}</h4>
      <dl className="kv">
        <dt>问题</dt>
        <dd>{data.question as string}</dd>
        <dt>生成的答案</dt>
        <dd>{answer || <span className="muted">（空）</span>}</dd>
      </dl>
      {!!data.error && <Failed error={String(data.error)} />}
      <h4>检索结果（{sources.length}）</h4>
      {sources.map((source, index) => <pre key={index} className="block">{JSON.stringify(source, null, 2)}</pre>)}
      <h4>引用（{citations.length}）</h4>
      <pre className="block">{JSON.stringify(citations, null, 2)}</pre>
      <h4>完整响应</h4>
      <pre className="block tall">{JSON.stringify(body, null, 2)}</pre>
      {data.audit ? (
        <>
          <h4>审计记录（retrievalDiagnostics）</h4>
          <p className="small muted">
            它不在 HTTP 响应里，controller 解构时排除了 —— 只写进
            <code>knowledge_query_audit.metadata</code>，这份是抄进库的存档。
          </p>
          <pre className="block">{JSON.stringify(data.audit, null, 2)}</pre>
        </>
      ) : (
        <p className="small muted">
          没有审计记录。要它得跑一次「审计归因」，且需要只读 Postgres 连接。
        </p>
      )}
    </div>
  )
}



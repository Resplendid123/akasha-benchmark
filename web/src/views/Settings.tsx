import { useEffect, useId, useState } from 'react'
import type { ComponentProps, FocusEvent } from 'react'
import { api } from '../api'
import type { AppConnection, ConnectionTest, ModelConfigsView, Provider } from '../types'
import { Failed, Loading, Pass, useAction, useAsync } from '../ui'

// 提示性 placeholder：聚焦时空字段不该用它们填充。
const PROMPTY_PLACEHOLDERS = new Set(['必填', '填一次即可', '已设置', '••••••••'])

// 聚焦时空字段用 placeholder 填充并全选；已有值时直接全选方便覆盖。
// 直接操作 DOM，避免触发受控 input 的 onChange 闭包——但 input 是受控的，
// 我们也手动同步触发 React 知道这件事。
function selectOrFill(event: FocusEvent<HTMLInputElement | HTMLTextAreaElement>) {
  const target = event.currentTarget
  const current = target.value ?? ''
  if (current) {
    target.select()
    return
  }
  const placeholder = target.placeholder.trim()
  if (!placeholder || PROMPTY_PLACEHOLDERS.has(placeholder)) {
    target.select()
    return
  }
  // 用原生 setter 写值，绕过 React 的 input value tracker，
  // 这样下一次 onChange 能正常触发（用户接着输入会替换）。
  const nativeSetter = Object.getOwnPropertyDescriptor(
    target.constructor.prototype,
    'value',
  )?.set
  nativeSetter?.call(target, placeholder)
  target.dispatchEvent(new Event('input', { bubbles: true }))
  target.select()
}

function SecretInput({
  secretLabel = '密码',
  ...props
}: Omit<ComponentProps<'input'>, 'type'> & { secretLabel?: string }) {
  const [visible, setVisible] = useState(false)
  const generatedId = useId()
  const id = props.id ?? generatedId
  const action = `${visible ? '隐藏' : '显示'}${secretLabel}`

  useEffect(() => {
    if (!props.value) setVisible(false)
  }, [props.value])

  return (
    <span className="secret-input">
      <input {...props} id={id} type={visible ? 'text' : 'password'} />
      <button
        type="button"
        className="secret-toggle"
        aria-label={action}
        aria-controls={id}
        aria-pressed={visible}
        title={action}
        disabled={props.disabled}
        onClick={() => setVisible((current) => !current)}
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"
          aria-hidden="true">
          <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z" />
          <circle cx="12" cy="12" r="3" />
          {!visible && <path d="m3 3 18 18" />}
        </svg>
      </button>
    </span>
  )
}

/** 配置层：四组配置，一处改完。
 *
 * 1. Akasha 连接 —— 只有一份，只能改
 * 2. Akasha 那边的模型配置（compiler / embedding / answer / image）
 * 3. judge 模型
 * 4. 归因分析模型
 *
 * 第 2 组原先在编译层。挪过来是因为「改一个模型要去哪」不该取决于它属于哪一层 ——
 * 用户想的是「我要改配置」。编译层保留只读的漂移提示，那是它该管的事。
 *
 * 配置只存在库里。曾经有过两条旁路（配置文件、环境变量覆盖），都删了：
 * 同一份配置有多个来源时，「我改了但没生效」是查不出来的。
 */
export function Settings() {
  return (
    <>
      <h2>配置</h2>
      <div className="note warn">
        <strong>当前配置连同密钥明文存储，只供测试环境使用。</strong>
      </div>
      <Connection />
      <AkashaModels />
      <ProviderPanel
        role="judge"
        title="评估模型"
        hint="LLM as a judge。Akasha 只回传 apiKeySet 布尔量、不回传 key，密钥在Akasha端配置。"
      />
      <ProviderPanel
        role="analysis"
        title="归因分析模型"
        hint="用于 badcase 归因。"
      />
    </>
  )
}

/** Akasha 连接。**只有一份**，只能改，没有新增与删除。 */
function Connection() {
  const { data, error, loading, reload } = useAsync(() => api.connection(), [])
  const [form, setForm] = useState<Record<string, unknown>>({})
  const save = useAction()
  const test = useAction<ConnectionTest>()

  useEffect(() => setForm({}), [data?.updated_at])

  if (loading) return <Loading what="连接配置" />
  if (error) return <Failed error={error} />
  if (!data) return null

  const value = (name: keyof AppConnection, fallback: unknown = '') =>
    (form[name] ?? data[name] ?? fallback) as string | number
  const set = (name: string, next: unknown) => setForm((f) => ({ ...f, [name]: next }))
  const dirty = Object.keys(form).length > 0

  return (
    <div className="panel">
      <div className="spread">
        <h3 style={{ margin: 0 }}>Akasha 连接</h3>
        <div className="row tight">
          <button className="action" disabled={test.busy} onClick={() => test.run(() => api.testConnection())}>
            {test.busy ? '测试中…' : '测试连接'}
          </button>
          <button
            className="action primary"
            disabled={save.busy || !dirty}
            onClick={() =>
              save.run(async () => {
                const result = await api.saveConnection(form)
                reload()
                return result
              })
            }
          >
            {save.busy ? '保存中…' : '保存'}
          </button>
        </div>
      </div>

      <div className="grid2" style={{ marginTop: 10 }}>
        <label className="field">
          base_url
          <input
            value={value('base_url') as string}
            placeholder="http://localhost:3000"
            onFocus={selectOrFill}
            onChange={(event) => set('base_url', event.target.value)}
          />
        </label>
        <label className="field">
          email（必须是 OWNER 账号）
          <input
            value={value('email') as string}
            placeholder="test@example.com"
            onFocus={selectOrFill}
            onChange={(event) => set('email', event.target.value)}
          />
        </label>
        <label className="field">
          password
          <input
            type="text"
            placeholder="12345678"
            value={value('password') as string}
            onFocus={selectOrFill}
            onChange={(event) => set('password', event.target.value)}
          />
        </label>
        <label className="field">
          database_url（只读 PG）
          <input
            type="text"
            placeholder="postgres://akasha:STRONG_DB_PASSWORD@localhost:5432/akasha"
            value={value('database_url') as string}
            onFocus={selectOrFill}
            onChange={(event) => set('database_url', event.target.value)}
          />
        </label>
      </div>

      <h4>查询与编译参数</h4>
      <div className="row">
        {(
          [
            ['concurrency', '查询并发数'],
            ['request_interval_seconds', '请求最小间隔（秒）'],
            ['timeout_seconds', '单次 HTTP 超时（秒）'],
            ['poll_interval_seconds', '编译状态轮询间隔（秒）'],
            ['poll_timeout_seconds', '编译等待上限（秒）'],
          ] as const
        ).map(([name, label]) => (
          <label key={name} className="field">
            {label}
            <input
              type="number"
              step="any"
              style={{ minWidth: 110 }}
              value={String(value(name))}
              onFocus={selectOrFill}
              onChange={(event) => set(name, Number(event.target.value))}
            />
          </label>
        ))}
      </div>

      {data.ingested_layers.length > 0 && (
        <div className="note plain small">
          <strong>{data.ingested_layers.length} 个层已入库：</strong>
          {data.ingested_layers.map((l) => ` #${l.id} ${l.label}（ws ${l.workspace_id ?? '—'}）`).join('、')}
          。改 base_url 或 email 可能落到另一个 workspace，那些层的 page_map 只在
          原来那个里有意义 —— 会有提示。
        </div>
      )}

      {save.error && <div className="note bad">{save.error}</div>}
      {test.error && <div className="note bad">连接失败：{test.error}</div>}
      {test.result && <TestResult result={test.result} />}
      {!test.result && data.last_checked_at && (
        <div className="row tight small muted" style={{ marginTop: 8 }}>
          <span>上次测试 {data.last_checked_at}</span>
          <Pass ok={Boolean(data.last_check_ok)} yes="通过" no="失败" />
          {data.last_check_role && data.last_check_role !== 'owner' && (
            <span className="tag bad">非 owner</span>
          )}
        </div>
      )}
    </div>
  )
}

function TestResult({ result }: { result: ConnectionTest }) {
  const bad = !result.is_owner || result.blocked_layers.length > 0
  return (
    <div className={`note ${bad ? 'bad' : ''}`}>
      <div className="row tight">
        <Pass ok={result.ok} yes="连接成功" no="失败" />
        <span className="small">
          {result.user.email} · 角色 {result.user.role}
        </span>
        <Pass ok={result.is_owner} yes="OWNER" no="非 OWNER" />
      </div>
      <div className="small muted" style={{ marginTop: 4 }}>
        服务端解析出的 workspace：<span className="mono">{result.workspace.name}</span>
        （<span className="mono">{result.workspace.id}</span>）—— 这一项由服务端决定，
        填不了也不用填。
      </div>
      {result.owner_warning && (
        <div className="small" style={{ marginTop: 6 }}>
          {result.owner_warning}
        </div>
      )}
      {result.blocked_layers.length > 0 && (
        <div className="small" style={{ marginTop: 6 }}>
          <strong>{result.blocked_layers.length} 个已入库的层跑不了：</strong>
          {result.blocked_layers.map((entry) => (
            <div key={entry.id} style={{ marginTop: 4 }}>
              #{entry.id} {entry.label} —— {entry.reason}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/** Akasha 那边的四项模型配置。改 compiler/embedding 要二次确认。 */
function AkashaModels() {
  const { data, error, loading, reload } = useAsync(() => api.modelConfigs(), [])
  const [pending, setPending] = useState<{ feature: string; payload: string } | null>(null)
  const action = useAction<{ impact: string }>()

  if (loading) return <Loading what="Akasha 模型配置" />
  if (error)
    return (
      <div className="panel">
        <h3 style={{ marginTop: 0 }}>Akasha 的模型配置</h3>
        <div className="note warn">
          <strong>拿不到。</strong> {error}
          <div className="small" style={{ marginTop: 4 }}>
            先把上面的连接填好并测试通过。其余视图不需要它。
          </div>
        </div>
      </div>
    )
  if (!data) return null

  const configs = extractConfigs(data.live)

  return (
    <div className="panel">
      <div className="spread">
        <h3 style={{ margin: 0 }}>Akasha 的模型配置</h3>
        <span className="small muted">编译、向量化与回答用的模型</span>
      </div>

      <table style={{ marginTop: 8 }}>
        <thead>
          <tr>
            <th>用途</th>
            <th>provider</th>
            <th>模型</th>
            <th>baseUrl</th>
            <th>影响</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {data.features.map((feature) => {
            const entry = configs[feature]
            const rebuild = feature === 'compiler' || feature === 'embedding'
            return (
              <tr key={feature}>
                <td>{feature}</td>
                <td className="small mono">{entry?.provider ?? '—'}</td>
                <td className="small mono">{entry?.model ?? '—'}</td>
                <td className="small mono muted truncate">{entry?.baseUrl ?? '—'}</td>
                <td className="small">
                  {rebuild ? (
                    <span className="tag warn">需重编译</span>
                  ) : (
                    <span className="tag">无影响</span>
                  )}
                </td>
                <td>
                  <button
                    className="action small"
                    onClick={() =>
                      setPending({ feature, payload: JSON.stringify(entry ?? {}, null, 2) })
                    }
                  >
                    修改
                  </button>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>

      <DriftTable layers={data.index_layers} />

      {pending && (
        <ConfirmModelChange
          feature={pending.feature}
          initial={pending.payload}
          busy={action.busy}
          error={action.error}
          result={action.result}
          onCancel={() => {
            setPending(null)
            action.reset()
          }}
          onConfirm={(payload) =>
            action.run(async () => {
              const result = await api.saveModelConfig(pending.feature, JSON.parse(payload))
              reload()
              return result
            })
          }
        />
      )}
    </div>
  )
}

function DriftTable({ layers }: { layers: ModelConfigsView['index_layers'] }) {
  if (layers.length === 0) return null
  return (
    <>
      <h4>与既有层的快照比对</h4>
      <table>
        <thead>
          <tr>
            <th>层</th>
            <th>embedding</th>
            <th>compiler</th>
          </tr>
        </thead>
        <tbody>
          {layers.map((layer) => (
            <tr key={layer.id}>
              <td>
                #{layer.id} <span className="tag accent">{layer.label}</span>
              </td>
              <td>
                <Pass ok={layer.embedding_matches} yes="一致" no="已漂移" />
              </td>
              <td>
                <Pass ok={layer.compiler_matches} yes="一致" no="已漂移" />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="small muted">
        embedding 漂移会让那一层的 chunk 永远召回不到，所以查询阶段对它
        <strong>拒绝执行</strong>，而不是给个警告。
      </p>
    </>
  )
}

/** 提交前的确认。影响范围与「必须新建一层」写在这里，不是提交之后才说。 */
function ConfirmModelChange({
  feature,
  initial,
  busy,
  error,
  result,
  onCancel,
  onConfirm,
}: {
  feature: string
  initial: string
  busy: boolean
  error: string | null
  result: { impact: string } | null
  onCancel: () => void
  onConfirm: (payload: string) => void
}) {
  const [payload, setPayload] = useState(initial)
  const rebuild = feature === 'compiler' || feature === 'embedding'
  let invalid: string | null = null
  try {
    JSON.parse(payload)
  } catch (exc) {
    invalid = String(exc)
  }

  return (
    <div className="panel" style={{ marginTop: 12 }}>
      <h4 style={{ marginTop: 0 }}>修改 {feature} 配置</h4>

      <div className={`note ${rebuild ? 'bad' : 'warn'}`}>
        <strong>改的是 Akasha 那个部署的设置，不是本平台的。</strong>
        同一部署上的其他账号也会受影响。
        {rebuild && (
          <div style={{ marginTop: 6 }}>
            改完之后既有的编译层与新配置<strong>不再可比</strong>，需要新建一层重编译。
            {feature === 'embedding' && (
              <>
                {' '}
                embedding 更硬：旧 chunk 带的是旧 profile，它们永远召回不到，而评测会
                照常算出一份看着合理的坏报告 —— 那种失效不会报错。
              </>
            )}
          </div>
        )}
      </div>

      <textarea
        value={payload}
        onChange={(event) => setPayload(event.target.value)}
        rows={8}
        style={{ width: '100%', fontFamily: 'var(--mono)', fontSize: 12 }}
      />
      {invalid && <div className="note bad small">JSON 解析失败：{invalid}</div>}
      {error && <div className="note bad">{error}</div>}
      {result && <div className="note">{result.impact}</div>}

      <div className="row" style={{ marginTop: 10 }}>
        <button
          className="action primary"
          disabled={busy || invalid !== null || result !== null}
          onClick={() => onConfirm(payload)}
        >
          {busy ? '提交中…' : '确认修改'}
        </button>
        <button className="action" onClick={onCancel}>
          {result ? '关闭' : '取消'}
        </button>
      </div>
    </div>
  )
}

function extractConfigs(live: unknown): Record<string, Record<string, string>> {
  const entries = (live as { configs?: unknown[] })?.configs ?? (Array.isArray(live) ? live : [])
  const map: Record<string, Record<string, string>> = {}
  for (const entry of entries as Array<Record<string, string>>) {
    if (entry?.feature) map[entry.feature] = entry
  }
  return map
}

function ProviderPanel({
  role,
  title,
  hint,
}: {
  role: 'judge' | 'analysis'
  title: string
  hint: string
}) {
  const { data, error, loading, reload } = useAsync(() => api.providers(role), [role])
  const [form, setForm] = useState({ label: 'default', base_url: '', model: '', api_key: '' })
  const save = useAction<{ id: number }>()
  const remove = useAction<{ deleted: number }>()

  useEffect(() => {
    const first = data?.[0]
    if (first) {
      setForm({ label: first.label, base_url: first.base_url, model: first.model, api_key: '' })
    }
  }, [data])

  if (loading) return <Loading what={title} />
  if (error) return <Failed error={error} />

  return (
    <div className="panel">
      <div className="spread">
        <h3 style={{ margin: 0 }}>{title}</h3>
        <button
          className="action primary"
          disabled={save.busy || !form.base_url || !form.model}
          onClick={() =>
            save.run(async () => {
              const result = await api.saveProvider(role, form)
              reload()
              return result
            })
          }
        >
          {save.busy ? '保存中…' : '保存'}
        </button>
      </div>
      <p className="small muted">{hint}</p>

      <div className="grid2">
        <label className="field">
          标签
          <input
            value={form.label}
            placeholder="default"
            onFocus={selectOrFill}
            onChange={(event) => setForm({ ...form, label: event.target.value })}
          />
        </label>
        <label className="field">
          base_url
          <input
            value={form.base_url}
            placeholder="https://api.openai.com/v1"
            onFocus={selectOrFill}
            onChange={(event) => setForm({ ...form, base_url: event.target.value })}
          />
        </label>
        <label className="field">
          模型
          <input
            value={form.model}
            placeholder="gpt-4o-mini"
            onFocus={selectOrFill}
            onChange={(event) => setForm({ ...form, model: event.target.value })}
          />
        </label>
        <label className="field">
          api_key
          <SecretInput
            secretLabel="API Key"
            value={form.api_key}
            placeholder="必填"
            onFocus={selectOrFill}
            onChange={(event) => setForm({ ...form, api_key: event.target.value })}
          />
        </label>
      </div>

      {save.error && <div className="note bad">{save.error}</div>}
      {save.result && <div className="note">已保存。</div>}
      {remove.error && <div className="note bad">{remove.error}</div>}

      {data && data.length > 0 && (
        <table style={{ marginTop: 10 }}>
          <thead>
            <tr>
              <th>标签</th>
              <th>模型</th>
              <th>端点</th>
              <th>密钥</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {data.map((provider: Provider) => (
              <tr key={provider.id}>
                <td>{provider.label}</td>
                <td className="small mono">{provider.model}</td>
                <td className="small mono muted truncate">{provider.base_url}</td>
                <td>
                  {provider.api_key_set ? (
                    <span className="tag ok">已设置</span>
                  ) : (
                    <span className="tag bad">未设置</span>
                  )}
                </td>
                <td>
                  <button
                    className="action small danger"
                    disabled={remove.busy}
                    onClick={() =>
                      remove.run(async () => {
                        const result = await api.deleteProvider(provider.id)
                        reload()
                        return result
                      })
                    }
                  >
                    删除
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

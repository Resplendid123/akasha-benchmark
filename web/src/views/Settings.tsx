import { useEffect, useState } from 'react'
import { api, getToken, setToken } from '../api'
import type { Connection, ConnectionTest, ModelConfig, Provider } from '../types'
import { Failed, Field, Loading, Pass, SecretField, useAction, useAsync } from '../ui'

/** 配置层：Akasha 连接、它那边的模型配置、本地 judge / 归因端点。 */
export function Settings() {
  return (
    <>
      <h2>配置</h2>
      <AccessToken />
      <ConnectionForm />
      <ModelConfigs />
      <Providers role="judge" title="评估模型 judge" />
      <Providers role="attribution" title="归因模型 attribute" />
    </>
  )
}

/** 令牌不对时 health 是 401，所以请求失败也要显示这一段 —— 否则没有地方改它。 */
function AccessToken() {
  const [saved, setSaved] = useState(getToken())
  const [value, setValue] = useState(saved)
  const health = useAsync(() => api.health(), [saved])
  const required = Boolean(health.data?.settings.auth_required)

  const apply = (next: string) => {
    setToken(next)
    setValue(next)
    setSaved(next)
  }

  if (!required && !saved && !health.error) return null
  return (
    <div className="panel">
      <div className="panel-head">
        <h3>访问令牌</h3>
        {saved && !health.error && <Pass ok yes="已生效" />}
      </div>
      {health.error && <Failed error={health.error} />}

      <div className="form-grid">
        <SecretField
          label="X-Auth-Token"
          value={value}
          onChange={setValue}
          placeholder="启动时设的那个令牌"
          wide
        />
      </div>

      <div className="panel-actions">
        <button className="action primary" disabled={value === saved} onClick={() => apply(value)}>
          保存
        </button>
        <button className="action" disabled={!saved} onClick={() => apply('')}>
          清除
        </button>
        {value !== saved && <span className="small muted">未保存</span>}
      </div>
    </div>
  )
}

// 数值字段各自的下限。间隔可以是 0（不等待），超时与并发不行。
const NUMBER_FIELDS = [
  ['timeout_seconds', '请求超时（秒）', 1],
  ['concurrency', '默认并发', 1],
  ['request_interval_seconds', '请求间隔（秒）', 0],
  ['poll_interval_seconds', '编译轮询间隔（秒）', 0],
  ['poll_timeout_seconds', '编译轮询超时（秒）', 1],
] as const

type Form = Record<string, string>

/** 数值也按字符串存：清空输入框要能留着空，不能悄悄变成 0。 */
function toForm(data: Connection): Form {
  const { compiles: _compiles, updated_at: _updated, ...rest } = data
  return Object.fromEntries(Object.entries(rest).map(([key, value]) => [key, String(value)]))
}

function ConnectionForm() {
  const { data, error, loading, reload } = useAsync(() => api.connection(), [])
  const [form, setForm] = useState<Form>({})
  const [loaded, setLoaded] = useState<Form>({})
  const save = useAction<{ updated: string[] }>()
  const test = useAction<ConnectionTest>()

  useEffect(() => {
    if (data) {
      setForm(toForm(data))
      setLoaded(toForm(data))
    }
  }, [data])

  if (loading && !data) return <Loading what="连接配置" />
  if (error) return <Failed error={error} />
  if (!data) return null

  // 改过之后原来的保存结果与测试结论都不再对应当前表单。
  const set = (key: string, value: string) => {
    setForm((f) => ({ ...f, [key]: value }))
    save.reset()
    test.reset()
  }
  const revert = () => {
    setForm(loaded)
    save.reset()
    test.reset()
  }

  const dirty = Object.keys(loaded).some((key) => form[key] !== loaded[key])
  const invalid = NUMBER_FIELDS.filter(([key, , min]) => {
    const value = Number(form[key])
    return form[key]?.trim() === '' || !Number.isFinite(value) || value < min
  })

  const payload = () => ({
    ...form,
    ...Object.fromEntries(NUMBER_FIELDS.map(([key]) => [key, Number(form[key])])),
  })

  return (
    <div className="panel">
      <div className="panel-head">
        <h3>Akasha 连接</h3>
        {dirty && <span className="tag warn">有未保存的改动</span>}
      </div>

      {save.error && <Failed error={save.error} />}
      {test.error && <Failed error={test.error} />}
      {save.result && (
        <div className="note">已保存 {save.result.updated.length} 个字段。</div>
      )}

      <div className="form-grid">
        <Field label="base_url">
          <input value={form.base_url ?? ''} onChange={(e) => set('base_url', e.target.value)} />
        </Field>
        <Field label="登录邮箱">
          <input value={form.email ?? ''} onChange={(e) => set('email', e.target.value)} />
        </Field>
        <SecretField
          label="登录密码"
          value={form.password ?? ''}
          onChange={(value) => set('password', value)}
        />
        <SecretField
          label="只读 PostgreSQL"
          hint="归因链路用，可不填"
          value={form.database_url ?? ''}
          onChange={(value) => set('database_url', value)}
          placeholder="postgresql://…"
        />
      </div>

      <div className="form-grid compact">
        {NUMBER_FIELDS.map(([key, label, min]) => (
          <Field key={key} label={label}>
            <input
              type="number"
              min={min}
              value={form[key] ?? ''}
              onChange={(e) => set(key, e.target.value)}
            />
          </Field>
        ))}
      </div>

      {invalid.length > 0 && (
        <div className="note bad">
          <strong>这些字段要填数字：</strong>
          {invalid.map(([, label, min]) => `${label}（≥ ${min}）`).join('、')}
        </div>
      )}

      <div className="panel-actions">
        <button
          className="action primary"
          disabled={save.busy || !dirty || invalid.length > 0}
          onClick={() =>
            save.run(async () => {
              const result = await api.saveConnection(payload())
              reload()
              return result
            })
          }
        >
          {save.busy ? '保存中…' : '保存'}
        </button>
        <button className="action" disabled={!dirty || save.busy} onClick={revert}>
          放弃改动
        </button>
        <button
          className="action"
          disabled={test.busy}
          onClick={() => test.run(() => api.testConnection())}
        >
          {test.busy ? '测试中…' : '测试连接'}
        </button>
        {dirty && <span className="small muted">测试连接用的是已保存的配置</span>}
      </div>

      {test.result && (
        <div className={`note${test.result.is_owner ? '' : ' warn'}`}>
          <strong>连接成功。</strong> {test.result.user.email} · 角色 {test.result.user.role} ·{' '}
          workspace {test.result.workspace.name ?? test.result.workspace.id}
          <span style={{ marginLeft: 8 }}>
            <Pass ok={test.result.is_owner} yes="owner" no="非 owner" />
          </span>
          {test.result.owner_warning && (
            <div className="small" style={{ marginTop: 4 }}>
              {test.result.owner_warning}
            </div>
          )}
        </div>
      )}

      {test.result && test.result.blocked_compiles.length > 0 && (
        <div className="note bad">
          <strong>{test.result.blocked_compiles.length} 次编译在当前连接下用不了。</strong>
          <ul>
            {test.result.blocked_compiles.map((entry) => (
              <li key={entry.id} className="small">
                <span className="mono">{entry.run_id}</span> · {entry.reason}
              </li>
            ))}
          </ul>
        </div>
      )}

    </div>
  )
}

/** 四项配置各自的标题与说明。改了要重新编译的那两项在 REBUILD 里。 */
const FEATURE_LABELS: Record<string, { title: string }> = {
  compiler: { title: '编译模型' },
  embedding: { title: '嵌入模型' },
  answer: { title: '回答模型' },
  image: { title: '图像模型' },
}

const REBUILD = ['compiler', 'embedding']

const BLANK_CONFIG = { model: '', baseUrl: '', apiKey: '' }

/** 四项配置一张表，与 judge / 归因同一套：点「编辑」在下方展开表单。
 *
 * 没有新建与删除 —— 这四项是 Akasha 固定的 feature，只能改，不能增删。
 */
function ModelConfigs() {
  const { data, error, loading, reload } = useAsync(() => api.modelConfigs(), [])
  const [editing, setEditing] = useState<string | null>(null)
  const [form, setForm] = useState(BLANK_CONFIG)
  const save = useAction<{ impact: string; requires_new_compile: boolean }>()

  // 保存后会 reload：只在首次加载时让位给 Loading，否则表格会连同刚出的结果一起闪掉。
  if (loading && !data) return <Loading what="模型配置" />
  if (error)
    return (
      <div className="panel">
        <h3>Akasha 模型配置</h3>
        <Failed error={error} />
        <p className="small muted">这一段需要连上 Akasha。先把上面的连接配好并测试通过。</p>
      </div>
    )
  if (!data) return null

  const live: ModelConfig[] = Array.isArray(data.live) ? data.live : (data.live.configs ?? [])
  const byFeature = new Map(live.map((entry) => [entry.feature, entry]))
  const entry = editing === null ? undefined : byFeature.get(editing)
  const meta = FEATURE_LABELS[editing ?? ''] ?? { title: editing ?? '' }

  const set = (patch: Partial<typeof form>) => {
    setForm({ ...form, ...patch })
    save.reset()
  }
  const edit = (feature: string) => {
    const current = byFeature.get(feature)
    setEditing(feature)
    setForm({ model: current?.model ?? '', baseUrl: current?.baseUrl ?? '', apiKey: '' })
    save.reset()
  }
  const close = () => {
    setEditing(null)
    setForm(BLANK_CONFIG)
    save.reset()
  }

  // 密钥留空且模型与 base_url 没动，这次保存什么也不会改。
  const dirty =
    form.apiKey !== '' ||
    form.model !== (entry?.model ?? '') ||
    form.baseUrl !== (entry?.baseUrl ?? '')

  return (
    <>
      <div className="panel">
        <h3>Akasha 模型配置</h3>

        {save.error && <Failed error={save.error} />}
        {save.result && (
          <div className={`note${save.result.requires_new_compile ? ' warn' : ''}`}>
            已保存。{save.result.impact}
          </div>
        )}

        <table>
          <thead>
            <tr>
              <th>配置项</th>
              <th>模型</th>
              <th>base_url</th>
              <th>密钥</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {data.features.map((feature) => {
              const row = byFeature.get(feature)
              const label = FEATURE_LABELS[feature] ?? { title: feature }
              return (
                <tr key={feature} className={feature === editing ? 'selected' : ''}>
                  <td>
                    <div className="stack">
                      <span>
                        {label.title}
                        {REBUILD.includes(feature) && (
                          <span className="tag warn" style={{ marginLeft: 6 }}>
                            需重编译
                          </span>
                        )}
                      </span>
                      <span className="mono muted">{feature}</span>
                    </div>
                  </td>
                  <td className="small mono">{row?.model ?? '—'}</td>
                  <td className="small mono muted truncate">{row?.baseUrl ?? '—'}</td>
                  <td>
                    <Pass ok={Boolean(row?.apiKeySet)} yes="已设置" no="缺失" />
                  </td>
                  <td>
                    <button className="action small" onClick={() => edit(feature)}>
                      编辑
                    </button>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>

        {editing !== null && (
          <>
            <h4>
              编辑{meta.title} <code className="small">{editing}</code>
            </h4>

            <div className="form-grid inline">
              <Field label="模型">
                <input value={form.model} onChange={(e) => set({ model: e.target.value })} />
              </Field>
              <Field label="base_url">
                <input
                  value={form.baseUrl}
                  onChange={(e) => set({ baseUrl: e.target.value })}
                  placeholder="https://api.example.com/v1"
                />
              </Field>
              <SecretField
                label="api_key"
                hint="留空保留原值"
                value={form.apiKey}
                onChange={(value) => set({ apiKey: value })}
              />
            </div>

            <div className="panel-actions">
              <button
                className="action primary"
                disabled={save.busy || !dirty || !form.model || !form.baseUrl}
                onClick={() => {
                  const feature = editing
                  if (
                    REBUILD.includes(feature) &&
                    !window.confirm(
                      `${feature} 改了必须重新编译，已有编译的结果不再可比。确定要改吗？`,
                    )
                  )
                    return
                  save.run(async () => {
                    const payload = Object.fromEntries(
                      Object.entries(form).filter(([, value]) => value !== ''),
                    )
                    const result = await api.saveModelConfig(feature, payload)
                    setEditing(null)
                    setForm(BLANK_CONFIG)
                    reload()
                    return result
                  })
                }}
              >
                {save.busy ? '保存中…' : '保存到 Akasha'}
              </button>
              <button className="action" disabled={save.busy} onClick={close}>
                取消
              </button>
              {!dirty && <span className="small muted">没有改动</span>}
            </div>
          </>
        )}
      </div>

      <ConfigDrift features={data.features} compiles={data.compiles} />
    </>
  )
}

/** 现在的配置与各次编译的快照是否一致。四项放一张表里才看得出差在哪一项。 */
function ConfigDrift({
  features,
  compiles,
}: {
  features: string[]
  compiles: { id: number; run_id: string; drift: Record<string, boolean> }[]
}) {
  if (!compiles.some((entry) => Object.values(entry.drift).some(Boolean))) return null
  return (
    <div className="panel">
      <h3>与已有编译的差异</h3>
      <table>
        <thead>
          <tr>
            <th>编译</th>
            {features.map((feature) => (
              <th key={feature}>{feature}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {compiles.map((entry) => (
            <tr key={entry.id}>
              <td className="mono small">{entry.run_id}</td>
              {features.map((feature) => (
                <td key={feature}>
                  <Pass ok={!entry.drift[feature]} yes="一致" no="已变" />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

const BLANK = { label: '', base_url: '', model: '', api_key: '' }

/** 表单默认收起，由「新建端点」或表格里的「编辑」打开。
 *
 * 编辑必须带上 id：后端按 (role, label) 做 upsert，只按 label 提交会让「改个名字」
 * 变成新增一条，原来那条还留着。
 */
function Providers({ role, title }: { role: 'judge' | 'attribution'; title: string }) {
  const { data, error, loading, reload } = useAsync<Provider[]>(() => api.providers(role), [role])
  // null 是收起来，'new' 是新建，数字是在改那一条。
  const [mode, setMode] = useState<number | 'new' | null>(null)
  const [form, setForm] = useState(BLANK)
  const save = useAction<{ id: number }>()
  const remove = useAction<unknown>()

  const providers = data ?? []
  const editing = typeof mode === 'number' ? mode : null
  const set = (patch: Partial<typeof form>) => {
    setForm({ ...form, ...patch })
    save.reset()
  }
  const close = () => {
    setMode(null)
    setForm(BLANK)
    save.reset()
    remove.reset()
  }
  const create = () => {
    setMode('new')
    setForm(BLANK)
    save.reset()
    remove.reset()
  }
  const edit = (provider: Provider) => {
    setMode(provider.id)
    setForm({
      label: provider.label,
      base_url: provider.base_url,
      model: provider.model,
      api_key: '',
    })
    save.reset()
    remove.reset()
  }

  // 新建时标签不能撞已有的，否则那一条会被悄悄覆盖。
  const label = form.label.trim() || 'default'
  const taken = providers.some((p) => p.label === label && p.id !== editing)

  if (loading && !data) return <Loading what={title} />

  return (
    <div className="panel">
      <div className="panel-head">
        <h3>{title}</h3>
        <span className="small muted">{providers.length} 个端点</span>
      </div>

      {error && <Failed error={error} />}
      {save.error && <Failed error={save.error} />}
      {remove.error && <Failed error={remove.error} />}

      {providers.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>标签</th>
              <th>模型</th>
              <th>base_url</th>
              <th>密钥</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {providers.map((provider) => (
              <tr key={provider.id} className={provider.id === editing ? 'selected' : ''}>
                <td>{provider.label}</td>
                <td className="small mono">{provider.model}</td>
                <td className="small mono muted truncate">{provider.base_url}</td>
                <td>
                  <Pass ok={provider.api_key_set} yes="已设置" no="缺失" />
                </td>
                <td>
                  <div className="row tight">
                    <button className="action small" onClick={() => edit(provider)}>
                      编辑
                    </button>
                    <button
                      className="action small danger"
                      disabled={remove.busy}
                      onClick={() => {
                        if (!window.confirm(`删除端点「${provider.label}」？\n\n引用它的评测与归因记录会失去关联。`))
                          return
                        remove.run(async () => {
                          const result = await api.deleteProvider(provider.id)
                          if (editing === provider.id) close()
                          reload()
                          return result
                        })
                      }}
                    >
                      删除
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {providers.length === 0 && !error && mode === null && (
        <p className="small muted">还没有配端点。</p>
      )}

      {save.result && <div className="note">已保存端点。</div>}

      {mode !== null && (
        <>
          <h4>{editing === null ? '新建端点' : `编辑端点 #${editing}`}</h4>
          <div className="form-grid inline">
            <Field label="标签" hint="同一角色下不重名">
              <input
                value={form.label}
                onChange={(e) => set({ label: e.target.value })}
                placeholder="default"
              />
            </Field>
            <Field label="模型">
              <input value={form.model} onChange={(e) => set({ model: e.target.value })} />
            </Field>
            <Field label="base_url">
              <input
                value={form.base_url}
                onChange={(e) => set({ base_url: e.target.value })}
                placeholder="https://api.example.com/v1"
              />
            </Field>
            <SecretField
              label="api_key"
              hint={editing === null ? undefined : '留空保留原值'}
              value={form.api_key}
              onChange={(value) => set({ api_key: value })}
            />
          </div>

          {taken && (
            <div className="note bad">
              已经有一个叫「{label}」的端点了。换个标签，或者在上表里点它的「编辑」。
            </div>
          )}
        </>
      )}

      <div className="panel-actions">
        {mode === null ? (
          <button className="action" onClick={create}>
            新建端点
          </button>
        ) : (
          <>
            <button
              className="action primary"
              disabled={save.busy || taken || !form.base_url || !form.model}
              onClick={() =>
                save.run(async () => {
                  const result = await api.saveProvider(role, {
                    ...form,
                    label,
                    ...(editing === null ? {} : { id: editing }),
                  })
                  setMode(null)
                  setForm(BLANK)
                  reload()
                  return result
                })
              }
            >
              {save.busy ? '保存中…' : editing === null ? '创建端点' : '保存修改'}
            </button>
            <button className="action" disabled={save.busy} onClick={close}>
              取消
            </button>
          </>
        )}
      </div>
    </div>
  )
}

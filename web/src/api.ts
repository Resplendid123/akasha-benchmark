import type {
  AttributionDetail,
  CompileDoc,
  CompileRun,
  Connection,
  ConnectionTest,
  CorpusList,
  DatasetEntry,
  EvalDetail,
  EvalSampleDetail,
  Lineage,
  MetricsView,
  ModelConfigsView,
  Paged,
  Provider,
  ProviderProbe,
  RawSamples,
  ResponseList,
  SampleDetail,
  SampleList,
  Stage,
  Task,
  TaskDetail,
} from './types'

// 令牌只在绑非回环地址时需要。放 sessionStorage：关掉标签页就没了。
const TOKEN_KEY = 'akasha-platform-token'

export const setToken = (token: string) => sessionStorage.setItem(TOKEN_KEY, token)
export const getToken = () => sessionStorage.getItem(TOKEN_KEY) ?? ''

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message)
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const token = getToken()
  const response = await fetch(path, {
    ...init,
    headers: {
      ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
      ...(token ? { 'X-Auth-Token': token } : {}),
      ...init?.headers,
    },
  })
  if (!response.ok) {
    let detail = `HTTP ${response.status}`
    try {
      const body = (await response.json()) as { detail?: string }
      if (body.detail) detail = body.detail
    } catch {
      // 响应不是 JSON，保留状态码即可。
    }
    throw new ApiError(response.status, detail)
  }
  return (await response.json()) as T
}

const post = <T>(path: string, body?: unknown) =>
  request<T>(path, {
    method: 'POST',
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  })

const put = <T>(path: string, body: unknown) =>
  request<T>(path, { method: 'PUT', body: JSON.stringify(body) })

const del = <T>(path: string) => request<T>(path, { method: 'DELETE' })

const query = (params: Record<string, string | number | boolean | undefined>) => {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== '') search.set(key, String(value))
  }
  const text = search.toString()
  return text ? `?${text}` : ''
}

export const api = {
  health: () =>
    request<{
      ok: boolean
      settings: Record<string, unknown>
      startup: { archived_legacy_db: string | null; recovered_tasks: number }
    }>('/api/health'),

  // --- 配置 ---
  connection: () => request<Connection>('/api/connection'),
  saveConnection: (payload: Record<string, unknown>) =>
    put<{ updated: string[]; connection: Connection }>('/api/connection', payload),
  testConnection: () => post<ConnectionTest>('/api/connection/test'),
  modelConfigs: () => request<ModelConfigsView>('/api/model-configs'),
  saveModelConfig: (feature: string, payload: Record<string, unknown>) =>
    put<{ feature: string; requires_new_compile: boolean; impact: string }>(
      `/api/model-configs/${feature}`,
      payload,
    ),
  providers: (role?: 'judge' | 'attribution') =>
    request<Provider[]>(`/api/providers${query({ role })}`),
  saveProvider: (role: 'judge' | 'attribution', payload: Record<string, unknown>) =>
    put<{ id: number; api_key_set: boolean }>(`/api/providers/${role}`, payload),
  deleteProvider: (id: number) => del<{ deleted: number }>(`/api/providers/${id}`),
  // 真调一次这个端点，发一句 hi。
  probeProvider: (id: number) => post<ProviderProbe>(`/api/providers/${id}/probe`),

  // --- 数据集层与归一化层 ---
  datasets: () => request<{ datasets: DatasetEntry[]; dataset_dir: string }>('/api/datasets'),
  deleteDataset: (name: string) =>
    del<{ deleted: number }>(`/api/datasets/${encodeURIComponent(name)}`),
  rawSamples: (
    name: string,
    params: { kind?: 'qa' | 'corpus'; limit?: number; offset?: number } = {},
  ) => request<RawSamples>(`/api/datasets/${name}/raw${query(params)}`),
  samples: (name: string, params: { q?: string; limit?: number; offset?: number } = {}) =>
    request<SampleList>(`/api/datasets/${name}/samples${query(params)}`),
  sample: (name: string, sampleId: string) =>
    request<SampleDetail>(`/api/datasets/${name}/samples/${encodeURIComponent(sampleId)}`),
  corpus: (name: string, params: { q?: string; limit?: number; offset?: number } = {}) =>
    request<CorpusList>(`/api/datasets/${name}/corpus${query(params)}`),
  metrics: (datasets: string[] = []) =>
    request<MetricsView>(`/api/metrics${query({ datasets: datasets.join(',') })}`),

  // --- 编译层 ---
  compiles: () => request<{ compiles: CompileRun[] }>('/api/compiles'),
  compileDocs: (
    id: number,
    params: { dataset?: string; gold_only?: boolean; limit?: number; offset?: number } = {},
  ) =>
    request<Paged & { docs: CompileDoc[]; imported: number }>(
      `/api/compiles/${id}/docs${query(params)}`,
    ),
  deleteCompile: (id: number) =>
    del<{ deleted: number; space_id: string | null; note: string }>(`/api/compiles/${id}`),

  // --- 查询层 ---
  responses: (
    id: number,
    params: { dataset?: string; answer_mode?: string; limit?: number; offset?: number } = {},
  ) => request<ResponseList>(`/api/queries/${id}/responses${query(params)}`),
  response: (id: number, sampleId: string) =>
    request<Record<string, unknown>>(
      `/api/queries/${id}/responses/${encodeURIComponent(sampleId)}`,
    ),
  deleteQuery: (id: number) => del<{ deleted: number }>(`/api/queries/${id}`),

  // --- 评测层 ---
  evalRun: (id: number) => request<EvalDetail>(`/api/evals/${id}`),
  evalSample: (id: number, sampleId: string) =>
    request<EvalSampleDetail>(`/api/evals/${id}/samples/${encodeURIComponent(sampleId)}`),
  deleteEval: (id: number) => del<{ deleted: number }>(`/api/evals/${id}`),

  // --- 归因层 ---
  attribution: (id: number) => request<AttributionDetail>(`/api/attributions/${id}`),
  deleteAttribution: (id: number) => del<{ deleted: number }>(`/api/attributions/${id}`),
  lineage: (pageId: string, question = '') =>
    request<Lineage>(`/api/lineage/${encodeURIComponent(pageId)}${query({ question })}`),

  // --- 任务层 ---
  stages: () => request<Stage[]>('/api/stages'),
  tasks: (status?: string) => request<Task[]>(`/api/tasks${query({ status })}`),
  task: (id: number, afterId = 0) =>
    request<TaskDetail>(`/api/tasks/${id}${query({ after_id: afterId })}`),
  startTask: (stage: string, args: Record<string, unknown>) =>
    post<Task>(`/api/tasks/${stage}`, args),
  // 起四条普通阶段任务，返回链首那条。
  startChain: (args: Record<string, unknown>) => post<Task>('/api/chain', args),
  pauseTask: (id: number) => post<Task>(`/api/tasks/${id}/pause`),
  resumeTask: (id: number) => post<Task>(`/api/tasks/${id}/resume`),
  deleteTask: (id: number) => del<{ deleted: number }>(`/api/tasks/${id}`),
  cleanupTasks: () => post<{ deleted: number }>('/api/tasks/cleanup/inactive'),
  audit: (stage?: string, limit = 200) =>
    request<TaskDetail['logs']>(`/api/audit${query({ stage, limit })}`),
}

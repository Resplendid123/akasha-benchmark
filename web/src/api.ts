import type {
  AdapterList,
  AvailableMetrics,
  BadcaseList,
  BatchAnalysis,
  AppConnection,
  ConnectionTest,
  CorpusDocs,
  Dataset,
  Diff,
  EvalLayerDetail,
  IndexLayer,
  LayerDocDetail,
  LayerDocs,
  Lineage,
  MetricDefinition,
  ModelConfigsView,
  NormalizedSampleDetail,
  NormalizedSamples,
  Provider,
  RawSamples,
  ResponseList,
  SampleDetail,
  SampleLineage,
  SampleList,
  Stage,
  Task,
  TaskDetail,
  WorstList,
} from './types'

// 令牌只在绑非回环地址时才需要。放 sessionStorage 而不是 localStorage：
// 关掉标签页就没了，少一个长期留在磁盘上的凭据。
const TOKEN_KEY = 'akasha-platform-token'

export function setToken(token: string): void {
  sessionStorage.setItem(TOKEN_KEY, token)
}

export function getToken(): string {
  return sessionStorage.getItem(TOKEN_KEY) ?? ''
}

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
  request<T>(path, { method: 'POST', ...(body === undefined ? {} : { body: JSON.stringify(body) }) })

const query = (params: Record<string, string | number | boolean | undefined>) => {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== '') search.set(key, String(value))
  }
  const text = search.toString()
  return text ? `?${text}` : ''
}

export const api = {
  health: () => request<{ ok: boolean; settings: Record<string, unknown> }>('/api/health'),

  // --- 连接（只有一份）---
  connection: () => request<AppConnection>('/api/connection'),
  saveConnection: (payload: Record<string, unknown>) =>
    request<{ updated: string[]; connection: AppConnection; warnings: string[] }>(
      '/api/connection',
      { method: 'PUT', body: JSON.stringify(payload) },
    ),
  testConnection: () => post<ConnectionTest>('/api/connection/test'),

  // --- Akasha 侧的模型配置 ---
  modelConfigs: () => request<ModelConfigsView>('/api/model-configs'),
  saveModelConfig: (feature: string, payload: Record<string, unknown>) =>
    request<{ feature: string; requires_new_index_layer: boolean; impact: string }>(
      `/api/model-configs/${feature}`,
      { method: 'PUT', body: JSON.stringify(payload) },
    ),
  discardIngest: (layerId: number, confirm: boolean) =>
    post<{ discarded: Record<string, number>; note: string }>(
      `/api/layers/index/${layerId}/discard-ingest`,
      { confirm },
    ),
  providers: (role?: 'judge' | 'analysis') =>
    request<Provider[]>(`/api/providers${role ? `/${role}` : ''}`),
  saveProvider: (role: 'judge' | 'analysis', payload: Record<string, unknown>) =>
    request<{ id: number; api_key_set: boolean }>(`/api/providers/${role}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),
  deleteProvider: (id: number) =>
    request<{ deleted: number }>(`/api/providers/${id}`, { method: 'DELETE' }),

  // --- 数据集层与归一化层 ---
  datasets: () => request<Dataset[]>('/api/datasets'),
  adapters: () => request<AdapterList>('/api/adapters'),
  rawSamples: (name: string, params: { limit?: number; offset?: number } = {}) =>
    request<RawSamples>(`/api/datasets/${name}/raw${query(params)}`),
  normalizedSamples: (
    name: string,
    params: { q?: string; limit?: number; offset?: number } = {},
  ) => request<NormalizedSamples>(`/api/datasets/${name}/samples${query(params)}`),
  normalizedSample: (name: string, sampleId: string) =>
    request<NormalizedSampleDetail>(
      `/api/datasets/${name}/samples/${encodeURIComponent(sampleId)}`,
    ),
  corpus: (name: string, params: { q?: string; limit?: number; offset?: number } = {}) =>
    request<CorpusDocs>(`/api/datasets/${name}/corpus${query(params)}`),

  // --- 指标 ---
  metricDefinitions: () => request<MetricDefinition[]>('/api/metrics/definitions'),
  availableMetrics: (datasets: string[]) =>
    request<AvailableMetrics>(`/api/metrics/available${query({ datasets: datasets.join(',') })}`),

  // --- 层 ---
  layers: () => request<{ index_layers: IndexLayer[] }>('/api/layers'),
  indexLayer: (id: number) => request<Record<string, unknown>>(`/api/layers/index/${id}`),
  layerDocs: (
    id: number,
    params: { dataset?: string; gold_only?: boolean; q?: string; limit?: number; offset?: number } = {},
  ) => request<LayerDocs>(`/api/layers/index/${id}/docs${query(params)}`),
  layerDoc: (id: number, dataset: string, docId: string) =>
    request<LayerDocDetail>(
      `/api/layers/index/${id}/docs/${dataset}/${encodeURIComponent(docId)}`,
    ),

  responses: (
    queryLayerId: number,
    params: { dataset?: string; answer_mode?: string; limit?: number; offset?: number } = {},
  ) => request<ResponseList>(`/api/layers/query/${queryLayerId}/responses${query(params)}`),
  response: (queryLayerId: number, sampleId: string) =>
    request<Record<string, unknown>>(
      `/api/layers/query/${queryLayerId}/responses/${encodeURIComponent(sampleId)}`,
    ),

  evalLayer: (id: number) => request<EvalLayerDetail>(`/api/layers/eval/${id}`),
  samples: (evalLayerId: number, params: { dataset?: string; limit?: number } = {}) =>
    request<SampleList>(
      `/api/layers/eval/${evalLayerId}/samples${query({ ...params, limit: params.limit ?? 500 })}`,
    ),
  worst: (evalLayerId: number, metric: string, dataset?: string, limit = 20) =>
    request<WorstList>(
      `/api/layers/eval/${evalLayerId}/worst${query({ metric, dataset, limit })}`,
    ),
  sample: (evalLayerId: number, sampleId: string) =>
    request<SampleDetail>(
      `/api/layers/eval/${evalLayerId}/samples/${encodeURIComponent(sampleId)}`,
    ),
  sampleLineage: (evalLayerId: number, sampleId: string) =>
    request<SampleLineage>(
      `/api/layers/eval/${evalLayerId}/samples/${encodeURIComponent(sampleId)}/lineage`,
    ),

  // --- 血缘与 diff ---
  lineage: (pageId: string) => request<Lineage>(`/api/lineage/${encodeURIComponent(pageId)}`),
  diff: (pageId: string, question = '') =>
    request<Diff>(`/api/diff/${encodeURIComponent(pageId)}${query({ question })}`),

  // --- 归因层 ---
  badcases: (evalLayerId: number) => request<BadcaseList>(`/api/badcase/${evalLayerId}`),
  analyzeSample: (evalLayerId: number, sampleId: string, useModel: boolean) =>
    post<BatchAnalysis['results'][number]>(
      `/api/badcase/${evalLayerId}/${encodeURIComponent(sampleId)}`,
      { use_model: useModel },
    ),
  analyzeWorst: (
    evalLayerId: number,
    params: { metric: string; dataset?: string; limit?: number },
    useModel: boolean,
  ) =>
    post<BatchAnalysis>(`/api/badcase/${evalLayerId}/batch/worst${query(params)}`, {
      use_model: useModel,
    }),

  // --- 任务层 ---
  stages: () => request<Stage[]>('/api/stages'),
  tasks: (status?: string) => request<Task[]>(`/api/tasks${query({ status })}`),
  task: (id: number, afterId = 0) =>
    request<TaskDetail>(`/api/tasks/${id}${query({ after_id: afterId })}`),
  startTask: (stage: string, args: Record<string, unknown>) => post<Task>(`/api/tasks/${stage}`, args),
  cancelTask: (id: number) => post<Task>(`/api/tasks/${id}/cancel`),
  deleteTask: (id: number) =>
    request<{ deleted: number }>(`/api/tasks/${id}`, { method: 'DELETE' }),
  cleanupFinished: () => post<{ deleted: number }>('/api/tasks/cleanup/finished'),

  // --- 标注 ---
  addAnnotation: (payload: Record<string, unknown>) =>
    post<{ id: number }>('/api/annotations', payload),
  agreement: (level = 'sample') =>
    request<{ compared: number; agreement: number | null; note: string }>(
      `/api/annotations/agreement${query({ level })}`,
    ),
}

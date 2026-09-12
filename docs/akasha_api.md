# Akasha 接口样例

核心流程是导入语料和查询问题，还会调用登录、Space 管理、编译、质量诊断及模型配置接口。客户端在
[akasha_client.py](../src/akasha_benchmark/akasha_client.py)。

| 阶段 | 接口 | 用途 |
| --- | --- | --- |
| 入库 | `POST /api/pages/import` | 把 `{doc_id}.md` 导入 space，拿 `page_id` |
| 查询 | `POST /api/llm-wiki/query` | 跑一条问题，落盘完整响应 |

下面的形状都对着服务端源码核过（`../Akasha`），字段名右侧标注的行号指向 Akasha 仓库。

## 零、所有响应都套一层信封

`main.ts:160` 给**所有**路由挂了 `TransformHttpResponseInterceptor`，把每个
handler 的返回值包成 `{data, success, status}`（`http-response.interceptor.ts:33-38`）：

```json
{ "data": { "...真正的载荷..." }, "success": true, "status": 200 }
```

只有标了 `@SkipTransform()` 的 handler 例外，全仓库共三个（mcp、health、
robots.txt），本基准一个都不用 —— 也就是说**下面每个端点都套着信封**。

`POST /api/auth/login` 是特例：handler 没有返回值，所以信封里没有 `data` 键，
只有 `{"success": true, "status": 200}`。

**本文后面所有响应样例写的都是剥掉信封之后的形状。**
客户端在 [`request()`](../src/akasha_benchmark/akasha_client.py) 里统一剥
（见 `unwrap_envelope`），所以各阶段可以照本文直接读字段。

不剥的后果不是报错，而是静默读空，而且每一处的症状都会指向错误的方向：

| 位置 | 症状 |
| --- | --- |
| `users/me` | `role` 取到 `None`，OWNER 闸门永远拒绝执行 |
| `pages/import` | `page.get("id")` 为 `None`，每篇都记进 `failures` |
| `diagnostics/quality` | `summary` 为 `None`，四项闸门全 `None`，`all(value == 0)` **假通过** |
| `llm-wiki/query` | `answerMode` / `retrievedSources` 全空，每个指标算成 0 |

第三行最危险：闸门假通过，然后拿一个空索引跑出一整份看起来正常的报告。

判据要收紧到信封自身的形状（`success` 是 bool、`status` 是 int、键集合
⊆ `{data, success, status}`），不能只看有没有 `data` —— 某个端点的正常载荷里
完全可以有一个叫 `data` 的字段，那种不能动。

## 认证

两个接口都要先登录。`POST /api/auth/login` 成功时**响应体只有信封**
（`{"success": true, "status": 200}`，剥出来是 `None`），
只 set 一个 httpOnly 的 `authToken` cookie（`auth.controller.ts:222`）。
httpx 的 cookie jar 直接带上就行，不用自己拼 bearer header。

```json
{ "email": "admin@example.com", "password": "..." }
```

登录返回 2xx 但没有 `authToken` cookie，说明这个账号开了 MFA ——
客户端会直接报错，而不是让后面每个请求都以 401 失败。

## 一、`POST /api/pages/import`

multipart 表单，两个字段：`file`（`.md` 文件）和 `spaceId`。

```
file:    hotpotqa_5a7a06935542990198eaf050.md   (text/markdown)
spaceId: 018f2c1e-....-....-....-............
```

导入服务会**取首个 Markdown heading 当 page title 并从正文里删掉**
（`import.service.ts:106`）。没有 heading 时回落到文件名。
所以文件名只承担 `doc_id` 的职责，不干扰标题。

### 响应

返回新建的 page 行，字段就是 `page.repo.ts:28-47` 的 `baseFields`：

```json
{
  "id": "0191f3a2-6b7c-7d8e-9f01-23456789abcd",
  "slugId": "x7Kp2mQ9",
  "title": "Ed Wood (film)",
  "icon": null,
  "coverPhoto": null,
  "position": "0|100000:",
  "parentPageId": null,
  "creatorId": "0191e0aa-1111-7222-8333-444455556666",
  "lastUpdatedById": "0191e0aa-1111-7222-8333-444455556666",
  "sourceCreatorName": null,
  "sourceLastUpdatedByName": null,
  "spaceId": "018f2c1e-7777-7888-8999-aaaabbbbcccc",
  "workspaceId": "018f2c1e-0000-7111-8222-333344445555",
  "isLocked": false,
  "createdAt": "2026-09-09T02:11:43.512Z",
  "updatedAt": "2026-09-09T02:11:43.512Z",
  "deletedAt": null,
  "contributorIds": ["0191e0aa-1111-7222-8333-444455556666"]
}
```

基准只用 `id` 和 `title`，其余字段不落盘。`content` / `ydoc` / `textContent`
**不在** `baseFields` 里，导入响应不回正文。

`id` 就是后面各阶段要的 `page_id`。[ingest.py:131](../src/akasha_benchmark/ingest.py#L131)
拿不到它会把该行记进 `failures` 而不是继续 —— 没有 page_id 的行等于永久丢失映射，
评测反查不到就只能当 `__unmapped__` 处理。

写进 `page_map.jsonl` 的是这一行（[ingest.py:136-150](../src/akasha_benchmark/ingest.py#L136-L150)）：

```json
{"dataset": "hotpotqa", "doc_id": "hotpotqa_5a7a0693...", "page_id": "0191f3a2-...", "space_id": "018f2c1e-...", "title": "Ed Wood (film)", "md_sha256": "3f1a...", "imported_at": "2026-09-09T02:11:43Z"}
```

`md_sha256` 是导入前重算的，和子集 manifest 里的值不符会直接抛错：
语料在两次运行之间被改过，这份 page_map 就不能用了。

## 二、`POST /api/llm-wiki/query`

请求体见 `query-knowledge.dto.ts`。基准只用前三个字段：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `query` | 是 | 1–4000 字符 |
| `spaceIds` | 是 | 非空、去重、全部是 UUID |
| `type` | 否 | `user` / `robot`，基准固定 `user` |
| `scoreThreshold` | 否 | 语义召回的最大余弦距离，**越小越严**，范围 0–2 |
| `chatContext` | 否 | 最多 30 条，每条 ≤4000 字符 |
| `generalKnowledgeEnabled` | 否 | 允许回落到通用知识 |
| `includeCitations` / `attachments` | 否 | 附件签名 URL |

```json
{
  "query": "Which film whose director was born first, Ed Wood or The Man Who Fell to Earth?",
  "spaceIds": ["018f2c1e-7777-7888-8999-aaaabbbbcccc"],
  "type": "user",
  "scoreThreshold": 0.35
}
```

`scoreThreshold` 是**距离**不是相似度，这点容易反过来理解。省略时用服务端默认值。

### 响应（`answerMode: "knowledge"`）

形状是 `AiKnowledgeChatResult`（`ai-knowledge-chat.service.ts:92-130`），
经 controller 加工后返回（`llm-wiki.controller.ts:242-275`）。为了可读，
下面每个数组只留一项：

```json
{
  "answer": "Ed Wood was directed by Tim Burton, born in 1958. The Man Who Fell to Earth was directed by Nicolas Roeg, born in 1928. Nicolas Roeg was born first.",
  "answerMode": "knowledge",
  "retrievalQuery": "Ed Wood director birth date; The Man Who Fell to Earth director",
  "citations": [
    {
      "sourcePageId": "0191f3a2-6b7c-7d8e-9f01-23456789abcd",
      "title": "Ed Wood (film)",
      "url": "https://akasha.example.com/p/x7Kp2mQ9",
      "images": []
    }
  ],
  "citationEvidence": [
    {
      "sourcePageId": "0191f3a2-6b7c-7d8e-9f01-23456789abcd",
      "title": "Ed Wood (film)",
      "url": "https://akasha.example.com/p/x7Kp2mQ9",
      "excerpts": [
        {
          "text": "Ed Wood is a 1994 American biographical film directed by Tim Burton.",
          "sourceRange": { "startOffset": 0, "endOffset": 67 },
          "quoteHash": "9c1b7f0e2a..."
        }
      ]
    }
  ],
  "retrievedSources": [
    {
      "sourcePageId": "0191f3a2-6b7c-7d8e-9f01-23456789abcd",
      "title": "Ed Wood (film)",
      "url": "https://akasha.example.com/p/x7Kp2mQ9"
    }
  ],
  "snippets": [
    {
      "id": "0191f3a1-4a5b-7c6d-8e9f-0123456789ab",
      "title": "Ed Wood (film)",
      "text": "Ed Wood is a 1994 American biographical film directed by Tim Burton...",
      "retrievalReasons": ["semantic", "graph-neighbor"],
      "sourceWindows": [
        {
          "sourcePageId": "0191f3a2-6b7c-7d8e-9f01-23456789abcd",
          "title": "Ed Wood (film)",
          "url": "https://akasha.example.com/p/x7Kp2mQ9",
          "text": "Ed Wood is a 1994 American biographical film directed by Tim Burton.",
          "sourceRange": { "startOffset": 0, "endOffset": 67 },
          "quoteHash": "9c1b7f0e2a..."
        }
      ]
    }
  ],
  "warnings": [],
  "retrievalReasons": ["semantic", "lexical", "graph-neighbor"],
  "budget": {
    "maxContextLength": 12000,
    "usedContextLength": 8421,
    "remainingContextLength": 3579,
    "includedItemCount": 9,
    "omittedItemCount": 3,
    "responseReserve": 0,
    "perItemMaxLength": 12000
  },
  "completenessNotice": "Some knowledge may be unavailable because access is permission-scoped."
}
```

`retrievalQuery` 只在带 `chatContext` 触发了查询重写时出现，基准不传它，所以实际不会有这个键。

`budget` 里的三个上限**永远是默认值**：没有任何调用点给 `buildContextPack` 传
`budget` 参数，所以 `maxContextLength` 固定 12000、`responseReserve` 固定 0、
`perItemMaxLength` 回落成等于 `maxContextLength`（`knowledge-context-pack.service.ts:242-262`）。
真正随查询变的只有 `usedContextLength` / `remainingContextLength` /
`includedItemCount` / `omittedItemCount` —— `omittedItemCount > 0` 就是上下文截断
把召回结果挤掉了，这是唯一有信息量的那个数。

`snippets[].id` 是 chunk 或 capsule 的原始 UUID，**不带类型前缀**
（`knowledge-context-pack.service.ts:210-239`）。内部的 `kind`（`chunk` / `capsule`）
没有出现在 snippet 里，所以光看响应分不出这条 snippet 来自原文块还是编译产物。

### 三个数组的关系

`retrievedSources` ⊇ `citations`，而 `citationEvidence` 覆盖的页**与 `citations` 完全相同**。

- `retrievedSources` —— 裁剪前的召回全集（`ai-knowledge-chat.service.ts:544`）。
- `citations` —— `resolveAnswerCitations` 收窄成「答案真的引了」∩「有证据支撑」。
- `citationEvidence` —— 对 `citations` **一对一** map，给每项配上 `excerpts`
  （`ai-knowledge-chat.service.ts:1036`）。所以
  `citationEvidence.length === citations.length` 恒成立，它不是更小的子集。

`excerpts` 可以是空数组 —— 那一页被引了但没配上证据窗口。这正是
`evidence_verifiable_rate` 在量的东西。另外每页的 `excerpts`
**最多 2 条**（`ai-knowledge-chat.service.ts:1030` 的 `windows.length < 2`），
按 `quoteHash` + `sourceRange` 去重，所以这个数不反映证据的丰富程度，只反映有无。

检索指标一律用 `retrievedSources`；拿 `citations` 算 Recall 会低估检索能力。
差集才是归因指标要的东西，见 [metrics.md](metrics.md)。

`citations` 是怎么从 `retrievedSources` 收窄的：模型在生成时按
`[[cite:{sourcePageId}]]` 标注引用，`extractCitedSourceIds` 把这些 id 抽出来，
再和 `sourceWindows` 求交集（`ai-knowledge-chat.service.ts:527-534`）。
**返回的 `answer` 里这些标记已经被 `stripCitationMarkers` 全部删掉了**
（`ai-knowledge-chat.service.ts:518`），所以答案是纯散文，没有行内出处 ——
想知道哪句话由哪一页支撑，只能看 `citationEvidence[].excerpts`。
答案 F1 因此不会被引用标记污染，这是好事。

### 四个坑

**`citations` 多一个 `images` 字段，`retrievedSources` 没有。**
controller 只给 `citations` 做图片富化（`llm-wiki.controller.ts:234-244`），
类型上就是 `KnowledgeQueryCitation` 与 `KnowledgeCitation` 之别。
拿两个数组做 diff 时按 `sourcePageId` 比，别整对象比。

`url` 在服务层是相对路径 `/p/{slugId}`，controller 才拼上 appUrl
（`iself-llm-wiki.controller.ts:250-263`）。没配 appUrl 时保持相对路径原样返回，
所以这个字段可能是绝对 URL 也可能是 `/p/x7Kp2mQ9`，不要拿它当稳定标识 ——
稳定标识是 `sourcePageId`。

**`retrievalReasons` 有两个，含义不同。** 顶层那个是全部入选条目的信号**去重合集**
（`knowledge-context-pack.service.ts:137`），多跳指标要的是
`snippets[].retrievalReasons` —— 只有后者能定位到具体哪条 snippet 由哪个信号产出。
已知取值：`semantic`、`lexical`、`exact-title`、`graph-neighbor`、`sidecar-prefiltered`。

**响应里没有 `retrievalDiagnostics` 和 `retrievalScope`。** controller 在
`llm-wiki.controller.ts:174` 解构时排除了这两个键，它们只写进
`knowledge_query_audit.metadata`。所以三段归因必须走 `audit_join`
读数据库，没有别的路径（见 [metrics.md](metrics.md) 的「三段归因」）。

### `no_match` 与 `general`

这两种模式**无条件**返回空的 `retrievedSources` / `citations` /
`citationEvidence` / `snippets`（`ai-knowledge-chat.service.ts:641,667`），
不管检索实际找到了什么：

```json
{
  "answer": "No usable knowledge was found. Try rephrasing the question, adding more context, choosing another knowledge space, or enabling general knowledge mode.",
  "answerMode": "no_match",
  "citations": [],
  "citationEvidence": [],
  "retrievedSources": [],
  "snippets": [],
  "warnings": [],
  "retrievalReasons": [],
  "budget": { "maxContextLength": 12000, "usedContextLength": 0, "remainingContextLength": 12000, "includedItemCount": 0, "omittedItemCount": 0, "responseReserve": 0, "perItemMaxLength": 12000 },
  "completenessNotice": "Some knowledge may be unavailable because access is permission-scoped."
}
```

`no_match` 的 `answer` 是**固定文案**，按问题里有没有汉字在中英两句之间二选一
（`ai-knowledge-chat.service.ts:1073-1079`）。四个数据集的问题都是英文，所以走英文那句。
这句话是常量，答案 F1 对它没有意义。

`general` 形状一样，区别是 `answerMode: "general"` 且 `answer` 前面拼了一段
通用知识免责声明。两者的 `budget` 都来自 `buildContextPack({})`，所以是空 pack 的默认值。

**这些行的检索分数天然是 0，不是检索失败。** 评测因此每个检索指标都出两份，
`retrieval` 和 `retrieval_knowledge_only`，差值就是生成端拒答的规模。

`completenessNotice` 是个**常量**（`knowledge-retrieval.service.ts:25`），
每次都返回同一句话，不代表本次真的有权限截断 —— 别把它当信号读。

## 三、`GET /api/llm-wiki/admin/model-configs`

入库与查询各拉一次做快照比对。剥掉信封后是：

```json
{
  "configs": [
    {
      "feature": "compiler",
      "provider": "openai-compatible",
      "model": "qwen3.8-flash",
      "baseUrl": "https://tokencheap.io/v1",
      "apiKeySet": true,
      "parameters": null
    },
    { "feature": "answer",    "...": "..." },
    { "feature": "image",     "...": "..." },
    { "feature": "embedding", "...": "...", "parameters": { "dimension": 1024, "supportsMrl": false } }
  ]
}
```

是**数组**不是按 feature 索引的对象，四项齐全。`provider` 只有
`openai-compatible` 一个合法取值（`20260808T100000-ai-model-configs.ts:18`
的 CHECK 约束），没有 Anthropic 驱动。

`apiKeySet` 是布尔量，不回传 key 本身。它为 `false` 时编译不会在配置阶段报错，
而是等真正调用时才失败，失败得很晚 —— 所以值得在入库前先看一眼。

**`baseUrl` 必须自己带上 `/v1`。** Akasha 用
`createOpenAICompatible({baseURL})`（`ai-model-factory.ts:21-25`），而该 SDK 是
`new URL(`${baseURL}${path}`)` 纯字符串拼接、只补 `/chat/completions` 或
`/embeddings`。所以填 `https://host` 会打到 `https://host/chat/completions`，
少一段 `/v1`，页级诊断只会给出 `errorCode: provider_error` /
"Knowledge compiler provider request failed."，看不出是路径问题。
判据是耗时：几百毫秒就失败是 404，真实推理是几十秒。

## 查询记录

每个 `(query_layer_id, sample_id)` 在 SQLite 的 `query_response` 中保存一行：
样本、问题、请求时间、耗时、HTTP 状态、错误及完整业务响应（`response_json`）。
响应信封由客户端剥除后保存。

失败同样入库：连接错误记 `http_status=0`，HTTP 错误保留真实状态码。
常规续跑跳过已有记录；需要重试失败样本时使用查询阶段的 `--retry-failed`。

## 平台验证与开发回归

日常验证从「评测层 → 小样本验证」启动，检查结果和失败原因进入任务日志，指标进入报告。
下面的 pytest 命令仅用于开发者核对服务端契约与离线重放。

本文写下的每条字段约定都由 [tests/test_live_akasha.py](../tests/test_live_akasha.py)
在真实 Akasha 上核对 —— 响应里没有 `retrievalDiagnostics`、`citationEvidence`
与 `citations` 等长、`budget` 上限恒为 12000、snippet id 不带类型前缀、
`no_match` 四个数组全空、质量计数是 camelCase，等等。这些约定一旦不成立，
评测算出来的数就是错的，而且不会有任何报错，所以它们需要断言而不是只写在文档里。

```bash
AKASHA_LIVE=1 uv run pytest tests/test_live_akasha.py -v  # 开发者接口回归
```

跑完的完整往返写进 `data/smoke/<ts>-roundtrip.json`，可以离线重放：

```bash
AKASHA_LIVE_REPLAY=data/smoke/<ts>-roundtrip.json uv run pytest tests/test_live_akasha.py -v
```

## 手动验一条

```bash
curl -s -c /tmp/ak.txt -X POST http://localhost:3000/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"admin@example.com","password":"..."}'

curl -s -b /tmp/ak.txt -X POST http://localhost:3000/api/llm-wiki/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"Who directed Ed Wood?","spaceIds":["<space-uuid>"],"type":"user"}' \
  | jq '.data | {answerMode, retrieved: (.retrievedSources|length), cited: (.citations|length), reasons: [.snippets[].retrievalReasons]|flatten|unique}'
```

注意开头的 `.data` —— 手敲 curl 时信封还在，得自己剥（见 §0）。漏了它每个字段都是
`null`，看起来像服务端没返回数据。

`space-uuid` 从入库的 `manifest.json` 里取（`spaces.<dataset>.id`）。

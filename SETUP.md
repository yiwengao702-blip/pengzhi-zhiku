# 鹏至智库 · 企业级智能知识库 RAG —— 部署与配置说明（SETUP）

> 本文档基于 `D:\knowledge_base` 参考工程整理，目标是让项目**开箱即可运行**。
> 已实测：Python 3.11.9 + 项目自带 `.venv`，两个 FastAPI 服务均可正常启动并响应请求。

---

## 一、项目简介与架构

「鹏至智库」是一个企业级智能知识库 RAG 系统，把 PDF / Markdown 文档导入后完成
**向量化 → 混合检索 → 大模型问答**的全链路。整体由 **两个 FastAPI 服务** 组成，
二者共用同一套本地业务包（`config` / `processor` / `utils` / `tool` / `web`）：

| 服务 | 端口 | 入口模块 | 职责 |
|---|---|---|---|
| 导入服务 | 8000 | `web.api.import_service:app` | 文件上传、LangGraph 导入工作流（切分→向量化→入库 Milvus） |
| 问答服务 | 8001 | `web.api.query_service:app` | 用户提问、混合检索（向量+HyDE+RRF+重排序）、流式生成答案 |

核心编排采用 **LangGraph 双工作流**：
- `processor/import_processor/main_graph.py`：导入流程（`node_entry` → `node_document_split` →
  `node_bge_embedding` → `node_import_milvus` → `node_item_name_recognition` → `node_md_img`）。
- `processor/query_processor/main_graph_v2.py`：问答流程（检索 → HyDE → RRF → 重排序 → 答案生成 → 联网补充）。

技术栈：FastAPI + uvicorn + pydantic + python-dotenv + colorlog + requests +
LangChain（`langchain-openai` / `langchain-text-splitters` / `langchain-core`）+
LangGraph + openai-agents（MCP 联网搜索）+ Milvus（向量库）+ MongoDB（元数据/历史）+
MinIO（对象存储）+ FlagEmbedding（BGE-M3）+ DashScope（重排序/LLM）。

---

## 二、目录结构

```
knowledge_base/
├── main.py                      # 一键启动器：同时拉起 8000/8001 两个服务
├── pyproject.toml               # uv 依赖声明（权威）
├── requirements.txt             # pip 等价导出（含 PyTorch cu128 额外索引）
├── uv.lock                      # 完整可复现锁定（159 个包）
├── DEPENDENCIES.md              # 依赖逐项说明（已审计去冗余）
├── .env.example                 # 环境变量模板（复制为 .env 后填写）
├── .env                         # 本地真实配置（不入库）
├── config/                      # 配置单例（dataclass + load_dotenv）
│   ├── embedding_config.py  milvus_config.py  minio_config.py
│   ├── lm_config.py  reranker_config.py  mineru_config.py  bailian_mcp_config.py
├── processor/
│   ├── import_processor/        # 导入工作流（base / state / nodes / main_graph）
│   └── query_processor/         # 问答工作流（base / state / nodes / prompt / main_graph_v2）
├── utils/                       # 公共能力（embedding / milvus / minio / mongo_history / sse / task / reranker / llm / json_format）
├── tool/                        # 日志 logger、修复脚本 fix_missing_items
├── web/
│   ├── api/                     # import_service.py / query_service.py
│   └── page/                    # import.html / chat.html 前端页面
└── doc/                         # 上传文档与解析结果落盘目录（运行时生成）
```

> ⚠️ `config` / `processor` / `utils` / `tool` / `web` 是**本项目自带源码目录**，
> 运行时要保证**项目根目录在 `PYTHONPATH`**，它们**不应**作为 PyPI 第三方包装进依赖。

---

## 三、Python 环境与依赖安装

- **Python 要求**：`>=3.11`（实测 3.11.9）。
- 项目已自带 `.venv`（依赖已安装）。若需重建环境，二选一：

### 方式 A：uv（推荐）
```bash
pip install uv
uv sync                       # 按 pyproject.toml 同步 .venv
.venv\Scripts\activate        # Windows 激活
```

### 方式 B：pip（通用）
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
> `requirements.txt` 顶部已写 `--extra-index-url https://download.pytorch.org/whl/cu128`
> 用于拉取 **GPU 版 torch**。纯 CPU 环境请删除该行，并把 torch 换成普通版本。

### 依赖审计结论（本次已验证）
声明在 `pyproject.toml` 的顶层依赖**全部可成功导入**，包括：
`fastapi / uvicorn / pydantic / python-dotenv / colorlog / requests /
langchain-openai / langchain-text-splitters / langchain-core / langgraph /
openai-agents(导入名 agents) / pymilvus / pymongo / minio /
FlagEmbedding(导入名 FlagEmbedding) / transformers / modelscope / torch / dashscope`。
（注：库名大小写与 PyPI 包名不同——`openai-agents`→`agents`，`flagembedding`→`FlagEmbedding`。）

> ⚠️ **额外依赖 `sakura_base`（兄弟工程，非 PyPI）**：LLM 网关融合后，文本路径的大模型调用收敛到
> `sakura_base.llm.gateway`，它来自本地兄弟工程 **sakura_platform**，需以可编辑模式安装进本工程 `.venv`：
> `pip install -e "<sakura_platform 路径>/packages/sakura_base[web]"`。未安装则服务在导入期即报错。
> 详细安装与排查见 `DEPENDENCIES.md` 第二节。

---

## 四、环境变量配置（.env）

把 `.env.example` 复制为 `.env`，并按真实环境填写。逐项说明如下：

### 1) 模型来源与缓存
| 变量 | 说明 | 默认值/示例 |
|---|---|---|
| `MINERU_MODEL_SOURCE` | MinerU 模型来源 | `modelscope` |
| `MODELSCOPE_OFFLINE` | ModelScope 离线模式（1=启用） | `1` |
| `MODELSCOPE_CACHE` | ModelScope 缓存目录 | `D:/ai_models/modelscope_cache` |
| `HF_HOME` | HuggingFace 缓存目录 | `D:/ai_models/huggingface_cache` |
| `MD_ROOT_DIR` | 临时文件根目录 | `./temp-files/` |

### 2) LLM / 重排序 API（DashScope 兼容 OpenAI 格式）
| 变量 | 说明 |
|---|---|
| `OPENAI_API_KEY` | API 密钥（**必填**，否则问答节点调用 LLM 失败） |
| `OPENAI_API_BASE` | API 基础地址（阿里云 DashScope / SiliconFlow 等兼容端点） |
| `LLM_DEFAULT_MODEL` | 默认文本模型，如 `qwen-flash` / `deepseek-ai/DeepSeek-V4-Pro` |
| `LLM_DEFAULT_TEMPERATURE` | 温度（0-1，越低越稳） | `0.1` |
| `VL_MODEL` | 视觉语言模型（图片总结节点用） |
| `ITEM_MODEL` | 商品名识别模型 |

### 3) BGE 向量化 / 重排序模型
| 变量 | 说明 |
|---|---|
| `BGE_M3_PATH` | **BGE-M3 本地路径（必填，向量化节点依赖）** |
| `BGE_M3` | 模型名 `BAAI/bge-m3` |
| `BGE_DEVICE` | 运行设备 `cuda:0` / `cpu`（无 GPU 改 `cpu`） |
| `BGE_FP16` | 是否半精度 `True`/`False` |
| `BGE_RERANKER_LARGE` | 重排序模型本地路径 |
| `BGE_RERANKER_DEVICE` | 重排序设备（同上，注意与 GPU 是否可用一致） |
| `BGE_RERANKER_FP16` | 重排序半精度 `1`/`0` |

### 4) 通用嵌入
| 变量 | 说明 |
|---|---|
| `EMBEDDING_DIM` | 嵌入维度（BGE-M3 实际为 1024；若用其它模型按模型填） |
| `EMBEDDING_MODEL` | 嵌入模型名（用于文档说明） |

### 5) 向量库 Milvus
| 变量 | 说明 | 示例 |
|---|---|---|
| `MILVUS_URL` | Milvus 地址 | `http://localhost:19530` |
| `CHUNKS_COLLECTION` | 切片集合名 | `kb_chunks` |
| `ITEM_NAME_COLLECTION` | 商品名集合名 | `kb_item_names` |
| `MILVUS_METRIC_TYPE` | 相似度度量 | `COSINE` |
| `MILVUS_MIN_COSINE_SCORE` | 最小余弦阈值 | `0.75` |

### 6) 文档库 MongoDB
| 变量 | 说明 | 示例 |
|---|---|---|
| `MONGO_URL` | MongoDB 地址 | `mongodb://localhost:27017` |
| `MONGO_DB_NAME` | 数据库名 | `kb001` |

### 7) 对象存储 MinIO
| 变量 | 说明 | 示例 |
|---|---|---|
| `MINIO_ENDPOINT` | 端点 | `localhost:9000` |
| `MINIO_ACCESS_KEY` | 访问密钥 | `minioadmin` |
| `MINIO_SECRET_KEY` | 私有密钥 | `minioadmin` |
| `MINIO_BUCKET_NAME` | 桶名 | `knowledge-base` |
| `MINIO_IMG_DIR` | 图片目录 | `/upload-images` |

### 8) 联网搜索 / MinerU（可选）
| 变量 | 说明 |
|---|---|
| `MCP_DASHSCOPE_BASE_URL` | DashScope MCP 联网搜索地址（query 流程的联网补充节点用） |
| `MCP_API_KEY` | MCP 鉴权密钥 |
| `MINERU_BASE_URL` / `MINERU_API_TOKEN` | MinerU PDF→MD 转换服务（导入流程的 PDF 解析节点用） |

---

## 五、外部中间件准备（运行前置）

本项目只负责 RAG 逻辑，以下中间件需自行准备且**可达**：

1. **Milvus**（默认 `localhost:19530`）：存放切片向量与商品名。首次运行由代码自动建集合。
2. **MongoDB**（默认 `localhost:27017`）：文档元数据与对话历史。已验证可自动连接。
3. **MinIO**（默认 `localhost:9000`）：对象存储（图片等）。需提前建桶 `knowledge-base`。
4. **BGE-M3 / bge-reranker-large 模型文件**：按 `BGE_M3_PATH` / `BGE_RERANKER_LARGE` 提前下载到本地。
5. **MinerU（可选）**：PDF→Markdown 转换服务；不配则用本地解析兜底。

> 若只验证「服务能起来」，只需 MongoDB 可达即可（导入/问答的后台任务在缺 Milvus/模型时会
> 把任务标记为 `failed`，但**接口本身仍正常返回 200**，不会让服务崩溃）。

---

## 六、启动方式

### 一键启动（推荐）
```bash
cd D:\knowledge_base
python main.py
# 同时拉起 导入(8000) + 问答(8001)，Ctrl+C 优雅停止
```

### 单独启动
```bash
# 必须在项目根目录执行，保证本地包在 PYTHONPATH
python -m uvicorn web.api.import_service:app --host 127.0.0.1 --port 8000
python -m uvicorn web.api.query_service:app  --host 127.0.0.1 --port 8001
```
> 若不在根目录运行，请先 `set PYTHONPATH=D:\knowledge_base`（Windows）或
> `export PYTHONPATH=/d/knowledge_base`（Linux/macOS）。

### 探活
```bash
curl http://127.0.0.1:8001/health        # -> {"ok":true}
curl -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/import.html   # -> 200
```
> ⏱️ **冷启动较慢**：导入服务需加载 torch/FlagEmbedding，问答服务需加载 agents(mcp) 等，
> 首次请求约需 **20–40 秒** 才就绪，属正常现象，请耐心等待后再探活。

---

## 七、接口一览

### 导入服务（8000）
| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/upload` | 多文件上传，触发 LangGraph 导入全流程，返回 `task_ids` |
| GET | `/status/{task_id}` | 轮询任务进度（done_list / running_list / status） |
| GET | `/import.html` | 文件导入前端页面 |
| GET | `/health` | 健康检查 |

### 问答服务（8001）
| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/query` | 提问（`query` / `session_id` / `is_stream`），同步或异步返回 |
| GET | `/stream/{session_id}` | SSE 流式结果推送 |
| GET | `/history/{session_id}` | 查询会话历史 |
| DELETE | `/history/{session_id}` | 清空会话历史 |
| GET | `/chat.html` | 对话前端页面 |
| GET | `/health` | 健康检查 |

---

## 八、已知事项与排错

1. **冷启动慢**：见第六节，非 bug。
2. **设备不匹配**：`BGE_DEVICE` / `BGE_RERANKER_DEVICE` 设为 `cuda:0` 但本机无 GPU 会报错，
   纯 CPU 环境请统一改为 `cpu`。
3. **缺 API Key**：`OPENAI_API_KEY` 为空时，问答/图片总结节点会失败（任务 `failed`），但服务不崩。
4. **模型文件缺失**：`BGE_M3_PATH` 指向不存在的目录，向量化节点会失败，请提前下载模型。
5. **MinerU/MCP 未配置**：对应节点会跳过或失败，不影响服务启动与基础检索问答。
6. **`.env` 优于系统环境变量**：各 config 模块用 `load_dotenv(override=True)`，`.env` 中的值会覆盖系统环境变量。

---

## 九、本次交付核对（已验证）

- ✅ 两个 FastAPI 应用均可 `import` 成功（无导入期错误）。
- ✅ `query_service` 冷启动约 30s 后 `GET /health` → `200 {"ok":true}`。
- ✅ `import_service` 冷启动约 20s 后 `GET /import.html` → `200`（前端页存在）。
- ✅ 声明依赖全部可导入（含大小写差异的 `agents` / `FlagEmbedding`）。
- ✅ 修复 `.env.example` 中游离字符 `卸` 导致的配置污染。
- ✅ 为 `import_service` 补充 `/health` 健康检查端点（与 `query_service` 对齐）。

> 说明：完整端到端跑通导入/问答，仍需按要求准备 Milvus、模型文件与有效的 LLM API Key。
> 服务本身已具备「直接运行」能力。

# 依赖说明（DEPENDENCIES）

> 鹏至智库 · 企业级智能知识库 RAG 项目
> 本文档由全仓库 `import` 审计 + `.venv` 实际版本核对生成，保证**完整、准确、可直接使用**。

---

## 一、怎么装（二选一）

### 方式 A：uv（推荐，项目默认）
```bash
# 1) 安装 uv（若尚未安装）
pip install uv
# 2) 按 pyproject.toml 创建/同步虚拟环境并安装全部依赖
uv sync
# 3) 激活环境后运行
.venv\Scripts\activate        # Windows
# 或 macOS/Linux: source .venv/bin/activate
```

### 方式 B：pip（无 uv 也能跑）
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
> ⚠️ `requirements.txt` 顶部已写入 `--extra-index-url https://download.pytorch.org/whl/cu128`，
> 用于拉取 GPU 版 `torch`；若你只需 CPU 版，删掉该行并把 `torch==...cu128` 换成普通版本即可。

---

## 二、第三方依赖清单（顶层，已按用途分组）

下表每一项都在代码里有真实 `import`，已剔除原 `pyproject.toml` 中的误配与冗余项。

| 依赖 | 版本约束 | 用途 | 主要使用位置 |
|---|---|---|---|
| `fastapi` | >=0.138.0 | Web API 框架 | `web/api/import_service.py`、`query_service.py` |
| `uvicorn[standard]` | >=0.51.0 | ASGI 服务器（启动入口） | 各服务的 `if __name__ == "__main__"` |
| `pydantic` | >=2.13.4 | 请求/响应数据模型 | `web/api/query_service.py`、测试 |
| `python-dotenv` | >=1.2.2 | 读取 `.env` 配置 | `config/*`、`processor/.../config_base.py` |
| `colorlog` | >=6.10.1 | 彩色日志 | `tool/logger.py`、`processor/.../base.py` |
| `requests` | >=2.34.2 | HTTP 客户端（MinerU 转换、联网请求） | `node_pdf_to_md.py`、测试 |
| `langchain-openai` | >=1.3.2 | `ChatOpenAI` 调用大模型 | `llm_utils.py`、识别/图片节点 |
| `langchain-text-splitters` | >=1.1.2 | 文档切分 | `node_document_split.py` |
| `langchain-core` | >=1.4.8 | `SystemMessage`/`HumanMessage` | 多个 query 节点（直接 import） |
| `langgraph` | >=1.2.6 | `StateGraph` 工作流编排 | `processor/*/main_graph.py` |
| `openai-agents` | >=0.18.3 | `agents.mcp` 联网搜索 | `node_web_search_mcp.py` |
| `pymilvus[model]` | >=3.0.0 | 向量库 + BGE-M3 嵌入函数 | `embedding_utils.py`、`milvus_utils.py` |
| `pymongo` | >=4.17.0 | 对话历史存储（`bson` 随附） | `mongo_history_utils.py` |
| `minio` | >=7.2.20 | 对象存储（图片等） | `minio_utils.py`、`node_md_img.py` |
| `flagembedding` | >=1.4.0 | BGE-M3 嵌入（由 `pymilvus[model]` 引入，运行必需） | `utils/embedding_utils.py` |
| `transformers` | >=4.45.0,<5.0.0 | `flagembedding` 底层模型加载依赖 | 运行时被 flagembedding 调用 |
| `modelscope` | >=1.38.1 | 模型下载 / 离线缓存（`.env` 中 `MODELSCOPE_OFFLINE`） | 模型管理阶段 |
| `torch` | >=2.11.0 | 深度学习后端（GPU cu128） | `flagembedding` / `transformers` 运行依赖 |
| `dashscope` | >=1.26.4 | 文本重排序服务 | `reranker_http_utils.py` |
| `sakura_base` | 本地可编辑安装（**非 PyPI 包**） | LLM 网关：文本路径大模型调用统一收敛到 `sakura_base.llm.gateway` | `utils/sakura_llm_bridge.py`、`utils/llm_utils.py` |

> ⚠️ **`sakura_base` 是本地兄弟工程依赖，不是 PyPI 包**（2026-07-17 LLM 网关融合引入）。
> 它来自兄弟工程 **sakura_platform**，未安装时 `import utils.sakura_llm_bridge` 会直接抛 `RuntimeError`
> 并提示安装方式。请在 knowledge_base 的 `.venv` 中执行：
> ```bash
> pip install -e "<sakura_platform 路径>/packages/sakura_base[web]"
> # 本机示例：pip install -e "D:\WorkBuddy\2026-10-14-11-06-38\sakura_platform/packages/sakura_base[web]"
> ```
> 安装后 `import sakura_base` 应可直接成功。多模态 / 视觉调用（vl_model）仍由各节点直连 `ChatOpenAI`，不依赖此包。

---

## 三、已被移除的"假依赖 / 冗余项"（重要）

原 `pyproject.toml` 存在以下问题，已全部修正：

| 原条目 | 问题 | 处理 |
|---|---|---|
| `config>=0.5.1` | 与项目本地包 `config/`（配置模块）同名，误当 PyPI 第三方包装 → 会装到无关包 | **删除** |
| `processor>=0.10.0` | 与项目本地包 `processor/`（工作流模块）同名，同上 | **删除** |
| `grandalf>=0.8` | 全仓库无任何 import | **删除** |
| `torchvision>=0.26.0` | 未使用，且体量巨大 | **删除** |
| `torchaudio>=2.11.0` | 未使用，且体量巨大 | **删除** |
| `flagembedding` 重复两条 | 重复声明 | **合并为一条** |
| 漏配 `requests` / `pydantic` / `langchain-core` | 代码已 import 但未声明 | **已补** |

> 注：`config`、`processor`、`utils`、`tool`、`web` 是**本项目自带源码目录**，运行时要保证
> 项目根目录在 `PYTHONPATH`（直接在根目录执行 `python web/api/...py` 即可，根目录默认在路径中）。
> 它们不应出现在外部依赖里。

---

## 四、运行前置（外部服务）

本项目只负责“知识库 RAG”逻辑，以下中间件需自行准备：

| 服务 | 用途 | 配置项（见 `.env`） |
|---|---|---|
| Milvus | 向量数据库 | `MILVUS_URL` |
| MongoDB | 文档元数据 / 对话历史 | `MONGO_URL`、`MONGO_DB_NAME` |
| MinIO | 对象存储（图片等） | `MINIO_ENDPOINT`、`MINIO_ACCESS_KEY`、`MINIO_SECRET_KEY` |
| DashScope（阿里云） | LLM / 重排序 API | `OPENAI_API_KEY`、`OPENAI_API_BASE`、`LLM_DEFAULT_MODEL` |
| MinerU（可选） | PDF→Markdown 转换服务 | `node_pdf_to_md.py` 通过 `requests` 调用 |

---

## 五、GPU 说明

`torch` 通过 `pyproject.toml` 的 `[tool.uv.sources]` 固定从
`https://download.pytorch.org/whl/cu128` 安装 CUDA 12.8 版本。
纯 CPU 环境可改为 `torch>=2.11.0`（去掉 sources 配置），相应调整 `requirements.txt`。

---

## 六、版本来源

- `pyproject.toml`：顶层依赖（带 `>=` 下限，供 `uv` 解析）。
- `uv.lock`：由 `uv lock` 生成的**完整、可复现**锁定文件（159 个包）。
- `requirements.txt`：由 `uv export` 导出的 pip 等价文件（含精确 `==` 版本 + PyTorch 额外索引）。

三者保持一致；优先使用 `uv sync`（开发）或 `pip install -r requirements.txt`（通用）。

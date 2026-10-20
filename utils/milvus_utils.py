# knowledge_base\utils\milvus_utils.py
import threading

from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker
from tool.logger import logger
from config.milvus_config import milvus_config

# 单例
_milvus_client = None
# 建客户端的过程需要互斥：Milvus Lite 在建连时会启动进程内服务并抢占 .db 文件锁，
# 问答流程里「稠密检索 / HyDE 检索」两条分支是并发的，若同时进入会有一个抢锁失败
# （DataDirLockedError: another process holds the lock）。加锁后串行化首次建连。
_milvus_lock = threading.Lock()


def is_lite_mode() -> bool:
    """
    判断当前是否运行在 Milvus Lite（本地 .db 文件）模式。

    判据：MILVUS_URL 不是 http(s):// 也不是 tcp:// 形式，即视为本地文件路径。
    该模式下索引类型受限（官方仅支持 FLAT），且不接受 IVF/AUTOINDEX 的调优参数，
    因此建表逻辑需要据此分支，否则会在 create_collection 阶段报错。
    """
    url = (milvus_config.milvus_url or "").strip()
    return not url.startswith(("http://", "https://", "tcp://"))


def dense_index_type() -> str:
    """当前环境可用的稠密索引类型：Lite 用 FLAT，服务端用 AUTOINDEX。"""
    return "FLAT" if is_lite_mode() else "AUTOINDEX"


def get_milvus_client():

    global _milvus_client
    if _milvus_client is not None:
        return _milvus_client# 存在直接返回

    # 双重检查 + 互斥锁：并发节点同时首次调用时只建一个客户端
    with _milvus_lock:
        if _milvus_client is None:
            # 创建对象（不存在就创建，客户端全局只会创建一次）
            _milvus_client = MilvusClient(milvus_config.milvus_url)
    return _milvus_client


# 已确认 load 过的集合（避免每次检索都多发一次 load 请求）
_loaded_collections = set()


def ensure_collection_loaded(client, collection_name: str) -> None:
    """
    确保集合处于 Loaded 状态（可被检索）。

    【为什么需要】
    Milvus Lite 的集合状态跟随进程：容器/进程重启后，磁盘上的集合是 'released' 状态，
    直接检索会报 `Collection 'xxx' is in state 'released'; call load() before search`。
    服务端 Milvus 在重启后同理。这里在检索前做一次幂等 load，并用本地集合去重，
    正常路径下开销约为 0（命中缓存直接返回）。
    """
    if not collection_name or collection_name in _loaded_collections:
        return
    try:
        if client.has_collection(collection_name):
            client.load_collection(collection_name)
        _loaded_collections.add(collection_name)
    except Exception as e:  # noqa: BLE001
        # load 失败不在这里吞掉：交给实际检索路径决定，避免掩盖真正的错误
        logger.warning(f"集合[{collection_name}] load 失败（将继续尝试检索）：{e}")


def escape_milvus_string(value: str) -> str:
    """
    Milvus数据库过滤表达式中字符串的安全转义函数（防止解析失败）
    作用：
        转义特殊字符（反斜杠、双引号），避免Milvus解析filter时报错
    参数：
        value: 需要转义的原始字符串
    返回：
        str: 转义后的安全字符串
    """
    # 转义反斜杠（\ → \\） 双引号（" → \"） 单引号（' → \'）
    value = value.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
    return value

# utils/milvus_utils.py
# 在这个文件中添加如下方法
# 创建了2个请求
def create_hybrid_search_requests(dense_vector, sparse_vector, dense_params=None, sparse_params=None, expr=None,
                                  limit=5):
    """
    构建Milvus混合搜索请求对象
    分别创建稠密/稀疏向量的搜索请求，用于后续混合搜索融合
    :param dense_vector: 文本生成的稠密向量
    :param sparse_vector: 文本生成的稀疏向量
    :param dense_params: 稠密向量搜索参数，默认使用余弦相似度
    :param sparse_params: 稀疏向量搜索参数，默认使用内积相似度
    :param expr: 搜索过滤表达式，用于精准筛选数据
    :param limit: 单向量搜索返回结果数量，默认5
    :return: 搜索请求列表，包含[dense_req, sparse_req]
    """
    # 稠密向量默认搜索参数：余弦相似度（COSINE），适配BGE-M3稠密向量
    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}
    # 稀疏向量默认搜索参数：内积（IP），适配BGE-M3稀疏向量
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    # 构建稠密向量搜索请求，关联Milvus的dense_vector字段 近似最近邻（ANN）检索请求的核心类
    dense_req = AnnSearchRequest(
        data=[dense_vector],# 数据（稠密向量）
        anns_field="dense_vector",# 要查的字段 名称
        param=dense_params,#相似度规则
        expr=expr,
        limit=limit # 召回条数
    )

    # 构建稀疏向量搜索请求，关联Milvus的sparse_vector字段
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=expr,
        limit=limit
    )
    # 配置参数
    return [dense_req, sparse_req]

# 开始 相应向量数据库的检索
# 两路检索  稠密向量检索：近似检索   稀疏向量检索：精确（关键字）检索
def hybrid_search(client, collection_name, reqs, ranker_weights=(0.5, 0.5), norm_score=False, limit=5,
                  output_fields=None, search_params=None):
    """
    执行Milvus稠密+稀疏向量混合搜索
    基于WeightedRanker实现双向量搜索结果加权融合，提升检索准确性
    :param client: MilvusClient实例
    :param collection_name: 集合名称
    :param reqs: 搜索请求列表，固定为[dense_req, sparse_req]
    :param ranker_weights: 加权融合权重，默认(0.5,0.5)，依次对应稠密/稀疏向量
    :param norm_score: 是否归一化评分后再融合，避免评分量级差异导致权重失效
    :param limit: 混合搜索最终返回结果数量，默认5
    :param output_fields: 需要返回的字段列表，默认返回item_name
    :param search_params: 搜索参数，如ef/topk等，默认None
    :return: 混合搜索结果列表，搜索失败返回None
    """
    try:
        # 检索前确保集合已 load（进程重启后集合为 released 状态，需先加载）
        ensure_collection_loaded(client, collection_name)
        # 初始化加权排名器：按权重融合稠密/稀疏向量的搜索结果
        # norm_score=True：先将两个向量评分归一化到0~1区间，再加权计算，避免一个得分特别大、另一个特别小导致权重失效。
        # 版本：V2.4
        # 创建排名器（加权排名）   RRF 倒排融合
        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)
        # 默认返回字段：文档标识字段
        if output_fields is None:
            output_fields = ["item_name"]

        # 执行混合搜索：融合稠密+稀疏向量结果，按权重重新排序
        # 开始执行  去向量数据库  里面去找 相似的片段 然后 融合这些结果（近似、关键字查询的结果） 重新排序
        res = client.hybrid_search(
            collection_name=collection_name,# 集合名称 表
            reqs=reqs,# 请求
            ranker=rerank,# 使用排序器
            limit=limit,# 返回条数限制
            output_fields=output_fields,# 输出的内容
            search_params=search_params# 搜索参数
        )

        logger.info(f"Milvus混合搜索完成，集合[{collection_name}]共检索到{len(res[0])}条结果")
        return res
    except Exception as e:
        msg = str(e)
        # 集合可能在本进程 load 之后又被 release（如导入流程重建集合）→ 重新 load 并重试一次
        if "released" in msg or "not loaded" in msg:
            try:
                _loaded_collections.discard(collection_name)
                ensure_collection_loaded(client, collection_name)
                rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)
                res = client.hybrid_search(
                    collection_name=collection_name,
                    reqs=reqs,
                    ranker=rerank,
                    limit=limit,
                    output_fields=output_fields or ["item_name"],
                    search_params=search_params
                )
                logger.info(f"重新 load 后混合搜索成功，集合[{collection_name}]共检索到{len(res[0])}条结果")
                return res
            except Exception as e2:  # noqa: BLE001
                logger.error(f"Milvus混合搜索重试仍失败，集合[{collection_name}]：{e2}", exc_info=True)
                return None
        logger.error(f"Milvus混合搜索执行失败，集合[{collection_name}]：{msg}", exc_info=True)
        return None
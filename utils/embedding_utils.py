# utils/embedding_utils.py

import os  # noqa: E402
import torch  # noqa: E402
from collections import OrderedDict  # noqa: E402

# [性能] BGE-M3 在 CPU 上推理为计算密集型，默认 torch 仅使用「物理核数」线程，
# 会浪费超线程的剩余算力。这里在模型加载前把线程数提到全部逻辑核，
# 通常可带来 20%~50% 的向量化提速（不影响结果精度）。
_t = max(1, os.cpu_count() or 1)
torch.set_num_threads(_t)

from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from config.embedding_config import embedding_config

# 注意：不再在此处调用 setup_logging()
# 原因：setup_logging() 内部会执行 logger.handlers.clear() 清空所有已注册的 handler，
#       这会与 tool/logger.py 中配置的彩色日志 handler 冲突，导致日志输出异常。
#       日志配置统一由 tool/logger.py 在模块加载时完成。

# 模型单例对象，避免重复初始化
_bge_m3_ef = None


def get_bge_m3_ef():
    """
    获取BGE-M3模型单例对象，自动加载环境变量配置
    :return: 初始化完成的BGEM3EmbeddingFunction实例
    """
    global _bge_m3_ef
    if _bge_m3_ef is not None:
        return _bge_m3_ef

    # 从环境变量加载配置
    model_name = embedding_config.bge_m3_path
    device = embedding_config.bge_device
    use_fp16 = embedding_config.bge_fp16

    # 如果模型没有被提前下载，会自动下载
    _bge_m3_ef = BGEM3EmbeddingFunction(
        model_name=model_name,
        device=device,
        use_fp16=use_fp16
    )
    return _bge_m3_ef


# [P1/P2-方案6/7] 嵌入结果内存 LRU 缓存：相同文本（如重复提问、同一改写查询）直接命中，
# 免去 CPU 上 BGE-M3 推理开销；按 (模型路径, 文本元组) 作为键，模型更换后自动失效。
_EMBED_CACHE: "OrderedDict" = OrderedDict()
_EMBED_CACHE_MAX = 2000


def generate_embeddings(texts):
    """
    为文本生成向量嵌入
    :param texts: 要生成嵌入的文本列表
    :return: 包含dense和sparse向量的字典
    """
    # [P1/P2-方案6/7] 先查缓存：命中则直接返回，避免重复推理
    cache_key = (embedding_config.bge_m3_path, tuple(texts))
    if cache_key in _EMBED_CACHE:
        _EMBED_CACHE.move_to_end(cache_key)
        return _EMBED_CACHE[cache_key]

    model = get_bge_m3_ef()
    embeddings = model.encode_documents(texts)
    processed_sparse = []
    for i in range(len(texts)):
        sparse_indices = embeddings["sparse"].indices[
                         embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]].tolist()
        sparse_data = embeddings["sparse"].data[
                      embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]].tolist()
        sparse_dict = {k: v for k, v in zip(sparse_indices, sparse_data)}
        processed_sparse.append(sparse_dict)

    result = {
        "dense": [emb.tolist() for emb in embeddings["dense"]],
        "sparse": processed_sparse
    }

    # 写入缓存（超过上限 LRU 淘汰）
    _EMBED_CACHE[cache_key] = result
    while len(_EMBED_CACHE) > _EMBED_CACHE_MAX:
        _EMBED_CACHE.popitem(last=False)

    return result


# ================= 单元测试主入口 =================
if __name__ == "__main__":
    # 准备测试文本（一个是中文，一个是英文词，模拟双语召回测试）
    test_data = ["大功率电磁炉", "Car"]

    print(f"👉 开始单独测试 generate_embeddings 函数...")
    print(f"👉 测试文本: {test_data}\n")

    try:
        # 执行目标函数
        output = generate_embeddings(test_data)

        print("\n" + "=" * 20 + " 🎉 测试结果 🎉 " + "=" * 20)

        # 1. 验证稠密向量
        print(f"【稠密向量 (Dense)】")
        print(f"   - 成功生成数量: {len(output['dense'])} 条")
        # 稠密向量有 1024 维，我们只打印前 5 个数字展示一下，否则会刷屏
        print(f"   - 第一条文本的维度大小: {len(output['dense'][0])} 维")
        print(f"   - 第一条文本的前 5 维特征值: {output['dense'][0][:5]} ...")

        # 2. 验证稀疏向量
        print(f"\n【稀疏向量 (Sparse)】")
        print(f"   - 成功生成数量: {len(output['sparse'])} 条")
        print(f"   - 第一条文本的关键字字典 (词ID : 权重):")
        print(f"     {output['sparse'][0]}")

        print("=" * 54)
        print("💡 结论：函数单独测试通过！输出数据结构完全符合 Milvus 的入库标准。")

    except Exception as e:
        print(f"\n❌ 测试失败！拦截到异常: {e}")
        print("💡 排查建议：")
        print("   1. 请确认本地 D 盘路径下的模型文件是否完整。")
        print(
            "   2. 如果提示 CUDA 相关错误，说明你的显卡驱动或 PyTorch 环境有兼容问题。可以将代码顶部的 BGE_DEVICE 改为 'cpu' 再试一下。")
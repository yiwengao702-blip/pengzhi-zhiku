# utils/minio_utils.py
"""
本地对象存储（替代 MinIO）

【为什么改】
原实现依赖 MinIO 服务（独立容器，常驻约 100MB 内存 + 额外端口）。
轻量变体用本地文件系统模拟 S3 语义，方法签名与 minio.Minio 客户端的子集保持一致，
上层 node_md_img 的调用（bucket_exists / make_bucket / set_bucket_policy /
fput_object / list_objects / remove_objects）无需改动即可运行。

【目录约定】
  存储根目录：KB_OBJECT_DIR（默认 ./data/objects）
  对象路径  ：<根目录>/<bucket>/<object_name>
  对外 URL  ：http://<MINIO_ENDPOINT>/<bucket>/<object_name>
              （导入服务已把 /<bucket> 挂载为静态目录，见 web/api/import_service.py）

【注意】
这是「单机可用」的实现：没有多副本、没有签名鉴权。
仅作轻量部署的等价替代，生产环境仍建议接真实对象存储。
"""

import logging
import os
import shutil
from types import SimpleNamespace
from typing import Iterable, List

from config.minio_config import minio_config

# 注意：不再在此处调用 setup_logging()
# 原因：setup_logging() 内部会执行 logger.handlers.clear() 清空所有已注册的 handler，
#       这会与 tool/logger.py 中配置的彩色日志 handler 冲突，导致日志输出异常。
#       日志配置统一由 tool/logger.py 在模块加载时完成。

# 存储根目录：可用 KB_OBJECT_DIR 覆盖，默认项目内 ./data/objects
OBJECT_ROOT = os.path.abspath(
    os.getenv("KB_OBJECT_DIR", os.path.join(os.getenv("KB_DATA_DIR", "./data"), "objects"))
)


class LocalObjectStore:
    """
    极简本地对象存储：以目录结构承载 bucket / object，接口对齐 minio.Minio 的常用子集。
    """

    def __init__(self, root_dir: str, default_bucket: str):
        self.root_dir = os.path.abspath(root_dir)
        self.default_bucket = default_bucket
        os.makedirs(self.root_dir, exist_ok=True)

    # ---------- bucket 管理 ----------
    def _bucket_dir(self, bucket: str) -> str:
        return os.path.join(self.root_dir, bucket)

    def bucket_exists(self, bucket_name: str = None) -> bool:
        return os.path.isdir(self._bucket_dir(bucket_name or self.default_bucket))

    def make_bucket(self, bucket_name: str) -> None:
        os.makedirs(self._bucket_dir(bucket_name), exist_ok=True)

    def set_bucket_policy(self, bucket_name: str, policy: str) -> None:
        """本地实现无鉴权概念，保留方法以兼容原有调用链（幂等空操作）。"""
        return None

    # ---------- 对象读写 ----------
    def fput_object(self, bucket_name: str, object_name: str,
                    file_path: str, content_type: str = None):
        """把本地文件复制进对象目录（等价于 minio 的 fput_object）。"""
        target = os.path.join(self._bucket_dir(bucket_name), object_name.replace("/", os.sep))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(file_path, target)
        return SimpleNamespace(object_name=object_name, bucket_name=bucket_name)

    def put_object(self, bucket_name: str, object_name: str, data, length: int,
                   content_type: str = None):
        """流式写入（对齐 minio 的 put_object，供未来扩展使用）。"""
        target = os.path.join(self._bucket_dir(bucket_name), object_name.replace("/", os.sep))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as f:
            f.write(data.read(length) if hasattr(data, "read") else data)
        return SimpleNamespace(object_name=object_name, bucket_name=bucket_name)

    def list_objects(self, bucket_name: str, prefix: str = "", recursive: bool = False):
        """
        列举对象，返回 SimpleNamespace(object_name=...) 序列（与 minio 的返回项兼容）。
        """
        base = self._bucket_dir(bucket_name)
        prefix = (prefix or "").strip("/")
        if not os.path.isdir(base):
            return []
        items: List[SimpleNamespace] = []
        if recursive:
            for dirpath, _dirnames, filenames in os.walk(base):
                for name in filenames:
                    full = os.path.join(dirpath, name)
                    rel = os.path.relpath(full, base).replace(os.sep, "/")
                    if (not prefix) or rel.startswith(prefix):
                        items.append(SimpleNamespace(object_name=rel))
        else:
            search_dir = os.path.join(base, prefix.replace("/", os.sep)) if prefix else base
            if os.path.isdir(search_dir):
                for name in os.listdir(search_dir):
                    full = os.path.join(search_dir, name)
                    if os.path.isfile(full):
                        rel = os.path.relpath(full, base).replace(os.sep, "/")
                        items.append(SimpleNamespace(object_name=rel))
        return items

    def remove_objects(self, bucket_name: str, delete_object_list: Iterable) -> List[str]:
        """
        批量删除对象（对齐 minio 的 remove_objects）。
        delete_object_list 支持两种元素：字符串对象名，或带 .object_name 属性的对象。
        返回错误信息列表（全部成功则为空列表）。
        """
        errors: List[str] = []
        base = self._bucket_dir(bucket_name)
        for item in delete_object_list:
            name = item if isinstance(item, str) else getattr(item, "object_name", "")
            target = os.path.join(base, str(name).replace("/", os.sep))
            try:
                if os.path.isfile(target):
                    os.remove(target)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{name}: {e}")
        return errors


# 模块级单例：与原实现保持一致，导入即完成「bucket 检查 + 创建」
try:
    minio_client = LocalObjectStore(
        root_dir=OBJECT_ROOT,
        default_bucket=minio_config.bucket_name or "knowledge-base",
    )
    if not minio_client.bucket_exists(minio_config.bucket_name):
        minio_client.make_bucket(minio_config.bucket_name)
    minio_client.set_bucket_policy(minio_config.bucket_name, "{}")
    logging.info(f"本地对象存储就绪：{OBJECT_ROOT}/{minio_config.bucket_name}")
except Exception as e:  # noqa: BLE001
    logging.error(f"本地对象存储初始化失败，错误原因：{e}")


def get_minio_client() -> LocalObjectStore:
    return minio_client


if __name__ == '__main__':
    print("root =", OBJECT_ROOT)
    print("bucket =", minio_config.bucket_name, "| exists =", minio_client.bucket_exists(minio_config.bucket_name))

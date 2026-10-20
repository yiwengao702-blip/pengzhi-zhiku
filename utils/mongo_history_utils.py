# utils/mongo_history_utils.py
"""
轻量历史会话存储（SQLite 实现）

【为什么改】
原实现依赖 MongoDB（pymongo + 独立 mongod 进程，常驻约 200MB 内存）。
轻量变体改为单文件 SQLite，零额外进程、零额外内存，接口与文档字段完全保持不变，
上层（node_answer_output / node_item_name_confirm / query_service 等）无需改动。

【兼容性保证】
  * 对外函数签名与原 mongo 版逐字一致：save_chat_message / get_recent_messages /
    clear_history / update_message_item_names / update_latest_assistant_meta /
    get_history_mongo_tool。
  * 记录主键仍返回 24 位十六进制字符串（与原 ObjectId 字符串形态一致），
    前端把它当作不透明 ID 使用，不会因存储层更换而失效。
  * get_recent_messages 的排序与 limit 语义与原实现一致（ts 升序取前 N 条）。

【存储位置】
  KB_HISTORY_DB 环境变量可覆盖，默认 <KB_DATA_DIR>/kb_history.db（KB_DATA_DIR 默认 ./data）。
"""

import json
import os
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List

from tool.logger import logger

_DEFAULT_DIR = os.getenv("KB_DATA_DIR", "./data")
DB_PATH = os.getenv("KB_HISTORY_DB") or os.path.join(_DEFAULT_DIR, "kb_history.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_message (
    id              TEXT PRIMARY KEY,           -- 24 位十六进制，形态对齐 ObjectId
    session_id      TEXT NOT NULL,
    role            TEXT NOT NULL,
    text            TEXT,
    rewritten_query TEXT DEFAULT '',
    item_names      TEXT,                       -- JSON 数组
    image_urls      TEXT,                       -- JSON 数组
    ts              REAL NOT NULL,
    meta            TEXT                        -- JSON 对象（检索遥测指标）
);
CREATE INDEX IF NOT EXISTS idx_session_ts ON chat_message (session_id, ts DESC);
"""


def _new_id() -> str:
    """生成 24 位十六进制主键，与 MongoDB ObjectId 的字符串形态保持一致。"""
    return os.urandom(12).hex()


def _loads(raw, default):
    """JSON 字段反序列化，脏数据不抛异常。"""
    if raw is None or raw == "":
        return default
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return default


def _dumps(val):
    """JSON 字段序列化，None 原样存 NULL。"""
    if val is None:
        return None
    return json.dumps(val, ensure_ascii=False)


class HistoryMongoTool:
    """
    历史对话记录读写工具类（SQLite 实现）
    类名保留 HistoryMongoTool，避免上层 import 失效。
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        # 数据库文件所在目录可能不存在（首次启动），先建目录
        parent = os.path.dirname(os.path.abspath(self.db_path))
        os.makedirs(parent, exist_ok=True)
        # SQLite 连接跨线程复用需要 check_same_thread=False，读写用锁串行化
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL 模式：读写不互相阻塞，适合 FastAPI 多 worker 线程场景
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info(f"Successfully opened history store (SQLite): {self.db_path}")

    # ---------- 底层读写 ----------
    def _execute(self, sql: str, params: tuple = ()):
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _query(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        """把表行还原成原 MongoDB 文档形状（_id + 各业务字段）。"""
        return {
            "_id": row["id"],
            "session_id": row["session_id"],
            "role": row["role"],
            "text": row["text"],
            "rewritten_query": row["rewritten_query"] or "",
            "item_names": _loads(row["item_names"], None),
            "image_urls": _loads(row["image_urls"], None),
            "ts": row["ts"],
            **({"meta": _loads(row["meta"], None)} if row["meta"] else {}),
        }

    # ---------- 与原 pymongo 版同名的工具方法 ----------
    def upsert(self, document: Dict[str, Any], message_id: str = None) -> str:
        """新增或更新一条记录，返回记录主键字符串。"""
        rec_id = message_id or _new_id()
        if message_id:
            cur = self._execute(
                """UPDATE chat_message
                      SET session_id=?, role=?, text=?, rewritten_query=?,
                          item_names=?, image_urls=?, ts=?, meta=?
                    WHERE id=?""",
                (
                    document.get("session_id"), document.get("role"), document.get("text"),
                    document.get("rewritten_query") or "",
                    _dumps(document.get("item_names")), _dumps(document.get("image_urls")),
                    document.get("ts"), _dumps(document.get("meta")), message_id,
                ),
            )
            # 传入的 message_id 在库中不存在（如前端换了浏览器）→ 退化为插入，避免消息丢失
            if cur.rowcount == 0:
                self._insert(rec_id, document)
            return message_id
        self._insert(rec_id, document)
        return rec_id

    def _insert(self, rec_id: str, document: Dict[str, Any]) -> None:
        self._execute(
            """INSERT INTO chat_message
                   (id, session_id, role, text, rewritten_query, item_names, image_urls, ts, meta)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                rec_id, document.get("session_id"), document.get("role"), document.get("text"),
                document.get("rewritten_query") or "",
                _dumps(document.get("item_names")), _dumps(document.get("image_urls")),
                document.get("ts"), _dumps(document.get("meta")),
            ),
        )

    def find_recent(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """按 ts 升序取该会话前 limit 条（与原实现语义一致）。"""
        rows = self._query(
            # rowid 作为同毫秒写入的稳定次序，避免 user/assistant 顺序在同步写入时被翻转
            "SELECT * FROM chat_message WHERE session_id=? ORDER BY ts ASC, rowid ASC LIMIT ?",
            (session_id, int(limit)),
        )
        return [self._row_to_dict(r) for r in rows]

    def update_item_names(self, ids: List[str], item_names: List[str]) -> int:
        if not ids:
            return 0
        placeholders = ",".join("?" * len(ids))
        cur = self._execute(
            f"UPDATE chat_message SET item_names=? WHERE id IN ({placeholders})",
            (_dumps(item_names), *ids),
        )
        return cur.rowcount

    def update_latest_assistant_meta(self, session_id: str, meta: Dict[str, Any]) -> bool:
        rows = self._query(
            "SELECT id FROM chat_message WHERE session_id=? AND role='assistant' "
            "ORDER BY ts DESC, rowid DESC LIMIT 1",
            (session_id,),
        )
        if not rows:
            return False
        self._execute(
            "UPDATE chat_message SET meta=? WHERE id=?",
            (_dumps(meta), rows[0]["id"]),
        )
        return True

    def delete_session(self, session_id: str) -> int:
        cur = self._execute("DELETE FROM chat_message WHERE session_id=?", (session_id,))
        return cur.rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# 定义全局变量：存储单例实例（模块加载即建立连接，提升首问响应速度）
try:
    _history_mongo_tool: HistoryMongoTool = HistoryMongoTool()
except Exception as _e:  # noqa: BLE001
    # 存储不可用不应导致整个服务起不来：降级为 None，调用时按空结果处理
    logger.error(f"历史存储初始化失败（将降级为空历史）：{_e}")
    _history_mongo_tool = None


def get_history_mongo_tool() -> HistoryMongoTool:
    """
    获取单例实例（懒加载模式）：全局实例为空时创建，保证整个程序只有一条连接。
    """
    global _history_mongo_tool
    if _history_mongo_tool is None:
        _history_mongo_tool = HistoryMongoTool()
    return _history_mongo_tool


def save_chat_message(
        session_id: str, role: str, text: str,
        rewritten_query: str = "",
        item_names: List[str] = None,
        image_urls: List[str] = None,
        message_id: str = None,
        meta: dict = None
) -> str:
    """
    写入/更新单条会话记录
    支持两种模式：无 message_id 时新增记录，有 message_id 时更新已有记录
    :return: 插入/更新的记录唯一标识
    """
    ts = datetime.now().timestamp()
    document = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query or "",
        "item_names": item_names,
        "image_urls": image_urls,
        "ts": ts,
    }
    # 检索遥测指标：有则随记录落库（历史恢复时前端据此重绘指标行/侧栏）
    if meta is not None:
        document["meta"] = meta

    try:
        return get_history_mongo_tool().upsert(document, message_id)
    except Exception as e:  # noqa: BLE001
        logger.error(f"写入历史会话失败: {e}")
        return ""


def update_message_item_names(ids: List[str], item_names: List[str]) -> int:
    """
    批量更新历史会话记录的关联商品名称
    :return: 实际更新的记录数量，更新失败返回 0
    """
    try:
        n = get_history_mongo_tool().update_item_names(ids, item_names)
        logger.info(f"Updated {n} records to item_names: {item_names}")
        return n
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error updating history item_names: {e}")
        return 0


def update_latest_assistant_meta(session_id: str, meta: dict) -> bool:
    """
    把本轮检索遥测指标回写进「该会话最近一条助手消息」，使其随历史持久化。
    :return: 是否成功补写
    """
    if not meta:
        return False
    try:
        return get_history_mongo_tool().update_latest_assistant_meta(session_id, meta)
    except Exception as e:  # noqa: BLE001
        logger.error(f"回写检索指标到历史失败: {e}")
        return False


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    查询指定会话的对话记录，返回原始字典格式（结果按时间正序排列，可直接喂给 LLM）
    :return: 对话记录列表，查询失败返回空列表
    """
    try:
        return get_history_mongo_tool().find_recent(session_id, limit)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error getting recent messages: {e}")
        return []


def clear_history(session_id: str) -> int:
    """
    清空指定会话的所有历史对话记录
    :return: 实际删除的记录数量，删除失败返回 0
    """
    try:
        n = get_history_mongo_tool().delete_session(session_id)
        logger.info(f"Deleted {n} messages for session {session_id}")
        return n
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error clearing history for session {session_id}: {e}")
        return 0


if __name__ == "__main__":
    # 冒烟测试：写 → 读 → 改 → 删 全链路
    sid = "test_session_sqlite"
    clear_history(sid)
    uid = save_chat_message(sid, "user", "你好，有烫金机吗？", "请问有烫金机吗")
    aid = save_chat_message(sid, "assistant", "有的，请问需要哪个型号？", meta={"hit": True})
    print("user_id =", uid, "| assistant_id =", aid)
    print("recent =", get_recent_messages(sid, 5))
    print("update_meta =", update_latest_assistant_meta(sid, {"hit": False, "score": 0.8}))
    print("update_names =", update_message_item_names([uid], ["HAK180 烫金机"]))
    print("after  =", get_recent_messages(sid, 5))
    print("cleared =", clear_history(sid))

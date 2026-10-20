# D:/knowledge_base/main.py
"""
鹏至智库 · 一键启动器

同时提供两个 FastAPI 服务：
  - 导入服务  web.api.import_service:app  -> http://<host>:8100
  - 问答服务  web.api.query_service:app   -> http://<host>:8101

【双模式启动】
  A. 单进程双端口（默认，Milvus Lite 场景必需）
     Milvus Lite 用「文件锁」独占 ./data/kb_milvus.db，一个进程拿到锁后，另一个进程
     再去开同一个 .db 会抛 DataDirLockedError。因此导入与问答必须跑在同一个 Python
     进程里（共享 utils.milvus_utils 的客户端单例），用 asyncio 并发托管两个 uvicorn。
  B. 子进程双服务（Milvus 服务端 / standalone 场景）
     MILVUS_URL 是 http(s):// 或 tcp:// 时无文件锁问题，仍按原方式起两个子进程，
     互不干扰、故障隔离更好。可用 KB_SPLIT_PROCESS=1 强制启用。

端口由 .env 的 KB_PORT_IMPORT / KB_PORT_QUERY 控制（默认 8100/8101；
8000/8001 已被「清衡舆情分析平台」占用，故本项目后移 100）。

用法：
  python main.py          # 启动两个服务
  Ctrl+C                  # 同时优雅停止
"""

import asyncio
import os
import signal
import subprocess
import sys
import time

from dotenv import load_dotenv

# 加载 .env，使 KB_PORT_IMPORT / KB_PORT_QUERY 等端口配置生效
load_dotenv()

# 项目根目录（保证 config / processor / utils / tool / web 可被导入）
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PORT_IMPORT = int(os.getenv("KB_PORT_IMPORT", 8100))
PORT_QUERY = int(os.getenv("KB_PORT_QUERY", 8101))
# 监听地址：本机开发用 127.0.0.1；容器 / 云主机对外提供服务时设 KB_HOST=0.0.0.0
HOST = os.getenv("KB_HOST", "127.0.0.1")

# Milvus Lite（本地 .db 文件）判定：非 http(s)/tcp 形式即本地文件
_MILVUS_URL = (os.getenv("MILVUS_URL") or "").strip()
_IS_LITE = not _MILVUS_URL.startswith(("http://", "https://", "tcp://"))
# 是否拆分为两个子进程（仅在非 Lite 环境下有意义）
SPLIT_PROCESS = os.getenv("KB_SPLIT_PROCESS", "0" if _IS_LITE else "1") in ("1", "true", "True")


def _banner() -> None:
    """启动成功后打印访问地址，避免用户走到错误的演示地址。"""
    print("\n" + "=" * 64)
    print("  鹏至智库已启动，请在浏览器打开以下地址（务必用这两个）：")
    print(f"    平台门户（着陆页）：  http://{HOST}:{PORT_IMPORT}/")
    print(f"    导入页（上传文档）：  http://{HOST}:{PORT_IMPORT}/import.html")
    print(f"    问答页（检索对话）：  http://{HOST}:{PORT_QUERY}/chat.html")
    print(f"    概览页：              http://{HOST}:{PORT_QUERY}/overview.html")
    print(f"  模式：{'单进程双端口（Milvus Lite 文件锁独占）' if not SPLIT_PROCESS else '双子进程'}")
    print("  提示：若页面打不开或显示「未能连接后端」，请确认本窗口仍在运行；")
    print("        PDF 导入需有效的 MINERU_API_TOKEN（见 .env），Markdown 导入不受限。")
    print("=" * 64)
    print(">>> 按 Ctrl+C 停止全部服务。\n")


# ==================== 模式 A：单进程双端口 ====================
async def _serve_both() -> None:
    """
    在同一个事件循环里并发托管两个 uvicorn 服务。
    同进程意味着两个服务共享 utils.milvus_utils 的 MilvusClient 单例，
    从而规避 Milvus Lite 的「一个 .db 只能被一个进程打开」限制。
    """
    import uvicorn

    cfg_import = uvicorn.Config(
        "web.api.import_service:app", host=HOST, port=PORT_IMPORT, log_level="info"
    )
    cfg_query = uvicorn.Config(
        "web.api.query_service:app", host=HOST, port=PORT_QUERY, log_level="info"
    )
    srv_import = uvicorn.Server(cfg_import)
    srv_query = uvicorn.Server(cfg_query)

    print(f">>> 启动导入服务 (端口 {PORT_IMPORT}) ...")
    print(f">>> 启动问答服务 (端口 {PORT_QUERY}) ...")
    _banner()

    # gather 并发运行；任一服务异常退出则整体结束
    await asyncio.gather(srv_import.serve(), srv_query.serve())


# ==================== 模式 B：双子进程 ====================
def _start_service(module_app: str, port: int) -> subprocess.Popen:
    """用当前 Python 解释器以 uvicorn 模块方式启动一个服务子进程。"""
    cmd = [
        sys.executable, "-m", "uvicorn",
        module_app,
        "--host", HOST,
        "--port", str(port),
    ]
    env = dict(os.environ)
    # 将项目根目录加入 PYTHONPATH，确保本地包可被导入
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(cmd, cwd=ROOT, env=env)


def _stop_service(proc: subprocess.Popen) -> None:
    """优雅停止子进程：先 SIGINT/SIGTERM，超时则强制结束。"""
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            proc.terminate()  # Windows: TerminateProcess
        else:
            proc.send_signal(signal.SIGINT)
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


def _run_split() -> None:
    procs = []
    try:
        print(f">>> 启动导入服务 (端口 {PORT_IMPORT}) ...")
        procs.append(_start_service("web.api.import_service:app", PORT_IMPORT))
        print(f">>> 启动问答服务 (端口 {PORT_QUERY}) ...")
        procs.append(_start_service("web.api.query_service:app", PORT_QUERY))
        _banner()

        # 轮询：任一子进程异常退出则整体退出
        while True:
            for p in procs:
                if p.poll() is not None:
                    print(f"!!! 检测到服务异常退出（returncode={p.returncode}），准备停止。")
                    raise KeyboardInterrupt
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n>>> 收到停止信号，正在关闭服务 ...")
    finally:
        for p in procs:
            _stop_service(p)
        print(">>> 已停止全部服务。")


def main() -> None:
    if SPLIT_PROCESS:
        _run_split()
        return
    try:
        asyncio.run(_serve_both())
    except KeyboardInterrupt:
        print("\n>>> 收到停止信号，正在关闭服务 ...")
    print(">>> 已停止全部服务。")


if __name__ == "__main__":
    main()

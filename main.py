# D:/knowledge_base/main.py
"""
鹏至智库 · 一键启动器（参照 web/api 模板编写）

同时拉起两个 FastAPI 服务：
  - 导入服务  web.api.import_service:app  -> http://127.0.0.1:8100
  - 问答服务  web.api.query_service:app   -> http://127.0.0.1:8101

说明：8000/8001 已被「清衡舆情分析平台」占用，本项目默认后移 100 用 8100/8101；
端口由 .env 的 KB_PORT_IMPORT / KB_PORT_QUERY 控制，启动后会按实际端口打印访问地址。

用法：
  python main.py          # 启动两个服务
  Ctrl+C                  # 同时优雅停止

说明：
  - 依赖见 pyproject.toml / requirements.txt / DEPENDENCIES.md
  - 也可单独运行：python web/api/import_service.py 或 query_service.py
"""
import os
import sys
import time
import signal
import subprocess
from dotenv import load_dotenv

# 加载 .env，使 KB_PORT_IMPORT / KB_PORT_QUERY 等端口配置生效
load_dotenv()

# 项目根目录（保证 config / processor / utils / tool / web 可被导入）
ROOT = os.path.dirname(os.path.abspath(__file__))

# 服务端口：默认 8100（导入）/ 8101（问答），与 .env 的 KB_PORT_IMPORT / KB_PORT_QUERY 一致。
# 8000/8001 已被「清衡舆情分析平台」占用，本项目统一后移 100。
PORT_IMPORT = int(os.getenv("KB_PORT_IMPORT", 8100))
PORT_QUERY = int(os.getenv("KB_PORT_QUERY", 8101))


def _start_service(module_app: str, port: int) -> subprocess.Popen:
    """用当前 Python 解释器以 uvicorn 模块方式启动一个服务子进程。"""
    cmd = [
        sys.executable, "-m", "uvicorn",
        module_app,
        "--host", "127.0.0.1",
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


def main() -> None:
    procs = []
    try:
        print(f">>> 启动导入服务 (端口 {PORT_IMPORT}) ...")
        procs.append(_start_service("web.api.import_service:app", PORT_IMPORT))
        print(f">>> 启动问答服务 (端口 {PORT_QUERY}) ...")
        procs.append(_start_service("web.api.query_service:app", PORT_QUERY))

        # 启动成功：大字打印正确的访问地址，避免用户再走到错误的静态演示地址
        print("\n" + "=" * 64)
        print("  鹏至智库已启动，请在浏览器打开以下地址（务必用这两个）：")
        print(f"    平台门户（着陆页）：  http://127.0.0.1:{PORT_IMPORT}/")
        print(f"    导入页（上传文档）：  http://127.0.0.1:{PORT_IMPORT}/import.html")
        print(f"    问答页（检索对话）：  http://127.0.0.1:{PORT_QUERY}/chat.html")
        print(f"    概览页：              http://127.0.0.1:{PORT_QUERY}/overview.html")
        print("  提示：若页面打不开或显示「未能连接后端」，请确认本窗口仍在运行；")
        print("        PDF 导入需有效的 MINERU_API_TOKEN（见 .env），Markdown 导入不受限。")
        print("=" * 64)
        print(">>> 按 Ctrl+C 停止全部服务。\n")

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


if __name__ == "__main__":
    main()

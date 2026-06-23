from typing import TypedDict,List
import copy

class ImportGraphState(TypedDict):
    task_id:str
    is_md_read_enabled: bool
    is_pdf_read_enabled: bool
    import_file_path: str
    file_dir: str
    pdf_path: str
    md_path: str
    file_title: str
    item_name: str
    md_content: str
    chunks:List

GRAPH_DEFAULT_STATE: ImportGraphState = {

    "task_id": "",

    "is_pdf_read_enabled": False,

    "is_md_read_enabled": False,

    "file_dir": "",

    "import_file_path": "",

    "pdf_path": "",

    "md_path": "",

    "file_title": "",

    "md_content": "",

    "chunks": [],

    "item_name": "",

}

def get_default_state() -> ImportGraphState:
    """
    获取默认状态副本
    :return: 状态副本（避免全局污染）
    """
    return copy.deepcopy(GRAPH_DEFAULT_STATE)
class MyState(TypedDict, total=False):
    """
    total=False 表示所有字段都是可选的
    """
    name: str           # 字符串类型
    count: int          # 整数类型
    items: List[str]    # 字符串列表

if __name__ == "__main__":
    print("--- 1. 初始化默认状态 ---")
    # 通过工厂函数获取干净的默认初始状态
    current_state = get_default_state()
    print(f"初始任务ID: '{current_state['task_id']}'")
    print(f"初始切片列表: {current_state['chunks']}")

    print("\n--- 2. 模拟节点 1：入口节点 (node_entry) 更新状态 ---")
    # 模拟用户上传了一个 PDF 文件
    input_file = "BOSS_Manual_2026.pdf"

    # 节点处理逻辑：解析路径、设置控制标志
    current_state["import_file_path"] = input_file
    current_state["file_title"] = input_file.rsplit(".", 1)[0]
    current_state["is_pdf_read_enabled"] = True  # 激活 PDF 读取开关

    print(f"[node_entry 完工] 当前状态:")
    print(f"  - 目标文件: {current_state['import_file_path']}")
    print(f"  - 文件标题: {current_state['file_title']}")
    print(f"  - PDF控制标志: {current_state['is_pdf_read_enabled']}")

    print("\n--- 3. 模拟节点 2：切片节点 (node_document_split) 追加数据 ---")
    # 模拟切片组件切出了 2 段内容
    mock_chunks = [
        {"chunk_id": 1, "text": "智库系统简介..."},
        {"chunk_id": 2, "text": "安全合规说明..."}
    ]
    # 更新状态中的列表
    current_state["chunks"] = mock_chunks

    print(f"[node_document_split 完工] 当前状态:")
    print(f"  - 已切片数量: {len(current_state['chunks'])} 段")

    print("\n--- 4. 验证深拷贝 (copy.deepcopy) 的安全性 ---")
    # 再次获取一个全新的默认状态，验证它有没有被上面几步的操作污染
    new_clean_state = get_default_state()
    print(f"全新实例的任务ID: '{new_clean_state['task_id']}' (依然为空，未被污染)")
    print(f"全新实例的切片列表: {new_clean_state['chunks']} (依然为空，未被污染)")
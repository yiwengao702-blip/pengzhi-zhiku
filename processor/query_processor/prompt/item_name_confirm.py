# processor/query_processor/prompt/item_name_confirm.py
# 系统提示词（人设）
ITEM_NAME_EXTRACT_SYSTEM_PROMPT = "你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"
# 用户提示词
ITEM_NAME_EXTRACT_TEMPLATE = """
历史会话：
{history_text}

当前用户问题：
{query}

请根据历史会话和当前问题，提取用户正在询问的商品名称（item_names）。
1. 如果用户明确提到了商品名称，请提取出来。可能有一个或多个，但不能重复。
2. 如果用户使用了代词（如"这个"、"它"），请结合历史会话指代消解，确定商品名称。
3. 如果无法确定商品名称，item_names 返回空列表。
4. 请重新改写用户的问题（rewritten_query），使其成为包含商品名称的独立完整问题。
5. 判断该问题是否需要"确认具体型号/商品"才能准确回答：
   - 若问题针对某个具体产品/型号（如"XX 怎么用/参数/故障"），need_item_confirm 为 true；
   - 若问题是通用知识或通用操作（如"万用表怎么测电压""短路怎么查"），与具体型号无关，need_item_confirm 为 false。

请直接返回JSON格式结果，格式如下：
{{
    "item_names": ["商品A", "商品B"],
    "rewritten_query": "关于商品A和商品B，...",
    "need_item_confirm": true
}}
"""
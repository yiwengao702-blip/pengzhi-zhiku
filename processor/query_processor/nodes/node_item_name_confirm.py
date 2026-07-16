# processor/query_processor/nodes/node_item_name_confirm.py

import json
import re
from typing import Tuple, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage

from config.lm_config import lm_config
from config.milvus_config import milvus_config
from config.retrieval_config import retrieval_config  # [P0] 检索阈值统一配置
from processor.query_processor.base import NodeBase
from processor.query_processor.prompt.item_name_confirm import ITEM_NAME_EXTRACT_TEMPLATE, \
    ITEM_NAME_EXTRACT_SYSTEM_PROMPT
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from utils.mongo_history_utils import save_chat_message, get_recent_messages, update_message_item_names
from utils.task_utils import add_done_task
# [融合] 文本 LLM 收敛到共享底座网关（R5），商品名抽取属于纯文本路径，不再直连 ChatOpenAI
from utils.llm_utils import get_llm_client


class NodeItemNameConfirm(NodeBase):
    """
    节点功能：确认用户问题中的核心商品名称。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_item_name_confirm"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 步骤0（前置）：先校验 session_id，再对原始问题做边界守卫。
        # [BUG修复] 原实现先调用 _step_1_validate_param（空/缺失 original_query 会直接 raise），
        # 导致下方输入守卫永远来不及拦截空/垃圾输入而抛异常，破坏边界健壮性。
        # 现把守卫前置：空、超长、纯标点、广告等边界输入由守卫直接给友好提示，不进入检索链路与参数校验。
        session_id = state.get("session_id")
        if not session_id:
            raise ValueError("核心参数session_id缺失")
        guard_msg = self._step_0_guard_query(state)
        if guard_msg:
            logger.info(f"步骤0：输入守卫拦截，提示={guard_msg}")
            # 仍记录用户消息与助手提示，保证历史连贯
            message_id = save_chat_message(session_id, "user", state.get("original_query") or "")
            state["answer"] = guard_msg
            state["item_names"] = []
            self._step_8_write_history(state, session_id, state.get("original_query") or "", message_id)
            add_done_task(session_id, self.name, state.get("is_stream"))
            return state

        # 步骤1：校验参数（守卫已确保 original_query 非空，此处不会再因空查询抛异常）
        session_id, original_query = self._step_1_validate_param(state)
        logger.info(f"步骤1：参数校验通过")

        # 步骤2：获取历史记录  校验完了 你有原始问题 和会话ID  我先MongoDB 找历史记录
        history = get_recent_messages(session_id)  # 根据你提供的会话ID 去Mongo找历史记录
        logger.info(f"步骤2：获取到 {len(history)} 条历史消息")
        # 更新状态
        state["history"] = history  # 很重要

        # 步骤3：用户初始消息保存   "BrotherHAK180烫金机如何调节温度？"
        # 保存用户这一次 问题
        message_id = save_chat_message(session_id, "user", original_query)
        logger.info(f"步骤3：用户消息已初始保存, ID: {message_id}")

        # 步骤4：提取信息  LLM 识别(商品名称)以后 消息历史
        extract_res = self._step_4_extract_info(original_query, history)
        item_names = extract_res.get("item_names")  # 第四步 识别商品名称
        rewritten_query = extract_res.get("rewritten_query", original_query)
        # [P1-方案5] 是否需要确认具体型号：通用问题(need_item_confirm=False)无需对齐型号
        need_item_confirm = extract_res.get("need_item_confirm", True)
        # 更新状态 状态
        state["rewritten_query"] = rewritten_query

        # 5. & 6. 型号对齐（仅在「需要确认型号」且「提取到型号」时执行）
        # [P1-方案5] 通用问题短路：need_item_confirm=False 表示问题与具体型号无关
        # （如"短路怎么查""万用表怎么测电压""U盘怎么格式化"），直接清空 item_names 走全库检索，
        # 跳过无谓的 kb_item_names 向量对齐查询，降低延迟与误拒率。
        align_result = {}
        if need_item_confirm and len(item_names) > 0:
            state["item_names"] = item_names
            query_results = self._step_5_vectorize_and_query(item_names)
            # logger.info(query_results)
            align_result = self._step_6_align_item_names(query_results)
        else:
            if not need_item_confirm:
                logger.info("Node: 通用问题(need_item_confirm=False)，短路跳过型号对齐，直接全库检索")
            else:
                logger.info("Node: 未提取到商品名，跳过向量检索")
            # 空列表 ⇒ 下游 node_search_embedding / hyde 走全库检索
            state["item_names"] = []

        # 7. 检查确认状态
        state = self._step_7_check_confirmation(state, align_result, history)

        # 8. 写入最终历史
        self._step_8_write_history(state, session_id, rewritten_query, message_id)
        add_done_task(state.get("session_id"), self.name, state.get("is_stream"))
        return state

    def _step_7_check_confirmation(self, state, align_result, history):
        """
        7 检查step6对齐后的商品名状态，分3种分支更新state，并同步更新历史消息的商品名关联
        :param state: 字典 - 原始会话状态，包含session_id/original_query等核心字段
        :param align_result: 字典 - step6的对齐结果
        :param history: 列表[字典] - 近期会话历史
        :return: 字典 - 更新后的会话状态，包含item_names/answer
        """
        # 从对齐结果中提取确认商品名列表，无则空列表
        confirmed = align_result.get("confirmed_item_names", [])
        # 从对齐结果中提取候选商品名列表，无则空列表
        options = align_result.get("options", [])

        # 分支A：有确认的商品名（高置信度，无需用户确认）
        if confirmed:
            # 收集历史消息中未关联商品名的消息ID（需批量更新关联）
            ids_to_update = []
            for msg in history:# 找到没有商品名称的  对应ID
                if not msg.get("item_names"):  # 仅更新item_names为空的历史消息
                    mid = msg.get("_id")  # 提取消息唯一ID
                    if mid:
                        ids_to_update.append(str(mid))  # 转为字符串，避免ID格式问题

            # 若存在需更新的消息ID，批量更新历史消息的商品名关联   （主体、关键词）
            if ids_to_update:
                update_message_item_names(ids_to_update, confirmed)

            # 更新会话状态：设置确认商品名、改写后的查询
            state["item_names"] = confirmed
            # [BUG修复] 使用 None 替代空字符串 ""，语义更清晰
            # None 明确表示"无需预置答案，走正常检索流程"，避免与空字符串的歧义性
            # 同时防止 "answer" in state 返回 True 导致下游误判
            state["answer"] = None
            # 返回更新后的状态
            return state

        # 分支B：无确认商品名，但有候选商品名（中置信度，需用户明确）
        if options:
            # 候选商品名拼接为字符串（取前3个，避免过长），格式："商品1、商品2、商品3"
            options_str = "、".join(options[:3])
            # 构造向用户确认的提示语
            answer = f"您是想问以下哪个产品：{options_str}？请明确一下型号。"
            # 更新会话状态：设置确认提示语、清空商品名列表
            state["answer"] = answer
            state["item_names"] = []
            return state

        # 分支C：无确认商品名，且无候选商品名
        # [P0-方案1] 不再直接拒答：清空 item_names（下游走全库检索），answer 置 None 使路由
        # 进入 node_multi_search；最终是否"未找到"由 node_answer_output 在检索仍为空时统一兜底。
        state["answer"] = None
        state["item_names"] = []  # 空列表 ⇒ 下游 node_search_embedding / hyde 走全库检索
        logger.info("分支C：未确认型号，降级为全库检索（不再直接拒答）")
        return state

    def _step_6_align_item_names(self, query_results) -> dict:
        """
        6 根据Milvus搜索评分，逐个对齐step4提取的item_names，生成「确认商品名」和「候选商品名」
        对齐规则（优先级a>b>c>d）：
                a  如果只有一个匹配结果评分高于0.85 → 直接确认该商品名
                b  如果多条匹配结果评分超过0.85 → 优先取与原始提取名相同的，无则取分数最高的
                c  如果无0.85分以上结果 → 取分数≥0.6的最高前5个作为候选
                d  如果无0.6分及以上结果 → 不返回任何商品名（确认+候选均为空）
        :param query_results: 列表[字典] - step5的返回结果，每个商品名的搜索匹配数据（格式同step5返回值）
        :return: 字典 - 商品名对齐结果，包含确认列表和候选列表，格式：
                 {
                     "confirmed_item_names": ["确认商品名1", "确认商品名2"],  # 去重后的确认商品名，无则空列表
                     "options": ["候选商品名1", "候选商品名2", ...]          # 去重后的候选商品名，无则空列表
                 }
        """
        # [{'extracted_name': 'HAK180烫金机', 'matches': [{'item_name': 'BrotherHAK180烫金扫描仪', 'score': 0.8573375344276428}]}]
        # query_results =[{'extracted_name': 'HAK180烫金机', 'matches': [{'item_name': 'BrotherHAK180烫金扫描仪', 'score': 0.8573375344276428}]}]
        # 创建两个变量  空
        # 1、初始化确认商品名列表（符合高置信度规则的商品名）
        confirmed_item_names: List[str] = []
        # 2、初始化候选商品名列表（低置信度，需用户确认的商品名）# 候选
        options: List[str] = []

        logger.info(f"步骤6：获得待处理的数据源：{query_results}")

        for res in query_results:
            # 提取原始的数据，商品名和匹配结果
            extracted_name = (res.get("extracted_name", "") or "").strip()
            # 获取匹配的商品名，无就获取空列表
            matches = res.get("matches", []) or []
            # 若无匹配结果，直接跳过当前商品名的对齐
            if not matches:
                continue

            # 筛选高置信度匹配结果：评分 > item_confirm_high（默认 0.85，由配置统一管理）
            high = [m for m in matches if m.get("score", 0) > retrieval_config.item_confirm_high]
            # 筛选中置信度匹配结果：评分 >= item_confirm_mid（默认 0.60，仅高置信度为空时生效）
            mid = [m for m in matches if m.get("score", 0) >= retrieval_config.item_confirm_mid]

            # 规则a: 只有一个高置信度结果（>0.85）→ 直接确认该商品名
            if len(high) == 1:
                confirmed_item_names.append(high[0].get("item_name"))
                continue  # 匹配到规则a，跳过后续规则判断

            # 规则b: 多条高置信度结果（>0.85）
            if len(high) > 1:
                # 初始化选中结果为None，优先匹配原始提取名
                picked = None
                # 若原始提取名非空，优先取与原始名相同的匹配结果
                if extracted_name:
                    for m in high:
                        if m.get("item_name") == extracted_name:
                            picked = m
                            break
                # 如果没有与原始名相同的结果，则取分数最高的第一个结果
                if not picked:
                    picked = high[0]

                # 将选中的结果加入确认商品名列表
                confirmed_item_names.append(picked.get("item_name"))
                continue  # 匹配到规则b，跳过后续规则判断

            # 规则c: 无0.85分以上结果，取≥0.6分的最高前5个作为候选
            # 注：高置信度列表high为空时才会走到此处（规则a/b均不满足）
            if len(mid) > 0:
                # 取中置信度结果的前5个，加入候选列表
                for m in mid[:5]:
                    options.append(m.get("item_name"))

            # 规则d: 无0.6分及以上结果 → 不做任何操作，确认+候选列表均为空
        # 返回最终对齐结果：确认列表和候选列表均做去重处理（list(set())）
        return {
            "confirmed_item_names": list(set(confirmed_item_names)),  # 去重，避免重复确认 []
            "options": list(set(options))  # 去重，避免重复候选 []
        }

    def _step_8_write_history(self, state, session_id, rewritten_query, message_id):
        """
         8 把本次处理的核心信息（用户问题、助手答案、商品名、改写查询）写入MongoDB的会话历史
         包含2个核心操作：1. 写入助手答案（若有）；2. 更新用户原始问题的关联信息
         :param state: 字典 - step6更新后的会话状态，包含answer/item_names等字段
         :param session_id: 字符串 - 会话唯一标识
         :param rewritten_query: 字符串 - step3改写后的完整问题
         :param message_id: 字符串 - 本次用户问题的消息唯一ID
         :return:
         """
        # 若会话状态中有助手答案（分支B/C），写入助手消息到历史
        if state.get("answer"):
            save_chat_message(
                session_id=session_id,  # 会话ID，关联所属会话
                role="assistant",  # 消息角色：助手
                text=state["answer"],  # 消息内容：向用户确认的提示语/无结果提示语
                rewritten_query="",  # 助手消息无需改写查询，设为空
                item_names=state.get("item_names", [])  # 关联的商品名列表（分支B/C均为空）
            )

        # 强制更新本次用户原始问题的关联信息（核心：补充改写查询、商品名）
        save_chat_message(
            session_id=session_id,  # 会话ID，关联所属会话
            role="user",  # 消息角色：用户
            text=state["original_query"],  # 消息内容：用户原始查询
            rewritten_query=rewritten_query,  # 补充step3改写后的完整问题
            item_names=state.get("item_names", []),  # 补充关联的商品名列表
            message_id=message_id  # 消息ID，指定更新已存在的用户消息（而非新增）
        )

        # 返回最终会话状态，供下游节点使用
        return state

    def _step_5_vectorize_and_query(self, item_names) -> List[Dict]:
        """
           把分析出的item_names逐个向量化（BGEM3模型），并在Milvus向量数据库(kb_item_names)中执行混合搜索，获取匹配评分
           :param item_names: 列表[字符串] - 步骤4中 提取的商品名列表（如["苹果15", "华为P60"]）
           :return: 列表[字典] - 格式：
                [
                    {
                        "extracted_name": "提取的原始商品名",  # 如"苹果15"
                        "matches": [                          # 该商品名的TopN匹配结果，无则空列表
                            {
                                "item_name": "数据库中的商品名",  # Milvus中存储的标准化商品名
                                "score": 0.98                  # 混合搜索的相似度评分（0-1，越高越相似）
                            },
                            ...
                        ]
                    },
                    ...
                ]
        """
        # 1、初始化最终返回结果列表，存储每个商品名的向量化查询结果
        # 查询结果  （向量数据库）
        results = []

        # 2、获取Milvus向量数据库的客户端连接对象
        client = get_milvus_client()

        # 3、校验Milvus客户端连接是否成功，失败则记录错误日志并返回空结果
        # 保证代码的鲁棒性（健壮性）
        if not client:
            logger.error("连接 Milvus 失败")
            return results

        # 4、从环境变量中获取Milvus中存储商品名称向量的集合名（表名）
        # 集合名 （表名） 自己识别出来 商品   向量数据库 里面存的商品 进行检索（相似度对比）
        collection_name = milvus_config.item_name_collection  # kb_item_names

        # 5、对所有商品名称批量生成BGEM3向量（稠密+稀疏），相比逐个生成提升处理效率
        # embeddings格式：{"dense": [向量1, 向量2,...], "sparse": [向量1, 向量2,...]}
        # 用户新提出来 问题里面  带的商品名称
        # 向量化稀疏向量+稠密向量
        embeddings = generate_embeddings(item_names)

        # 6、遍历每个商品名称，逐个执行向量搜索（保证结果与原始商品名一一对应）
        for i in range(len(item_names)):
            try:
                # 从批量生成的向量结果中，取出当前商品名对应的稠密向量（高维连续值，如[0.12, 0.35,...]）
                dense_vector = embeddings.get("dense")[i]
                # 从批量生成的向量结果中，取出当前商品名对应的稀疏向量（键值对，如{100:0.747, 205:0.664}）
                sparse_vector = embeddings.get("sparse")[i]

                # 构造Milvus混合搜索请求对象，传入稠/稀疏向量，指定返回Top5匹配结果
                # reqs返回格式：[稠密向量搜索请求, 稀疏向量搜索请求]
                # 调请求对象
                reqs = create_hybrid_search_requests(
                    dense_vector=dense_vector,
                    sparse_vector=sparse_vector,
                    limit=5  # 5条
                )

                # 执行BGEM3混合向量搜索，获取数据库中的匹配结果和评分
                # 默认配置：稠/稀疏向量权重各0.8/0.2，开启评分归一化（将距离值转为0-1相似度评分）
                search_res = hybrid_search(
                    client=client,  # Milvus客户端连接实例
                    collection_name=collection_name,  # 目标向量集合名（存储商品向量的表）
                    reqs=reqs,  # 混合搜索请求对象列表
                    ranker_weights=(0.8, 0.2),  # 稠/稀疏向量评分权重配比（和为1最佳）
                    limit=5,  # 最终返回Top5匹配结果
                    norm_score=True,  # 开启评分归一化，统一评分量级为0-1
                    output_fields=["item_name"]  # 指定返回Milvus中存储的商品名字段（业务字段）
                )
                # 判断有没有召回
                # 初始化当前商品名的匹配结果列表，存储匹配到的商品名+对应相似度评分
                matches = []
                # 校验搜索结果是否有效（非空且包含数据，适配Milvus批量搜索格式）
                if search_res and len(search_res) > 0:
                    # 遍历当前商品名的Top5匹配结果（search_res[0]为该商品的独立搜索结果集）
                    for hit in search_res[0]:
                        # 提取匹配结果中的商品名和评分，做防KeyError处理（设置默认空字典）
                        # hit格式：{"id": 数据库ID, "distance": 相似度评分, "entity": {"item_name": "标准化商品名"}}
                        matches.append(
                            {
                                "item_name": hit.get("entity", {}).get("item_name"),  # 数据库标准化商品名
                                "score": hit.get("distance"),  # 0-1相似度评分
                            }
                        )

                # 将当前商品名的原始名称+匹配结果，封装后加入最终结果列表
                results.append({
                    "extracted_name": item_names[i],  # step4提取的原始商品名称
                    "matches": matches  # 该商品名的Top5匹配结果（含评分）
                })

            # 捕获单个商品名处理的异常（不中断其他商品名执行），仅记录错误日志
            except Exception as e:
                logger.error(f"查询商品名 '{item_names[i]}' 时出错: {e}")
                # [BUG修复] 异常时补���空结果条目，保持 results 与 item_names 数量一致
                # 避免下游 _step_6_align_item_names 遍历时索引对应错乱
                results.append({
                    "extracted_name": item_names[i],
                    "matches": []
                })

        # 返回所有商品名的向量化+搜索结果列表
        return results

    def _step_4_extract_info(self, query, history) -> Dict:
        """
        利用LLM从当前问题以及历史会话中提取出主要询问的商品名称item_names（可多个，JSON列表形式）
        若商品名不够明确则返回空列表，同时根据上下文重新改写问题，保证问题独立完整
        :param query: 字符串 - 用户当前原始查询问题（如："这个多少钱？"）
        :param history: 列表[字典] - 近期会话历史，每条消息含role/text等字段，
                        格式：[{"role": "user/assistant", "text": "消息内容", "_id": "消息ID"}, ...]
        :return: 字典 - 提取结果，固定包含2个字段，格式：
                 {
                     "item_names": ["商品名1", "商品名2", ...],  # 提取的商品名列表，无则空列表
                     "rewritten_query": "改写后的完整问题"       # 包含商品名的独立问题，无则返回原始query
                 }
        """

        try:
            # 1. 获取llm客户端 商品名称
            # [融合 R5] 纯文本路径统一走共享网关 SakuraGatewayChatModel，json_mode=True 透传
            # response_format={"type":"json_object"}，与原 ChatOpenAI 行为一致；模型由 item_model 指定。
            # [兼容性提醒] response_format={"type": "json_object"} 要求 LLM 提供商支持该参数
            # 部分模型（如旧版 DeepSeek）可能忽略此参数，需依赖后续代码块的 JSON 清理逻辑兜底
            chat_model = get_llm_client(model=lm_config.item_model, json_mode=True)

            # 2. 构造历史对话文本，拼接为"角色: 内容"的格式，供LLM做上下文理解
            history_text = ""
            for msg in history:
                role = msg.get("role")
                content = msg.get("text")
                history_text += f"{role}: {content}\n"

            # 3. 处理和动态拼接提示词
            # 为了把大括号当作 “普通字符” 保留下来，用双大括号 {{ 表示普通的左大括号 {，双大括号 }} 表示普通的右大括号 }。
            user_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
                history_text=history_text,
                query=query
            )

            # 4. 构造LLM调用的消息列表，包含系统角色（定义助手身份）和用户角色（传入提示词）
            messages = [
                SystemMessage(content=ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ]

            # 5. 调用LLM客户端，发起请求获取结果
            response = chat_model.invoke(messages)
            content = response.content

            # 6. 数据清洗：处理LLM可能返回的代码块格式（如 ```json ... ``` ），去除包裹符
            #    兼容多种 LLM 的输出格式：```json / ```JSON / ``` (无语言标记) / 前后空白
            content = content.strip()
            # 去除开头的代码块标记：```json, ```JSON, ``` 等
            if content.startswith("```"):
                # 切掉第一行（```json 或 ```）
                first_newline = content.find("\n")
                if first_newline != -1:
                    content = content[first_newline + 1:]
                else:
                    content = content[3:]  # 单行 ```xxx``` 的情况
            # 去除结尾的 ```
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            # 7. 数据解析：将JSON字符串转为字典
            result = json.loads(content)

            # 8. 健壮性处理：确保字段存在
            # 确保返回结果包含item_names字段，无则设为空列表
            # 大模型调用结束以后 有没有商品名称 有没有原始问题
            if "item_names" not in result:
                result["item_names"] = []  #
            # 确保返回结果包含rewritten_query字段，无则复用原始查询
            if "rewritten_query" not in result:
                result["rewritten_query"] = query

            # 9. 给item_names 去除空格
            result["item_names"] = [
                name.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")
                for name in result["item_names"]
            ]

            # [P1-方案5] 解析 need_item_confirm：判断是否需要"确认具体型号"才能准确回答。
            # 该字段由 prompt 第 5 条产出；缺省默认 True（保持原行为：不明确时仍尝试对齐型号）。
            if "need_item_confirm" not in result:
                result["need_item_confirm"] = True
            else:
                # 兼容字符串/数字等形式，统一为布尔
                v = result["need_item_confirm"]
                if isinstance(v, str):
                    result["need_item_confirm"] = v.strip().lower() in ("true", "1", "yes")
                elif isinstance(v, (int, float)):
                    result["need_item_confirm"] = bool(v)
                else:
                    result["need_item_confirm"] = bool(v)

            # 10、返回解析后的提取结果
            return result

        except Exception as e:
            # 捕获所有异常（如LLM调用失败、JSON解析失败等），记录错误日志
            logger.error(f"大模型调用异常：{e}")
            # [P0-方案4] LLM 提取失败时，除返回空外，再用正则兜底抓取疑似型号，提升召回
            fallback_names = self._fallback_extract_item_names(query)
            # [P1-方案5] 兜底分支：正则已抓到疑似型号 ⇒ 仍需确认具体型号（True）；
            # 未抓到则走全库检索，need_item_confirm 在此不影响走向，统一置 True 保持原行为。
            return {"item_names": fallback_names, "rewritten_query": query, "need_item_confirm": True}

    def _step_1_validate_param(self, state: QueryGraphState) -> Tuple[str, str]:

        session_id = state.get("session_id")
        if not session_id:
            raise ValueError("核心参数session_id缺失")

        original_query = state.get("original_query")
        if not original_query:
            raise ValueError("核心参数original_query缺失")

        return session_id, original_query

    @staticmethod
    def _step_0_guard_query(state: QueryGraphState) -> str:
        """
        [P0-方案4] 输入守卫：对原始问题进行边界与异常校验。
        返回 None 表示通过；返回非空字符串表示拦截原因（作为友好提示直接输出，不进入检索链路）。
        """
        q = (state.get("original_query") or "").strip()
        # 1. 空白/纯空白
        if not q:
            return "请输入您想了解的产品或问题。"
        # 2. 超长（避免 token 浪费与异常）
        if len(q) > 2000:
            return "您的问题过长，请精简后重试（建议不超过 2000 字）。"
        # 3. 纯标点/空白/无有效字符（中文、字母、数字）
        if not re.search(r"[\u4e00-\u9fa5A-Za-z0-9]", q):
            return "请输入包含有效文字或型号的问题。"
        # 4. 简单安全拦截：广告/外链/诱导类（可按业务扩展关键词）
        if re.search(r"(加微信|代购|刷单|免费领|http[s]?://|点击链接|加我)", q, re.IGNORECASE):
            return "您的问题包含不安全内容，无法处理。"
        return None

    @staticmethod
    def _fallback_extract_item_names(query: str) -> List[str]:
        """
        [P0-方案4] LLM 提取失败时的兜底：用正则从 query 中抓取疑似型号/连续字母数字串。
        命中后上游仍会走 kb_item_names 向量确认；未命中则降级全库检索。
        """
        candidates = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{1,}", query or "")
        out, seen = [], set()
        for c in candidates:
            # 跳过纯数字（避免把年份/数量误当型号）
            if c.isdigit():
                continue
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out[:5]


if __name__ == "__main__":
    # print(lm_config.item_model)
    # from config.lm_config import lm_config
    # print(lm_config.api_key)
    #  # 初始化图状态
    #  # "HAK 180 烫金机怎么用？"
    #  # "怎么用呢？"
    #  # 初始化状态
    init_state = {
        "session_id": "test_session_010",  # 会话ID
        "original_query": "HAK 180 烫金机怎么用？"
    }

    # 创建节点对象
    node_item_name_confirm = NodeItemNameConfirm()  # __call__
    # 执行节点的单元测试
    result = node_item_name_confirm(init_state)
    # 将返回的图状态进行json序列化
    json_state = json.dumps(result, ensure_ascii=False, indent=4)
    # 输出
    # logger.info(json_state)
    from utils.json_format_utils import format_json

    logger.info(format_json(result))

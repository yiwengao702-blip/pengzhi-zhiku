# processor/import_processor/nodes/node_pdf_to_md.py
import json
import logging
import os
import shutil
import time
import zipfile
from pathlib import Path

import requests

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError, ConfigurationError, \
    PdfConversionError
from processor.import_processor.state import ImportGraphState


class NodePDFToMD(BaseNode):
    """
    PDF 转 Markdown 节点：PDF结构化解析
    """

    name = "node_pdf_to_md"

    def process(self, state: ImportGraphState):
        """

        :param state:
        :return:
        """

        # 1. 参数校验并返回Path结果
        pdf_path_obj, output_dir_obj = self._step_1_validate_paths(state)

        # 2. PDF → Markdown：优先 MinerU 云端解析；若失败（如 token 过期返回 401）
        #    则自动回退到「本地 PDF 解析」，保证 PDF 导入不被单一外部服务/Token 状态卡死。
        try:
            # 2.1 将PDF上传到MinerU并轮询结果，得到结果 ZIP 下载链接
            zip_url = self._step_2_upload_and_poll(pdf_path_obj)
            self.logger.info(zip_url)
            # 2.2 下载ZIP包并解压，得到 md 的绝对路径
            md_path = self._step_3_download_and_extract(zip_url, output_dir_obj, pdf_path_obj.stem)
        except Exception as e:
            # 2.3 是否允许本地兜底（可通过 .env 的 PDF_LOCAL_FALLBACK=false 关闭，默认开启）
            fallback_on = os.getenv("PDF_LOCAL_FALLBACK", "true").strip().lower() not in ("0", "false", "no", "off")
            if not fallback_on:
                raise
            self.logger.warning(
                "MinerU 云端解析失败，自动切换到【本地 PDF 解析】兜底（不影响后续 切片/向量化/入库）。失败原因：%s", e
            )
            md_path = self._local_pdf_to_md(pdf_path_obj, output_dir_obj)

        # 3. 将MD文件的内容都取出来
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # 4. 返回结果
        state["md_content"] = md_content
        state["md_path"] = md_path
        return state

    def _local_pdf_to_md(self, pdf_path_obj: Path, output_dir_obj: Path) -> str:
        """
        兜底步骤：本地 PDF → Markdown（不依赖 MinerU / 外部 Token）

        说明：
          - 使用 PyMuPDF(fitz) 逐页抽取文本，按「## 第 N 页」分节，生成可直接进入
            「文档切分 → 向量化 → 入库」链路的纯文本 Markdown。
          - 相比 MinerU 不含图片/复杂版面还原，但能保证 Token 失效时 PDF 导入依然可用。
          - 输出文件名与 MinerU 分支保持一致（<pdf_stem>.md），且不产生 images/ 目录，
            下游 node_md_img 会自动跳过图片处理、直接透传正文，不会因此报错。

        :param pdf_path_obj: PDF 的 Path 对象
        :param output_dir_obj: 输出目录 Path 对象
        :return: 生成的 MD 文件绝对路径（字符串）
        """
        try:
            import pymupdf as fitz  # PyMuPDF 1.24+ 推荐导入名
        except ImportError:
            try:
                import fitz  # 兼容旧版本 PyMuPDF
            except ImportError as e:
                raise PdfConversionError(
                    "本地 PDF 解析兜底不可用：未安装 PyMuPDF。请在项目根目录执行 "
                    "`uv pip install --python .venv/Scripts/python.exe pymupdf`（或 pip install pymupdf）后重试；"
                    "或到 https://mineru.net 获取有效的 MINERU_API_TOKEN 以使用云端解析。"
                ) from e

        # 1. 确保输出目录存在
        output_dir_obj.mkdir(parents=True, exist_ok=True)

        # 2. 打开 PDF 并逐页抽取文本，按「约 1800 字」为单位合并成更大的节，
        #    以减少切片数量（每个切片需一次 BGE-M3 向量化，CPU 上约 2s；
        #    合并多页可显著缩短整体导入耗时，且对检索问答无实质影响）。
        doc = fitz.open(str(pdf_path_obj))
        page_count = doc.page_count
        TARGET = 1800  # 单节目标字符数（接近 document_split 的 max_content_length=2000，避免被二次细分）

        sections = []  # 每个元素: (起始页, 结束页, 文本)
        buf_start, buf_end, buf_text, buf_len = None, None, [], 0
        for index in range(page_count):
            page_text = (doc.load_page(index).get_text("text") or "").strip()
            if not page_text:
                page_text = "（本页无可提取文本，可能为扫描件或纯图形页）"
            # 仍能塞进当前节则并入；否则先结算当前节，本页另起新节
            if buf_len == 0 or buf_len + len(page_text) + 2 <= TARGET:
                if buf_start is None:
                    buf_start, buf_end = index + 1, index + 1
                else:
                    buf_end = index + 1
                buf_text.append(page_text)
                buf_len += len(page_text) + 2
            else:
                sections.append((buf_start, buf_end, "\n\n".join(buf_text)))
                buf_start, buf_end = index + 1, index + 1
                buf_text, buf_len = [page_text], len(page_text) + 2
        if buf_start is not None:
            sections.append((buf_start, buf_end, "\n\n".join(buf_text)))
        doc.close()

        # 2.x 组装 Markdown：标题 + 各合并节（节标题标注页范围，便于溯源）
        lines = [f"# {pdf_path_obj.stem}", ""]
        for start, end, text in sections:
            title = f"## 第 {start} 页" if start == end else f"## 第 {start}-{end} 页"
            lines.append(title)
            lines.append("")
            lines.append(text)
            lines.append("")

        # 3. 落盘（文件名与 MinerU 分支一致）
        md_path_obj = output_dir_obj / f"{pdf_path_obj.stem}.md"
        md_path_obj.write_text("\n".join(lines), encoding="utf-8")

        self.log_step("本地PDF解析", f"完成，共 {page_count} 页 -> {md_path_obj.name}")
        return str(md_path_obj.absolute())

    def _step_3_download_and_extract(self, zip_url: str, output_dir_obj: Path, pdf_stem: str) -> str:
        """
        步骤3：下载MinerU解析结果ZIP包并解压，提取目标MD文件
        核心流程：下载ZIP → 清理旧目录并解压 → 查找MD文件 → 重命名统一为PDF同名
        参数：zip_url-ZIP包下载链接；output_dir_obj-输出目录Path；pdf_stem-PDF无后缀纯名称
        返回：最终MD文件的字符串格式绝对路径
        异常：RuntimeError(下载失败)
        """

        # 1.下载zip  包含压缩包 服务器返回的内容
        response = requests.get(zip_url)

        # 防御性编程  （目的是为了代码的鲁棒性（健壮性））
        if response.status_code != 200:
            raise FileProcessingError("下载失败")

        #定义做包的保存路径
        zip_save_path = output_dir_obj / f"{pdf_stem}_result.zip"
        with open(zip_save_path, "wb") as f:
            f.write(response.content)

        self.log_step("ZIP下载", "下载成功")# 打印

        # 2. 解压ZIP
        # 解压目录
        extract_target_dir = output_dir_obj / pdf_stem

        # 删除已有目录  防御性编程
        if extract_target_dir.exists():
            shutil.rmtree(extract_target_dir)

        # 创建新目录
        extract_target_dir.mkdir(parents=True, exist_ok=True)

        # 解压ZIP
        with zipfile.ZipFile(zip_save_path, "r") as zip_file_obj:
            zip_file_obj.extractall(extract_target_dir)

        self.log_step("ZIP下载", "解压完成")
        # 读extract_target_dir full.md
        target_md_file = extract_target_dir / "full.md"# 路径拼接
        new_md_path = target_md_file.with_name(f"{pdf_stem}.md")# 新的md的名字
        target_md_file.rename(new_md_path)# 执行

        return str(new_md_path.absolute())# 绝对路径


    def _step_2_upload_and_poll(self, pdf_path_obj: Path):
        """
        步骤2：上传PDF至MinerU并轮询解析任务状态
        核心流程：配置校验 → 获取上传链接 → 文件上传 → 任务轮询（直至完成/失败/超时）
        参数：pdf_path_obj-已校验的PDF Path对象
        返回：解析结果ZIP包下载链接full_zip_url
        异常：ValueError(配置缺失)、RuntimeError(请求/上传失败)、TimeoutError(任务超时)
        """


        # 1、配置文件校验
        if not self.config.mineru_base_url:
            raise ConfigurationError("MinerU配置缺失：请在 .env 文件中正确配置 MINERU_BASE_URL 参数")
        if not self.config.mineru_api_token:
            raise ConfigurationError("MinerU配置缺失：请在 .env 文件中正确配置 MINERU_API_TOKEN 参数")

        # 2. 调用MinerU的远程API接口，获取上传链接
        # 2.1 组织上传需要的数据
        token = self.config.mineru_api_token
        url = f"{self.config.mineru_base_url}/file-urls/batch"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        data = {
            "files": [
                {"name": pdf_path_obj.name}
            ],
            "model_version": "vlm"
        }

        # 2.2 发送请求
        response = requests.post(url, headers=header, json=data)

        # 2.3 获取响应结果
        if response.status_code != 200:
            # 解析错误体，把「高频卡点」翻译成可执行提示，避免用户只看到笼统的「失败」
            err_code = None
            err_msg = ""
            try:
                err_body = response.json()
                err_code = err_body.get("msgCode") or err_body.get("code")
                err_msg = err_body.get("msg") or err_body.get("message") or ""
            except Exception:
                err_body = response.text
                err_msg = response.text
            # MinerU 鉴权失败：401/403 或 msgCode=A0202 或包含 authenticate 字样，均视为 token 过期/无效
            if response.status_code in (401, 403) or str(err_code) == "A0202" or "authenticate" in str(err_msg).lower():
                raise PdfConversionError(
                    "【MinerU 鉴权失败】MINERU_API_TOKEN 已过期或无效（HTTP {}，msgCode={}）。\n"
                    "请到 https://mineru.net 重新获取 API Token，替换项目根目录 .env 中的 "
                    "MINERU_API_TOKEN 后重启 python main.py 再试。\n"
                    "注意：token 失效期间 PDF 导入会一直失败，但 Markdown 导入不受影响，可照常使用。"
                    .format(response.status_code, err_code)
                )
            raise PdfConversionError(
                '获取上传链接失败. 响应状态:{} ,响应结果:{}'.format(response.status_code, err_body))

        result = response.json()
        if result["code"] != 0:
            raise PdfConversionError('获取上传链接失败. 返回数据：{}'.format(result["msg"]))

        # 2.4 获取到上传链接
        # 批量提取任务 id，可用于批量查询解析结果
        batch_id = result["data"]["batch_id"]
        # 文件上传链接
        urls = result["data"]["file_urls"]

        # 3. 文件上传
        with open(pdf_path_obj, 'rb') as f:
            res_upload = requests.put(urls[0], data=f)
            if res_upload.status_code != 200:
                raise PdfConversionError(f'{urls[0]}文件上传失败')

            self.logger.info('上传文件成功!')

        # 4. 获取解析结果
        poll_url = f"{self.config.mineru_base_url}/extract-results/batch/{batch_id}"

        # 轮询获取
        start_time = time.time()  # 记录当前时间
        timeout_seconds = 600  # 最大超时时间
        poll_interval = 3  # 轮询间隔
        self.log_step(
            step_name="轮询开始",
            message=f"轮询间隔: {poll_interval}s, 超时时间: {timeout_seconds}s，batch_id:{batch_id}"
        )
        while True:
            #已消耗时间
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
              raise TimeoutError(f"[任务轮询]超时，已消耗时间: {int(elapsed_time)}s，batch_id:{batch_id}")

            # 获取任务结果
            try:
                poll_res = requests.get(poll_url, headers=header, timeout=10)
            except Exception as e:
                self.logger.warning(f"网络请求异常，{poll_interval}s后重试，batch_id:{batch_id}")
                self.logger.warning(f"异常信息：{str(e)}")
                time.sleep(poll_interval)
                continue

            # print(poll_res.status_code)
            # print(poll_res.json())
            # print(poll_res.json()["data"])
            if poll_res.status_code != 200:
                raise PdfConversionError(f'[任务轮询]失败，状态码：{poll_res.status_code}')

            poll_res_json = poll_res.json()
            if poll_res_json["code"] != 0:
                raise PdfConversionError(f'[任务轮询]失败，错误码：{poll_res_json["code"]}')

            extract_results = poll_res_json["data"]["extract_result"]
            extract_result = extract_results[0]
            # 获取任务的状态值
            data_state = extract_result["state"]
            if data_state == "done":
                self.log_step(
                    step_name="任务轮询",
                    message=f"解析完成s, 总耗时: {int(elapsed_time)}s，batch_id:{batch_id}"
                )

                full_zip_url = extract_result["full_zip_url"]
                self.log_step(
                    step_name="任务轮询",
                    message=f"获取全量zip地址成功: {full_zip_url}, 总耗时: {int(elapsed_time)}s，batch_id:{batch_id}"
                )

                return full_zip_url

            elif data_state == "failed":
                raise PdfConversionError(f'[任务轮询]失败，错误信息: {extract_result["err_msg"]}')

            else:
                self.log_step(
                    step_name="任务轮询",
                    message=f"处理中...... 已耗时: {int(elapsed_time)}s，状态：{data_state}，batch_id:{batch_id}"
                )
                time.sleep(poll_interval)


    def _step_1_validate_paths(self, state: ImportGraphState):
        """
        步骤1：校验PDF文件路径和输出目录
        核心职责：参数非空校验 | 路径转换 | PDF文件有效性校验 | 输出目录自动创建
        返回：合法的PDF文件Path对象、输出目录Path对象
        异常：StateFieldError(参数缺失)、FileNotFoundError(文件无效)
        """

        # 1.参数的非空校验
        pdf_path = state.get("pdf_path")
        if not pdf_path:
            raise StateFieldError(field_name="pdf_path", message="PDF路径不能为空", expected_type=str)

        file_dir = state.get("file_dir")
        if not file_dir:
            raise StateFieldError(field_name="file_dir", message="输出路径不能为空", expected_type=str)

        # 2. 转换为Path对象
        pdf_path_obj = Path(pdf_path)
        file_dir_obj = Path(file_dir)

        # 3.pdf是否存在
        if not pdf_path_obj.exists():
            raise FileProcessingError(f"文件{pdf_path_obj.name}不存在")

        # 4.输出目录不存在则创建
        if not file_dir_obj.exists():
            self.logger.info(f"输出目录{file_dir_obj.absolute()}不存在，开始创建...")
            file_dir_obj.mkdir(parents=True, exist_ok=True)

        return pdf_path_obj, file_dir_obj


if __name__ == "__main__":
    # 激活日志
    setup_logging()

    init_state = {
        "pdf_path": r"D:\python\13-企业级智能知识库系统项目\day01\资料\04-设备手册汇总\doc\联想海豚用户手册.pdf",
        "file_dir": r"D:\python"
    }
    node_pdf_to_md = NodePDFToMD()
    # 使用 对象() 的方式相当于调用了 对象的__call__()
    result = node_pdf_to_md(init_state)

    # 将返回的图状态进行json序列化
    json_state = json.dumps(result, ensure_ascii=False, indent=4)
    # 输出
    logging.getLogger().info(json_state)


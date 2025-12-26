import json
import os
import time
import mimetypes
import base64
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class MineruParser:
    """调用 MinerU /file_parse 接口的服务类"""
    
    # 文件扩展名到 MIME type 的映射
    CONTENT_TYPE_MAP = {
        '.pdf': 'application/pdf',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.tiff': 'image/tiff',
        '.tif': 'image/tiff',
    }
    
    # 支持的文档类型扩展名
    SUPPORTED_EXTENSIONS = {
        'pdf': ['.pdf'],
        'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif'],
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化解析器
        
        Args:
            config_path: 配置文件路径，默认为 server/config.json
        """
        if config_path is None:
            # 获取当前文件所在目录的父目录的父目录（server/parser -> server -> 项目根目录）
            current_dir = Path(__file__).parent.parent.parent
            config_path = os.path.join(current_dir, "server", "config.json")
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        self.config = config.get("mineru_parse", {})
        self.service_url = self.config.get("service_url", "http://127.0.0.1:8000")
        self.output_dir = self.config.get("output_dir", "./output")
        self.timeout = self.config.get("timeout", 300)
        self.retry_times = self.config.get("retry_times", 1)
        
        # 创建带重试机制的会话
        self.session = requests.Session()
        retry_strategy = Retry(
            total=self.retry_times,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def parse_files(
        self,
        file_paths: List[str],
        output_dir: Optional[str] = None,
        lang_list: List[str] = None,
        backend: str = "pipeline",
        parse_method: str = "auto",
        formula_enable: bool = True,
        table_enable: bool = True,
        server_url: Optional[str] = None,
        start_page_id: int = 0,
        end_page_id: int = 99999,
    ) -> Dict[str, Any]:
        """
        解析文件，只返回 markdown 内容
        
        Args:
            file_paths: 要解析的文件路径列表
            output_dir: 输出目录，默认使用配置文件中的值
            lang_list: 语言列表，默认 ["ch"]
            backend: 后端类型，默认 "pipeline"
            parse_method: 解析方法，默认 "auto"
            formula_enable: 是否启用公式解析
            table_enable: 是否启用表格解析
            server_url: 服务器URL（仅用于vlm-http-client后端）
            start_page_id: 起始页码
            end_page_id: 结束页码
        
        Returns:
            解析结果字典，包含 success 和 data 字段
            data 格式: {"文件名": {"md_content": "markdown内容"}}
        """
        if lang_list is None:
            lang_list = ["ch"]
        
        if output_dir is None:
            output_dir = self.output_dir
        
        # 构建请求URL
        url = f"{self.service_url}/file_parse"
        
        # 准备文件
        files = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            file_name = os.path.basename(file_path)
            # 根据文件扩展名获取 MIME type
            content_type, _ = mimetypes.guess_type(file_path)
            if content_type is None:
                # 如果无法识别，根据常见扩展名设置
                ext = os.path.splitext(file_path)[1].lower()
                content_type = self.CONTENT_TYPE_MAP.get(ext, 'application/octet-stream')
            
            with open(file_path, "rb") as f:
                files.append(("files", (file_name, f.read(), content_type)))
        
        # 准备表单数据，只返回 markdown 内容
        data = {
            "output_dir": output_dir,
            "backend": backend,
            "parse_method": parse_method,
            "formula_enable": formula_enable,
            "table_enable": table_enable,
            "return_md": True,  # 只返回 markdown
            "return_middle_json": False,
            "return_model_output": False,
            "return_content_list": False,
            "return_images": True,  # 返回图片
            "response_format_zip": False,
            "f_draw_layout_bbox": False,
            "f_draw_span_bbox": False,
            "f_dump_orig_pdf": False,
            "start_page_id": start_page_id,
            "end_page_id": end_page_id,
        }
        
        # 添加可选参数
        if server_url:
            data["server_url"] = server_url
        
        # 添加语言列表（需要为每个文件添加）
        for lang in lang_list:
            data.setdefault("lang_list", []).append(lang)
        
        # 发送请求
        try:
            response = self.session.post(
                url,
                files=files,
                data=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # 返回JSON响应
            return {
                "success": True,
                "data": response.json()
            }
        
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": f"请求超时（{self.timeout}秒）"
            }
        except requests.exceptions.HTTPError as e:
            try:
                error_detail = e.response.json()
            except:
                error_detail = {"error": str(e)}
            return {
                "success": False,
                "error": f"HTTP错误: {e.response.status_code}",
                "detail": error_detail
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"请求失败: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"未知错误: {str(e)}"
            }
    
    def _collect_supported_files(
        self, 
        dir_path: str, 
        supported_types: List[str]
    ) -> List[tuple]:
        """
        递归收集目录中支持的文件
        
        Args:
            dir_path: 目录路径
            supported_types: 支持的文档类型列表，如 ['pdf', 'image']
        
        Returns:
            文件列表，每个元素为 (文件绝对路径, 相对于dir_path的相对路径)
        """
        dir_path = os.path.abspath(dir_path)
        if not os.path.isdir(dir_path):
            raise ValueError(f"目录不存在或不是目录: {dir_path}")
        
        # 收集所有支持的扩展名
        supported_extensions = set()
        for doc_type in supported_types:
            if doc_type.lower() in self.SUPPORTED_EXTENSIONS:
                supported_extensions.update(self.SUPPORTED_EXTENSIONS[doc_type.lower()])
        
        if not supported_extensions:
            raise ValueError(f"没有找到支持的类型: {supported_types}")
        
        files = []
        for root, dirs, filenames in os.walk(dir_path):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in supported_extensions:
                    file_path = os.path.join(root, filename)
                    # 计算相对于dir_path的相对路径
                    rel_path = os.path.relpath(file_path, dir_path)
                    files.append((file_path, rel_path))
        
        return files
    
    def parse_dir(
        self,
        dir_path: str,
        supported_types: List[str] = None,
        local_output_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        lang_list: List[str] = None,
        backend: str = "pipeline",
        parse_method: str = "auto",
        formula_enable: bool = True,
        table_enable: bool = True,
        server_url: Optional[str] = None,
        start_page_id: int = 0,
        end_page_id: int = 99999,
        batch_size: int = 10,
    ) -> Dict[str, Any]:
        """
        解析目录中的所有文档，保持目录结构
        
        Args:
            dir_path: 要解析的目录路径
            supported_types: 支持的文档类型列表，如 ['pdf', 'image']，默认 ['pdf', 'image']
            local_output_dir: 本地输出目录，解析后的文档保存到此目录
            output_dir: 远程服务端的输出目录，默认使用配置文件中的值
            lang_list: 语言列表，默认 ["ch"]
            backend: 后端类型，默认 "pipeline"
            parse_method: 解析方法，默认 "auto"
            formula_enable: 是否启用公式解析
            table_enable: 是否启用表格解析
            server_url: 服务器URL（仅用于vlm-http-client后端）
            start_page_id: 起始页码
            end_page_id: 结束页码
            batch_size: 每批处理的文件数量，默认 10
        
        Returns:
            解析结果字典，包含 success、data 和 file_mapping 字段
            file_mapping: 原始文件路径到相对路径的映射
        """
        if supported_types is None:
            supported_types = ['pdf', 'image']
        
        if local_output_dir is None:
            raise ValueError("local_output_dir 参数必须指定")
        
        if output_dir is None:
            output_dir = self.output_dir
        
        if lang_list is None:
            lang_list = ["ch"]
        
        # 收集所有支持的文件
        print(f"正在扫描目录: {dir_path}")
        file_list = self._collect_supported_files(dir_path, supported_types)
        
        if not file_list:
            return {
                "success": False,
                "error": f"目录中没有找到支持的文件类型: {supported_types}"
            }
        
        print(f"找到 {len(file_list)} 个文件需要解析")
        
        # 获取 dir_path 的最后一级目录名，并在 local_output_dir 下创建同名子目录
        dir_path_abs = os.path.abspath(dir_path)
        dir_name = os.path.basename(dir_path_abs)
        
        # 创建本地输出目录，在 local_output_dir 下创建与 dir_path 最后一级目录名相同的子目录
        local_output_dir = os.path.abspath(local_output_dir)
        local_output_dir = os.path.join(local_output_dir, dir_name)
        os.makedirs(local_output_dir, exist_ok=True)
        
        # 批量处理文件
        all_results = {}
        file_mapping = {}  # 原始文件名到相对路径的映射
        
        total_batches = (len(file_list) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(file_list), batch_size):
            batch_files = file_list[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            
            print(f"\n处理批次 {batch_num}/{total_batches} ({len(batch_files)} 个文件)...")
            
            # 准备文件路径和相对路径映射
            file_paths = [fp for fp, _ in batch_files]
            batch_rel_paths = [rel_path for _, rel_path in batch_files]
            
            # 调用 parse_files 解析这批文件
            result = self.parse_files(
                file_paths=file_paths,
                output_dir=output_dir,
                lang_list=lang_list,
                backend=backend,
                parse_method=parse_method,
                formula_enable=formula_enable,
                table_enable=table_enable,
                server_url=server_url,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
            )
            
            if not result.get("success"):
                print(f"批次 {batch_num} 解析失败: {result.get('error')}")
                continue
            
            # 提取结果并保存到本地，保持目录结构
            data = result.get("data", {})
            results = data.get("results", {})
            
            # 使用索引来匹配结果和文件（顺序应该一致）
            result_items = list(results.items())
            
            for idx, (file_name, file_result) in enumerate(result_items):
                if idx >= len(batch_rel_paths):
                    print(f"警告: 结果数量 ({len(result_items)}) 与文件数量 ({len(batch_rel_paths)}) 不匹配")
                    continue
                
                rel_path = batch_rel_paths[idx]
                # 更新全局映射
                file_mapping[rel_path] = file_result
                
                # 构建本地保存路径，保持目录结构
                rel_dir = os.path.dirname(rel_path)
                file_stem = os.path.splitext(os.path.basename(rel_path))[0]
                
                local_file_dir = os.path.join(local_output_dir, rel_dir) if rel_dir else local_output_dir
                os.makedirs(local_file_dir, exist_ok=True)
                
                # 保存 markdown 内容
                md_content = file_result.get("md_content")
                if md_content:
                    md_path = os.path.join(local_file_dir, f"{file_stem}.md")
                    with open(md_path, "w", encoding="utf-8") as f:
                        f.write(md_content)
                    print(f"  已保存: {md_path}")
                
                # 保存图片
                images = file_result.get("images", {})
                if images:
                    images_dir = os.path.join(local_file_dir, "images")
                    os.makedirs(images_dir, exist_ok=True)
                    
                    # 解析并保存图片
                    data_uri_pattern = re.compile(r'data:image/(\w+);base64,(.+)')
                    for image_name, data_uri in images.items():
                        match = data_uri_pattern.match(data_uri)
                        if match:
                            image_format = match.group(1)
                            base64_content = match.group(2)
                            
                            try:
                                image_data = base64.b64decode(base64_content)
                                ext = image_format if image_format in ['jpeg', 'jpg', 'png', 'gif', 'bmp'] else 'jpg'
                                if ext == 'jpeg':
                                    ext = 'jpg'
                                
                                image_path = os.path.join(images_dir, image_name)
                                if not os.path.splitext(image_name)[1]:
                                    image_path = f"{image_path}.{ext}"
                                
                                with open(image_path, "wb") as f:
                                    f.write(image_data)
                            except Exception as e:
                                print(f"  警告: 保存图片 {image_name} 失败: {str(e)}")
                
                # 保存结果到 all_results
                all_results[rel_path] = file_result
        
        return {
            "success": True,
            "data": {
                "backend": backend,
                "results": all_results
            },
            "file_mapping": file_mapping,
            "total_files": len(file_list),
            "local_output_dir": local_output_dir
        }
    
    def get_md_content(self, file_paths: List[str], **kwargs) -> Dict[str, Optional[str]]:
        """
        便捷方法：直接获取文件的 markdown 内容
        
        Args:
            file_paths: 要解析的文件路径列表
            **kwargs: 其他参数，传递给 parse_files
        
        Returns:
            字典，键为文件名，值为 markdown 内容（如果解析失败则为 None）
        """
        result = self.parse_files(file_paths, **kwargs)
        
        if not result.get("success"):
            # 如果解析失败，返回所有文件名为 None
            return {os.path.basename(fp): None for fp in file_paths}
        
        data = result.get("data", {})
        results = data.get("results", {})
        
        md_contents = {}
        for file_name, file_result in results.items():
            md_contents[file_name] = file_result.get("md_content")
        
        return md_contents
    
    def save_md_content(self, md_contents: Dict[str, Optional[str]], output_dir: str):
        """
        保存 markdown 内容到文件
        
        Args:
            md_contents: 文件名到 markdown 内容的字典
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for file_name, md_content in md_contents.items():
            if md_content is None:
                print(f"警告: {file_name} 的 markdown 内容为空，跳过保存")
                continue
            
            output_path = os.path.join(output_dir, f"{file_name}.md")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            print(f"已保存: {output_path}")
    
    def save_images(self, images_dict: Dict[str, Dict[str, str]], output_dir: str):
        """
        保存图片到文件
        
        Args:
            images_dict: 文件名到图片字典的映射，格式: {"文件名": {"图片名": "data:image/jpeg;base64,..."}}
            output_dir: 输出目录，图片将保存到 output_dir/images/ 目录
        """
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # 匹配 data URI 格式的正则表达式
        data_uri_pattern = re.compile(r'data:image/(\w+);base64,(.+)')
        
        total_saved = 0
        for file_name, images in images_dict.items():
            if not images:
                continue
            
            for image_name, data_uri in images.items():
                match = data_uri_pattern.match(data_uri)
                if match:
                    image_format = match.group(1)  # jpeg, png 等
                    base64_content = match.group(2)
                    
                    # 解码 base64
                    try:
                        image_data = base64.b64decode(base64_content)
                        
                        # 确定文件扩展名
                        ext = image_format if image_format in ['jpeg', 'jpg', 'png', 'gif', 'bmp'] else 'jpg'
                        if ext == 'jpeg':
                            ext = 'jpg'
                        
                        # 保存图片
                        image_path = os.path.join(images_dir, image_name)
                        # 如果图片名没有扩展名，添加扩展名
                        if not os.path.splitext(image_name)[1]:
                            image_path = f"{image_path}.{ext}"
                        
                        with open(image_path, "wb") as f:
                            f.write(image_data)
                        total_saved += 1
                    except Exception as e:
                        print(f"警告: 保存图片 {image_name} 失败: {str(e)}")
                else:
                    print(f"警告: 图片 {image_name} 的格式不正确")
        
        if total_saved > 0:
            print(f"已保存 {total_saved} 张图片到: {images_dir}")


def test_parse_files():
    """测试接口的主方法"""
    # 初始化解析器
    parser = MineruParser()
    backend_list = ["pipeline", "vlm-transformers", "vlm-mlx-engine", "vlm-vllm-async-engine", "vlm-lmdeploy-engine", "vlm-http-client"]
    backend = "vlm-vllm-async-engine"
    
    print(f"服务URL: {parser.service_url}")
    print(f"输出目录: {parser.output_dir}")
    print(f"后端引擎: {backend}")
    print(f"超时时间: {parser.timeout}秒")
    print(f"重试次数: {parser.retry_times}")
    print("-" * 50)
    
    # 测试文件路径（请根据实际情况修改）
    # 这里使用项目中的示例PDF文件
    test_files = [
        # 示例：使用项目中的PDF文件
        "/Users/ailabuser7-1/Documents/cursor-workspace/xiaobankuaipao/data/样品课件/样品课件-会计/课件/ACCT 6002 Accountability S2 2022.pdf"
    ]
    
    # 如果没有指定测试文件，提示用户
    if not test_files:
        print("请指定要测试的文件路径")
        print("示例：")
        print('  test_files = ["/path/to/your/file.pdf"]')
        return
    
    # 检查文件是否存在
    for file_path in test_files:
        if not os.path.exists(file_path):
            print(f"警告: 文件不存在: {file_path}")
    
    print(f"\n开始解析 {len(test_files)} 个文件...")
    start_time = time.time()
    
    # 调用解析接口，只获取 markdown 内容
    result = parser.parse_files(
        file_paths=test_files,
        lang_list=["ch"],
        backend=backend,
        parse_method="auto",
    )
    
    elapsed_time = time.time() - start_time
    
    # 打印结果
    print(f"\n解析完成，耗时: {elapsed_time:.2f}秒")
    print("-" * 50)
    
    if result.get("success"):
        print("✓ 解析成功")
        data = result.get("data", {})
        
        if isinstance(data, dict):
            print(f"后端: {data.get('backend', 'N/A')}")
            print(f"版本: {data.get('version', 'N/A')}")
            
            results = data.get("results", {})
            print(f"\n解析结果数量: {len(results)}")
            
            # 提取 markdown 内容和图片用于显示和保存
            md_contents = {}
            images_dict = {}
            for file_name, file_result in results.items():
                print(f"\n文件: {file_name}")
                md_content = file_result.get("md_content")
                md_contents[file_name] = md_content
                if md_content:
                    md_len = len(md_content)
                    print(f"  - Markdown内容长度: {md_len} 字符")
                    # 显示前200个字符作为预览
                    preview = md_content[:200].replace("\n", " ")
                    print(f"  - 内容预览: {preview}...")
                else:
                    print(f"  - Markdown内容: 无")
                
                # 提取图片
                images = file_result.get("images", {})
                if images:
                    images_dict[file_name] = images
                    img_count = len(images)
                    print(f"  - 图片数量: {img_count}")
            
            # 保存 markdown 内容和图片到 docs_parsed/ 目录
            # 获取项目根目录
            project_root = Path(__file__).parent.parent.parent
            docs_parsed_dir = project_root / "docs_parsed"
            
            parser.save_md_content(md_contents, str(docs_parsed_dir))
            print(f"\nMarkdown 文件已保存到: {docs_parsed_dir}")
            
            # 保存图片
            if images_dict:
                parser.save_images(images_dict, str(docs_parsed_dir))
    else:
        print("✗ 解析失败")
        print(f"错误: {result.get('error', '未知错误')}")
        if "detail" in result:
            print(f"详情: {result['detail']}")


def test_parse_dir():
    """测试 parse_dir 方法"""
    # 初始化解析器
    parser = MineruParser()
    backend_list = ["pipeline", "vlm-transformers", "vlm-mlx-engine", "vlm-vllm-async-engine", "vlm-lmdeploy-engine", "vlm-http-client"]
    backend = "vlm-vllm-async-engine"
    
    print(f"服务URL: {parser.service_url}")
    print(f"输出目录: {parser.output_dir}")
    print(f"后端引擎: {backend}")
    print(f"超时时间: {parser.timeout}秒")
    print(f"重试次数: {parser.retry_times}")
    print("-" * 50)
    
    # 测试目录路径（请根据实际情况修改）
    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent
    #test_dir = project_root / "data" / "样品课件" / "样品课件-会计" / "课件"
    test_dir = project_root / "data" / "金融实证方法" 
    
    # 检查目录是否存在
    if not os.path.exists(str(test_dir)):
        print(f"警告: 测试目录不存在: {test_dir}")
        print("请修改 test_dir 变量指向一个包含 PDF 或图片文件的目录")
        return
    
    # 本地输出目录
    local_output_dir = project_root / "docs_parsed"
    
    print(f"\n测试目录: {test_dir}")
    print(f"本地输出目录: {local_output_dir}")
    print(f"支持的文档类型: ['pdf', 'image']")
    print("-" * 50)
    
    print(f"\n开始解析目录...")
    start_time = time.time()
    
    # 调用 parse_dir 方法
    result = parser.parse_dir(
        dir_path=str(test_dir),
        supported_types=['pdf', 'image'],
        local_output_dir=str(local_output_dir),
        lang_list=["ch"],
        backend=backend,
        parse_method="auto",
        batch_size=5,  # 每批处理5个文件
    )
    
    elapsed_time = time.time() - start_time
    
    # 打印结果
    print(f"\n解析完成，耗时: {elapsed_time:.2f}秒")
    print("-" * 50)
    
    if result.get("success"):
        print("✓ 解析成功")
        print(f"总文件数: {result.get('total_files', 0)}")
        print(f"本地输出目录: {result.get('local_output_dir', 'N/A')}")
        
        data = result.get("data", {})
        results = data.get("results", {})
        print(f"\n解析结果数量: {len(results)}")
        
        # 显示部分结果统计
        if results:
            print("\n部分解析结果:")
            for idx, (rel_path, file_result) in enumerate(list(results.items())[:5]):
                print(f"\n  {idx + 1}. {rel_path}")
                md_content = file_result.get("md_content")
                if md_content:
                    md_len = len(md_content)
                    print(f"     - Markdown内容长度: {md_len} 字符")
                images = file_result.get("images", {})
                if images:
                    print(f"     - 图片数量: {len(images)}")
            
            if len(results) > 5:
                print(f"\n  ... 还有 {len(results) - 5} 个文件")
    else:
        print("✗ 解析失败")
        print(f"错误: {result.get('error', '未知错误')}")
        if "detail" in result:
            print(f"详情: {result['detail']}")


if __name__ == "__main__":
    # test_parse_files()
    test_parse_dir()
    pass

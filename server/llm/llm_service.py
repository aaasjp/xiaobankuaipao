import copy
import json
import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Generator, Union
from openai import OpenAI

import logging

logger = logging.getLogger(__name__)

# 默认配置文件路径
# 从 server/llm/llm_service.py 出发，parent.parent 是 server/，所以是 server/config.json
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / 'config.json'


def iter_response(response) -> Generator[str, None, None]:
    """
    处理流式响应，逐步生成累积的文本内容
    
    Args:
        response: OpenAI流式响应对象
        
    Yields:
        str: 累积的响应文本
    """
    res = ""
    try:
        for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content is not None:
                    res += delta.content
                    yield res
    except Exception as e:
        logger.error(f"处理流式响应时出错: {str(e)}")
        raise


class LlmService:
    """
    大语言模型服务类，封装OpenAI兼容API的调用
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        初始化LLM服务
        
        Args:
            config_path: 配置文件路径，默认为项目config目录下的config.json
        """
        # 加载配置文件
        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"配置文件JSON格式错误: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
            raise
        
        # 验证并获取LLM配置
        if 'llm' not in config:
            raise ValueError("配置文件中缺少'llm'配置项")
        
        self.cfg = config['llm']
        self.model = self.cfg.get('model')
        self.base_url = self.cfg.get('base_url')
        self.api_key = self.cfg.get('api_key', 'EMPTY')
        
        # 验证必需配置
        if not self.model:
            raise ValueError("配置文件中'llm.model'不能为空")
        if not self.base_url:
            raise ValueError("配置文件中'llm.base_url'不能为空")
        
        # 初始化OpenAI客户端
        try:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key if self.api_key != 'EMPTY' else None
            )
        except Exception as e:
            logger.error(f"初始化OpenAI客户端失败: {str(e)}")
            raise
        
        # 默认请求参数
        self.default_params: Dict[str, Any] = {
            'temperature': 0.01,
            'max_tokens': 8192,
            #'repetition_penalty': 1.05,
            'extra_body': {
                "chat_template_kwargs": {"enable_thinking": False}
            },
            'stream': False
        }
        
        logger.info(f"LLM服务初始化成功: model={self.model}, base_url={self.base_url}")

    def inference(
        self,
        prompt: str,
        system: str = '',
        history: Optional[List[Tuple[str, str]]] = None,
        stream: bool = True,
        generate_params: Optional[Dict[str, Any]] = None
    ) -> Union[str, Generator[str, None, None]]:
        """
        调用大模型API进行推理
        
        Args:
            prompt: 用户输入的提示词
            system: 系统提示词，默认为空
            history: 历史对话记录，格式为[(question, answer), ...]，默认为None
            stream: 是否使用流式响应，默认为True
            generate_params: 额外的生成参数，会覆盖默认参数，默认为None
            
        Returns:
            如果stream=True，返回生成器；否则返回完整的响应字符串
            
        Raises:
            ValueError: 当prompt为空时
            Exception: 当API调用失败时
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt不能为空")
        
        # 使用None作为默认值，避免可变默认参数问题
        if history is None:
            history = []
        if generate_params is None:
            generate_params = {}
        
        logger.info(f"LLM推理请求 - prompt长度: {len(prompt)}, system长度: {len(system)}, history轮数: {len(history)}")
        
        # 构建请求参数
        req_params = copy.deepcopy(self.default_params)
        req_params.update(generate_params)  # 用户参数会覆盖默认参数
        req_params['stream'] = stream
        req_params['model'] = self.model
        
        # 构建messages
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        
        # 添加历史对话
        for tup in history:
            if len(tup) >= 2:
                q, a = tup[0], tup[1]
                messages.append({"role": "user", "content": str(q)})
                messages.append({"role": "assistant", "content": str(a)})
            else:
                logger.warning(f"历史对话记录格式不正确，已跳过: {tup}")
        
        # 添加当前用户消息
        messages.append({"role": "user", "content": prompt})
        req_params['messages'] = messages
        
        # 记录请求参数（脱敏处理，不记录完整的messages内容）
        log_params = copy.deepcopy(req_params)
        if 'messages' in log_params:
            # 只记录messages的数量和角色信息，不记录完整内容
            messages_summary = [
                {"role": msg.get("role"), "content_length": len(str(msg.get("content", "")))}
                for msg in log_params['messages']
            ]
            log_params['messages'] = messages_summary
        #logger.info(f"请求参数: {json.dumps(log_params, ensure_ascii=False, indent=2)}")
        
        try:
            # 使用OpenAI客户端调用API
            response = self.client.chat.completions.create(**req_params)
            
            if stream:
                # 流式响应
                return iter_response(response)
            else:
                # 非流式响应
                if not response.choices or len(response.choices) == 0:
                    raise ValueError("API响应中没有choices")
                
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("API响应中content为空")
                
                logger.info(f"LLM响应长度: {len(content)}")
                return content
                
        except Exception as e:
            error_msg = f"调用大模型API失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg) from e


class LLMClient:
    """LLM客户端封装 - 基于LlmService的适配器，提供简化的调用接口"""
    
    def __init__(self, llm_service: Optional[LlmService] = None, config_path: Optional[Union[str, Path]] = None):
        """
        初始化LLM客户端
        
        Args:
            llm_service: LlmService实例，如果为None则创建新实例
            config_path: 配置文件路径，如果为None则使用默认的server/config.json
        """
        if llm_service is None:
            # 如果没有提供llm_service，使用默认配置文件
            if config_path is None:
                config_path = DEFAULT_CONFIG_PATH
            else:
                config_path = Path(config_path)
            
            self.llm_service = LlmService(config_path=config_path)
        else:
            self.llm_service = llm_service
    
    def call(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7) -> str:
        """
        调用LLM
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            temperature: 温度参数
        
        Returns:
            LLM响应文本
        """
        try:
            response = self.llm_service.inference(
                prompt=prompt,
                system=system_prompt or "",
                stream=False,
                generate_params={"temperature": temperature}
            )
            return response
        except Exception as e:
            raise Exception(f"LLM调用失败: {str(e)}")
    
    def _fix_latex_escapes(self, text: str) -> str:
        """
        修复JSON字符串中的LaTeX转义序列
        
        将LaTeX转义序列（如 \{, \}, \ldots等）转换为有效的JSON转义序列（\\{, \\}, \\ldots等）
        只在JSON字符串值内部修复，不破坏JSON结构
        
        Args:
            text: 需要修复的JSON字符串
        
        Returns:
            修复后的JSON字符串
        """
        # 使用正则表达式匹配JSON字符串值（在双引号内的内容）
        # 模式：从 " 开始，到未转义的 " 结束
        def fix_string_content(match):
            """修复字符串内容中的LaTeX转义序列"""
            string_content = match.group(1)  # 获取引号内的内容（不包括引号）
            
            # 修复字符串内容中的转义序列
            result = []
            i = 0
            while i < len(string_content):
                if string_content[i] == '\\':
                    if i + 1 < len(string_content):
                        next_char = string_content[i + 1]
                        # 有效的JSON转义序列，保留原样
                        if next_char in ['"', '\\', '/', 'b', 'f', 'n', 'r', 't']:
                            result.append(string_content[i:i+2])
                            i += 2
                        # Unicode转义序列 \uXXXX
                        elif next_char == 'u' and i + 5 < len(string_content):
                            unicode_part = string_content[i+2:i+6]
                            if re.match(r'[0-9a-fA-F]{4}', unicode_part):
                                result.append(string_content[i:i+6])
                                i += 6
                            else:
                                # 不是有效的Unicode转义，转义反斜杠和u
                                result.append('\\\\u')
                                i += 2
                        else:
                            # 其他转义序列（如LaTeX的 \{, \}, \ldots等），需要转义为双反斜杠
                            result.append('\\\\' + next_char)
                            i += 2
                    else:
                        # 反斜杠在字符串末尾，转义它
                        result.append('\\\\')
                        i += 1
                else:
                    result.append(string_content[i])
                    i += 1
            
            return '"' + ''.join(result) + '"'
        
        # 匹配JSON字符串：从 " 开始，到未转义的 " 结束
        # 使用负向前瞻来确保匹配到正确的结束引号
        pattern = r'"((?:[^"\\]|\\.)*)"'
        return re.sub(pattern, fix_string_content, text)
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        解析LLM返回的JSON响应
        
        处理可能被markdown代码块包裹的JSON响应，并修复LaTeX转义序列
        
        Args:
            response: LLM返回的原始响应字符串
        
        Returns:
            解析后的JSON字典
        
        Raises:
            Exception: 当无法解析JSON时
        """
        # 清理响应：去除 markdown 代码块标记
        cleaned_response = response.strip()
        
        # 去除 markdown 代码块标记（```json 或 ```）
        if cleaned_response.startswith("```"):
            # 找到第一个换行符
            first_newline = cleaned_response.find("\n")
            if first_newline != -1:
                cleaned_response = cleaned_response[first_newline + 1:]
            else:
                cleaned_response = cleaned_response[3:]
            
            # 去除结尾的 ```
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
        
        cleaned_response = cleaned_response.strip()
        
        # 尝试提取JSON（使用更精确的正则表达式）
        json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                # 如果解析失败，尝试修复LaTeX转义序列后重新解析
                try:
                    fixed_json = self._fix_latex_escapes(json_str)
                    return json.loads(fixed_json)
                except json.JSONDecodeError:
                    pass
        
        # 如果提取失败，尝试直接解析清理后的响应
        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            # 如果解析失败，尝试修复LaTeX转义序列后重新解析
            try:
                fixed_response = self._fix_latex_escapes(cleaned_response)
                return json.loads(fixed_response)
            except json.JSONDecodeError as e:
                # 只显示前500字符，避免错误信息过长
                #error_preview = response[:500] + "..." if len(response) > 500 else response
                error_preview = response
                raise Exception(f"无法解析LLM返回的JSON: {str(e)}\n原始响应: {error_preview}")
    
    def call_json(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.3) -> Dict[str, Any]:
        """
        调用LLM并解析JSON响应
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            temperature: 温度参数
        
        Returns:
            解析后的JSON字典
        
        Raises:
            Exception: 当无法解析JSON时
        """
        json_prompt = prompt + "\n\nPlease return the result in JSON format, ensuring that the returned content is valid JSON."
        response = self.call(json_prompt, system_prompt, temperature)
        return self._parse_json_response(response)
"""
智能考点提取智能体系统

基于智能体架构的考点提取系统，能够：
1. 根据文档结构和内容动态构建提取规划策略
2. 自动评估和反思提取结果
3. 自动修正抽取结果
4. 支持用户自定义抽取目标和要求
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime

from server.llm.llm_service import LLMClient

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ImportanceLevel(Enum):
    """重要程度等级"""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5


@dataclass
class KeyPoint:
    """考点数据结构"""
    topic: str  # 考点
    description: str  # 考点说明
    importance: int  # 重要程度（1-5）
    importance_reason: str  # 重要原因
    source: str  # 来源（章节/页码/片段）
    source_file: str  # 来源文件
    source_page: Optional[int] = None  # 来源页码（如果有）
    source_snippet: Optional[str] = None  # 来源片段（如果有）


@dataclass
class ExtractionGoal:
    """抽取目标和要求"""
    goal: str  # 抽取目标描述
    requirements: List[str]  # 具体要求列表
    focus_areas: Optional[List[str]] = None  # 重点关注领域
    importance_criteria: Optional[str] = None  # 重要性评估标准


@dataclass
class ExtractionStrategy:
    """动态提取策略"""
    strategy_name: str  # 策略名称
    approach: str  # 提取方法
    file_priority: List[str]  # 文件优先级
    chunk_size: int  # 分块大小
    extraction_focus: List[str]  # 提取重点
    reasoning: str  # 策略推理过程


@dataclass
class ReflectionResult:
    """反思结果"""
    quality_score: float  # 质量评分（0-1）
    issues: List[str]  # 发现的问题
    improvements: List[str]  # 改进建议
    corrections: List[Dict[str, Any]]  # 修正建议
    duplicates: List[Dict[str, Any]] = None  # 重复/相似考点信息


class MaterialReader:
    """资料读取工具 - 直接读取已解析好的markdown文件"""
    
    def __init__(self, parsed_dir: Optional[str] = None):
        """
        初始化资料读取器
        
        Args:
            parsed_dir: 已解析文件的目录路径，如果为None则使用默认的docs_parsed目录
        """
        self.parsed_dir = Path(parsed_dir) if parsed_dir else Path("docs_parsed")
    
    def read_directory(self, dir_path: str) -> Dict[str, str]:
        """
        读取目录中的所有已解析资料文件
        
        Args:
            dir_path: 课程资料目录路径（用于匹配对应的已解析目录）
        
        Returns:
            文件路径到markdown内容的映射
        """
        # 根据课程目录名找到对应的已解析目录
        course_dir_name = Path(dir_path).name
        parsed_course_dir = self.parsed_dir / course_dir_name
        
        if not parsed_course_dir.exists():
            raise FileNotFoundError(
                f"未找到已解析的课程目录: {parsed_course_dir}\n"
                f"请确保课程资料已解析并保存在 {parsed_course_dir}"
            )
        
        # 读取所有markdown文件
        md_files = {}
        for md_file in parsed_course_dir.rglob("*.md"):
            # 计算相对于课程目录的相对路径
            rel_path = md_file.relative_to(parsed_course_dir)
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():  # 只添加非空文件
                        md_files[str(rel_path)] = content
            except Exception as e:
                logger.warning(f"读取文件失败 {md_file}: {e}")
                continue
        
        if not md_files:
            raise ValueError(f"目录中没有找到有效的markdown文件: {parsed_course_dir}")

        return md_files
    
    def read_file(self, file_path: str) -> str:
        """
        读取单个markdown文件
        
        Args:
            file_path: markdown文件路径（可以是绝对路径或相对于parsed_dir的路径）
        
        Returns:
            markdown内容
        """
        file_path_obj = Path(file_path)
        
        # 如果是相对路径，尝试在parsed_dir中查找
        if not file_path_obj.is_absolute():
            file_path_obj = self.parsed_dir / file_path
        
        if not file_path_obj.exists():
            raise FileNotFoundError(f"文件不存在: {file_path_obj}")
        
        if not file_path_obj.suffix == ".md":
            raise ValueError(f"文件不是markdown格式: {file_path_obj}")
        
        with open(file_path_obj, "r", encoding="utf-8") as f:
            return f.read()


class IntelligentPlanner:
    """智能规划器 - 根据文档结构和内容动态构建提取策略"""
    
    def __init__(self, llm_client: LLMClient):
        """
        初始化智能规划器
        
        Args:
            llm_client: LLM客户端
        """
        self.llm_client = llm_client
    
    def analyze_document_structure(self, material_files: Dict[str, str]) -> Dict[str, Any]:
        """
        分析文档结构
        
        Args:
            material_files: 文件字典（文件路径 -> 内容）
        
        Returns:
            文档结构分析结果
        """
        file_list = list(material_files.keys())
        
        # 采样分析文件内容
        sample_files = {}
        for file_path, content in list(material_files.items())[:15]:  # 分析前15个文件
            # 提取文件的前1000字符和后500字符作为样本
            preview = content[:1000] if len(content) > 1000 else content
            if len(content) > 1500:
                preview += "\n\n... [中间内容省略] ...\n\n" + content[-500:]
            sample_files[file_path] = {
                "preview": preview,
                "length": len(content),
                "has_tables": "|" in content[:500] or "<table>" in content[:500].lower(),
                "has_code": "```" in content[:500],
                "has_formulas": "$" in content[:500] or "\\[" in content[:500]
            }
        
        prompt = f"""
Please analyze the structure and characteristics of the following course materials, identify the document organization, content types, and key features.

File list:
{json.dumps(file_list, ensure_ascii=False, indent=2)}

File sample analysis:
{json.dumps(sample_files, ensure_ascii=False, indent=2)}

Please analyze and return in JSON format:
{{
    "document_structure": {{
        "organization_type": "chapter-based/topic-based/mixed",
        "content_types": ["text", "formulas", "tables", "code", "diagrams"],
        "hierarchy_levels": number,
        "file_naming_pattern": "description of file naming pattern"
    }},
    "content_characteristics": {{
        "average_file_length": number,
        "has_formulas": true/false,
        "has_tables": true/false,
        "has_code": true/false,
        "complexity_level": "simple/medium/complex"
    }},
    "chapters": [
        {{
            "name": "chapter name",
            "files": ["file_path1", "file_path2"],
            "description": "chapter description",
            "estimated_keypoints": number
        }}
    ],
    "key_observations": ["observation1", "observation2", "observation3"]
}}
"""
        
        system_prompt = """You are a document structure analysis expert, skilled in identifying and analyzing document organization, content characteristics, and extraction strategies.
Please carefully analyze the document structure, identify chapter organization, content types, and key features."""
        
        try:
            result = self.llm_client.call_json(prompt, system_prompt, temperature=0.3)
            return result
        except Exception as e:
            logger.warning(f"文档结构分析失败: {e}")
            return {
                "document_structure": {"organization_type": "unknown", "content_types": []},
                "content_characteristics": {},
                "chapters": [],
                "key_observations": []
            }
    
    def create_dynamic_strategy(
        self, 
        document_analysis: Dict[str, Any],
        extraction_goal: ExtractionGoal,
        material_files: Dict[str, str]
    ) -> ExtractionStrategy:
        """
        根据文档分析和用户目标动态创建提取策略
        
        Args:
            document_analysis: 文档结构分析结果
            extraction_goal: 用户定义的抽取目标
            material_files: 文件字典
        
        Returns:
            动态提取策略
        """
        file_list = list(material_files.keys())
        
        prompt = f"""
Based on the following information, please create an intelligent keypoint extraction strategy:

Document structure analysis:
{json.dumps(document_analysis, ensure_ascii=False, indent=2)}

User extraction goals:
- Goal: {extraction_goal.goal}
- Requirements: {json.dumps(extraction_goal.requirements, ensure_ascii=False, indent=2)}
- Focus areas: {json.dumps(extraction_goal.focus_areas or [], ensure_ascii=False, indent=2)}
- Importance criteria: {extraction_goal.importance_criteria or "default criteria"}

File list:
{json.dumps(file_list, ensure_ascii=False, indent=2)}

Please create a dynamic and intelligent extraction strategy, considering:
1. Document structure characteristics (chapter-based/topic-based)
2. Content types (formulas, tables, code, etc.)
3. User's specific requirements and focus areas
4. File priority and order
5. Chunk size and extraction methods

Please return in JSON format:
{{
    "strategy_name": "strategy name",
    "approach": "detailed description of extraction method",
    "file_priority": ["file_path1", "file_path2", ...],
    "chunk_size": number (suggested chunk size in characters, at least 8000),
    "extraction_focus": ["focus1", "focus2", ...],
    "reasoning": "reasoning process for strategy creation"
}}
"""
        
        system_prompt = """You are an intelligent strategy planning expert, skilled in creating optimal extraction strategies based on document characteristics and user requirements.
Please carefully analyze the document structure and user requirements to create a dynamic and intelligent extraction strategy."""
        
        try:
            result = self.llm_client.call_json(prompt, system_prompt, temperature=0.5)
            
            return ExtractionStrategy(
                strategy_name=result.get("strategy_name", "Default Strategy"),
                approach=result.get("approach", "Sequential Extraction"),
                file_priority=result.get("file_priority", file_list),
                chunk_size=max(12000, result.get("chunk_size", 8000)),
                extraction_focus=result.get("extraction_focus", []),
                reasoning=result.get("reasoning", "")
            )
        except Exception as e:
            logger.warning(f"策略创建失败，使用默认策略: {e}")
            return ExtractionStrategy(
                strategy_name="Default Strategy",
                approach="Sequential File Extraction",
                file_priority=file_list,
                chunk_size=max(12000, 8000),
                extraction_focus=[],
                reasoning="Using default strategy"
            )


class IntelligentExtractor:
    """智能提取器 - 根据策略动态提取考点"""
    
    def __init__(self, llm_client: LLMClient):
        """
        初始化智能提取器
        
        Args:
            llm_client: LLM客户端
        """
        self.llm_client = llm_client
    
    def extract_keypoints(
        self,
        content: str,
        source_file: str,
        strategy: ExtractionStrategy,
        extraction_goal: ExtractionGoal,
        context: Optional[str] = None
    ) -> List[KeyPoint]:
        """
        根据策略和目标智能提取考点
        
        Args:
            content: 要分析的内容
            source_file: 来源文件路径
            strategy: 提取策略
            extraction_goal: 抽取目标
            context: 上下文信息（可选）
        
        Returns:
            考点列表
        """
        # 根据策略中的chunk_size进行分块
        max_chunk_size = strategy.chunk_size
        chunks = self._split_content(content, max_chunk_size)
        
        if len(chunks) > 1:
            logger.info(f"      内容分块: {len(chunks)} 个片段 (每块约 {max_chunk_size:,} 字符)")
            logger.info(f"      片段处理进度: [0/{len(chunks)}] 0.0%")
        
        all_keypoints = []
        for i, chunk in enumerate(chunks):
            chunk_start_time = time.time()
            chunk_progress = (i + 1) / len(chunks) * 100 if len(chunks) > 1 else 100
            
            if len(chunks) > 1:
                logger.info(f"      处理片段 {i + 1}/{len(chunks)} ({chunk_progress:.1f}%) (长度: {len(chunk):,} 字符)")
            keypoints = self._extract_from_chunk(
                chunk, 
                source_file, 
                i + 1, 
                len(chunks),
                strategy,
                extraction_goal,
                context
            )
            chunk_time = time.time() - chunk_start_time
            all_keypoints.extend(keypoints)
            
            if len(chunks) > 1:
                if keypoints:
                    logger.info(f"        ✓ 片段 {i + 1} 提取到 {len(keypoints)} 个考点 (耗时: {chunk_time:.2f}秒)")
                else:
                    logger.info(f"        ⚠ 片段 {i + 1} 未提取到考点 (耗时: {chunk_time:.2f}秒)")
                logger.info(f"        片段处理进度: [{i + 1}/{len(chunks)}] {chunk_progress:.1f}%")
        
        return all_keypoints
    
    def _split_content(self, content: str, max_chunk_size: int) -> List[str]:
        """智能分块内容"""
        if len(content) <= max_chunk_size:
            return [content]
        
        chunks = []
        # 按段落分割
        paragraphs = content.split('\n\n')
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _extract_from_chunk(
        self,
        chunk: str,
        source_file: str,
        chunk_num: int,
        total_chunks: int,
        strategy: ExtractionStrategy,
        extraction_goal: ExtractionGoal,
        context: Optional[str]
    ) -> List[KeyPoint]:
        """从单个chunk中智能提取考点"""
        # 根据chunk长度和策略计算应该提取的考点数量
        chunk_length = len(chunk)
        base_keypoints = max(1, chunk_length // 2000)
        max_keypoints = min(4, base_keypoints)  # 最多4个
        
        # 限制内容长度
        content_preview = chunk[:min(6000, len(chunk))]
        context_info = f"\nContext information: {context}" if context else ""
        
        focus_areas_text = ""
        if extraction_goal.focus_areas:
            focus_areas_text = f"\nFocus areas: {', '.join(extraction_goal.focus_areas)}"
        
        prompt = f"""
Please extract important keypoints from the course content based on the following strategy and goals.

Extraction strategy:
- Strategy name: {strategy.strategy_name}
- Extraction method: {strategy.approach}
- Extraction focus: {', '.join(strategy.extraction_focus) if strategy.extraction_focus else 'no specific focus'}

User extraction goals:
- Goal: {extraction_goal.goal}
- Requirements: {json.dumps(extraction_goal.requirements, ensure_ascii=False, indent=2)}
{focus_areas_text}
- Importance criteria: {extraction_goal.importance_criteria or 'default criteria'}

Course content (chunk {chunk_num}/{total_chunks}):
{content_preview}
{context_info}

Please extract up to {max_keypoints} keypoints, each keypoint should include:
1. Keypoint name (concise and clear)
2. Keypoint description (detailed explanation)
3. Importance level (1-5, where 5 is most important)
4. Importance reason (why it's important)
5. Source snippet (original text quote)

Please return in JSON format:
{{
    "keypoints": [
        {{
            "topic": "keypoint name",
            "description": "keypoint description",
            "importance": 3,
            "importance_reason": "importance reason",
            "source_snippet": "source snippet (original text quote)"
        }}
    ]
}}
"""
        
        system_prompt = """You are an education expert, skilled in identifying and extracting important keypoints from course materials based on strategies and goals.
Please focus on:
- Core concepts and definitions
- Important formulas and theorems
- Key methods and techniques
- Common exam points and error-prone areas
- Exam-focused content

Importance level evaluation criteria:
- Level 5: Core concepts, exam essentials, fundamentals of fundamentals
- Level 4: Important knowledge points, high-frequency exam questions
- Level 3: Generally important, needs to be mastered
- Level 2: Understanding is sufficient
- Level 1: Supplementary knowledge, not essential"""
        
        try:
            result = self.llm_client.call_json(prompt, system_prompt, temperature=0.3)
            keypoints_data = result.get("keypoints", [])
            
            # 如果返回的考点数量超过限制，按重要性排序后只取前max_keypoints个
            if len(keypoints_data) > max_keypoints:
                keypoints_data.sort(key=lambda x: x.get("importance", 0), reverse=True)
                keypoints_data = keypoints_data[:max_keypoints]
                logger.warning(f"片段 {chunk_num} 提取的考点数量超过限制，已截取前{max_keypoints}个最重要的考点")
            
            keypoints = []
            for kp_data in keypoints_data:
                keypoint = KeyPoint(
                    topic=kp_data.get("topic", ""),
                    description=kp_data.get("description", ""),
                    importance=kp_data.get("importance", 3),
                    importance_reason=kp_data.get("importance_reason", ""),
                    source=f"{source_file} (chunk {chunk_num}/{total_chunks})",
                    source_file=source_file,
                    source_snippet=kp_data.get("source_snippet")
                )
                keypoints.append(keypoint)
            
            return keypoints
        except Exception as e:
            logger.error(f"提取考点失败: {e}")
            return []


class ReflectionEvaluator:
    """反思评估器 - 自动评估、反思和修正提取结果"""
    
    def __init__(self, llm_client: LLMClient):
        """
        初始化反思评估器
        
        Args:
            llm_client: LLM客户端
        """
        self.llm_client = llm_client
    
    def reflect_and_evaluate(
        self,
        keypoints: List[KeyPoint],
        extraction_goal: ExtractionGoal,
        document_analysis: Dict[str, Any],
        strategy: ExtractionStrategy
    ) -> ReflectionResult:
        """
        反思和评估提取结果
        
        Args:
            keypoints: 提取的考点列表
            extraction_goal: 抽取目标
            document_analysis: 文档分析结果
            strategy: 使用的策略
        
        Returns:
            反思结果
        """
        if not keypoints:
            return ReflectionResult(
                quality_score=0.0,
                issues=["No keypoints extracted"],
                improvements=["Check document content and extraction strategy"],
                corrections=[],
                duplicates=[]
            )
        
        keypoints_json = [asdict(kp) for kp in keypoints]
        
        prompt = f"""
Please evaluate the quality of the following keypoint extraction results and identify issues and improvement directions.

User extraction goals:
- Goal: {extraction_goal.goal}
- Requirements: {json.dumps(extraction_goal.requirements, ensure_ascii=False, indent=2)}
- Focus areas: {json.dumps(extraction_goal.focus_areas or [], ensure_ascii=False, indent=2)}

Strategy used:
- Strategy name: {strategy.strategy_name}
- Extraction method: {strategy.approach}

Extracted keypoints list:
{json.dumps(keypoints_json, ensure_ascii=False, indent=2)}

Please evaluate and return in JSON format:
{{
    "quality_score": number between 0.0-1.0 (quality score),
    "issues": [
        "issue1",
        "issue2"
    ],
    "improvements": [
        "improvement suggestion1",
        "improvement suggestion2"
    ],
    "corrections": [
        {{
            "keypoint_index": number (keypoint index, starting from 0),
            "issue": "identified issue",
            "corrected_topic": "corrected keypoint name (if needed)",
            "corrected_description": "corrected description (if needed)",
            "corrected_importance": number (corrected importance level, if needed),
            "corrected_reason": "correction reason"
        }}
    ],
    "duplicates": [
        {{
            "keypoint_indices": [number1, number2, ...] (indices of duplicate/similar keypoints),
            "similarity_reason": "reason why these keypoints are similar or duplicate",
            "suggested_merge": {{
                "topic": "merged topic name",
                "description": "merged description",
                "importance": number (highest importance among duplicates),
                "importance_reason": "merged importance reason"
            }}
        }}
    ]
}}

Evaluation criteria:
1. Whether keypoints meet user goals and requirements
2. Whether importance level assessment is reasonable
3. Whether keypoint descriptions are clear and accurate
4. Whether there are duplicate or overly similar keypoints (IMPORTANT: identify and suggest merges)
5. Whether important content is missing
6. Whether source information is accurate

IMPORTANT: Carefully identify duplicate or similar keypoints. For each group of similar/duplicate keypoints, provide:
- The indices of all similar keypoints
- The reason why they are similar
- A suggested merged keypoint that combines the best information from all duplicates
"""
        
        system_prompt = """You are a quality evaluation expert, skilled in evaluating the quality of keypoint extraction results, identifying issues and proposing improvement suggestions.
Please carefully evaluate the extraction results, identify all issues and propose specific improvement suggestions and correction plans."""
        
        try:
            result = self.llm_client.call_json(prompt, system_prompt, temperature=0.4)
            
            return ReflectionResult(
                quality_score=result.get("quality_score", 0.5),
                issues=result.get("issues", []),
                improvements=result.get("improvements", []),
                corrections=result.get("corrections", []),
                duplicates=result.get("duplicates", [])  # 添加重复考点信息
            )
        except Exception as e:
            logger.warning(f"反思评估失败: {e}")
            return ReflectionResult(
                quality_score=0.5,
                issues=["Evaluation process error"],
                improvements=["Re-evaluate"],
                corrections=[],
                duplicates=[]
            )
    
    def apply_corrections(
        self,
        keypoints: List[KeyPoint],
        corrections: List[Dict[str, Any]]
    ) -> List[KeyPoint]:
        """
        应用修正建议
        
        Args:
            keypoints: 原始考点列表
            corrections: 修正建议列表
        
        Returns:
            修正后的考点列表
        """
        corrected_keypoints = keypoints.copy()
        
        for correction in corrections:
            idx = correction.get("keypoint_index")
            if idx is None or idx < 0 or idx >= len(corrected_keypoints):
                continue
            
            kp = corrected_keypoints[idx]
            
            # 应用修正
            if "corrected_topic" in correction and correction["corrected_topic"]:
                kp.topic = correction["corrected_topic"]
            
            if "corrected_description" in correction and correction["corrected_description"]:
                kp.description = correction["corrected_description"]
            
            if "corrected_importance" in correction:
                new_importance = correction["corrected_importance"]
                if isinstance(new_importance, int) and 1 <= new_importance <= 5:
                    kp.importance = new_importance
            
            if "corrected_reason" in correction and correction["corrected_reason"]:
                kp.importance_reason = correction["corrected_reason"]
        
        return corrected_keypoints
    
    def refine_keypoints(
        self,
        keypoints: List[KeyPoint],
        reflection_result: ReflectionResult,
        extraction_goal: ExtractionGoal
    ) -> List[KeyPoint]:
        """
        根据反思结果优化考点
        
        Args:
            keypoints: 原始考点列表
            reflection_result: 反思结果
            extraction_goal: 抽取目标
        
        Returns:
            优化后的考点列表
        """
        # 应用修正
        refined_keypoints = self.apply_corrections(keypoints, reflection_result.corrections)
        
        # 处理LLM识别的重复考点
        if reflection_result.duplicates:
            refined_keypoints = self._merge_duplicates(refined_keypoints, reflection_result.duplicates)
        
        # 合并相似考点（使用算法检测）
        refined_keypoints = self._merge_similar_keypoints(refined_keypoints)
        
        # 如果质量评分较低，使用LLM进一步优化
        if reflection_result.quality_score < 0.6:
            refined_keypoints = self._llm_refine(refined_keypoints, extraction_goal, reflection_result)
        
        return refined_keypoints
    
    def _merge_duplicates(
        self,
        keypoints: List[KeyPoint],
        duplicates: List[Dict[str, Any]]
    ) -> List[KeyPoint]:
        """
        合并LLM识别的重复考点
        
        Args:
            keypoints: 考点列表
            duplicates: LLM识别的重复考点信息
        
        Returns:
            合并后的考点列表
        """
        # 创建索引到考点的映射
        kp_dict = {i: kp for i, kp in enumerate(keypoints)}
        indices_to_remove = set()
        merged_keypoints = []
        
        for dup_info in duplicates:
            indices = dup_info.get("keypoint_indices", [])
            if not indices or len(indices) < 2:
                continue
            
            # 验证索引有效性
            valid_indices = [idx for idx in indices if 0 <= idx < len(keypoints)]
            if len(valid_indices) < 2:
                continue
            
            # 获取要合并的考点
            kps_to_merge = [kp_dict[idx] for idx in valid_indices]
            
            # 使用LLM建议的合并结果，如果没有则自动合并
            suggested_merge = dup_info.get("suggested_merge")
            if suggested_merge:
                merged_kp = KeyPoint(
                    topic=suggested_merge.get("topic", kps_to_merge[0].topic),
                    description=suggested_merge.get("description", kps_to_merge[0].description),
                    importance=suggested_merge.get("importance", max(kp.importance for kp in kps_to_merge)),
                    importance_reason=suggested_merge.get("importance_reason", kps_to_merge[0].importance_reason),
                    source=kps_to_merge[0].source,
                    source_file=kps_to_merge[0].source_file,
                    source_page=kps_to_merge[0].source_page,
                    source_snippet=kps_to_merge[0].source_snippet
                )
            else:
                # 自动合并：选择重要程度最高的，合并描述
                best_kp = max(kps_to_merge, key=lambda x: x.importance)
                descriptions = [kp.description for kp in kps_to_merge if kp.description]
                merged_description = best_kp.description
                if len(descriptions) > 1:
                    # 合并描述，去重
                    unique_descriptions = []
                    for desc in descriptions:
                        if desc not in unique_descriptions and desc != best_kp.description:
                            unique_descriptions.append(desc)
                    if unique_descriptions:
                        merged_description = best_kp.description + " " + " ".join(unique_descriptions[:2])
                
                merged_kp = KeyPoint(
                    topic=best_kp.topic,
                    description=merged_description[:500],  # 限制长度
                    importance=best_kp.importance,
                    importance_reason=best_kp.importance_reason,
                    source=best_kp.source,
                    source_file=best_kp.source_file,
                    source_page=best_kp.source_page,
                    source_snippet=best_kp.source_snippet
                )
            
            merged_keypoints.append(merged_kp)
            indices_to_remove.update(valid_indices)
            
            logger.info(f"合并 {len(valid_indices)} 个重复考点: {[kp.topic for kp in kps_to_merge]}")
        
        # 添加未合并的考点
        for i, kp in enumerate(keypoints):
            if i not in indices_to_remove:
                merged_keypoints.append(kp)
        
        return merged_keypoints
    
    def _merge_similar_keypoints(self, keypoints: List[KeyPoint]) -> List[KeyPoint]:
        """
        合并相似的考点（使用算法检测）
        使用更智能的相似度检测方法
        """
        if not keypoints:
            return []
        
        merged = []
        used_indices = set()
        
        for i, kp1 in enumerate(keypoints):
            if i in used_indices:
                continue
            
            # 查找相似的考点
            similar_group = [kp1]
            similar_indices = [i]
            
            for j, kp2 in enumerate(keypoints[i+1:], start=i+1):
                if j in used_indices:
                    continue
                
                if self._are_similar(kp1, kp2):
                    similar_group.append(kp2)
                    similar_indices.append(j)
            
            # 如果找到相似的考点，合并它们
            if len(similar_group) > 1:
                # 选择重要程度最高的，合并描述
                best_kp = max(similar_group, key=lambda x: x.importance)
                descriptions = [kp.description for kp in similar_group if kp.description and kp.description != best_kp.description]
                
                merged_description = best_kp.description
                if descriptions:
                    # 合并前2个不同的描述
                    unique_descriptions = list(dict.fromkeys(descriptions))[:2]
                    merged_description = best_kp.description + " " + " ".join(unique_descriptions)
                    merged_description = merged_description[:500]  # 限制长度
                
                merged_kp = KeyPoint(
                    topic=best_kp.topic,
                    description=merged_description,
                    importance=best_kp.importance,
                    importance_reason=best_kp.importance_reason,
                    source=best_kp.source,
                    source_file=best_kp.source_file,
                    source_page=best_kp.source_page,
                    source_snippet=best_kp.source_snippet
                )
                merged.append(merged_kp)
                used_indices.update(similar_indices)
                
                if len(similar_group) > 1:
                    logger.debug(f"算法检测到 {len(similar_group)} 个相似考点并合并: {[kp.topic for kp in similar_group]}")
            else:
                merged.append(kp1)
                used_indices.add(i)
        
        return merged
    
    def _are_similar(self, kp1: KeyPoint, kp2: KeyPoint, similarity_threshold: float = 0.7) -> bool:
        """
        判断两个考点是否相似
        
        Args:
            kp1: 考点1
            kp2: 考点2
            similarity_threshold: 相似度阈值（0-1）
        
        Returns:
            是否相似
        """
        # 方法1: 主题名称相似度
        topic1 = kp1.topic.lower().strip()
        topic2 = kp2.topic.lower().strip()
        
        # 完全相同的主题
        if topic1 == topic2:
            return True
        
        # 一个主题包含另一个
        if topic1 in topic2 or topic2 in topic1:
            return True
        
        # 计算词汇重叠度
        words1 = set(topic1.split())
        words2 = set(topic2.split())
        
        if not words1 or not words2:
            return False
        
        # 计算Jaccard相似度
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        jaccard_similarity = intersection / union if union > 0 else 0
        
        if jaccard_similarity >= similarity_threshold:
            return True
        
        # 方法2: 描述相似度（如果主题相似度不够）
        if jaccard_similarity >= 0.5:  # 中等相似的主题
            desc1 = kp1.description.lower().strip()[:200]  # 只比较前200字符
            desc2 = kp2.description.lower().strip()[:200]
            
            if desc1 and desc2:
                desc_words1 = set(desc1.split())
                desc_words2 = set(desc2.split())
                
                if desc_words1 and desc_words2:
                    desc_intersection = len(desc_words1 & desc_words2)
                    desc_union = len(desc_words1 | desc_words2)
                    desc_similarity = desc_intersection / desc_union if desc_union > 0 else 0
                    
                    # 如果描述相似度也高，认为是相似的
                    if desc_similarity >= 0.6:
                        return True
        
        return False
    
    def _llm_refine(
        self,
        keypoints: List[KeyPoint],
        extraction_goal: ExtractionGoal,
        reflection_result: ReflectionResult
    ) -> List[KeyPoint]:
        """使用LLM进一步优化考点"""
        keypoints_json = [asdict(kp) for kp in keypoints]
        
        prompt = f"""
Please optimize the keypoints list based on the following reflection results.

User extraction goals:
- Goal: {extraction_goal.goal}
- Requirements: {json.dumps(extraction_goal.requirements, ensure_ascii=False, indent=2)}

Reflection results:
- Quality score: {reflection_result.quality_score}
- Issues identified: {json.dumps(reflection_result.issues, ensure_ascii=False, indent=2)}
- Improvement suggestions: {json.dumps(reflection_result.improvements, ensure_ascii=False, indent=2)}

Current keypoints list:
{json.dumps(keypoints_json, ensure_ascii=False, indent=2)}

Please optimize the keypoints list and return in JSON format (same format as input):
{{
    "keypoints": [
        {{
            "topic": "keypoint name",
            "description": "keypoint description",
            "importance": 3,
            "importance_reason": "importance reason",
            "source": "source",
            "source_file": "source file",
            "source_snippet": "source snippet"
        }}
    ]
}}
"""
        
        system_prompt = """You are a keypoint optimization expert, skilled in optimizing keypoint lists based on reflection results.
Please optimize the keypoints list based on the issues and improvement suggestions in the reflection results to ensure quality improvement."""
        
        try:
            result = self.llm_client.call_json(prompt, system_prompt, temperature=0.3)
            refined_data = result.get("keypoints", keypoints_json)
            
            refined_keypoints = []
            for kp_data in refined_data:
                keypoint = KeyPoint(
                    topic=kp_data.get("topic", ""),
                    description=kp_data.get("description", ""),
                    importance=kp_data.get("importance", 3),
                    importance_reason=kp_data.get("importance_reason", ""),
                    source=kp_data.get("source", ""),
                    source_file=kp_data.get("source_file", ""),
                    source_page=kp_data.get("source_page"),
                    source_snippet=kp_data.get("source_snippet")
                )
                refined_keypoints.append(keypoint)
            
            return refined_keypoints
        except Exception as e:
            logger.warning(f"LLM优化失败，使用原始结果: {e}")
            return keypoints


class IntelligentKeyPointAgent:
    """智能考点提取智能体 - 主智能体类"""
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        material_reader: Optional[MaterialReader] = None,
        parsed_dir: Optional[str] = None
    ):
        """
        初始化智能体
        
        Args:
            llm_client: LLM客户端，如果为None则创建新实例
            material_reader: 资料读取器，如果为None则创建新实例
            parsed_dir: 已解析文件的目录路径
        """
        self.llm_client = llm_client or LLMClient()
        self.material_reader = material_reader or MaterialReader(parsed_dir=parsed_dir)
        
        # 初始化各个组件
        self.planner = IntelligentPlanner(self.llm_client)
        self.extractor = IntelligentExtractor(self.llm_client)
        self.evaluator = ReflectionEvaluator(self.llm_client)
    
    def extract(
        self,
        course_dir: str,
        extraction_goal: ExtractionGoal,
        output_path: Optional[str] = None,
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        执行智能提取流程
        
        Args:
            course_dir: 课程资料目录路径
            extraction_goal: 用户定义的抽取目标和要求
            output_path: 输出文件路径（可选）
            max_iterations: 最大迭代次数（用于反思和修正）
        
        Returns:
            提取结果字典
        """
        # 记录开始时间
        start_time = time.time()
        total_steps = 7  # 总步骤数
        
        logger.info("=" * 60)
        logger.info("开始智能提取课程考点")
        logger.info("=" * 60)
        logger.info(f"课程目录: {course_dir}")
        logger.info(f"抽取目标: {extraction_goal.goal}")
        logger.info(f"具体要求: {', '.join(extraction_goal.requirements)}")
        if extraction_goal.focus_areas:
            logger.info(f"重点关注: {', '.join(extraction_goal.focus_areas)}")
        if extraction_goal.importance_criteria:
            logger.info(f"重要性标准: {extraction_goal.importance_criteria}")
        logger.info(f"最大迭代次数: {max_iterations}")
        if output_path:
            logger.info(f"输出路径: {output_path}")
        logger.info(f"总体进度: [0/{total_steps}] 0.0%")
        
        # 步骤1: 读取资料
        step_start_time = time.time()
        logger.info("-" * 60)
        logger.info(f"步骤1/7: 读取课程资料 [进度: {1/total_steps*100:.1f}%]")
        logger.info("-" * 60)
        material_files = self.material_reader.read_directory(course_dir)
        step_time = time.time() - step_start_time
        logger.info(f"✓ 已读取 {len(material_files)} 个文件 (耗时: {step_time:.2f}秒)")
        
        # 统计文件信息
        total_content_length = sum(len(content) for content in material_files.values())
        avg_file_length = total_content_length / len(material_files) if material_files else 0
        logger.info(f"  总内容长度: {total_content_length:,} 字符")
        logger.info(f"  平均文件长度: {avg_file_length:,.0f} 字符")
        
        # 步骤2: 分析文档结构
        step_start_time = time.time()
        logger.info("-" * 60)
        logger.info(f"步骤2/7: 分析文档结构 [进度: {2/total_steps*100:.1f}%]")
        logger.info("-" * 60)
        logger.info("正在分析文档结构和内容特征...")
        document_analysis = self.planner.analyze_document_structure(material_files)
        step_time = time.time() - step_start_time
        
        # 记录文档结构分析结果
        doc_structure = document_analysis.get("document_structure", {})
        content_chars = document_analysis.get("content_characteristics", {})
        chapters = document_analysis.get("chapters", [])
        observations = document_analysis.get("key_observations", [])
        
        logger.info(f"✓ 文档结构分析完成 (耗时: {step_time:.2f}秒)")
        logger.info(f"  组织类型: {doc_structure.get('organization_type', '未知')}")
        logger.info(f"  内容类型: {', '.join(doc_structure.get('content_types', []))}")
        logger.info(f"  层级数: {doc_structure.get('hierarchy_levels', '未知')}")
        logger.info(f"  识别章节数: {len(chapters)}")
        
        if content_chars:
            logger.info(f"  平均文件长度: {content_chars.get('average_file_length', 0):,.0f} 字符")
            logger.info(f"  包含公式: {'是' if content_chars.get('has_formulas') else '否'}")
            logger.info(f"  包含表格: {'是' if content_chars.get('has_tables') else '否'}")
            logger.info(f"  包含代码: {'是' if content_chars.get('has_code') else '否'}")
            logger.info(f"  复杂度: {content_chars.get('complexity_level', '未知')}")
        
        if observations:
            logger.info(f"  关键观察:")
            for obs in observations[:5]:  # 只显示前5个观察
                logger.info(f"    - {obs}")
        
        if chapters:
            logger.info(f"  章节列表:")
            for i, chapter in enumerate(chapters[:10], 1):  # 只显示前10个章节
                logger.info(f"    {i}. {chapter.get('name', '未知')} ({len(chapter.get('files', []))} 个文件)")
        
        # 保存中间结果：文档结构分析
        if output_path:
            self._save_intermediate_results(
                output_path,
                "step2_document_analysis",
                {
                    "step": "文档结构分析",
                    "document_analysis": document_analysis,
                    "file_count": len(material_files),
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # 步骤3: 动态创建提取策略
        step_start_time = time.time()
        logger.info("-" * 60)
        logger.info(f"步骤3/7: 动态创建提取策略 [进度: {3/total_steps*100:.1f}%]")
        logger.info("-" * 60)
        logger.info("正在根据文档分析和用户目标创建提取策略...")
        strategy = self.planner.create_dynamic_strategy(
            document_analysis,
            extraction_goal,
            material_files
        )
        step_time = time.time() - step_start_time
        logger.info(f"✓ 策略创建完成 (耗时: {step_time:.2f}秒)")
        logger.info(f"  策略名称: {strategy.strategy_name}")
        logger.info(f"  提取方法: {strategy.approach}")
        logger.info(f"  分块大小: {strategy.chunk_size:,} 字符")
        logger.info(f"  文件优先级: {len(strategy.file_priority)} 个文件")
        if strategy.extraction_focus:
            logger.info(f"  提取重点: {', '.join(strategy.extraction_focus)}")
        logger.info(f"  策略推理: {strategy.reasoning}")
        
        # 保存中间结果：提取策略
        if output_path:
            self._save_intermediate_results(
                output_path,
                "step3_extraction_strategy",
                {
                    "step": "提取策略创建",
                    "strategy": asdict(strategy),
                    "extraction_goal": asdict(extraction_goal),
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # 步骤4: 根据策略提取考点
        step_start_time = time.time()
        logger.info("-" * 60)
        logger.info(f"步骤4/7: 根据策略提取考点 [进度: {4/total_steps*100:.1f}%]")
        logger.info("-" * 60)
        all_keypoints = []
        total_files = len(strategy.file_priority)
        processed_files = []
        failed_files = []
        
        logger.info(f"开始处理 {total_files} 个文件...")
        logger.info(f"文件处理进度: [0/{total_files}] 0.0%")
        
        for idx, file_path in enumerate(strategy.file_priority, 1):
            file_start_time = time.time()
            file_progress = (idx - 1) / total_files * 100
            
            if file_path not in material_files:
                logger.warning(f"  [{idx}/{total_files}] ({file_progress:.1f}%) ⚠ 文件不在材料列表中，跳过: {file_path}")
                continue
            
            logger.info(f"  [{idx}/{total_files}] ({file_progress:.1f}%) 处理文件: {file_path}")
            try:
                content = material_files[file_path]
                content_length = len(content)
                logger.info(f"    文件长度: {content_length:,} 字符")
                
                keypoints = self.extractor.extract_keypoints(
                    content,
                    file_path,
                    strategy,
                    extraction_goal
                )
                
                file_time = time.time() - file_start_time
                all_keypoints.extend(keypoints)
                processed_files.append(file_path)
                
                # 记录提取的考点详情
                if keypoints:
                    importance_dist = {}
                    for kp in keypoints:
                        importance_dist[kp.importance] = importance_dist.get(kp.importance, 0) + 1
                    
                    logger.info(f"    ✓ 提取到 {len(keypoints)} 个考点 (耗时: {file_time:.2f}秒)")
                    logger.info(f"      重要程度分布: {dict(sorted(importance_dist.items()))}")
                    
                    # 显示前3个最重要的考点
                    sorted_kps = sorted(keypoints, key=lambda x: x.importance, reverse=True)
                    for i, kp in enumerate(sorted_kps[:3], 1):
                        logger.info(f"      {i}. [{kp.importance}级] {kp.topic[:50]}...")
                else:
                    logger.warning(f"    ⚠ 未提取到考点 (耗时: {file_time:.2f}秒)")
                
                # 显示文件处理进度
                current_progress = idx / total_files * 100
                elapsed_time = time.time() - step_start_time
                avg_time_per_file = elapsed_time / idx if idx > 0 else 0
                remaining_files = total_files - idx
                estimated_remaining_time = avg_time_per_file * remaining_files
                
                logger.info(f"    文件处理进度: [{idx}/{total_files}] {current_progress:.1f}% | "
                          f"已用: {elapsed_time:.1f}秒 | "
                          f"预计剩余: {estimated_remaining_time:.1f}秒")
                    
            except Exception as e:
                file_time = time.time() - file_start_time
                logger.error(f"    ✗ 提取失败 (耗时: {file_time:.2f}秒): {e}")
                failed_files.append(file_path)
                processed_files.append(file_path)
                continue
        
        step_time = time.time() - step_start_time
        
        logger.info("-" * 60)
        logger.info(f"✓ 文件处理完成 (总耗时: {step_time:.2f}秒, 平均: {step_time/total_files:.2f}秒/文件)")
        logger.info(f"  成功处理: {len(processed_files) - len(failed_files)} 个文件")
        logger.info(f"  失败文件: {len(failed_files)} 个")
        if failed_files:
            logger.warning(f"  失败文件列表: {', '.join(failed_files[:5])}")
        
        # 统计初始提取结果
        total_kp = len(all_keypoints)
        if total_kp > 0:
            importance_stats = {}
            for kp in all_keypoints:
                importance_stats[kp.importance] = importance_stats.get(kp.importance, 0) + 1
            avg_importance = sum(kp.importance for kp in all_keypoints) / total_kp
            
            logger.info(f"  共提取到 {total_kp} 个考点")
            logger.info(f"  平均重要程度: {avg_importance:.2f}")
            logger.info(f"  重要程度分布:")
            for level in sorted(importance_stats.keys(), reverse=True):
                count = importance_stats[level]
                percentage = (count / total_kp) * 100
                logger.info(f"    {level}级: {count} 个 ({percentage:.1f}%)")
        else:
            logger.warning(f"  ⚠ 未提取到任何考点")
        
        # 保存中间结果：初始提取的考点
        if output_path:
            self._save_intermediate_results(
                output_path,
                "step4_initial_extraction",
                {
                    "step": "初始考点提取",
                    "total_keypoints": len(all_keypoints),
                    "processed_files": processed_files,
                    "keypoints": [asdict(kp) for kp in all_keypoints],
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # 步骤5: 反思、评估和修正（迭代）
        step_start_time = time.time()
        logger.info("-" * 60)
        logger.info(f"步骤5/7: 反思、评估和修正 [进度: {5/total_steps*100:.1f}%]")
        logger.info("-" * 60)
        current_keypoints = all_keypoints
        
        for iteration in range(max_iterations):
            iter_start_time = time.time()
            iter_progress = (iteration + 1) / max_iterations * 100
            logger.info(f"\n  迭代 {iteration + 1}/{max_iterations} ({iter_progress:.1f}%): 评估和反思")
            logger.info(f"    当前考点数: {len(current_keypoints)}")
            logger.info(f"    正在评估考点质量...")
            
            reflection_result = self.evaluator.reflect_and_evaluate(
                current_keypoints,
                extraction_goal,
                document_analysis,
                strategy
            )
            
            iter_time = time.time() - iter_start_time
            logger.info(f"    ✓ 评估完成 (耗时: {iter_time:.2f}秒)")
            logger.info(f"      质量评分: {reflection_result.quality_score:.2f} ({'优秀' if reflection_result.quality_score >= 0.8 else '良好' if reflection_result.quality_score >= 0.6 else '需改进'})")
            
            if reflection_result.issues:
                logger.info(f"      发现问题 ({len(reflection_result.issues)} 个):")
                for i, issue in enumerate(reflection_result.issues[:5], 1):  # 只显示前5个问题
                    logger.warning(f"        {i}. {issue}")
                if len(reflection_result.issues) > 5:
                    logger.info(f"        ... 还有 {len(reflection_result.issues) - 5} 个问题")
            
            if reflection_result.improvements:
                logger.info(f"      改进建议 ({len(reflection_result.improvements)} 个):")
                for i, improvement in enumerate(reflection_result.improvements[:5], 1):  # 只显示前5个建议
                    logger.info(f"        {i}. {improvement}")
                if len(reflection_result.improvements) > 5:
                    logger.info(f"        ... 还有 {len(reflection_result.improvements) - 5} 个建议")
            
            if reflection_result.corrections:
                logger.info(f"      修正建议 ({len(reflection_result.corrections)} 个):")
                for i, correction in enumerate(reflection_result.corrections[:5], 1):  # 只显示前5个修正
                    idx = correction.get("keypoint_index", "?")
                    issue = correction.get("issue", "")
                    logger.info(f"        {i}. 考点 #{idx}: {issue}")
                if len(reflection_result.corrections) > 5:
                    logger.info(f"        ... 还有 {len(reflection_result.corrections) - 5} 个修正建议")
            
            if reflection_result.duplicates:
                logger.info(f"      重复/相似考点 ({len(reflection_result.duplicates)} 组):")
                total_duplicates = sum(len(dup.get("keypoint_indices", [])) for dup in reflection_result.duplicates)
                logger.info(f"        共识别到 {total_duplicates} 个重复/相似考点需要合并")
                for i, dup in enumerate(reflection_result.duplicates[:3], 1):  # 只显示前3组
                    indices = dup.get("keypoint_indices", [])
                    reason = dup.get("similarity_reason", "")
                    logger.info(f"        {i}. 考点 {indices}: {reason}")
                if len(reflection_result.duplicates) > 3:
                    logger.info(f"        ... 还有 {len(reflection_result.duplicates) - 3} 组重复考点")
            
            # 应用修正和优化
            refine_start_time = time.time()
            logger.info(f"    正在应用修正和优化...")
            refined_keypoints = self.evaluator.refine_keypoints(
                current_keypoints,
                reflection_result,
                extraction_goal
            )
            refine_time = time.time() - refine_start_time
            
            logger.info(f"    ✓ 优化完成 (耗时: {refine_time:.2f}秒)")
            logger.info(f"      优化前: {len(current_keypoints)} 个考点")
            logger.info(f"      优化后: {len(refined_keypoints)} 个考点")
            if len(refined_keypoints) != len(current_keypoints):
                diff = len(refined_keypoints) - len(current_keypoints)
                logger.info(f"      变化: {diff:+d} 个考点")
            
            # 显示迭代进度
            total_iter_time = time.time() - iter_start_time
            logger.info(f"    迭代总耗时: {total_iter_time:.2f}秒")
            
            # 保存中间结果：每次迭代的反思评估结果
            if output_path:
                self._save_intermediate_results(
                    output_path,
                    f"step5_iteration_{iteration + 1}",
                    {
                        "step": f"反思评估迭代 {iteration + 1}",
                        "iteration": iteration + 1,
                        "quality_score": reflection_result.quality_score,
                        "issues": reflection_result.issues,
                        "improvements": reflection_result.improvements,
                        "corrections": reflection_result.corrections,
                        "duplicates": reflection_result.duplicates if reflection_result.duplicates else [],
                        "keypoints_before": len(current_keypoints),
                        "keypoints_after": len(refined_keypoints),
                        "refined_keypoints": [asdict(kp) for kp in refined_keypoints],
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            # 如果质量评分足够高，或者没有新的修正，可以提前结束
            if reflection_result.quality_score >= 0.8 and len(reflection_result.corrections) == 0:
                logger.info("    质量评分已达到要求，提前结束迭代")
                current_keypoints = refined_keypoints
                break
            
            current_keypoints = refined_keypoints
        
        step_time = time.time() - step_start_time
        logger.info(f"\n✓ 反思评估完成 (总耗时: {step_time:.2f}秒)")
        
        # 步骤6: 整理结果
        step_start_time = time.time()
        logger.info("-" * 60)
        logger.info(f"步骤6/7: 整理结果 [进度: {6/total_steps*100:.1f}%]")
        logger.info("-" * 60)
        logger.info("正在整理和统计提取结果...")
        results = self._organize_results(
            current_keypoints,
            document_analysis,
            strategy,
            extraction_goal,
            processed_files
        )
        step_time = time.time() - step_start_time
        
        # 记录最终统计信息
        logger.info(f"✓ 结果整理完成 (耗时: {step_time:.2f}秒)")
        logger.info(f"  课程名称: {results.get('course_name', '未知')}")
        logger.info(f"  总考点数: {results.get('total_keypoints', 0)}")
        logger.info(f"  重要程度统计:")
        importance_stats = results.get('importance_statistics', {})
        for level in sorted(importance_stats.keys(), reverse=True):
            count = importance_stats[level]
            total = results.get('total_keypoints', 1)
            percentage = (count / total * 100) if total > 0 else 0
            logger.info(f"    {level}级: {count} 个 ({percentage:.1f}%)")
        
        chapter_keypoints = results.get('keypoints_by_chapter', {})
        logger.info(f"  章节分布: {len(chapter_keypoints)} 个章节")
        for chapter_name, kps in list(chapter_keypoints.items())[:5]:
            logger.info(f"    - {chapter_name}: {len(kps)} 个考点")
        if len(chapter_keypoints) > 5:
            logger.info(f"    ... 还有 {len(chapter_keypoints) - 5} 个章节")
        
        # 步骤7: 保存结果
        step_start_time = time.time()
        logger.info("-" * 60)
        logger.info(f"步骤7/7: 保存结果 [进度: {7/total_steps*100:.1f}%]")
        logger.info("-" * 60)
        if output_path:
            logger.info(f"正在保存最终结果到: {output_path}")
            self._save_results(results, output_path)
            step_time = time.time() - step_start_time
            logger.info(f"✓ 最终结果已保存 (耗时: {step_time:.2f}秒)")
        else:
            logger.info("未指定输出路径，跳过保存")
        
        # 显示总体统计
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("智能提取完成！")
        logger.info("=" * 60)
        logger.info(f"总体进度: [{total_steps}/{total_steps}] 100.0%")
        logger.info(f"总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
        logger.info(f"平均每文件耗时: {total_time/total_files:.2f}秒" if total_files > 0 else "")
        logger.info(f"共提取考点: {len(current_keypoints)} 个")
        logger.info("=" * 60)
        return results
    
    def _organize_results(
        self,
        keypoints: List[KeyPoint],
        document_analysis: Dict[str, Any],
        strategy: ExtractionStrategy,
        extraction_goal: ExtractionGoal,
        processed_files: List[str]
    ) -> Dict[str, Any]:
        """整理提取结果"""
        # 按重要程度排序
        sorted_keypoints = sorted(keypoints, key=lambda x: x.importance, reverse=True)
        
        # 按章节分组
        chapters = document_analysis.get("chapters", [])
        chapter_keypoints = {}
        
        for kp in sorted_keypoints:
            chapter_name = "未分类"
            for chapter in chapters:
                for file_path in chapter.get("files", []):
                    if file_path in kp.source_file or kp.source_file in file_path:
                        chapter_name = chapter.get("name", "未分类")
                        break
                if chapter_name != "未分类":
                    break
            
            if chapter_name not in chapter_keypoints:
                chapter_keypoints[chapter_name] = []
            chapter_keypoints[chapter_name].append(kp)
        
        # 统计信息
        importance_stats = {}
        for level in range(1, 6):
            count = sum(1 for kp in keypoints if kp.importance == level)
            importance_stats[level] = count
        
        result = {
            "course_name": Path(processed_files[0]).parent.name if processed_files else "未知",
            "extraction_goal": {
                "goal": extraction_goal.goal,
                "requirements": extraction_goal.requirements,
                "focus_areas": extraction_goal.focus_areas,
                "importance_criteria": extraction_goal.importance_criteria
            },
            "strategy": {
                "strategy_name": strategy.strategy_name,
                "approach": strategy.approach,
                "reasoning": strategy.reasoning,
                "extraction_focus": strategy.extraction_focus
            },
            "total_keypoints": len(keypoints),
            "importance_statistics": importance_stats,
            "keypoints_by_chapter": {
                chapter: [asdict(kp) for kp in kps]
                for chapter, kps in chapter_keypoints.items()
            },
            "all_keypoints": [asdict(kp) for kp in sorted_keypoints],
            "document_analysis": document_analysis,
            "processed_files": processed_files
        }
        
        return result
    
    def _save_results(self, results: Dict[str, Any], output_path: str):
        """保存结果到文件"""
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path_obj, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到: {output_path_obj}")
    
    def _save_intermediate_results(
        self,
        output_path: str,
        step_name: str,
        data: Dict[str, Any]
    ):
        """
        保存中间结果到文件
        
        Args:
            output_path: 最终输出文件路径
            step_name: 步骤名称（用于生成中间文件名）
            data: 要保存的数据
        """
        output_path_obj = Path(output_path)
        output_dir = output_path_obj.parent
        output_stem = output_path_obj.stem
        
        # 生成中间结果文件名：原文件名.step_name.json
        intermediate_path = output_dir / f"{output_stem}.{step_name}.json"
        
        # 确保目录存在
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存中间结果
        with open(intermediate_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"中间结果已保存到: {intermediate_path}")


def main():
    """主函数 - 示例用法"""
    import sys
    
    if len(sys.argv) < 2:
        logger.error("用法: python intelligent_exctract_keypoint.py <课程目录路径> [输出文件路径]")
        logger.error("示例: python intelligent_exctract_keypoint.py data/金融实证方法 output/keypoints.json")
        sys.exit(1)
    
    course_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output/keypoints.json"
    
    # 创建抽取目标（示例）
    extraction_goal = ExtractionGoal(
        goal="提取课程中的重要考点，重点关注考试相关内容。",
        requirements=[
            "考点名称要简洁明确",
            "重要程度评估要合理",
            "来源信息要准确",
            "避免重复和过于相似的考点"
        ],
        focus_areas=["核心概念", "重要公式", "考试重点"],
        importance_criteria="根据考试频率和重要性评估"
    )
    
    # 创建智能体并执行
    agent = IntelligentKeyPointAgent()
    results = agent.extract(course_dir, extraction_goal, output_path)
    
    # 打印摘要
    logger.info("\n" + "="*50)
    logger.info("提取摘要")
    logger.info("="*50)
    logger.info(f"课程名称: {results['course_name']}")
    logger.info(f"总考点数: {results['total_keypoints']}")
    logger.info(f"重要程度统计:")
    for level, count in results['importance_statistics'].items():
        logger.info(f"  {level}级: {count}个")
    logger.info(f"章节数: {len(results['keypoints_by_chapter'])}")


if __name__ == "__main__":
    main()


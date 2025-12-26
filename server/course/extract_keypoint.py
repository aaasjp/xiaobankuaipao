"""
课程考点提取 Agent Workflow

基于课程资料目录，提取整个课程的重要考点。
每个重要考点包括：
- 考点
- 考点说明
- 重要程度（1-5个等级）
- 重要原因
- 来源（章节/页码/片段）
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

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
class ExtractionPlan:
    """提取计划"""
    course_name: str  # 课程名称
    total_files: int  # 总文件数
    file_list: List[str]  # 文件列表
    chapters: List[Dict[str, Any]]  # 章节信息
    extraction_strategy: str  # 提取策略


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
        
        logger.info(f"已读取 {len(md_files)} 个文件")
        logger.info(f"文件列表: {md_files.keys()}")
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


class Planner:
    """规划 - 负责任务规划"""
    
    def __init__(self, llm_client: LLMClient):
        """
        初始化规划组件
        
        Args:
            llm_client: LLM客户端
        """
        self.llm_client = llm_client
    
    def create_plan(self, course_dir: str, material_files: Dict[str, str]) -> ExtractionPlan:
        """
        创建提取计划
        
        Args:
            course_dir: 课程目录路径
            material_files: 资料文件字典（文件路径 -> markdown内容）
        
        Returns:
            提取计划
        """
        course_name = Path(course_dir).name
        
        # 分析目录结构
        file_list = list(material_files.keys())
        
        # 使用LLM分析章节结构
        file_summaries = {}
        for file_path, content in list(material_files.items())[:10]:  # 限制前10个文件用于分析
            preview = content[:500] if len(content) > 500 else content
            file_summaries[file_path] = preview
        
        prompt = f"""
请分析以下课程资料的文件结构，识别章节信息：

课程名称：{course_name}
文件列表：
{json.dumps(file_list, ensure_ascii=False, indent=2)}

文件内容预览：
{json.dumps(file_summaries, ensure_ascii=False, indent=2)}

请分析并返回JSON格式：
{{
    "chapters": [
        {{
            "name": "章节名称",
            "files": ["文件路径1", "文件路径2"],
            "description": "章节描述"
        }}
    ],
    "extraction_strategy": "提取策略说明"
}}
"""
        
        system_prompt = "你是一个课程分析专家，擅长分析课程资料的结构和组织方式。"
        
        try:
            result = self.llm_client.call_json(prompt, system_prompt)
            chapters = result.get("chapters", [])
            strategy = result.get("extraction_strategy", "按章节顺序提取")
        except Exception as e:
            logger.warning(f"LLM分析失败，使用默认策略: {e}")
            chapters = [{"name": "全部内容", "files": file_list, "description": "未分类"}]
            strategy = "按文件顺序提取"
        
        return ExtractionPlan(
            course_name=course_name,
            total_files=len(file_list),
            file_list=file_list,
            chapters=chapters,
            extraction_strategy=strategy
        )


class Extractor:
    """提取 - 负责提取考点"""
    
    def __init__(self, llm_client: LLMClient):
        """
        初始化提取组件
        
        Args:
            llm_client: LLM客户端
        """
        self.llm_client = llm_client
    
    def extract_keypoints(self, content: str, source_file: str, context: Optional[str] = None) -> List[KeyPoint]:
        """
        从内容中提取考点
        
        Args:
            content: 要分析的内容
            source_file: 来源文件路径
            context: 上下文信息（可选）
        
        Returns:
            考点列表
        """
        # 如果内容太长，分段处理
        max_chunk_size = 8000  # 每个chunk的最大字符数
        chunks = []
        if len(content) > max_chunk_size:
            # 按段落分割
            paragraphs = content.split('\n\n')
            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) > max_chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = para
                else:
                    current_chunk += "\n\n" + para if current_chunk else para
            if current_chunk:
                chunks.append(current_chunk)
        else:
            chunks = [content]
        
        all_keypoints = []
        for i, chunk in enumerate(chunks):
            keypoints = self._extract_from_chunk(chunk, source_file, i + 1, len(chunks), context)
            all_keypoints.extend(keypoints)
        
        return all_keypoints
    
    def _extract_from_chunk(self, chunk: str, source_file: str, chunk_num: int, total_chunks: int, context: Optional[str]) -> List[KeyPoint]:
        """从单个chunk中提取考点"""
        # 限制内容长度
        content_preview = chunk[:6000]
        context_info = f"\n上下文信息：{context}" if context else ""
        
        prompt = f"""
请从以下课程内容中提取重要考点。每个考点应该包括：
1. 考点名称（简洁明确）
2. 考点说明（详细描述）
3. 重要程度（1-5级，5为最重要）
4. 重要原因（为什么这个考点重要）
5. 来源位置（在内容中的具体位置描述）

课程内容：
{content_preview}
{context_info}

请以JSON格式返回，格式如下：
{{
    "keypoints": [
        {{
            "topic": "考点名称",
            "description": "考点说明",
            "importance": 3,
            "importance_reason": "重要原因",
            "source_snippet": "来源片段（原文引用）"
        }}
    ]
}}
"""
        
        system_prompt = """你是一个教育专家，擅长从课程资料中识别和提取重要考点。
请关注：
- 核心概念和定义
- 重要公式和定理
- 关键方法和技巧
- 常见考点和易错点
- 考试重点内容

重要程度评估标准：
5级：核心概念，考试必考，基础中的基础
4级：重要知识点，高频考点
3级：一般重要，需要掌握
2级：了解即可
1级：补充知识，非重点
"""
        
        try:
            result = self.llm_client.call_json(prompt, system_prompt, temperature=0.3)
            keypoints_data = result.get("keypoints", [])
            
            keypoints = []
            for kp_data in keypoints_data:
                keypoint = KeyPoint(
                    topic=kp_data.get("topic", ""),
                    description=kp_data.get("description", ""),
                    importance=kp_data.get("importance", 3),
                    importance_reason=kp_data.get("importance_reason", ""),
                    source=f"{source_file} (片段 {chunk_num}/{total_chunks})",
                    source_file=source_file,
                    source_snippet=kp_data.get("source_snippet")
                )
                keypoints.append(keypoint)
            
            return keypoints
        except Exception as e:
            logger.error(f"提取考点失败: {e}")
            return []


class Evaluator:
    """评估 - 负责评估和优化考点"""
    
    def __init__(self, llm_client: LLMClient):
        """
        初始化评估组件
        
        Args:
            llm_client: LLM客户端
        """
        self.llm_client = llm_client
    
    def evaluate_and_refine(self, keypoints: List[KeyPoint], course_context: str) -> List[KeyPoint]:
        """
        评估和优化考点列表
        
        Args:
            keypoints: 原始考点列表
            course_context: 课程上下文信息
        
        Returns:
            优化后的考点列表
        """
        if not keypoints:
            return []
        
        # 合并相似考点
        refined_keypoints = self._merge_similar_keypoints(keypoints)
        
        # 使用LLM进一步评估和优化
        keypoints_json = [asdict(kp) for kp in refined_keypoints]
        
        prompt = f"""
请评估和优化以下考点列表。要求：
1. 检查重要程度是否合理，必要时调整
2. 完善重要原因说明
3. 确保考点说明清晰准确
4. 去除重复或过于相似的考点
5. 确保来源信息准确

课程上下文：{course_context}

考点列表：
{json.dumps(keypoints_json, ensure_ascii=False, indent=2)}

请返回优化后的JSON格式，格式相同。
"""
        
        system_prompt = "你是一个教育评估专家，擅长评估和优化课程考点的质量和重要性。"
        
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
            logger.warning(f"评估优化失败，使用原始结果: {e}")
            return refined_keypoints
    
    def _merge_similar_keypoints(self, keypoints: List[KeyPoint]) -> List[KeyPoint]:
        """合并相似的考点"""
        # 简单的相似度合并：如果主题相似且来源相同，合并为一个
        merged = []
        seen_topics = set()
        
        for kp in keypoints:
            # 检查是否有相似主题
            topic_lower = kp.topic.lower()
            is_similar = False
            
            for seen_topic in seen_topics:
                # 简单的相似度检查：如果主题包含关系或高度相似
                if topic_lower in seen_topic.lower() or seen_topic.lower() in topic_lower:
                    is_similar = True
                    break
            
            if not is_similar:
                merged.append(kp)
                seen_topics.add(topic_lower)
            else:
                # 如果相似，选择重要程度更高的
                for i, existing_kp in enumerate(merged):
                    if existing_kp.topic.lower() in topic_lower or topic_lower in existing_kp.topic.lower():
                        if kp.importance > existing_kp.importance:
                            merged[i] = kp
                        break
        
        return merged


class Organizer:
    """整理 - 负责整理和输出结果"""
    
    def organize_results(self, keypoints: List[KeyPoint], plan: ExtractionPlan) -> Dict[str, Any]:
        """
        整理提取结果
        
        Args:
            keypoints: 考点列表
            plan: 提取计划
        
        Returns:
            整理后的结果字典
        """
        # 按重要程度排序
        sorted_keypoints = sorted(keypoints, key=lambda x: x.importance, reverse=True)
        
        # 按章节分组
        chapter_keypoints = {}
        for kp in sorted_keypoints:
            # 尝试匹配章节
            chapter_name = "未分类"
            for chapter in plan.chapters:
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
            "course_name": plan.course_name,
            "total_keypoints": len(keypoints),
            "importance_statistics": importance_stats,
            "keypoints_by_chapter": {
                chapter: [asdict(kp) for kp in kps]
                for chapter, kps in chapter_keypoints.items()
            },
            "all_keypoints": [asdict(kp) for kp in sorted_keypoints],
            "extraction_plan": {
                "total_files": plan.total_files,
                "strategy": plan.extraction_strategy,
                "chapters": plan.chapters
            }
        }
        
        return result
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        保存结果到文件
        
        Args:
            results: 结果字典
            output_path: 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到: {output_path}")
    
    def save_intermediate_results(
        self, 
        keypoints: List[KeyPoint], 
        plan: ExtractionPlan,
        processed_files: List[str],
        output_path: str,
        status: str = "extracting"
    ):
        """
        保存中间结果（提取过程中的临时结果）
        
        Args:
            keypoints: 已提取的考点列表
            plan: 提取计划
            processed_files: 已处理的文件列表
            output_path: 最终输出文件路径（用于生成中间结果文件路径）
            status: 当前状态，可选值：extracting（提取中）、evaluating（评估中）、completed（已完成）
        """
        output_path_obj = Path(output_path)
        # 生成中间结果文件路径：原文件名.intermediate.json
        intermediate_path = output_path_obj.parent / f"{output_path_obj.stem}.intermediate.json"
        
        # 整理中间结果
        sorted_keypoints = sorted(keypoints, key=lambda x: x.importance, reverse=True)
        
        # 统计信息
        importance_stats = {}
        for level in range(1, 6):
            count = sum(1 for kp in keypoints if kp.importance == level)
            importance_stats[level] = count
        
        intermediate_result = {
            "course_name": plan.course_name,
            "status": status,  # 状态：extracting（提取中）、evaluating（评估中）、completed（已完成）
            "progress": {
                "processed_files": len(processed_files),
                "total_files": plan.total_files,
                "processed_file_list": processed_files
            },
            "current_keypoints_count": len(keypoints),
            "importance_statistics": importance_stats,
            "all_keypoints": [asdict(kp) for kp in sorted_keypoints],
            "extraction_plan": {
                "total_files": plan.total_files,
                "strategy": plan.extraction_strategy,
                "chapters": plan.chapters
            }
        }
        
        intermediate_path.parent.mkdir(parents=True, exist_ok=True)
        with open(intermediate_path, "w", encoding="utf-8") as f:
            json.dump(intermediate_result, f, ensure_ascii=False, indent=2)
        
        status_msg = {
            "extracting": "提取中",
            "evaluating": "评估中",
            "completed": "已完成"
        }.get(status, status)
        
        logger.info(f"中间结果已保存到: {intermediate_path} ({status_msg}, 已处理 {len(processed_files)}/{plan.total_files} 个文件, {len(keypoints)} 个考点)")


class KeyPointExtractionWorkflow:
    """考点提取工作流主类"""
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        material_reader: Optional[MaterialReader] = None,
        parsed_dir: Optional[str] = None
    ):
        """
        初始化工作流
        
        Args:
            llm_client: LLM客户端，如果为None则创建新实例
            material_reader: 资料读取器，如果为None则创建新实例
            parsed_dir: 已解析文件的目录路径，如果为None则使用默认的docs_parsed目录
        """
        self.llm_client = llm_client or LLMClient()
        self.material_reader = material_reader or MaterialReader(parsed_dir=parsed_dir)
        
        # 初始化各个组件
        self.planner = Planner(self.llm_client)
        self.extractor = Extractor(self.llm_client)
        self.evaluator = Evaluator(self.llm_client)
        self.organizer = Organizer()
    
    def extract(self, course_dir: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        执行完整的提取流程
        
        Args:
            course_dir: 课程资料目录路径
            output_path: 输出文件路径（可选）
        
        Returns:
            提取结果字典
        """
        logger.info(f"开始提取课程考点: {course_dir}")
        
        # 步骤1: 读取资料
        logger.info("步骤1: 读取课程资料...")
        material_files = self.material_reader.read_directory(course_dir)
        logger.info(f"已读取 {len(material_files)} 个文件")
        
        # 步骤2: 创建提取计划
        logger.info("步骤2: 创建提取计划...")
        plan = self.planner.create_plan(course_dir, material_files)
        logger.info(f"提取计划已创建: {plan.extraction_strategy}")
        logger.info(f"识别到 {len(plan.chapters)} 个章节")
        
        # 步骤3: 提取考点
        logger.info("步骤3: 提取考点...")
        all_keypoints = []
        total_files = len(material_files)
        processed_files = []
        
        for idx, (file_path, content) in enumerate[tuple[str, str]](material_files.items(), 1):
            logger.info(f"  处理文件 {idx}/{total_files}: {file_path}")
            try:
                keypoints = self.extractor.extract_keypoints(content, file_path)
                all_keypoints.extend(keypoints)
                processed_files.append(file_path)
                logger.info(f"    提取到 {len(keypoints)} 个考点")
                
                # 每处理完一个文件就保存中间结果
                if output_path:
                    self.organizer.save_intermediate_results(
                        all_keypoints, 
                        plan, 
                        processed_files,
                        output_path,
                        status="extracting"
                    )
            except Exception as e:
                logger.error(f"    提取失败: {e}")
                # 即使失败也记录已处理的文件
                processed_files.append(file_path)
                # 保存中间结果（包含失败的文件信息）
                if output_path:
                    self.organizer.save_intermediate_results(
                        all_keypoints, 
                        plan, 
                        processed_files,
                        output_path,
                        status="extracting"
                    )
                continue
        
        logger.info(f"共提取到 {len(all_keypoints)} 个考点")
        
        # 步骤4: 评估和优化
        logger.info("步骤4: 评估和优化考点...")
        course_context = f"课程: {plan.course_name}, 共{plan.total_files}个文件"
        refined_keypoints = self.evaluator.evaluate_and_refine(all_keypoints, course_context)
        logger.info(f"优化后剩余 {len(refined_keypoints)} 个考点")
        
        # 保存评估优化后的中间结果
        if output_path:
            self.organizer.save_intermediate_results(
                refined_keypoints, 
                plan, 
                processed_files,
                output_path,
                status="evaluating"
            )
        
        # 步骤5: 整理结果
        logger.info("步骤5: 整理结果...")
        results = self.organizer.organize_results(refined_keypoints, plan)
        
        # 步骤6: 保存结果
        if output_path:
            logger.info(f"步骤6: 保存结果到 {output_path}...")
            self.organizer.save_results(results, output_path)
        
        logger.info("提取完成！")
        return results


def main():
    """主函数 - 示例用法"""
    import sys
    
    if len(sys.argv) < 2:
        logger.error("用法: python extract_keypoint.py <课程目录路径> [输出文件路径]")
        logger.error("示例: python extract_keypoint.py data/金融实证方法 output/keypoints.json")
        sys.exit(1)
    
    course_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output/keypoints.json"
    
    # 创建工作流并执行
    workflow = KeyPointExtractionWorkflow()
    results = workflow.extract(course_dir, output_path)
    
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


"""
智能体驱动的考点提取系统
支持动态调整提取流程和评估策略
"""

import json
import logging
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod

from server.llm.llm_service import LLMClient
from server.course.extract_keypoint import (
    KeyPoint,
    ExtractionPlan,
    Planner,
    Extractor,
    Organizer,
    MaterialReader
)

# 配置日志
logger = logging.getLogger(__name__)

# 状态枚举
class AgentState(Enum):
    """智能体状态"""
    INITIALIZING = "initializing"  # 初始化
    PLANNING = "planning"  # 规划中
    EXTRACTING = "extracting"  # 提取中
    EVALUATING = "evaluating"  # 评估中
    REFINING = "refining"  # 优化中
    ADAPTING = "adapting"  # 自适应调整中
    COMPLETED = "completed"  # 已完成
    ERROR = "error"  # 错误状态


@dataclass
class ExecutionContext:
    """执行上下文 - 记录当前状态和历史信息"""
    state: AgentState
    plan: Optional[ExtractionPlan] = None
    extracted_keypoints: List[KeyPoint] = field(default_factory=list)
    processed_files: List[str] = field(default_factory=list)
    failed_files: List[str] = field(default_factory=list)
    evaluation_history: List[Dict[str, Any]] = field(default_factory=list)
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    current_strategy: str = ""
    iteration_count: int = 0
    max_iterations: int = 100


@dataclass
class ActionDecision:
    """行动决策"""
    action: str  # 行动类型
    target: Any  # 行动目标
    reason: str  # 决策原因
    confidence: float  # 置信度 (0-1)
    parameters: Dict[str, Any] = field(default_factory=dict)


class DecisionMaker:
    """决策器 - 基于当前状态和历史信息做出决策"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def decide_next_action(
        self, 
        context: ExecutionContext,
        material_files: Dict[str, str]
    ) -> ActionDecision:
        """
        决定下一步行动
        
        Args:
            context: 当前执行上下文
            material_files: 材料文件字典
        
        Returns:
            行动决策
        """
        # 分析当前状态和历史
        state_analysis = self._analyze_state(context)
        
        # 使用LLM进行智能决策
        prompt = self._build_decision_prompt(context, state_analysis, material_files)
        
        system_prompt = """你是一个智能决策系统，负责分析当前状态并决定下一步行动。
你需要考虑：
1. 当前提取进度和质量
2. 已提取考点的分布情况
3. 失败或低质量的文件
4. 是否需要调整提取策略
5. 是否需要重新评估已提取的考点

请根据情况做出最优决策。"""
        
        try:
            result = self.llm_client.call_json(prompt, system_prompt, temperature=0.3)
            
            return ActionDecision(
                action=result.get("action", "continue_extracting"),
                target=result.get("target"),
                reason=result.get("reason", ""),
                confidence=result.get("confidence", 0.7),
                parameters=result.get("parameters", {})
            )
        except Exception as e:
            logger.error(f"决策失败: {e}")
            # 默认决策：继续提取
            return self._default_decision(context, material_files)
    
    def _analyze_state(self, context: ExecutionContext) -> Dict[str, Any]:
        """分析当前状态"""
        analysis = {
            "current_state": context.state.value,
            "progress": {
                "processed": len(context.processed_files),
                "failed": len(context.failed_files),
                "extracted_count": len(context.extracted_keypoints)
            },
            "quality": context.quality_metrics,
            "iteration": context.iteration_count
        }
        
        # 分析考点分布
        if context.extracted_keypoints:
            importance_dist = {}
            for kp in context.extracted_keypoints:
                level = kp.importance
                importance_dist[level] = importance_dist.get(level, 0) + 1
            analysis["importance_distribution"] = importance_dist
            
            # 计算质量指标
            avg_importance = sum(kp.importance for kp in context.extracted_keypoints) / len(context.extracted_keypoints)
            high_importance_ratio = sum(1 for kp in context.extracted_keypoints if kp.importance >= 4) / len(context.extracted_keypoints)
            analysis["quality"]["avg_importance"] = avg_importance
            analysis["quality"]["high_importance_ratio"] = high_importance_ratio
        
        return analysis
    
    def _build_decision_prompt(
        self, 
        context: ExecutionContext,
        state_analysis: Dict[str, Any],
        material_files: Dict[str, str]
    ) -> str:
        """构建决策提示词"""
        remaining_files = [
            f for f in material_files.keys() 
            if f not in context.processed_files and f not in context.failed_files
        ]
        
        return f"""
当前执行上下文：
状态: {context.state.value}
已处理文件: {len(context.processed_files)}/{len(material_files)}
已提取考点: {len(context.extracted_keypoints)} 个
失败文件: {len(context.failed_files)} 个
迭代次数: {context.iteration_count}

状态分析：
{json.dumps(state_analysis, ensure_ascii=False, indent=2)}

剩余待处理文件: {len(remaining_files)} 个
当前策略: {context.current_strategy}

请分析当前情况并决定下一步行动。可选行动：
1. continue_extracting - 继续提取下一个文件
2. re_extract_file - 重新提取某个文件（如果质量不佳）
3. refine_keypoints - 优化已提取的考点
4. adapt_strategy - 调整提取策略
5. evaluate_quality - 评估当前质量并决定是否需要改进
6. complete - 完成提取

请返回JSON格式：
{{
    "action": "行动类型",
    "target": "行动目标（如文件路径、考点列表等）",
    "reason": "决策原因",
    "confidence": 0.8,
    "parameters": {{"key": "value"}}
}}
"""
    
    def _default_decision(
        self, 
        context: ExecutionContext,
        material_files: Dict[str, str]
    ) -> ActionDecision:
        """默认决策逻辑"""
        remaining_files = [
            f for f in material_files.keys() 
            if f not in context.processed_files and f not in context.failed_files
        ]
        
        if remaining_files:
            return ActionDecision(
                action="continue_extracting",
                target=remaining_files[0],
                reason="默认继续提取",
                confidence=0.5
            )
        else:
            return ActionDecision(
                action="complete",
                target=None,
                reason="所有文件已处理",
                confidence=1.0
            )


class AdaptiveEvaluator:
    """自适应评估器 - 根据提取结果动态调整评估标准"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def evaluate_with_adaptation(
        self,
        keypoints: List[KeyPoint],
        context: ExecutionContext
    ) -> Tuple[List[KeyPoint], Dict[str, Any]]:
        """
        自适应评估和优化
        
        Args:
            keypoints: 待评估的考点列表
            context: 执行上下文
        
        Returns:
            (优化后的考点列表, 评估报告)
        """
        # 分析当前考点的质量分布
        quality_analysis = self._analyze_quality(keypoints, context)
        
        # 根据分析结果调整评估策略
        evaluation_strategy = self._adapt_evaluation_strategy(quality_analysis, context)
        
        # 执行评估
        refined_keypoints = self._evaluate_keypoints(
            keypoints, 
            evaluation_strategy,
            context
        )
        
        # 生成评估报告
        report = {
            "original_count": len(keypoints),
            "refined_count": len(refined_keypoints),
            "quality_analysis": quality_analysis,
            "strategy_used": evaluation_strategy,
            "changes_made": self._compare_keypoints(keypoints, refined_keypoints)
        }
        
        return refined_keypoints, report
    
    def _analyze_quality(
        self, 
        keypoints: List[KeyPoint],
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """分析考点质量"""
        if not keypoints:
            return {"status": "empty", "recommendation": "需要重新提取"}
        
        # 统计信息
        importance_dist = {}
        for kp in keypoints:
            importance_dist[kp.importance] = importance_dist.get(kp.importance, 0) + 1
        
        avg_importance = sum(kp.importance for kp in keypoints) / len(keypoints)
        
        # 检查问题
        issues = []
        if avg_importance < 2.5:
            issues.append("平均重要程度偏低，可能需要提高筛选标准")
        if importance_dist.get(5, 0) == 0:
            issues.append("缺少最高重要程度的考点，可能需要深入挖掘")
        if len(keypoints) < 10:
            issues.append("考点数量偏少，可能需要扩大提取范围")
        
        return {
            "total_count": len(keypoints),
            "importance_distribution": importance_dist,
            "average_importance": avg_importance,
            "issues": issues,
            "status": "good" if not issues else "needs_improvement"
        }
    
    def _adapt_evaluation_strategy(
        self,
        quality_analysis: Dict[str, Any],
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """根据质量分析调整评估策略"""
        strategy = {
            "strictness": "normal",  # strict, normal, lenient
            "focus_areas": [],
            "adjustment_needed": False
        }
        
        if quality_analysis["status"] == "needs_improvement":
            if quality_analysis["average_importance"] < 2.5:
                strategy["strictness"] = "strict"
                strategy["focus_areas"].append("提高重要程度阈值")
                strategy["adjustment_needed"] = True
            
            if "缺少最高重要程度的考点" in str(quality_analysis.get("issues", [])):
                strategy["focus_areas"].append("重点关注核心概念")
                strategy["adjustment_needed"] = True
        
        return strategy
    
    def _evaluate_keypoints(
        self,
        keypoints: List[KeyPoint],
        strategy: Dict[str, Any],
        context: ExecutionContext
    ) -> List[KeyPoint]:
        """执行评估"""
        # 先合并相似考点
        refined = self._merge_similar_keypoints(keypoints)
        
        # 根据策略调整
        if strategy["strictness"] == "strict":
            # 提高重要程度阈值
            refined = [kp for kp in refined if kp.importance >= 3]
        
        # 使用LLM进一步优化
        if refined:
            refined = self._llm_refine(refined, strategy, context)
        
        return refined
    
    def _llm_refine(
        self,
        keypoints: List[KeyPoint],
        strategy: Dict[str, Any],
        context: ExecutionContext
    ) -> List[KeyPoint]:
        """使用LLM优化考点"""
        keypoints_json = [asdict(kp) for kp in keypoints]
        
        focus_instructions = ""
        if strategy["focus_areas"]:
            focus_instructions = f"\n重点关注: {', '.join(strategy['focus_areas'])}"
        
        prompt = f"""
请评估和优化以下考点列表。当前质量分析显示需要改进的地方：
{json.dumps(strategy, ensure_ascii=False, indent=2)}
{focus_instructions}

考点列表：
{json.dumps(keypoints_json, ensure_ascii=False, indent=2)}

请根据策略要求进行优化，返回优化后的JSON格式（保持相同结构）。
"""
        
        system_prompt = "你是教育评估专家，擅长根据质量分析结果优化考点列表。"
        
        try:
            result = self.llm_client.call_json(prompt, system_prompt, temperature=0.3)
            refined_data = result.get("keypoints", keypoints_json)
            
            refined_keypoints = []
            for kp_data in refined_data:
                refined_keypoints.append(KeyPoint(**kp_data))
            
            return refined_keypoints
        except Exception as e:
            logger.warning(f"LLM优化失败，使用原始结果: {e}")
            return keypoints
    
    def _merge_similar_keypoints(self, keypoints: List[KeyPoint]) -> List[KeyPoint]:
        """合并相似考点（复用原有逻辑）"""
        # 这里可以复用原有的 _merge_similar_keypoints 逻辑
        merged = []
        seen_topics = set()
        
        for kp in keypoints:
            topic_lower = kp.topic.lower()
            is_similar = False
            
            for seen_topic in seen_topics:
                if topic_lower in seen_topic.lower() or seen_topic.lower() in topic_lower:
                    is_similar = True
                    break
            
            if not is_similar:
                merged.append(kp)
                seen_topics.add(topic_lower)
            else:
                for i, existing_kp in enumerate(merged):
                    if existing_kp.topic.lower() in topic_lower or topic_lower in existing_kp.topic.lower():
                        if kp.importance > existing_kp.importance:
                            merged[i] = kp
                        break
        
        return merged
    
    def _compare_keypoints(
        self,
        original: List[KeyPoint],
        refined: List[KeyPoint]
    ) -> Dict[str, Any]:
        """比较优化前后的变化"""
        return {
            "removed_count": len(original) - len(refined),
            "importance_adjustments": self._count_importance_changes(original, refined)
        }
    
    def _count_importance_changes(
        self,
        original: List[KeyPoint],
        refined: List[KeyPoint]
    ) -> int:
        """统计重要程度调整的数量"""
        # 简化实现：统计调整的数量
        original_map = {kp.topic: kp.importance for kp in original}
        changes = 0
        for kp in refined:
            if kp.topic in original_map:
                if kp.importance != original_map[kp.topic]:
                    changes += 1
        return changes


class AdaptiveExtractor:
    """自适应提取器 - 根据上下文动态调整提取策略"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.base_extractor = Extractor(llm_client)
    
    def extract_with_context(
        self,
        content: str,
        source_file: str,
        context: ExecutionContext
    ) -> List[KeyPoint]:
        """
        基于上下文进行自适应提取
        
        Args:
            content: 文件内容
            source_file: 文件路径
            context: 执行上下文
        """
        # 分析已提取的考点，调整提取策略
        extraction_params = self._adapt_extraction_params(context)
        
        # 执行提取
        keypoints = self.base_extractor.extract_keypoints(
            content, 
            source_file,
            context=self._build_extraction_context(context, extraction_params)
        )
        
        # 根据策略调整结果
        if extraction_params.get("focus_high_importance"):
            keypoints = [kp for kp in keypoints if kp.importance >= 4]
        
        return keypoints
    
    def _adapt_extraction_params(
        self,
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """根据上下文调整提取参数"""
        params = {
            "focus_high_importance": False,
            "min_importance": 2,
            "max_keypoints_per_chunk": 5
        }
        
        # 如果已提取的考点平均重要程度偏低，提高标准
        if context.extracted_keypoints:
            avg_importance = sum(
                kp.importance for kp in context.extracted_keypoints
            ) / len(context.extracted_keypoints)
            
            if avg_importance < 2.5:
                params["focus_high_importance"] = True
                params["min_importance"] = 3
        
        # 如果考点数量过多，减少提取数量
        if len(context.extracted_keypoints) > 200:
            params["max_keypoints_per_chunk"] = 3
        
        return params
    
    def _build_extraction_context(
        self,
        context: ExecutionContext,
        params: Dict[str, Any]
    ) -> str:
        """构建提取上下文信息"""
        context_info = []
        
        if context.extracted_keypoints:
            # 提供已提取的考点作为参考
            recent_topics = [kp.topic for kp in context.extracted_keypoints[-10:]]
            context_info.append(f"已提取的相关考点: {', '.join(recent_topics)}")
        
        if params.get("focus_high_importance"):
            context_info.append("重点关注高重要程度的考点（4-5级）")
        
        return "\n".join(context_info) if context_info else None


class IntelligentKeyPointAgent:
    """智能考点提取智能体 - 主控制器"""
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        material_reader: Optional[MaterialReader] = None,
        parsed_dir: Optional[str] = None
    ):
        self.llm_client = llm_client or LLMClient()
        self.material_reader = material_reader or MaterialReader(parsed_dir=parsed_dir)
        
        # 初始化组件
        self.planner = Planner(self.llm_client)
        self.decision_maker = DecisionMaker(self.llm_client)
        self.adaptive_extractor = AdaptiveExtractor(self.llm_client)
        self.adaptive_evaluator = AdaptiveEvaluator(self.llm_client)
        self.organizer = Organizer()
        
        # 状态处理器映射
        self.state_handlers = {
            AgentState.INITIALIZING: self._handle_initializing,
            AgentState.PLANNING: self._handle_planning,
            AgentState.EXTRACTING: self._handle_extracting,
            AgentState.EVALUATING: self._handle_evaluating,
            AgentState.REFINING: self._handle_refining,
            AgentState.ADAPTING: self._handle_adapting,
        }
    
    def extract(self, course_dir: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        执行智能提取流程
        
        Args:
            course_dir: 课程目录路径
            output_path: 输出文件路径
        
        Returns:
            提取结果字典
        """
        logger.info(f"开始智能提取课程考点: {course_dir}")
        
        # 初始化上下文
        context = ExecutionContext(state=AgentState.INITIALIZING)
        material_files = {}
        
        # 主循环：根据状态和决策动态执行
        while context.state != AgentState.COMPLETED and context.iteration_count < context.max_iterations:
            context.iteration_count += 1
            logger.info(f"\n=== 迭代 {context.iteration_count}: 当前状态 {context.state.value} ===")
            
            # 执行当前状态的处理器
            if context.state in self.state_handlers:
                try:
                    next_state = self.state_handlers[context.state](context, material_files, course_dir, output_path)
                    context.state = next_state
                except Exception as e:
                    logger.error(f"状态处理失败: {e}")
                    context.state = AgentState.ERROR
                    break
            else:
                logger.warning(f"未知状态: {context.state}")
                break
            
            # 检查是否应该完成
            if context.state == AgentState.COMPLETED:
                break
        
        # 整理最终结果
        if context.state == AgentState.COMPLETED:
            results = self.organizer.organize_results(context.extracted_keypoints, context.plan)
            if output_path:
                self.organizer.save_results(results, output_path)
            return results
        else:
            logger.error(f"提取未正常完成，最终状态: {context.state.value}")
            return {"error": f"提取失败，状态: {context.state.value}"}
    
    def _handle_initializing(
        self,
        context: ExecutionContext,
        material_files: Dict[str, str],
        course_dir: str,
        output_path: Optional[str]
    ) -> AgentState:
        """处理初始化状态"""
        logger.info("初始化: 读取课程资料...")
        material_files.update(self.material_reader.read_directory(course_dir))
        logger.info(f"已读取 {len(material_files)} 个文件")
        return AgentState.PLANNING
    
    def _handle_planning(
        self,
        context: ExecutionContext,
        material_files: Dict[str, str],
        course_dir: str,
        output_path: Optional[str]
    ) -> AgentState:
        """处理规划状态"""
        logger.info("规划: 创建提取计划...")
        context.plan = self.planner.create_plan(course_dir, material_files)
        context.current_strategy = context.plan.extraction_strategy
        logger.info(f"提取计划已创建: {context.current_strategy}")
        return AgentState.EXTRACTING
    
    def _handle_extracting(
        self,
        context: ExecutionContext,
        material_files: Dict[str, str],
        course_dir: str,
        output_path: Optional[str]
    ) -> AgentState:
        """处理提取状态 - 使用决策器决定下一步"""
        # 获取决策
        decision = self.decision_maker.decide_next_action(context, material_files)
        logger.info(f"决策: {decision.action} - {decision.reason} (置信度: {decision.confidence:.2f})")
        
        if decision.action == "continue_extracting":
            # 继续提取下一个文件
            target_file = decision.target
            if target_file and target_file in material_files:
                logger.info(f"提取文件: {target_file}")
                try:
                    keypoints = self.adaptive_extractor.extract_with_context(
                        material_files[target_file],
                        target_file,
                        context
                    )
                    context.extracted_keypoints.extend(keypoints)
                    context.processed_files.append(target_file)
                    logger.info(f"提取到 {len(keypoints)} 个考点")
                    
                    # 保存中间结果
                    if output_path:
                        self.organizer.save_intermediate_results(
                            context.extracted_keypoints,
                            context.plan,
                            context.processed_files,
                            output_path,
                            status="extracting"
                        )
                    
                    # 每提取几个文件后评估一次
                    if len(context.processed_files) % 5 == 0:
                        return AgentState.EVALUATING
                    
                except Exception as e:
                    logger.error(f"提取失败: {e}")
                    context.failed_files.append(target_file)
            
            return AgentState.EXTRACTING
        
        elif decision.action == "evaluate_quality":
            return AgentState.EVALUATING
        
        elif decision.action == "refine_keypoints":
            return AgentState.REFINING
        
        elif decision.action == "adapt_strategy":
            return AgentState.ADAPTING
        
        elif decision.action == "complete":
            return AgentState.EVALUATING  # 完成前先评估
        
        else:
            return AgentState.EXTRACTING
    
    def _handle_evaluating(
        self,
        context: ExecutionContext,
        material_files: Dict[str, str],
        course_dir: str,
        output_path: Optional[str]
    ) -> AgentState:
        """处理评估状态"""
        logger.info("评估: 自适应评估考点质量...")
        
        refined_keypoints, report = self.adaptive_evaluator.evaluate_with_adaptation(
            context.extracted_keypoints,
            context
        )
        
        context.extracted_keypoints = refined_keypoints
        context.evaluation_history.append(report)
        context.quality_metrics.update(report.get("quality_analysis", {}))
        
        logger.info(f"评估完成: 原始 {report['original_count']} 个，优化后 {report['refined_count']} 个")
        
        # 保存评估后的中间结果
        if output_path:
            self.organizer.save_intermediate_results(
                context.extracted_keypoints,
                context.plan,
                context.processed_files,
                output_path,
                status="evaluating"
            )
        
        # 检查是否需要进一步处理
        remaining_files = [
            f for f in material_files.keys() 
            if f not in context.processed_files and f not in context.failed_files
        ]
        
        if remaining_files:
            return AgentState.EXTRACTING
        else:
            return AgentState.COMPLETED
    
    def _handle_refining(
        self,
        context: ExecutionContext,
        material_files: Dict[str, str],
        course_dir: str,
        output_path: Optional[str]
    ) -> AgentState:
        """处理优化状态"""
        logger.info("优化: 进一步优化考点...")
        # 可以添加额外的优化逻辑
        return AgentState.EVALUATING
    
    def _handle_adapting(
        self,
        context: ExecutionContext,
        material_files: Dict[str, str],
        course_dir: str,
        output_path: Optional[str]
    ) -> AgentState:
        """处理自适应调整状态"""
        logger.info("自适应: 调整提取策略...")
        
        # 使用LLM分析并调整策略
        prompt = f"""
当前提取情况：
- 已处理文件: {len(context.processed_files)}
- 已提取考点: {len(context.extracted_keypoints)} 个
- 失败文件: {len(context.failed_files)} 个
- 当前策略: {context.current_strategy}

质量指标：
{json.dumps(context.quality_metrics, ensure_ascii=False, indent=2)}

请分析当前情况，提出改进的提取策略。
"""
        
        system_prompt = "你是策略优化专家，擅长根据执行情况调整提取策略。"
        
        try:
            result = self.llm_client.call_json(prompt, system_prompt, temperature=0.3)
            new_strategy = result.get("strategy", context.current_strategy)
            context.current_strategy = new_strategy
            context.adaptation_history.append({
                "iteration": context.iteration_count,
                "old_strategy": context.current_strategy,
                "new_strategy": new_strategy,
                "reason": result.get("reason", "")
            })
            logger.info(f"策略已调整: {new_strategy}")
        except Exception as e:
            logger.warning(f"策略调整失败: {e}")
        
        return AgentState.EXTRACTING
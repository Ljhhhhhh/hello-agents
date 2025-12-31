import ast
from dotenv import load_dotenv
from BaseAgent import BaseAgent


""" 
graph TD
    A[用户问题] --> B["Planner.plan(question, history='无', failure_info='无')"]
    B --> C{计划生成成功?}
    C -->|否| Z[返回失败]
    C -->|是| D[取出当前步骤]
    
    D --> E["_execute_step(...)"]
    E --> F["Evaluator.evaluate(...)"]
    
    F --> |SUCCESS| G[记录到 history, step_index++]
    G --> H{还有更多步骤?}
    H -->|是| D
    H -->|否| Y[返回最终答案]
    
    F --> |FAILURE_RETRY| I{重试次数 < MAX_RETRIES?}
    I -->|是| E
    I -->|否| J[降级为 FAILURE_REPLAN]
    
    F --> |FAILURE_REPLAN| K{重规划次数 < MAX_REPLANS?}
    J --> K
    K -->|是| L["Planner.plan(question, history, failure_info)"]
    L --> D
    K -->|否| Z
"""


# --- Prompts ---
PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。

问题: {question}

# 已完成的历史记录 (如果有):
{history}

# 失败信息 (如果有):
{failure_info}

请根据以上信息，生成一个新的、可行的行动计划。
注意事项:
1. 如果有已完成的历史记录，你的新计划应该从这些已完成的步骤之后继续，不要重复已完成的工作。
2. 如果有失败信息，请分析失败原因，并设计一个能够绕过或解决该问题的新计划。
3. 确保计划中的每个步骤都是独立的、可执行的子任务，并严格按照逻辑顺序排列。

输出必须是一个Python列表，格式如下:
```python
["步骤1", "步骤2", ...]
```
"""

EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""

EVALUATOR_PROMPT_TEMPLATE = """
你是一位严格的AI评估专家。你的任务是评估一个子任务的执行结果。
请根据原始问题、当前步骤以及执行结果，判断该步骤是否成功完成。

原始问题:
{question}

当前步骤:
{current_step}

执行结果:
{result}

请分析该结果是否合理、正确，并判断是否能够支持后续步骤的进行。
你的输出必须是以下三种之一，请直接输出，不要有任何其他内容:

SUCCESS - 如果步骤成功完成且结果合理。
FAILURE_RETRY - 如果步骤失败，但可以通过重新尝试（例如，换一种方式表述）来解决。
FAILURE_REPLAN - 如果步骤失败，且问题出在计划本身，需要回退并重新规划。

输出: 
"""


class Planner:
    """
    规划器：负责根据问题、历史记录和失败信息生成（或重新生成）行动计划。
    """
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def plan(self, question: str, history: str = "无", failure_info: str = "无") -> list[str]:
        """
        根据用户问题、历史记录和失败信息生成一个行动计划。
        
        参数:
        - question: 用户的原始问题
        - history: 已成功完成的步骤和结果的字符串
        - failure_info: 如果发生了失败，这里包含失败的详细信息
        
        返回:
        - 一个包含步骤描述的字符串列表
        """
        prompt = PLANNER_PROMPT_TEMPLATE.format(
            question=question,
            history=history,
            failure_info=failure_info
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        print("--- 正在生成计划 ---")
        response_text = self.llm_client.think(messages=messages) or ""
        
        print(f"✅ 计划已生成:\n{response_text}")
        
        # 解析LLM输出的列表字符串
        try:
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {response_text}")
            return []
        except Exception as e:
            print(f"❌ 解析计划时发生未知错误: {e}")
            return []


class Evaluator:
    """
    评估器：负责评估执行结果，并决定下一步动作。
    """
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def evaluate(self, question: str, current_step: str, result: str) -> str:
        """
        根据问题、当前步骤和执行结果，评估该步骤是否成功完成。
        
        返回:
        - "SUCCESS": 步骤成功
        - "FAILURE_RETRY": 步骤失败，可重试
        - "FAILURE_REPLAN": 步骤失败，需要重新规划
        """
        prompt = EVALUATOR_PROMPT_TEMPLATE.format(
            question=question, current_step=current_step, result=result
        )
        messages = [{"role": "user", "content": prompt}]
        response_text = self.llm_client.think(messages=messages) or "FAILURE_REPLAN"
        
        # 解析评估结果
        if "SUCCESS" in response_text.upper():
            return "SUCCESS"
        elif "FAILURE_RETRY" in response_text.upper():
            return "FAILURE_RETRY"
        else:
            return "FAILURE_REPLAN"


class DynamicPlanAndSolveAgent:
    """
    支持动态重规划的 Plan-and-Solve Agent。
    
    核心特性:
    1. 每执行完一个步骤，由 Evaluator 评估结果
    2. 如果评估失败，可以触发重试或重规划
    3. 重规划时会带上历史记录和失败信息，让 Planner 生成更智能的新计划
    """
    MAX_REPLANS = 3  # 最大重规划次数
    MAX_RETRIES = 2  # 单步骤最大重试次数

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.planner = Planner(self.llm_client)
        self.evaluator = Evaluator(self.llm_client)
        
        self.history = []  # 成功完成的步骤和结果
        self.replan_count = 0

    def run(self, question: str) -> str:
        """
        运行智能体的完整流程。
        
        参数:
        - question: 用户的问题
        
        返回:
        - 最终答案字符串
        """
        print(f"\n{'='*50}")
        print(f"[动态规划Agent] 开始处理问题")
        print(f"{'='*50}")
        print(f"问题: {question}\n")
        
        # 重置状态
        self.history = []
        self.replan_count = 0
        
        # 1. 生成初始计划
        plan = self._generate_plan(question, failure_info="无")
        if not plan:
            return "无法生成初始计划。"

        step_index = 0
        retry_count = 0  # 当前步骤的重试计数器
        
        while step_index < len(plan):
            current_step = plan[step_index]
            print(f"\n{'─'*40}")
            print(f"📌 正在执行步骤 {step_index + 1}/{len(plan)}: {current_step}")
            print(f"{'─'*40}")

            # 2. 执行单个步骤
            result = self._execute_step(question, plan, current_step)
            
            # 安全地截取结果用于显示
            display_result = result[:150] + "..." if len(result) > 150 else result
            print(f"   📋 步骤结果: {display_result}")

            # 3. 评估执行结果
            evaluation = self.evaluator.evaluate(question, current_step, result)
            print(f"   🔍 评估结论: {evaluation}")

            if evaluation == "SUCCESS":
                self.history.append({"step": current_step, "result": result})
                step_index += 1
                retry_count = 0  # 重置重试计数器
                print(f"   ✅ 步骤 {step_index} 已成功完成")

            elif evaluation == "FAILURE_RETRY":
                retry_count += 1
                if retry_count > self.MAX_RETRIES:
                    print(f"   ❌ 步骤重试次数已达上限 ({self.MAX_RETRIES})，触发重规划...")
                    evaluation = "FAILURE_REPLAN"  # 降级为重规划
                else:
                    print(f"   ⚠️ 步骤失败，正在重试 ({retry_count}/{self.MAX_RETRIES})...")
                    continue  # 重新执行当前步骤

            # 处理重规划（包括从 FAILURE_RETRY 降级来的情况）
            if evaluation == "FAILURE_REPLAN":
                self.replan_count += 1
                if self.replan_count > self.MAX_REPLANS:
                    print(f"\n❌ 达到最大重规划次数 ({self.MAX_REPLANS})，任务失败。")
                    return "达到最大重规划次数，任务失败。"

                print(f"\n   🔄 触发重规划 (第 {self.replan_count}/{self.MAX_REPLANS} 次)...")
                failure_info = f"在执行步骤 '{current_step}' 时失败。\n执行结果/原因: {result}"
                
                # 4. 动态重规划
                plan = self._generate_plan(question, failure_info=failure_info)
                if not plan:
                    return "重规划失败，无法生成新计划。"
                    
                step_index = 0  # 从新计划的第一步开始
                retry_count = 0  # 重置重试计数器

        # 所有步骤执行完成
        final_answer = self.history[-1]["result"] if self.history else "未能完成任务"
        print(f"\n{'='*50}")
        print(f"🎉 任务完成!")
        print(f"{'='*50}")
        print(f"最终答案: {final_answer}")
        return final_answer

    def _generate_plan(self, question: str, failure_info: str) -> list[str]:
        """
        生成或重新生成计划。
        
        参数:
        - question: 用户的原始问题
        - failure_info: 失败信息（如果是重规划）
        
        返回:
        - 计划步骤列表
        """
        # 将历史记录格式化为字符串
        if self.history:
            history_str = "\n".join([
                f"- 步骤: {h['step']}\n  结果: {h['result']}" 
                for h in self.history
            ])
        else:
            history_str = "无"
        
        # 调用 Planner 生成计划
        return self.planner.plan(
            question=question,
            history=history_str,
            failure_info=failure_info
        )

    def _execute_step(self, question: str, plan: list[str], step: str) -> str:
        """
        执行单个步骤。
        
        参数:
        - question: 用户的原始问题
        - plan: 当前的完整计划
        - step: 当前要执行的步骤
        
        返回:
        - 执行结果字符串
        """
        # 格式化历史记录
        if self.history:
            history_str = "\n".join([
                f"步骤: {h['step']}\n结果: {h['result']}" 
                for h in self.history
            ])
        else:
            history_str = "无"
        
        # 格式化计划列表为可读字符串
        plan_str = "\n".join([f"{i+1}. {s}" for i, s in enumerate(plan)])
        
        prompt = EXECUTOR_PROMPT_TEMPLATE.format(
            question=question,
            plan=plan_str,
            history=history_str,
            current_step=step
        )
        messages = [{"role": "user", "content": prompt}]
        return self.llm_client.think(messages=messages) or ""


if __name__ == "__main__":
    # 加载环境变量
    load_dotenv()
    
    # 初始化LLM客户端
    llm_client = BaseAgent()
    
    # 创建动态规划智能体实例
    agent = DynamicPlanAndSolveAgent(llm_client)
    
    # ============================================================
    # 测试案例集：展现动态重规划Agent的威力
    # ============================================================
    
    # 案例1：信息不完整/需要调整策略的问题
    # 场景：初始计划可能因为缺少关键信息而失败，需要重规划补充信息获取步骤
    case_1 = """
    任务：为一家初创公司制定进入中国市场的策略。
    要求：
    1. 分析目标市场的竞争格局
    2. 提出差异化定位建议
    3. 给出具体的进入市场行动计划
    
    注意：该公司是一家AI教育科技公司，主打产品是AI辅导机器人。
    """
    
    # 案例2：多路径探索问题 (最能体现重规划价值)
    # 场景：问题有多种解法，如果第一种方法行不通，需要换一种方法
    case_2 = """
    任务：帮我找出以下逻辑谜题的答案。

    有5个人：Alice, Bob, Carol, David, Eve
    他们分别住在5栋不同颜色的房子里：红、蓝、绿、黄、白
    已知条件：
    1. Alice不住红房子
    2. Bob住在Alice的左边（相邻）
    3. Carol住蓝房子
    4. David不和Eve相邻
    5. 绿房子在红房子的右边（相邻）
    6. Eve不住黄房子也不住白房子

    请推理出每个人住什么颜色的房子？
    
    如果某个推理路径走不通，请尝试其他的推理顺序。
    """
    
    # 案例3：复杂依赖链问题
    # 场景：后续步骤依赖前序步骤的正确结果，如果发现矛盾需要回溯修正
    case_3 = """
    任务：设计一个微服务架构的技术方案。
    
    背景：
    - 现有单体应用需要拆分为微服务
    - 日活用户约100万
    - 需要支持高并发订单处理
    - 团队有10名开发人员
    
    要求：
    1. 确定服务拆分边界
    2. 设计服务间通信方案
    3. 规划数据库拆分策略
    4. 考虑服务发现和负载均衡
    5. 设计监控和日志方案
    
    如果在设计过程中发现某个决策与其他部分冲突，需要回退并重新调整方案。
    """
    
    # 案例4 (推荐): 需要"试错"的研究型问题
    # 场景：问题本身没有标准答案，需要尝试多种方法并根据结果调整
    case_4 = """
    任务：分析"为什么90%的创业公司会失败"这个问题。
    
    要求：
    1. 从资金、团队、市场、产品、时机等多个维度分析
    2. 找出最关键的3个失败原因
    3. 针对每个原因给出可操作的规避建议
    4. 如果分析过程中发现某个维度证据不足，需要调整分析框架
    
    预期：在分析过程中可能需要多次调整分析角度和权重。
    """
    
    # ============================================================
    # 选择要运行的案例
    # ============================================================
    print("\n" + "="*60)
    print("可用测试案例:")
    print("="*60)
    print("1. 市场进入策略 (信息不完整场景)")
    print("2. 逻辑谜题推理 (多路径探索场景) ⭐推荐")
    print("3. 微服务架构设计 (复杂依赖链场景)")
    print("4. 创业失败分析 (试错研究场景) ⭐推荐")
    print("="*60)
    
    # 默认运行案例2（最能体现动态重规划的价值）
    selected_case = case_2
    
    # 运行选中的案例
    agent.run(selected_case)
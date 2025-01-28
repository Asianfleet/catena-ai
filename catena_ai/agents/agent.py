from typing import Any, Dict, Iterator, List, Optional
from enum import Enum

class AgentState(Enum):
    READY = "ready"
    RUNNING = "running" 
    FINISHED = "finished"
    ERROR = "error"

class Agent:
    """
    基础Agent类，实现迭代执行的核心逻辑
    """
    def __init__(
        self,
        max_iterations: int = 10,
        early_stopping_method: str = "force",
        **kwargs
    ):
        """
        初始化Agent

        Args:
            max_iterations: 最大迭代次数
            early_stopping_method: 提前停止方法 ("force" 强制停止/"soft" 软停止)
            **kwargs: 其他参数
        """
        self.max_iterations = max_iterations
        self.early_stopping_method = early_stopping_method
        self.iteration_count = 0
        self.state = AgentState.READY
        self.intermediate_steps: List[Dict[str, Any]] = []
        
    def _should_continue(self) -> bool:
        """
        判断是否应该继续执行
        
        Returns:
            bool: 是否继续执行
        """
        # 检查最大迭代次数
        if self.iteration_count >= self.max_iterations:
            if self.early_stopping_method == "force":
                return False
            elif self.early_stopping_method == "soft":
                # 软停止时可以根据其他条件决定是否继续
                return self._check_soft_stopping()
        
        # 检查状态
        if self.state in [AgentState.FINISHED, AgentState.ERROR]:
            return False
            
        return True
    
    def _check_soft_stopping(self) -> bool:
        """
        软停止检查逻辑，子类可以重写此方法实现自定义的停止条件
        
        Returns:
            bool: 是否继续执行
        """
        return False

    def _before_iteration(self) -> None:
        """
        每次迭代前的处理，子类可以重写此方法
        """
        pass

    def _run_iteration(self) -> Dict[str, Any]:
        """
        执行单次迭代的核心逻辑，子类必须重写此方法
        
        Returns:
            Dict[str, Any]: 迭代执行结果
        """
        raise NotImplementedError("子类必须实现_run_iteration方法")

    def _after_iteration(self, result: Dict[str, Any]) -> None:
        """
        每次迭代后的处理，子类可以重写此方法
        
        Args:
            result: 迭代执行结果
        """
        self.intermediate_steps.append(result)
        self.iteration_count += 1

    def run(self) -> Iterator[Dict[str, Any]]:
        """
        运行Agent，返回一个迭代器
        
        Yields:
            Dict[str, Any]: 每次迭代的执行结果
        """
        self.state = AgentState.RUNNING
        
        try:
            while self._should_continue():
                self._before_iteration()
                
                result = self._run_iteration()
                
                self._after_iteration(result)
                
                yield result
                
            self.state = AgentState.FINISHED
            
        except Exception as e:
            self.state = AgentState.ERROR
            raise e
    
    def reset(self) -> None:
        """
        重置Agent状态
        """
        self.iteration_count = 0
        self.state = AgentState.READY
        self.intermediate_steps = []

    @property
    def has_finished(self) -> bool:
        """
        检查Agent是否已完成执行
        
        Returns:
            bool: 是否已完成
        """
        return self.state in [AgentState.FINISHED, AgentState.ERROR]


class NextAction(str, Enum):
    CONTINUE = "continue"
    VALIDATE = "validate"
    FINAL_ANSWER = "final_answer"


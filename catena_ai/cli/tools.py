import re
from functools import wraps
from rich.tree import Tree
from rich.console import Console
from typing import Optional, Union
from enum import Enum
from contextlib import contextmanager
from ..settings import settings

class Style(Enum):
    R = "[reset]"          # 重置所有样式
    B = "[bold]"           # 粗体
    BW = "[bold white]"    # 粗体白色
    BG = "[bold green]"    # 粗体绿色
    BGR = "[bold gray]"    # 粗体灰色
    GR = "[gray]"          # 灰色
    GR_L = "[#C9C9C9]"     # 浅灰色
    BBL = "[bold black]"   # 粗体黑色
    BO = "[bold orange]"   # 粗体橙色
    BR = "[bold red]"      # 粗体红色
    BY = "[bold yellow]"   # 粗体黄色
    BB = "[bold blue]"     # 粗体蓝色
    BC = "[bold cyan]"     # 粗体青色
    BM = "[bold magenta]"  # 粗体品红色
    U = "[underline]"      # 下划线
    I = "[italic]"         # 斜体
    S = "[strikethrough]"  # 删除线
    H = "[highlight]"      # 高亮
    L = "[lowlight]"       # 低亮
    D = "[dim]"            # 暗淡
    BL = "[blink]"         # 闪烁
    RE = "[reverse]"        # 反转颜色
    C = "[conceal]"        # 隐藏文本
    
    DOT = "•"               # 圆点
    SDOT = "▪"              # 小方点
    HB = "▬"                # 水平线
    RHT = "▷"               # 空心右三角
    RT = "▶"                # 实心右三角
    RH = "◈"                # 菱形
    SQR = "▫"               # 小空心方块
    CHECK = "✓"             # 勾选标记
    TOKEN = "🎫"            # token 符号

# 颜色样式
class Formatter:
      
    @classmethod
    def fc(
        cls, text: str, style: Optional[Union[str, Style]] = None
    ) -> str:
        """使用指定样式格式化给定文本。

        参数:
            text (str): 要格式化的文本。
            style (str): 要应用于文本的样式（例如，'bold'，'italic'）。

        返回:
            str: 带有样式标签的格式化文本。
        """
        if not style:
            return text
        style = style if isinstance(style, str) else style.value
        style = "[" + style.strip('[]') + "]"
        return f"{style}{text}[/{style.strip('[]')}]"
    
    @classmethod
    def printf(cls, *values, style: str):
        """使用指定样式将格式化文本打印到控制台。

        参数:
            *values: 要打印的值，这些值将连接成一个字符串。
            style (str): 要应用于文本的样式（例如，'bold'，'underline'）。
        """

        console = Console()
        values = [str(value) for value in values]
        console.print(cls.fc("".join(values), style))
      
condition = "None"
extra_condition = {}

@contextmanager
def info_condition(temp_condition, **kwargs):
    if kwargs.get("level", "User") == "Code":
        global condition
        original_condition = condition
        condition = temp_condition
    else:
        original_condition="None"

    global extra_condition
    original_extra = extra_condition.copy()
    extra_condition.update(kwargs)
    try:
        yield
    finally:
        condition = original_condition
        extra_condition = original_extra
      
def info(*values, prefix: Optional[str] = None):
    """ 输出运行信息 """
    global condition
    global extra_condition
    condition = (
        condition 
        if condition != "None" 
        else settings.log.enable_info
    )
    if condition:
        if prefix:
            pre = f"\[{prefix}|info] " 
        else:
            match = re.search(r'^\[(.*?)\]', values[0])
            if match:
                pre = f"\[{match.group(1)}|info] "
                newvalue = re.sub(r'\[(.*?)\]', '', values[0], count=1)
                newvalues = (newvalue, *values[1:])
            else:
                pre = "\[info] "
                newvalues = values
        if extra_condition.get("pre", True) == False:
            pre = ""
        Formatter.printf(pre, *newvalues, style=None)
 
def debug(*values, prefix: Optional[str] = None):
    """ 输出调试信息 """
    if settings.log.enable_debug:
        pre = f"\[{prefix}.info] " if prefix else "\[info] "
        Formatter.printf(pre, *values, style="bold")       

def warning(*values, prefix: Optional[str] = None):
    """ 输出警告信息 """
    if settings.log.enable_warning:
        if prefix:
            pre = f"\[{prefix}|warning] " 
            newvalues = values
        else:
            match = re.search(r'^\[(.*?)\]', values[0])
            if match:
                pre = f"\[{match.group(1)}|warning] "
                newvalue = re.sub(r'\[(.*?)\]', '', values[0], count=1)
                newvalues = (newvalue, *values[1:])
            else:
                pre = "\[warning] "
                newvalues = values
        if extra_condition.get("pre", True) == False:
            pre = ""
        Formatter.printf(pre, *newvalues, style=None) 

def error(*values, prefix: Optional[str] = None):
    """ 输出错误信息 """
    if settings.log.enable_error:
        pre = f"\[{prefix}.info] " if prefix else "\[info] "
        Formatter.printf(pre, *values, style="bold red")
       
if __name__ == "__main__":
    # python -m catena_ai.cli.tools
    #print(Style.fc("Hello, world!", "bold underline"))
    #Style.printf("Hello, world!", style="bold underline")
    
    """ console = Console()

    import time
    from rich.tree import Tree
    from rich.live import Live    
    tree = Tree("🎄")
    live = Live(tree, console=Console())
    
    with console.status("[bold green]Loading...") as status:
        
        live.start()
        # 模拟一系列耗时操作
        for i in range(5):
            tree.add(f"Red {i}")
            live.update(tree)
            time.sleep(3)
            status.update(f"[bold green]Processing... {i+1}/3")
        live.stop()
        
    console.print("Done!") """
    with info_condition(settings.visualize.message_metrics, pre=False):
        info("[111]warning")

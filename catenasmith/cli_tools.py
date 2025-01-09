from functools import wraps
from rich.tree import Tree
from rich.console import Console
from typing import Union
from enum import Enum
from ..catena_core.settings import settings

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
    def fc(cls, text: str, style: str | Style) -> str:
        """使用指定样式格式化给定文本。

        参数:
            text (str): 要格式化的文本。
            style (str): 要应用于文本的样式（例如，'bold'，'italic'）。

        返回:
            str: 带有样式标签的格式化文本。
        """
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
      
def info(*values):
    """ 输出运行信息 """
    if settings.debug.enable_func_level_info:
        print(*values)
 
def debug(*values):
    """ 输出调试信息 """
    if settings.debug.enable_func_level_debug:
        print(*values)       
       
        
if __name__ == "__main__":
    print(Style.fc("Hello, world!", "bold underline"))
    Style.printf("Hello, world!", style="bold underline")
    
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
    
    
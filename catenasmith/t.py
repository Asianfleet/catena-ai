""" from rich.console import Console
from rich.status import Status
import time



console = Console()

with console.status("[bold green]Working on tasks...") as status:
    for step in range(3):
        print("fiwfgwigiwg")
        time.sleep(1)  # 模拟任务处理时间
        if step == 0:
            status.update("[bold yellow]Step 1 complete")
        elif step == 1:
            status.update("[bold magenta]Step 2 complete")
        else:
            status.update("[bold cyan]Step 3 complete")

console.print("[bold green]All tasks completed!") """

""" from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

# 自定义标题样式
title_style = "bold magenta"
subtitle_style = "italic blue"

# 创建一个带有自定义样式的标题
title_text = Text("Title 1", style=title_style)
subtitle_text = Text("Subtitle 1", style=subtitle_style)

# 创建 Panel 并设置边框颜色
panel = Panel(
    "Region 1",
    title=title_text,
    subtitle=subtitle_text,
    border_style="green"  # 设置边框颜色为绿色
)

console.print(panel) """



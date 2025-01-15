from rich.tree import Tree
from rich.live import Live
from rich.status import Status
from rich.console import Console, Group
import time

console = Console()

# 初始化 Status 和 Tree
status = Status("Thinking...", spinner="aesthetic", speed=2.0, refresh_per_second=10)
tree = Tree("Dynamic Tree")

list_of_nodes = [status, tree]

# 创建一个 Group 包含 Status 和 Tree
group = Group(status, tree)

# 创建 Live 对象
live = Live(Group(*list_of_nodes), console=console, auto_refresh=True)

# 启动 Live 对象
live.start()

try:
    while True:
        # 更新 Tree
        tree.add(f"Node {time.time()}")
        
        # 更新 Group 并刷新 Live
        #group = Group(status, tree)
        live.update(Group(*list_of_nodes))
        
        # 等待1秒
        time.sleep(1)
except KeyboardInterrupt:
    # 停止 Live 对象
    live.stop()
    print("\n程序已终止")

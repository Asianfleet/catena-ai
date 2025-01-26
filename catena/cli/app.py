import typer

from .commands import generate_prompt_template

# 创建 typer 应用
app = typer.Typer()

# 注册命令
app.command(
    name="create_template", help="交互式创建提示词模板"
)(
    generate_prompt_template
)

# 主函数
def main():
    app()

if __name__ == "__main__":
    # python -m catena.cli.app
    main()

import os
import yaml
import json
import typer
import questionary
from typing import Dict
from questionary import Style
from rich.console import Console

from ..catena_core.utils.utils import TemplateCreater as TC

console = Console()
# 自定义样式
custom_style = Style([
    ('qmark', 'fg:#00ff00 bold'),  # 修改提示符的颜色和样式
    ('pointer', 'fg:#00ff00 bold'),  # 修改箭头指针的颜色和样式
    ('selected', 'fg:#00ff00 bold'),  # 修改选中项的颜色和样式
])



def generate_prompt_template(
    name: str = typer.Option(None, prompt="Please enter your prompt name")
):
    """  """
    
    # 1、设置模板名称
    if not name:
        name = typer.prompt("Please enter your prompt name (cannot be empty)")
    prompt_template = TC(name)
    
    # 2、设置角色面具
    mask = typer.prompt("Please enter your prompt mask", default="你是一个强大的助手")
    prompt_template.new_message(role="system", content=mask)
    
    # 3、设置消息
    while True:
        role = questionary.select(
            "Choose message role:",
            choices=["system", "user"],
            default="user",  # 默认选中第一个选项
            style=custom_style,  # 使用自定义样式
            qmark="❯",  # 将问号替换为其他符号
        ).ask()  # 获取用户选择 
        content = typer.prompt("message content")        
        prompt_template.new_message(role=role, content=content)
        
        # 询问是否继续
        continue_interaction = questionary.select(
            "Add a new message?",
            choices=["Yes", "No"],
            default="Yes",
            style=custom_style,
            qmark="❯",
        ).ask()
        
        if continue_interaction == "No":
            console.print("-----------------------------------------")
            break
    # 4、设置模板元数据    
    response_type = questionary.select(
        "Choose model response type",
        choices=["test", "json_object"],
        default="test",  # 默认选中第一个选项
        style=custom_style,  # 使用自定义样式
        qmark="❯",  # 将问号替换为其他符号
    ).ask()  # 获取用户选择
    prompt_template.add_meta({"output": {"type": response_type}})
    
    # 是否创建新文件
    create = questionary.select(
        "Create new file?",
        choices=["Yes", "No(Add to existing file)"],
        default="No(Add to existing file)",
        style=custom_style,
        qmark="❯",
    ).ask()
    
    if create == "Yes":
        file_type = questionary.select(
            "Select file type",
            choices=["json", "yaml"],
            default="yaml",
            style=custom_style,
            qmark="❯",
        ).ask()

        location = questionary.select(
            "Select file location",
            choices=["custom", "cwd"],
            default="cwd",
            style=custom_style,
            qmark="❯",
        ).ask()
        
        if location == "custom":
            file_location = os.path.expanduser(
                typer.prompt("Please enter your file location")
            )
        else:
            file_location = os.getcwd()
        if file_type == "json":
            file_location = os.path.join(file_location, f"{name}.json")
            with open(file_location, "w") as f:
                json.dump(prompt_template.create_template(), f, indent=4)
        if file_type == "yaml":
            file_location = os.path.join(file_location, f"{name}.yaml")
            with open(file_location, "w") as f:
                yaml.dump(prompt_template.create_template(), f)
        console.print(f"[green]Prompt template created successfully![/green]")
    else:
        location = questionary.select(
            "Select file location",
            choices=["custom", "cwd"],
            default="cwd",
            style=custom_style,
            qmark="❯",
        ).ask()
        if location == "custom":
            file_location = os.path.expanduser(
                typer.prompt("Please enter your file location")
            )
        else:
            file_location = os.getcwd()
        
        while True:
            # 读取目录中的json/yaml文件
            files = []
            for f in os.listdir(file_location):
                if f.endswith('.json') or f.endswith('.yaml'):
                    files.append(f)
            
            if not files:
                console.print("[red]当前目录为空，请重新选择文件位置[/red]")
                location = questionary.select(
                    "选择文件位置",
                    choices=["重新选择位置", "输入新路径"],
                    default="重新选择位置",
                    style=custom_style,
                    qmark="❯",
                ).ask()
                
                if location == "重新选择位置":
                    location = questionary.select(
                        "选择文件位置",
                        choices=["custom", "cwd"],
                        default="cwd",
                        style=custom_style,
                        qmark="❯",
                    ).ask()
                    if location == "custom":
                        file_location = typer.prompt("请输入文件位置")
                    else:
                        file_location = os.getcwd()
                else:
                    file_location = typer.prompt("请输入新的文件路径")
                continue
            
            # 选择文件
            selected_file: str = questionary.select(
                "选择要写入的文件:",
                choices=files,
                default=files[0],
                style=custom_style,
                qmark="❯"
            ).ask()
            break
        
        file_path = os.path.join(file_location, selected_file)
        
        # 写入文件
        template = prompt_template.create_template()
        if selected_file.endswith('.json'):
            with open(file_path, 'a') as f:
                json.dump(template, f, indent=4)
        else:
            with open(file_path, 'a') as f:
                yaml.dump(template, f, default_flow_style=False)
        
        console.print(f"[green]成功写入文件: {selected_file}[/green]")

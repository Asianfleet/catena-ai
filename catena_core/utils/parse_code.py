import ast

# 定义代码字符串
code = """
def find_max(numbers):
    return max(numbers)
"""

def parse_func_object(code):

    # 解析代码字符串为 AST
    tree = ast.parse(code)

    # 确保只有一个函数定义
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Code must contain exactly one function definition.")

    # 获取函数定义节点
    func_def = tree.body[0]

    # 获取函数名称
    func_name = func_def.name

    # 编译 AST 为字节码（使用 'exec' 模式）
    compiled_code = compile(tree, filename="<ast>", mode="exec")

    # 创建一个新的全局命名空间来执行代码
    global_namespace = {}

    # 执行编译后的代码以在全局命名空间中定义函数
    exec(compiled_code, global_namespace)

    # 从全局命名空间中获取动态创建的函数
    dynamic_function = global_namespace[func_name]
    
    return dynamic_function



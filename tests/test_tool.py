import unittest
from typing import Any

from catena.error import toolserr
from catena.agents.tools import tool, Tool
from catena.catena_core.tools.base import BaseTool
from catena.catena_core.utils.timer import record_time
from catena.catena_core.tools.tool_registry import ToolRegistry

class TestToolDecorator(unittest.TestCase):
    """测试工具装饰器"""
    
    def setUp(self):
        # 清空工具注册表
        #ToolRegistry.clear()
        pass
        
    def test_basic_decorator(self):
        """测试基本装饰器功能"""
        
        @tool
        def sample_tool():
            """示例工具"""
            return "success"
            
        self.assertTrue(hasattr(sample_tool, "__is_registered__"))
        self.assertTrue(sample_tool.__is_registered__)
        self.assertEqual(sample_tool(), "success")
        
    def test_decorator_with_parameters(self):
        """测试带参数的装饰器"""
        
        @tool(name="custom_name", description="custom description")
        def sample_tool():
            """ return "success" """
            
        registered_tool = ToolRegistry.get_tool("custom_name")
        self.assertIsNotNone(registered_tool)
        self.assertEqual(registered_tool.description, "custom description")
        
    def test_error_invalid_kwargs(self):
        """测试无效参数"""
        
        with self.assertRaises(ValueError):
            @tool(invalid_param=True)
            def sample_tool():
                pass
                
    def test_error_no_description(self):
        """测试没有文档字符串引发异常"""
        
        with self.assertRaises(ValueError):
            @tool
            def failing_tool():
                pass
            failing_tool()

class TestToolConstructor(unittest.TestCase):
    """测试工具构造器"""
    
    def test_error_metadata_validation(self):
        """测试元数据验证"""
        
        with self.assertRaises(toolserr.ToolMetaDataInitializeError):
            Tool(name="test")
            
    # TODO: 未测试
    def test_execute_method(self):
        """测试执行方法"""
        
        def sample_func(a: int = 1):
            return "executed"
        
        class TestTool_entry(Tool[int]):
            @record_time
            def _execute(self) -> Any:
                """ 执行工具的核心方法的抽象接口 """   
                return "executed"
            
            def validate_input(self, *args, **kwargs) -> bool:
                """ 验证输入参数 """
                return True
            
            def validate_output(self, output: Any) -> bool:
                """ 验证输出结果 """
                return True
            
        class TestTool_noentry(Tool[int]):
            
            @record_time
            def _execute(self) -> Any:
                """ 执行工具的核心方法的抽象接口 """   
                return "executed"
            
            def validate_input(self, *args, **kwargs) -> bool:
                """ 验证输入参数 """
                return True
            
            def validate_output(self, output: Any) -> bool:
                """ 验证输出结果 """
                return True    
            
        tool_instance_entry = TestTool_entry(
            name="test_tool",
            description="test description",
            entry_function=sample_func
        )
        
        tool_instance_noentry = TestTool_noentry(
            name="test_tool",
            description="test description"
        )
        
        result_entry = tool_instance_entry.execute(a=1)
        result_ne = tool_instance_noentry.execute()
        self.assertEqual(result_entry.output, "executed")
        self.assertEqual(result_entry.status, "success")
        self.assertEqual(result_ne.output, "executed")
        self.assertEqual(result_ne.status, "success")
        
    def test_error_implement(self):
        """测试输入验证"""
        
        class TestTool_noimpl(BaseTool[str]):
            pass
                
        with self.assertRaises(TypeError):
            tool_instance_noimpl = TestTool_noimpl(
            name="test_tool",
            description="test description"
        )

    def test_error_validation(self):
        """测试输入验证"""
        
        class TestTool_inp(BaseTool[str]):

            @record_time
            def _execute(self) -> Any:
                """ 执行工具的核心方法的抽象接口 """   
                return "executed"    
                    
            def validate_input(self, *args, **kwargs) -> bool:
                """ 验证输入参数 """
                raise toolserr.ToolInputValidateError("输入参数验证失败")
        
            def validate_output(self, output: Any) -> bool:
                """ 验证输出结果 """
                pass
                #raise toolserr.ToolOutputValidateError("输出结果验证失败")
                
        class TestTool_op(BaseTool[str]):

            @record_time
            def _execute(self) -> Any:
                """ 执行工具的核心方法的抽象接口 """   
                return "executed"    
                    
            def validate_input(self, *args, **kwargs) -> bool:
                """ 验证输入参数 """
                pass
        
            def validate_output(self, output: Any) -> bool:
                """ 验证输出结果 """
                raise toolserr.ToolOutputValidateError("输出结果验证失败")
                
        class TestTool_exec(BaseTool[str]):
            @record_time
            def _execute(self) -> Any:
                """ 执行工具的核心方法的抽象接口 """   
                raise Exception 
            
            def validate_output(self, output: Any) -> bool:
                """ 验证输出结果 """
                pass
            
            def validate_input(self, *args, **kwargs) -> bool:
                """ 验证输入参数 """
                pass
                
        with self.assertRaises(toolserr.ToolInputValidateError):
            tool_instance = TestTool_inp(
                name="test_tool",
                description="test description"
            )
            tool_instance.execute()
        with self.assertRaises(toolserr.ToolOutputValidateError):
            tool_instance = TestTool_op(
                name="test_tool",
                description="test description"
            )
            tool_instance.execute()
        with self.assertRaises(Exception):
            tool_instance = TestTool_exec(
                name="test_tool",
                description="test description"
            )
            tool_instance.execute()

    def test_return_type_inference(self):
        """测试返回类型推断"""
        
        class TestTool_int(Tool[int]):
            
            def _execute(self, a: int) -> Any:
                return 1
            
            def validate_input(self, *args, **kwargs):
                return True
            
            def validate_output(self, output: Any) -> bool:
                return True
            
        tool_instance_int = TestTool_int(
            name="test_tool",
            description="test description"
        )
        
        self.assertEqual(tool_instance_int.return_type, int)

    def test_parameter_schema_generation(self):
        """测试参数schema生成"""
        
        def sample_func(a: int, b: str = "default") -> None:
            pass
            
        tool_instance = Tool(
            name="test_tool",
            description="test description",
            entry_function=sample_func
        )
        
        expected_schema = {
            'a': {
                'type': int,
                'default': None,
                'required': True
            },
            'b': {
                'type': str,
                'default': 'default',
                'required': False
            },
            'return': None
        }
        
        self.assertEqual(tool_instance.parameters_schema, expected_schema)

    #def test_auto_create_functionality(self):
    #    """测试自动创建功能"""
    #    
    #    class TestTool(Tool[str]):
    #        auto_create = True
    #        
    #        def create_tool_auto(self):
    #            return "auto_created"
    #            
    #    tool_instance = TestTool(
    #        name="test_tool",
    #        description="test description"
    #    )
    #    
    #    result = tool_instance.execute()
    #    self.assertEqual(result.output, "auto_created")

    def test_error_invalid_parameter_names(self):
        """测试无效参数名"""
        
        def invalid_func(args, kwargs):
            pass
            
        with self.assertRaises(toolserr.ToolMetaDataInitializeError):
            Tool(
                name="test_tool",
                description="test description",
                entry_function=invalid_func
            )

    def test_missing_type_annotations(self):
        """测试缺少类型注解"""
        
        def invalid_func(a):
            pass
            
        with self.assertRaises(toolserr.ToolMetaDataInitializeError):
            Tool(
                name="test_tool",
                description="test description",
                entry_function=invalid_func
            )

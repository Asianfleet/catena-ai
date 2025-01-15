""" ``catena-core`` 定义了 CatenaAI 智能体框架的基本接口。

包含回调管理、配置管理、工具管理等核心组件的基本接口，包括：
 - 通用节点调用协议
 - 流式调用语法支持
 - 智能体框架的工具基本接口
 - 智能体框架的配置管理
 - 智能体框架的回调管理
 - 智能体框架的代理基类

为减少基本受依赖库版本的影响，catena-core 将保持极少数的第三方库引入，
这些引入的第三方库通常是有强大的社区基础，使用非常广泛且确定将会长期维护的库，包括：
 - `pydantic`
 - `pandas`
 - `numpy`
 - `opencv`
 - `requests`
 """
""" ``catena-core`` 定义了 CatenaAI 系统的基本抽象。

包含回调管理、配置管理等核心组件的接口
节点通用调用协议（Nodes）以及组合组件的语法（CatenaAI 流式调用语言）。

为减少底层受依赖库版本的影响，catena-core 将保持极少数的第三方库引入，
这些引入的第三方库通常是使用非常广泛且确定将会长期维护的库，
如 ``pydantic`` 、``pandas``、``numpy``、``opencv`` 等。

 """
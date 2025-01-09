import re
from pydantic import BaseModel
from pydantic.fields import FieldInfo



def _default_instructions(cls) -> str:
    inputs_ = ", ".join([f"`{field}`" for field in cls.input_fields])
    outputs_ = ", ".join([f"`{field}`" for field in cls.output_fields])
    return f"Given the fields {inputs_}, produce the fields {outputs_}."


class SignatureMeta(type(BaseModel)):
    def __call__(cls, *args, **kwargs):  # noqa: ANN002
        return super().__call__(*args, **kwargs)

    def __new__(mcs, signature_name, bases, namespace, **kwargs):  # noqa: N804
        # 将 `str` 设置为所有字段的默认类型
        raw_annotations = namespace.get("__annotations__", {})
        for name, field in namespace.items():
            if not isinstance(field, FieldInfo):
                continue  # Don't add types to non-field attributes
            if not name.startswith("__") and name not in raw_annotations:
                raw_annotations[name] = str
        namespace["__annotations__"] = raw_annotations

        # Let Pydantic do its thing
        cls = super().__new__(mcs, signature_name, bases, namespace, **kwargs)

        # If we don't have instructions, it might be because we are a derived generic type.
        # In that case, we should inherit the instructions from the base class.
        if cls.__doc__ is None:
            for base in bases:
                if isinstance(base, SignatureMeta):
                    doc = getattr(base, "__doc__", "")
                    if doc != "":
                        cls.__doc__ = doc

        # The more likely case is that the user has just not given us a type.
        # In that case, we should default to the input/output format.
        if cls.__doc__ is None:
            cls.__doc__ = _default_instructions(cls)

        # Ensure all fields are declared with InputField or OutputField
        cls._validate_fields()

        # Ensure all fields have a prefix
        for name, field in cls.model_fields.items():
            if "prefix" not in field.json_schema_extra:
                field.json_schema_extra["prefix"] = infer_prefix(name) + ":"
            if "desc" not in field.json_schema_extra:
                field.json_schema_extra["desc"] = f"${{{name}}}"

        return cls

    def _validate_fields(cls):
        for name, field in cls.model_fields.items():
            extra = field.json_schema_extra or {}
            field_type = extra.get("__dspy_field_type")
            if field_type not in ["input", "output"]:
                raise TypeError(
                    f"Field '{name}' in '{cls.__name__}' must be declared with "
                    "InputField or OutputField. {field.json_schema_extra=}",
                )
            
# 预测器的签名。
#
# 您通常会对其进行子类化，如下所示：
# 类我的签名（签名）：
# 输入：str = InputField(desc="...") # noqa：ERA001
# 输出: int = OutputField(desc="...") # noqa: ERA001
#
# 您可以调用Signature("input1, input2 -> output1, output2") 创建新的签名类型。
# 您还可以包含指令 Signature("input -> output", "This is a test")。
# 但通常最好使用 make_signature 函数。
#
# 如果您不确定您的输入是否是字符串表示形式（例如“input1，input2 -> output1，output2”），
# 或者签名，可以使用ensure_signature函数。
#
# 为了兼容旧版 dsp 格式，可以使用signature_to_template 函数。
#

class Signature(BaseModel, metaclass=SignatureMeta):
    ""  # noqa: D419

    # Note: Don't put a docstring here, as it will become the default instructions
    # for any signature that doesn't define it's own instructions.
    pass




def infer_prefix(attribute_name: str) -> str:
    """Infer a prefix from an attribute name."""
    # Convert camelCase to snake_case, but handle sequences of capital letters properly
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", attribute_name)
    intermediate_name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)

    # Insert underscores around numbers to ensure spaces in the final output
    with_underscores_around_numbers = re.sub(
        r"([a-zA-Z])(\d)",
        r"\1_\2",
        intermediate_name,
    )
    with_underscores_around_numbers = re.sub(
        r"(\d)([a-zA-Z])",
        r"\1_\2",
        with_underscores_around_numbers,
    )








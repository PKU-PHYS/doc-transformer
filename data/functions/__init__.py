from .registry import MathRelation, FunctionRegistry

# 导入各函数模块以触发注册
from . import unary
from . import binary
from . import multivar
from . import implicit
from . import composite

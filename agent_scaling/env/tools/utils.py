import functools
import inspect
from typing import (
    Annotated,
    Any,  # for typing for StructureTool (keep here)
    Callable,
    Literal,
    Optional,
    TypeVar,
    Union,
    overload,
)

from langchain_core.callbacks.manager import (
    CallbackManagerForToolRun,
)  # for typing for StructureTool (keep here)
from langchain_core.runnables import (
    RunnableConfig,
)  # for typing for StructureTool (keep here)
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.tools.base import ArgsSchema
from langfuse._client.observe import observe
from pydantic import Field


def enhance_tool(
    tool_instance: BaseTool,
    use_langfuse: bool = False,
    output_parse_func: Optional[Callable] = None,
):

    if hasattr(tool_instance, "use_langfuse") and hasattr(
        tool_instance, "output_parse_func"
    ):
        tool_instance.use_langfuse = use_langfuse  # type: ignore[assignment]
        # tool_instance.output_parse_func = output_parse_func
        return tool_instance

    tool_cls = tool_instance.__class__

    # Inspect parent signatures
    run_sig = inspect.signature(tool_cls._run)
    arun_sig = inspect.signature(tool_cls._arun)

    class EnhancedTool(tool_cls):
        """Enhanced tool class that adds Langfuse observation and output parsing."""

        use_langfuse: bool = False
        output_parse_func: Optional[Callable] = Field(
            default=None, description="Function to parse tool output.", exclude=True
        )
        parent_tool_cls: Optional[str] = tool_cls.__name__

        def _run(self, *args, **kwargs):
            def helper(*h_args, **h_kwargs):
                ret = super(EnhancedTool, self)._run(*h_args, **h_kwargs)
                if self.output_parse_func is not None:
                    ret = self.output_parse_func(ret)
                return ret

            if self.use_langfuse:
                func = observe(helper, name=f"tool: {self.name}")  # type: ignore
            else:
                func = helper
            return func(*args, **kwargs)

        async def _arun(self, *args, **kwargs):
            async def helper(*h_args, **h_kwargs):
                ret = await super(EnhancedTool, self)._arun(*h_args, **h_kwargs)
                if self.output_parse_func is not None:
                    ret = self.output_parse_func(ret)
                return ret

            if self.use_langfuse:
                func = observe(helper, name=f"tool: {self.name}")  # type: ignore
            else:
                func = helper
            return await func(*args, **kwargs)

    EnhancedTool._run.__signature__ = run_sig
    EnhancedTool._arun.__signature__ = arun_sig

    # needed for StructuredTool to work correctly, as they add inputs based on the signature and type hints
    EnhancedTool._run.__annotations__ = tool_cls._run.__annotations__.copy()
    EnhancedTool._arun.__annotations__ = tool_cls._arun.__annotations__.copy()

    return EnhancedTool(
        use_langfuse=use_langfuse,
        output_parse_func=output_parse_func,
        **tool_instance.model_dump(),
    )


class _ToolDescriptor:
    def __init__(
        self,
        fn: Callable,
        *,
        name: Optional[str],
        description: Optional[str],
        return_direct: bool,
        args_schema: Optional[ArgsSchema],
        infer_schema: bool,
        response_format: Literal["content", "content_and_artifact"],
        parse_docstring: bool,
        error_on_invalid_docstring: bool,
    ):
        self.original_fn = fn
        self.meta = dict(
            name=name or fn.__name__,
            description=description,
            return_direct=return_direct,
            args_schema=args_schema,
            infer_schema=infer_schema,
            response_format=response_format,
            parse_docstring=parse_docstring,
            error_on_invalid_docstring=error_on_invalid_docstring,
        )
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self.__annotations__ = getattr(fn, "__annotations__", {})

    def __get__(self, instance, owner):
        # when accessed on the class, return the descriptor itself
        if instance is None:
            return self

        # bind fn to the instance
        bound_method = self.original_fn.__get__(instance, owner)

        # wrap to accept the StandardTool args signature
        @functools.wraps(bound_method)
        def wrapper(*args, **kwargs):
            return bound_method(*args, **kwargs)

        # pick async vs sync
        if inspect.iscoroutinefunction(bound_method):
            tool = StructuredTool.from_function(
                func=None, coroutine=wrapper, **self.meta  # type: ignore[call-arg]
            )
            tool.coroutine = wrapper
        else:
            tool = StructuredTool.from_function(
                func=wrapper, coroutine=None, **self.meta  # type: ignore[call-arg]
            )
            tool.func = wrapper

        return tool


_F = TypeVar("_F", bound=Callable[..., object])


@overload
def cls_tool(_fn: _F) -> _ToolDescriptor: ...


@overload
def cls_tool(
    *,
    name: Optional[str] = ...,
    description: Optional[str] = ...,
    return_direct: bool = ...,
    args_schema: Optional[ArgsSchema] = ...,
    infer_schema: bool = ...,
    response_format: Literal["content", "content_and_artifact"] = ...,
    parse_docstring: bool = ...,
    error_on_invalid_docstring: bool = ...,
) -> Callable[[_F], _ToolDescriptor]: ...


def cls_tool(
    _fn: Optional[_F] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    return_direct: bool = False,
    args_schema: Optional[ArgsSchema] = None,
    infer_schema: bool = True,
    response_format: Literal["content", "content_and_artifact"] = "content",
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = True,
) -> Union[_ToolDescriptor, Callable[[_F], _ToolDescriptor]]:
    """
    Decorator that turns a method into a StructuredTool descriptor,
    binding `self` at access‐time and carrying through all metadata.
    """

    def wrap(fn: Callable) -> _ToolDescriptor:
        return _ToolDescriptor(
            fn,
            name=name,
            description=description,
            return_direct=return_direct,
            args_schema=args_schema,
            infer_schema=infer_schema,
            response_format=response_format,
            parse_docstring=parse_docstring,
            error_on_invalid_docstring=error_on_invalid_docstring,
        )

    if _fn is None:
        return wrap
    else:
        return wrap(_fn)

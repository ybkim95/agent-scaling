from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

from agent_scaling.logger import configure_logger
from agent_scaling.utils import read_yaml

configure_logger(level=10)


def same_prompts(
    p1: List[Dict[Literal["role", "content"], str]] | str,
    p2: List[Dict[Literal["role", "content"], str]] | str,
) -> bool:
    if len(p1) != len(p2):
        return False
    for i in range(len(p1)):
        if isinstance(p1[i], dict) and isinstance(p2[i], dict):
            if p1[i]["role"] != p2[i]["role"] or p1[i]["content"] != p2[i]["content"]:  # type: ignore
                return False
        elif isinstance(p1[i], str) and isinstance(p2[i], str):
            if p1[i] != p2[i]:
                return False
        else:
            return False
    return True


class NamedPrompt(BaseModel):
    """A specific named template extracted from a multi-template prompt"""

    name: str
    prompt_template: List[Dict[Literal["role", "content"], str]]

    def compile(self, **kwargs: Any) -> List[dict]:
        """Compile this specific template with given variables"""
        # For now, do simple template variable replacement manually
        compiled_messages = []
        for msg in self.prompt_template:
            compiled_msg = {}
            for key, value in msg.items():
                if isinstance(value, str):
                    # Replace template variables like {{variable}}
                    compiled_value = value
                    for var_name, var_value in kwargs.items():
                        compiled_value = compiled_value.replace(
                            f"{{{{{var_name}}}}}", str(var_value)
                        )
                    compiled_msg[key] = compiled_value
                else:
                    compiled_msg[key] = value
            compiled_messages.append(compiled_msg)

        return compiled_messages


class Prompt(BaseModel):
    """A prompt template with named templates"""

    name: str
    local_path: Optional[str] = None
    prompt_template: Optional[List[Dict[Literal["role", "content"], str]]] = None
    named_templates: Optional[Dict[str, NamedPrompt]] = None

    def model_post_init(self, __context: Any) -> None:
        if (
            not self.prompt_template
            and not self.named_templates
            and not self.local_path
        ):
            raise ValueError(
                f"Prompt {self.name} must have a prompt_template, named_templates, or local_path"
            )
        if self.local_path:
            res = read_yaml(self.local_path)
            if isinstance(res, dict):
                if "base" in res:
                    base_prompt = res["base"]
                else:
                    base_prompt = []
                named_templates_w_base = {
                    f"base_{k}": NamedPrompt(name=k, prompt_template=base_prompt + v)
                    for k, v in res.items()
                    if k != "base"
                }
                named_templates_wo_base = {
                    k: NamedPrompt(name=k, prompt_template=v) for k, v in res.items()
                }
                self.named_templates = {
                    **named_templates_w_base,
                    **named_templates_wo_base,
                }
            else:
                self.prompt_template = res

    def compile(self, **kwargs: Any) -> List[Dict[Literal["role", "content"], str]]:
        if not self.prompt_template:
            raise ValueError(f"Prompt {self.name} has no prompt_template to compile")

        return NamedPrompt(
            name=self.name,
            prompt_template=self.prompt_template,
        ).compile(**kwargs)

    def get_template(self, template_name: str, with_base: bool = True) -> NamedPrompt:
        if self.named_templates is None:
            raise ValueError(f"Named templates are not defined for {self.name}")
        if with_base:
            template_name = f"base_{template_name}"
        return self.named_templates[template_name]


if __name__ == "__main__":
    prompt = Prompt(
        name="movie-critic-chat",
        prompt_template=[
            {"role": "system", "content": "You are a bad {{criticlevel}} movie critic"},
            {"role": "user", "content": "Do you like {{movie}}?"},
        ],
    )
    print(prompt.prompt_template)
    s = prompt.compile(criticlevel="low")
    print("here")

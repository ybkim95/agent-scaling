import random
from copy import deepcopy
from typing import Dict, Literal, Union

from plancraft.environment.actions import MoveAction, SmeltAction, StopAction
from plancraft.environment.env import PlancraftEnvironment as PlancraftEnv
from plancraft.environment.env import target_and_inventory_to_text_obs
from plancraft.environment.search import gold_search_recipe

from agent_scaling.datasets.plancraft import PlancraftInstance
from agent_scaling.env.base import AgentEnvironmentTools

from .registry import register_env
from .tools import cls_tool

Action = Union[MoveAction, SmeltAction, StopAction]


@register_env("plancraft")
class PlancraftEnvironment(AgentEnvironmentTools):
    def __init__(
        self,
        *args,
        resolution: Literal["low", "medium", "high"] = "high",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataset_instance: PlancraftInstance
        self.resolution = resolution
        self.is_done = False
        self.success = False
        self.num_steps = 0
        if self.dataset_instance is not None:
            self.init_environment()

    def init_environment(self):
        self.num_steps = 0
        self.environment: PlancraftEnv = PlancraftEnv(
            inventory=deepcopy(self.dataset_instance.slotted_inventory),
            resolution=self.resolution,
        )

    def get_instance_prompt_info(self) -> Dict[str, str]:
        observation = self.environment.step()
        return {
            "observation": target_and_inventory_to_text_obs(
                self.dataset_instance.target, observation["inventory"]
            ),
            "tools_description": self.tools_description,
        }

    def env_done(self) -> bool:
        return self.is_done

    def _execute_action(self, action: Action):
        if isinstance(action, StopAction):
            observation = {"message": ""}
            success = self.dataset_instance.impossible
        else:
            observation = self.environment.step(action)
            observation["target"] = self.dataset_instance.target
            observation["message"] = target_and_inventory_to_text_obs(
                observation["target"], observation["inventory"]
            )
            success = self.check_done(
                observation["inventory"], self.dataset_instance.target
            )
        self.success = success
        return observation, success

    def check_done(self, inventory: dict, target: str):
        """
        Check that target object is obtained
        """
        for slot, item in inventory.items():
            # ensure the target is in the inventory (not in slot 0)
            if target == item["type"] and slot != 0:
                return True
        return False

    @cls_tool
    def search(self, recipe_name: str) -> str:
        """
        Search for recipes to craft a specific item.
        """
        random.seed(42)
        return gold_search_recipe(recipe_name)

    @cls_tool
    def move(self, slot_from: str, slot_to: str, quantity: int) -> str:
        """
        Transfer a specific quantity of items from one slot to another.
        Specifically, move from [slot_from] to [slot_to] with target quantity [quantity].

        Example:
        - move(slot_from="[I2]", slot_to="[A1]", quantity=3) to move 3 items from slot I2 to A1
        """

        action = MoveAction(
            **{"slot_from": slot_from, "slot_to": slot_to, "quantity": quantity}
        )
        observation, success = self._execute_action(action)
        if success:
            self.is_done = True
        return observation["message"]

    @cls_tool
    def smelt(self, slot_from: str, slot_to: str, quantity: int) -> str:
        """
        Smelt an item in a furnace and moves the output to a specific slot.
        Specifically, smelt from [slot_from] to [slot_to] with target quantity [quantity].

        Example:
        - smelt(slot_from="[I5]", slot_to="[I6]", quantity=1)
        """

        action = SmeltAction(
            **{"slot_from": slot_from, "slot_to": slot_to, "quantity": quantity}
        )
        observation, success = self._execute_action(action)
        if success:
            self.is_done = True
        return observation["message"]

    @cls_tool
    def impossible(self, reason: str) -> str:
        """
        Stop task if it is certain that it is impossible with given inventory.
        Specifically, indicate the task is impossible with reason [reason].
        """

        action = StopAction(reason=reason)
        observation, _ = self._execute_action(action)
        self.is_done = True
        return observation["message"] or ""

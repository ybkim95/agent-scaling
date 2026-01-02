import concurrent.futures
import contextlib
import os
import traceback
from typing import Any, Dict, List, Optional, Union

from langfuse._client.span import LangfuseSpan
from pydantic import BaseModel
from tqdm import tqdm

from agent_scaling.agents import AgentSystem
from agent_scaling.config.run import RunConfig
from agent_scaling.datasets import Dataset, DatasetInstance
from agent_scaling.logger import logger
from agent_scaling.utils import write_json, write_yaml


class InstanceSave(BaseModel):
    inp: Dict[str, Any]
    output: Dict[str, Any]
    metrics: Dict[str, Union[int, float, str]]
    expected_output: Optional[Any] = None


class ExperimentRunner:
    def __init__(self, config: RunConfig):
        self.config = config
        self.log_langfuse = config.log_langfuse
        self.output_dir = config.save_dir
        self.dataset: Dataset = self.config.dataset.dataset
        self.lf_dataset = self.config.dataset.langfuse_dataset
        self.agent: AgentSystem = self.config.get_agent()
        logger.debug(f"Logging to langfuse: {self.log_langfuse}")

    def _get_context_manager(self, index: int) -> contextlib.AbstractContextManager:
        if self.log_langfuse:
            assert self.lf_dataset is not None, (
                "Langfuse dataset must be set for logging"
            )
            return self.lf_dataset.items[index].run(
                run_name=self.config.run_name,
                run_description=self.config.llm.model,
                run_metadata=self.config.get_run_metadata(),
            )
        return contextlib.nullcontext()

    def _get_instances(self) -> List[DatasetInstance]:
        max_instances = 10 if self.config.debug else len(self.dataset.instances)
        max_instances = (
            min(max_instances, self.config.max_instances)
            if self.config.max_instances is not None
            else max_instances
        )

        if self.config.dataset.dataset_filter is not None:
            instances = [
                instance
                for i, instance in enumerate(self.dataset.instances)
                if eval(self.config.dataset.dataset_filter, {}, {"x": instance, "i": i})
            ]
        else:
            instances = self.dataset.instances
        instances = instances[:max_instances]
        return instances

    def run(self):
        metrics = []
        instances = self._get_instances()

        if self.log_langfuse:
            iterator = tqdm(
                enumerate(instances),
                total=len(instances),
                desc=f"Evaluating {self.dataset.dataset_id} dataset instances",
            )
        else:
            iterator = enumerate(instances)

        for i, instance in iterator:
            if self.config.debug and i >= 10:
                break
            instance_dir = None
            if self.output_dir is not None:
                instance_dir = os.path.join(
                    self.output_dir, "instance_runs", f"{i:04d}"
                )
                os.makedirs(instance_dir, exist_ok=True)

            # Use the core worker function
            inst_metrics = self._process_single_instance(i, instance, instance_dir)
            metrics.append(inst_metrics)

        all_metrics = self.dataset.get_metrics(metrics)
        if self.output_dir is not None:
            write_json(
                all_metrics,
                os.path.join(self.output_dir, "dataset_eval_metrics.json"),
                indent=True,
            )
        return all_metrics

    def _process_single_instance(
        self,
        i: int,
        instance,
        instance_dir: Optional[str] = None,
    ) -> Dict[str, Union[int, float]]:
        """
        Core worker function to process a single instance.
        This can be reused for both sequential and parallel processing.

        Args:
            i: Index of the instance
            instance: Dataset instance to process
            instance_dir: Optional directory to save outputs

        Returns:
            Tuple of (index, metrics) to maintain order
        """

        context_manager = self._get_context_manager(i)
        with context_manager as span:
            agent = self.config.get_agent()
            output = agent.run_agent(
                instance,
                instance_dir=instance_dir,
                llm_params=self.config.llm.params,
                instance_idx=i,
            )
            inst_metrics = self.dataset.get_instance_eval_metrics(output)
            inst_output = self.dataset.get_instance_eval_output(output)
            if isinstance(span, LangfuseSpan):
                span.update(
                    input=instance.get_prompt_info(),
                    output=inst_output,
                )
                for name, metric in inst_metrics.items():
                    span.score_trace(
                        name=name,
                        value=metric,
                    )

            if instance_dir is not None:
                output_save = InstanceSave(
                    inp=instance.get_prompt_info(),
                    output=inst_output,
                    metrics=inst_metrics,  # type: ignore
                    expected_output=instance.expected_output,
                )
                write_yaml(
                    output_save.model_dump(),
                    os.path.join(instance_dir, "instance_save.yaml"),
                    indent=True,
                )

            return inst_metrics

    def run_parallel(self, num_workers: int = 4):
        """
        Run the experiment in parallel using multiprocessing.

        Args:
            num_workers: Maximum number of worker processes
        """
        # Prepare work items
        instances = self._get_instances()

        work_items = []
        for i, instance in enumerate(instances):
            instance_dir = None
            if self.output_dir is not None:
                instance_dir = os.path.join(
                    self.output_dir, "instance_runs", f"{i:04d}"
                )
                os.makedirs(instance_dir, exist_ok=True)

            work_items.append((i, instance, instance_dir))

        # Process in parallel
        metrics: List[Dict[str, int | float] | str] = [""] * len(
            work_items
        )  # Pre-allocate to maintain order

        # Parallel processing with progress bar

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    self._process_single_instance,
                    *work_item,
                ): i
                for i, work_item in enumerate(work_items)
            }
            with tqdm(
                total=len(work_items),
                desc=f"Evaluating {self.dataset.dataset_id} dataset instances (parallel)",
            ) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    i = futures[future]
                    try:
                        result = future.result()
                        metrics[i] = result
                    except Exception as exc:
                        tb_str = "".join(
                            traceback.format_exception(
                                type(exc), exc, exc.__traceback__
                            )
                        )
                        logger.error(
                            f"Error generating output: {exc}\nTraceback:\n{tb_str}"
                        )
                        metrics[i] = (
                            f"Failed with exception: {exc}\nTraceback:\n{tb_str}"
                        )

                    pbar.update(1)

        all_metrics = self.dataset.get_metrics(metrics)
        if self.output_dir is not None:
            write_json(
                all_metrics,
                os.path.join(self.output_dir, "dataset_eval_metrics.json"),
                indent=True,
            )
        return all_metrics

import json
import os.path as osp
import time

import hydra
from dotenv import load_dotenv
from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from agent_scaling.config.run import RunConfig
from agent_scaling.exp_runner import ExperimentRunner
from agent_scaling.logger import add_sink, logger
from agent_scaling.utils import get_run_conf_dir, write_yaml

load_dotenv(override=True)


@hydra.main(
    config_path=get_run_conf_dir(), config_name="run_exp.yaml", version_base=None
)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    hydra_cfg: HydraConf = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    cfg_dict = dict(cfg)
    cfg_dict["save_dir"] = output_dir

    config = RunConfig(**{str(k): v for k, v in cfg_dict.items()})

    logger.info(f"log_langfuse={config.log_langfuse}")

    logger.info(
        f"Running experiment with config: {json.dumps(config.get_run_metadata(), indent=2)}"
    )
    write_yaml(config.get_run_metadata(), osp.join(output_dir, "run_config.yaml"))
    runner = ExperimentRunner(config)
    lptr = add_sink(osp.join(output_dir, "run.log"))
    if config.run_parallel:
        logger.info(f"Running experiment in parallel with {config.num_workers} workers")
        all_metrics = runner.run_parallel(num_workers=config.num_workers)
    else:
        logger.info("Running experiment sequentially")
        all_metrics = runner.run()
    time.sleep(2)  # make sure the logs below show up last
    logger.success(
        f"Experiment completed with metrics:\n{json.dumps(all_metrics, indent=2)}"
    )
    logger.info("Results saved to: " + output_dir)
    logger.remove(lptr)


if __name__ == "__main__":
    main()

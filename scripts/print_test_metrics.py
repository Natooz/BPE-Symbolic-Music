#!/usr/bin/python3 python

"""

"""

import json
import logging
from pathlib import Path

from exp_cla_finetune import experiments as exp_cla
from exp_generation import experiments as exp_gen


if __name__ == "__main__":
    NB_DECIMALS = 2
    logger = logging.getLogger("res")
    (fh := logging.FileHandler(Path("runs", "all_metrics.log"))).setLevel(logging.DEBUG)
    (sh := logging.StreamHandler()).setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG)

    for exp in exp_cla + exp_gen:
        logger.debug(f"\n{exp.name}")
        metrics = {}
        for baseline in exp.baselines:
            if not (baseline.run_path / "test_results.json").is_file():
                continue
            with open(baseline.run_path / "test_results.json") as file:
                results = json.load(file)
            # round results
            for key, val in results.items():
                if isinstance(val, float):
                    if "tse" in key.split("_"):
                        results[key] = round(val * 1000, NB_DECIMALS)
                    else:
                        results[key] = round(val, NB_DECIMALS)

            logger.debug(f"{baseline.name} - {results}")

#!/usr/bin/env python3
"""Print the effective configured ID cuts and save a Markdown table."""
import argparse
from pathlib import Path

from workflow.working_points import configured_wp_sources, load_configured_wps, load_id_config, print_id_table


def print_configured_id(config_path, output, training_dir=None):
    config_path = Path(config_path).resolve()
    config, base, _ = load_id_config(config_path)
    training_dir = Path(training_dir or base["workflow"]["TrainCutBasedID_TMVA"]["outputdir"]).resolve()
    sources = configured_wp_sources(base, config)
    wps = {}
    for region in config.get("regions", base["workflow"]["InspectDistribution"]["regions"]):
        wps[region] = {}
        for source, options in sources.items():
            path = options["path"].format(training_dir=training_dir, region=region, config_dir=config_path.parent)
            wps[region][source] = load_configured_wps(path, options)
    return print_id_table(wps, output, config.get("correctionlib", {}).get("requirement"), base["region_definition"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Validation YAML or standalone pipeline with an evaluation section")
    parser.add_argument("--output", required=True, help="Markdown output path")
    parser.add_argument("--training-dir", help="Directory containing the completed regional WP JSON files")
    args = parser.parse_args()
    print_configured_id(args.config, args.output, args.training_dir)


if __name__ == "__main__":
    main()

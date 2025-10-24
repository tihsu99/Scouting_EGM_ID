import luigi
import yaml
import importlib
from datetime import datetime
from copy import deepcopy

class WorkflowManager(luigi.WrapperTask):
    config_path = luigi.Parameter()

    def requires(self):
        # --- Load YAML ---
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        inputdir = config["inputdir"]
        process = config["process"]
        workflow_cfg = config.get("workflow", {})
        shared_config = {k: v for k, v in config.items() if k != "workflow"}

        tasks = []

        # --- Iterate over workflow blocks ---
        for workflow_name, workflow_params in workflow_cfg.items():
            try:
                module = importlib.import_module(f"workflow.{workflow_name}")
                workflow_func = getattr(module, f"{workflow_name}Analysis")
            except (ModuleNotFoundError, AttributeError):
                raise ImportError(
                    f"Could not import workflow '{workflow_name}' or find '{workflow_name}Analysis'"
                )

            # --- Merge top-level and workflow-specific args ---
            args = deepcopy(shared_config)
            args.update(workflow_params)
            args.setdefault("n_workers", 8)

            # Wrap as Luigi Task using a simple wrapper
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            class TaskWrapper(luigi.Task):
                def output(self):
                    return luigi.LocalTarget(f".{timestamp}done")  # Dummy target

                def run(self):
                    workflow_func(**args)
                    with open(self.output().path, "w") as f:
                        f.write("done")

            tasks.append(TaskWrapper())
            print(f"✅ [Workflow] add {workflow_name} with {workflow_params}")
        return tasks


if __name__ == "__main__":
    luigi.run(main_task_cls=WorkflowManager)


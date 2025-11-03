import luigi
import yaml
import importlib
from datetime import datetime
from copy import deepcopy


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
class TaskWrapper(luigi.Task):
    name = luigi.Parameter()
    workflow_func_path = luigi.Parameter()
    args = luigi.DictParameter()

    def output(self):
        return luigi.LocalTarget(f".{self.name}_{timestamp}.done")

    def run(self):
        module_name, func_name = self.workflow_func_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        workflow_func = getattr(module, func_name)
        workflow_func(**self.args)
        with open(self.output().path, "w") as f:
            f.write("done")


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
        itask = 1
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
            module_path = f"workflow.{workflow_name}"
            func_path = f"{module_path}.{workflow_name}Analysis"

            args = deepcopy(shared_config)
            args.update(workflow_params)
            args.setdefault("n_workers", 8)

            # Wrap as Luigi Task using a simple wrapper

            tasks.append(TaskWrapper(
                name=workflow_name,
                workflow_func_path=func_path,
                args=args,
            ))

            print(f"✅ [Workflow-{itask}] add {workflow_name} with {workflow_params}")
            itask += 1
        return tasks


if __name__ == "__main__":
    luigi.run(main_task_cls=WorkflowManager)


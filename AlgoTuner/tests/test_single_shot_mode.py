from types import SimpleNamespace
import unittest

from AlgoTuner.interfaces.llm_interface import LLMInterface


class SingleShotModeTests(unittest.TestCase):
    def test_extract_single_shot_solver_code_prefers_python_solver_block(self):
        response = """
Here is the solution.

```text
not the solver
```

```python
class Solver:
    def solve(self, problem, **kwargs):
        return problem
```
"""

        solver_code = LLMInterface._extract_single_shot_solver_code(response)

        self.assertIn("class Solver:", solver_code)
        self.assertIn("def solve", solver_code)
        self.assertNotIn("not the solver", solver_code)

    def test_run_single_shot_task_writes_solver_and_runs_train_eval(self):
        events = []

        def add_message(role, content):
            events.append(("message", role, content))

        def read_file(_path):
            return []

        def edit_file(file_path, start_line, end_line, new_content):
            events.append(("edit", str(file_path), start_line, end_line, new_content))
            return {"success": True, "message": "ok"}

        def runner_eval_dataset(data_subset, command_source):
            events.append(("eval", data_subset, command_source))
            return SimpleNamespace(
                message=f"{data_subset}:{command_source}",
                success=True,
                data={"aggregate_metrics": {"mean_speedup": 1.5}},
                error=None,
            )

        interface = SimpleNamespace(
            message_writer=SimpleNamespace(
                format_task_status=lambda *args: "task-status",
                format_warning=lambda msg: msg,
                format_error=lambda msg, _ctx: msg,
            ),
            get_response=lambda: {
                "message": """```python
class Solver:
    def solve(self, problem, **kwargs):
        return problem
```"""
            },
            message_handler=SimpleNamespace(add_message=add_message),
            editor=SimpleNamespace(
                file_manager=SimpleNamespace(read_file=read_file),
                edit_file=edit_file,
            ),
            command_handlers=SimpleNamespace(_runner_eval_dataset=runner_eval_dataset),
            _extract_single_shot_solver_code=LLMInterface._extract_single_shot_solver_code,
            _replace_solver_file=lambda solver_code: edit_file("solver.py", 0, 0, solver_code),
            _finalize_task_run=lambda should_terminate: events.append(("finalize", should_terminate)),
        )

        LLMInterface.run_single_shot_task(interface)

        self.assertIn(("eval", "train", "single_shot"), events)
        self.assertIn(("finalize", False), events)
        edit_events = [event for event in events if event[0] == "edit"]
        self.assertEqual(len(edit_events), 1)
        self.assertIn("class Solver:", edit_events[0][4])

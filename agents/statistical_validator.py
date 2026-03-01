"""Agent used for interpreting permutation-test results."""

from typing import Any, Dict, Iterable, List, Union


class StatisticalValidatorAgent:
    """Simple validator that compares p-values to a threshold.

    The agent accepts either a single result dictionary returned by
    ``run_permutation_test`` or a list of such dictionaries (when multiple
    metrics are evaluated).  The ``evaluate`` method returns ``True`` when
    *all* p-values are below or equal to the configured threshold.

    When the test fails the caller is expected to trigger a kill-switch
    (e.g. writing ``kill_switch_active.txt`` or aborting a model save).
    """

    def __init__(self, p_value_threshold: float = 0.05):
        self.threshold = float(p_value_threshold)

    def evaluate(
        self,
        results: Union[Dict[str, Any], Iterable[Dict[str, Any]]],
    ) -> bool:
        if isinstance(results, dict):
            results = [results]
        for r in results:
            pv = r.get("p_value")
            if pv is None:
                # missing value is treated as failure
                return False
            if pv > self.threshold:
                return False
        return True

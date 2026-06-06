from genesys.manager.lro import LROCollector


class Test1:
    """Describe what this component does."""

    def execute(self, *, lro_collector: LROCollector, node_id: str, **inputs) -> dict:
        # inputs come from the pipeline YAML inputs: block.
        # Return a dict — every key is reachable as ${test_1.output.key}.
        op_id = lro_collector.start(
            node_id=node_id, component="test_1", kind="run", ref="v1",
        )
        try:
            result = {"output_key": "output_value"}
            lro_collector.complete(op_id)
        except Exception as exc:
            lro_collector.fail(op_id, str(exc))
            raise
        return result
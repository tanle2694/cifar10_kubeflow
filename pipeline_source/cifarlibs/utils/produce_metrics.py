from typing import NamedTuple
from kfp.components import create_component_from_func


@create_component_from_func
def display_metrics(result) -> NamedTuple('Outputs', [
  ('mlpipeline_metrics', 'Metrics'),
]):
    import json
    output = json.loads(result)
    metrics = []
    for key, value in output.items():
        metrics.append({
            'name': key,
            'numberValue': value,
            'format': "RAW",
        })
    return_metrics = {
        "metrics": metrics
    }
    return [json.dumps(return_metrics)]


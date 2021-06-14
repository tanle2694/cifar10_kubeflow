import os
import abc
import logging
import time
import json
import errno

from kubeflow.katib import KatibClient
from kubernetes.client import V1ObjectMeta
from kubeflow.katib import V1beta1Experiment
from kubeflow.katib import V1beta1AlgorithmSpec
from kubeflow.katib import V1beta1ObjectiveSpec
from kubeflow.katib import V1beta1FeasibleSpace
from kubeflow.katib import V1beta1MetricsCollectorSpec, V1beta1CollectorSpec, V1beta1SourceSpec
from kubeflow.katib import V1beta1FilterSpec, V1beta1ExperimentSpec
from kubeflow.katib import V1beta1ObjectiveSpec
from kubeflow.katib import V1beta1ParameterSpec
from kubeflow.katib import V1beta1TrialTemplate
from kubeflow.katib import V1beta1TrialParameterSpec
from cifarlibs.hypertune.hypertune_utils import trial_spec_base
logger = logging.getLogger(__name__)


class AbstractHyperTunner(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def start_experiments(self):
        pass

    @abc.abstractmethod
    def wait_for_completion(self):
        pass


class HyperTunner(AbstractHyperTunner):
    def __init__(self, namespace, experiment_name, algorithm_name, objective_spec, parameters_spec, container, pvcs,
                 max_trial_count, parallel_trial_count, max_failed_trial_count, best_hp_file):
        """
        HyperTunner class
            namespace: str
                namespace kubeflow
            experiment_name: str
                name of this experiment
            algorithm_name: str
                algorithm for tunning
            objective_spec: json
                objective specification find best hyperparameters
            parameters: json
                parameters of model for find
        Example:
            objective_spec = {
                    "type": "minimize",
                    "goal": 0.001,
                    "objective_metric_name": "loss",
                    "additional_metric_names": ["accuracy"]
                }

            parameters_spec = [
                {

                    "name": "lr",
                    "parameter_type": "double",
                    "feasible_space": { "min": "0.01",
                                        "max": "0.06",
                                        "list": None,
                                        "step": None},
                    "description": "learning rate"
                },
                {
                    "name": "momentum",
                    "parameter_type": "double",
                    "feasible_space": {"min": "0.5",
                                       "max": "0.9",
                                       "list": None,
                                       "step": None},
                    "description": "momentum"
                },
            ]

            container = {
                "name": "pytorch",
                "image": "pytorch-mnist:latest",
                "command": ["python", "/opt/pytorch-mnist/mnist.py", "--epochs=1",
                            "--lr=${trialParameters.lr}",
                            "--momentum=${trialParameters.momentum}"],
                "envs": [{
                    "name": "PYTHONPATH",
                    "value": "/opt/pytorch-mnist/"
                }]
            }

            pvcs: [
                {
                    "claimName" : "",
                    "mountPath": ""
                }
            ]
        """
        self._best_hp_file = best_hp_file
        self._namespace = namespace
        self._experiment_name = experiment_name

        metadata = self.setup_metadata(experiment_name, namespace)
        algorithm = self.setup_algorithm(algorithm_name)

        objective = self.setup_objective(objective_spec)
        parameters = self.setup_parameters(parameters_spec)

        trial_parameters = self.setup_trial_parameters(parameters_spec)
        trial_spec = self.setup_trial_spec(pvcs, container)
        trial_template = self.setup_trial_template(trial_parameters, trial_spec)

        metric_collector_spec = self.setup_metric_collector()

        self._experiment = V1beta1Experiment(
            api_version="kubeflow.org/v1beta1",
            kind="Experiment",
            metadata=metadata,
            spec=V1beta1ExperimentSpec(
                max_trial_count=max_trial_count,
                parallel_trial_count=parallel_trial_count,
                max_failed_trial_count=max_failed_trial_count,
                metrics_collector_spec=metric_collector_spec,
                algorithm=algorithm,
                objective=objective,
                parameters=parameters,
                trial_template=trial_template
            )
        )
        self._kclient = KatibClient()

    def start_experiments(self):
        """
        Submit experiment into katib for hyperparameter tunning

        """
        submit_result = self._kclient.create_experiment(self._experiment, namespace=self._namespace)
        return submit_result

    def wait_for_completion(self, timeout=None):
        time.sleep(10)
        start_time = time.time()
        while True:

            if self._kclient.get_experiment_status(name=self._experiment_name, namespace=self._namespace) == "Succeeded":
                logger.info("Hyper-tune experiments succeeded")
                print("Hyper-tune experiments succeeded")
                break
            current_time = time.time() - start_time
            if timeout and (current_time - start_time > timeout):
                logger.error("Timeout")
                raise TimeoutError("Hyper-tune experiments timeout")
            print(self._kclient.get_experiment_status(name=self._experiment_name, namespace=self._namespace))
            time.sleep(10)
        return

    def get_optimal_hyperparameters(self):
        opt_trial = self._kclient.get_optimal_hyperparameters(name=self._experiment_name,
                                                         namespace=self._namespace)
        best_params = opt_trial["currentOptimalTrial"]["parameterAssignments"]
        best_param_json = {}
        print("Best params:")
        for hp in best_params:
            best_param_json[hp['name']] = hp['value']
            print(hp['name'], hp['value'])
        return best_param_json

    def delete_experiment(self):
        self._kclient.delete_experiment(name=self._experiment_name, namespace=self._namespace)

    @staticmethod
    def setup_metadata(experiment_name, namespace):
        return V1ObjectMeta(name=experiment_name, namespace=namespace)

    @staticmethod
    def write_json_to_file(json_input, file_output):
        if not os.path.exists(os.path.dirname(file_output)):
            try:
                os.makedirs(os.path.dirname(file_output))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        with open(file_output, 'w', encoding='utf-8') as f:
            f.write(json.dumps(json_input))

    @staticmethod
    def setup_algorithm(algorithm_name):
        return V1beta1AlgorithmSpec(algorithm_name=algorithm_name)

    @staticmethod
    def setup_parameters(parameters_spec):
        parameters = []
        for para in parameters_spec:
            feasible_space = V1beta1FeasibleSpace(list=para['feasible_space']['list'],
                                                  min=para['feasible_space']['min'],
                                                  max=para['feasible_space']['max'],
                                                  step=para['feasible_space']['step'])
            parameters.append(V1beta1ParameterSpec(name=para['name'],
                                                   parameter_type=para['parameter_type'],
                                                   feasible_space=feasible_space)
                              )
        return parameters

    @staticmethod
    def setup_trial_parameters(parameters_spec):
        trial_parameters = []
        for para in parameters_spec:
            trial_param_spec = V1beta1TrialParameterSpec(
                name=para['name'],
                description=para['description'],
                reference=para['name']
            )
            trial_parameters.append(trial_param_spec)
        return trial_parameters

    @staticmethod
    def setup_trial_spec(pvcs, container):
        """

        Args:
            pvcs:
            container:

        Returns:

        """
        trial_spec = trial_spec_base.copy()
        volumes = []
        volumes_mount = []

        for i, pvc in enumerate(pvcs):
            volume_spec = {"name": f"pvc-{i}",
                           "persistentVolumeClaim": {"claimName": pvc['claimName']}
                           }
            volumes.append(volume_spec)
            volume_mount_spec = {"name": f"pvc-{i}",
                                 "mountPath": pvc['mountPath']}
            volumes_mount.append(volume_mount_spec)
        if (len(volumes) > 0) and (len(volumes_mount) > 0):
            trial_spec['spec']['template']['spec']['volumes'] = volumes
            trial_spec['spec']['template']['spec']['containers'][0]['volumeMounts'] = volumes_mount
        trial_spec['spec']['template']['spec']['containers'][0]['name'] = container['name']
        trial_spec['spec']['template']['spec']['containers'][0]['image'] = container['image']
        trial_spec['spec']['template']['spec']['containers'][0]['command'] = container['command']
        if len(container["envs"]) > 0:
            trial_spec['spec']['template']['spec']['containers'][0]['env'] = container['envs']
        return trial_spec

    @staticmethod
    def setup_metric_collector():
        filter = V1beta1FilterSpec(metrics_format=["{metricName: ([\\w|-]+), metricValue: ((-?\\d+)(\\.\\d+)?)}"])
        source = V1beta1SourceSpec(filter=filter)

        collector = V1beta1CollectorSpec(kind='StdOut')
        metric_collector_spec = V1beta1MetricsCollectorSpec(collector=collector, source=source)
        return metric_collector_spec

    @staticmethod
    def setup_trial_template(trial_parameters, trial_spec):
        return V1beta1TrialTemplate(
            primary_container_name="pytorch",
            trial_parameters=trial_parameters,
            trial_spec=trial_spec
        )

    @staticmethod
    def setup_objective(objective_spec):
        return V1beta1ObjectiveSpec(type=objective_spec['type'],
                                    goal=objective_spec['goal'],
                                    objective_metric_name=objective_spec['objective_metric_name'],
                                    additional_metric_names=objective_spec['additional_metric_names'])

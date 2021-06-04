from cifarlibs.hypertune.hypertunner import HyperTunner

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
pvcs = [
        {
            "claimName": "cifar-git-src" ,
            "mountPath": "/tmp/gitcifarsrc"
        }
    ]

katib_tunner = HyperTunner(
                    namespace="kubeflow-user-example-com",
                    experiment_name="testkatib",
                    algorithm_name="random",
                    objective_spec=objective_spec,
                    parameters_spec=parameters_spec,
                    container=container,
                    pvcs=pvcs,
                    max_trial_count=3,
                    max_failed_trial_count=22,
                    parallel_trial_count=1
                    )

# print(katib_tunner._experiment)
katib_tunner.start_experiments()
katib_tunner.wait_for_completion()
print(katib_tunner.get_optimal_hyperparameters())
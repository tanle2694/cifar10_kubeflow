import argparse
import logging
import cifarlibs.utils.logging_handler
logger = logging.getLogger(__name__)
import json
from cifarlibs.hypertune.hypertunner import HyperTunner


def get_pipeline_args():
    parser = argparse.ArgumentParser("Hyper parameters tunning step")
    parser.add_argument("--learning_config", type=str, default="/home/tan.le2/PycharmProjects/cifar10_kubeflow/pipe"
                                                               "line_source/hyper_tunning_step/learning_config.json",
                        help="Learning config for hyperparameter tunning")
    parser.add_argument("--namespace", type=str, default="kubeflow-user-example-com", help="namespace kubeflow")
    parser.add_argument("--experiment_name", type=str, default="testkatibstep", help="Experiment name of hyper-tunning")
    parser.add_argument("--algorithm_name", type=str, default="random", choices=["random", "grid", ""])
    parser.add_argument("--best_hp_file", type=str, help="File save best hyper-parameters after tunning")
    parser.add_argument("--max_trial_count", type=int, default=3, help="max trial experiment")
    parser.add_argument("--max_failed_trial_count", type=int, default=2, help="max trial failed experiment")
    parser.add_argument("--parallel_trial_count", type=int, default=2, help="parallel_trial_count")
    parser.add_argument("--pvc_gitsrc_name", type=str, default="cifar-git-src", help="")
    parser.add_argument("--pvc_gitsrc_mount", type=str, default="/tmp/src", help="")
    parser.add_argument("--pvc_data_name", type=str, default="cifar-nfs-pvc", help="")
    parser.add_argument("--pvc_data_mount", type=str, default="/tmp/data", help="")
    args = parser.parse_args()
    return args


def main(args):
    with open(args.learning_config, 'r') as f:
        learning_config = json.load(f)
    print(learning_config)
    pvcs = [
        {
            "claimName": args.pvc_gitsrc_name,
            "mountPath": args.pvc_gitsrc_mount
        },
        {
            "claimName": args.pvc_data_name,
            "mountPath": args.pvc_data_mount
        }
    ]
    katib_tunner = HyperTunner(
        namespace=args.namespace,
        experiment_name=args.experiment_name,
        algorithm_name=args.algorithm_name,
        objective_spec=learning_config["objective_spec"],
        parameters_spec=learning_config["parameters_spec"],
        container=learning_config["container"],
        pvcs=pvcs,
        max_trial_count=args.max_trial_count,
        max_failed_trial_count=args.max_failed_trial_count,
        parallel_trial_count=args.parallel_trial_count,
        best_hp_file=args.best_hp_file
    )

    katib_tunner.start_experiments()
    katib_tunner.wait_for_completion()
    best_hyper_parameter = katib_tunner.get_optimal_hyperparameters()
    print(best_hyper_parameter)
    katib_tunner.write_json_to_file(best_hyper_parameter, args.best_hp_file)
    print('Saved hyper-parameter to file')

args = get_pipeline_args()
main(args)

import argparse
import logging
import json
import kfp
import os
import kfp.dsl as dsl
import time

from utils import use_k8s_secret
from kubernetes import client as k8s_client

from kfp.dsl._pipeline import PipelineConf
from kfp.components import func_to_container_op
from kfp.components import create_component_from_func, load_component_from_text, InputPath, OutputPath

from pipeline_orchestration.kube_ops import clean_gitpvc_op, gitsync_op, datapreparation_op, hypertune_op, train_op, \
    evaluation_op
from cifarlibs.utils.produce_metrics import display_metrics


SOURCE_FOLDER = "git_clone_folder"
PIPELINE_FOLDER = "pipeline_source"
INPUT_FOLDER = "cifar_data/data"
CHECKPOINT_PATH = "/tmp/checkpoints"
EVAL_OUTPUT = "/tmp/data/eval_result/eval_result.json"

pipeline_conf = PipelineConf()
pipeline_conf.ttl_seconds_after_finished = 86400 * 3

@dsl.pipeline(
    name="Cifar10 Pipeline",
    description=""
)
def cifar10_pipeline(git_pvc_mount,
                     git_pvc_name,
                     pvc_data,
                     pvc_data_mount,
                     git_repo,
                     rev,
                     branch,
                     git_secret,
                     namespace,
                     experiment_name,
                     hypertune_algorithm,
                     max_trial_count,
                     max_failed_trial_count,
                     parallel_trial_count,
                     training_batchsize,
                     eval_batchsize
                     ):

    root_source = os.path.join(str(git_pvc_mount), SOURCE_FOLDER, PIPELINE_FOLDER)

    clean_git_pvc_step = clean_gitpvc_op(gitpvc_name=git_pvc_name, git_pvc_mount=git_pvc_mount)

    gitsync_step = gitsync_op(git_repo=git_repo, rev=rev, branch=branch, gitpvc_name=git_pvc_name,
                              git_pvc_mount=git_pvc_mount, dest_source=SOURCE_FOLDER,
                              git_secret=git_secret)
    gitsync_step.after(clean_git_pvc_step)

    datapreparation_step = datapreparation_op(root_source=root_source, pvc_gitsrc=git_pvc_name,
                                              pvc_gitsrc_mount=git_pvc_mount,
                                              pvc_data=pvc_data, pvc_data_mount=pvc_data_mount,
                                              input_data=os.path.join(str(pvc_data_mount), INPUT_FOLDER),
                                              )
    datapreparation_step.after(gitsync_step)

    hypertune_step = hypertune_op(root_source=root_source, namespace=namespace,
                                  experiment_name="%s%s" % (experiment_name, time.strftime("%Y%m%d%H%M%S")),
                                  algorithm_name=hypertune_algorithm,
                                  max_trial_count=max_trial_count, max_failed_trial_count=max_failed_trial_count,
                                  parallel_trial_count=parallel_trial_count,
                                  pvc_gitsrc=git_pvc_name, pvc_gitsrc_mount=git_pvc_mount,
                                  pvc_data=pvc_data, pvc_data_mount=pvc_data_mount)
    hypertune_step.after(datapreparation_step)
    
    train_step = train_op(root_source=root_source, input_data=os.path.join(str(pvc_data_mount), INPUT_FOLDER),
                          best_hp=hypertune_step.outputs['best_hp_file'], checkpoint_path=CHECKPOINT_PATH,
                          git_pvc=git_pvc_name, git_pvc_mount=git_pvc_mount,
                          data_pvc=pvc_data, data_pvc_mount=pvc_data_mount)
    train_step.after(hypertune_step)

    evaluation_step = evaluation_op(root_source=root_source, input_data=os.path.join(str(pvc_data_mount), INPUT_FOLDER),
                                    batch_size=eval_batchsize, loss_name="cross_entropy", eval_result=EVAL_OUTPUT,
                                    model_path=dsl.InputArgumentPath(train_step.outputs['best_model']),
                                    git_pvc=git_pvc_name, git_pvc_mount=git_pvc_mount,
                                    data_pvc=pvc_data, data_pvc_mount=pvc_data_mount)
    evaluation_step.after(train_step)

    display_metrics(evaluation_step.outputs['metrics_score'])


def get_args():
    parser = argparse.ArgumentParser("Setup pipeline")

    parser.add_argument("--api_endpoint", type=str, default="http://10.0.19.82:8080/pipeline", help='api to connect '
                                                                                                    'kubeflow pipeline')
    parser.add_argument("--cookie_file", type=str, default="/home/tan.le2/PycharmProjects/cifar10_kubeflow/session_cookie.txt", help="")
    parser.add_argument("--namespace", type=str, default='kubeflow-user-example-com', help="")
    parser.add_argument("--experiment_name", type=str, default="tanlmtest", help="")
    parser.add_argument("--ttl_pod", type=float, default=1.0, help="")
    parser.add_argument("--git_repo", type=str, default="git@github.com:tanle2694/cifar10_kubeflow.git")
    parser.add_argument("--rev", type=str, default='HEAD')
    parser.add_argument("--branch", type=str, default="master")
    
    parser.add_argument("--git_pvc_name", type=str, default="cifar-git-src")
    parser.add_argument("--git_pvc_mount", type=str, default="/tmp/src")
    parser.add_argument("--git_secret", type=str, default="git-creds")
    parser.add_argument("--pvc_data", type=str, default="cifar-nfs-pvc")
    parser.add_argument("--pvc_data_mount", type=str, default="/tmp/data")
    parser.add_argument("--source_folder", type=str, default="pipeline_source")
    parser.add_argument("--cpu_image", type=str, default="envcpu")
    parser.add_argument("--gpu_image", type=str, default="envgpu")
    parser.add_argument("--hypertune_algorithm", type=str, choices=["random", "grid", "TPE", "Bayes", 'hyperband',
                                                                    "cmaes"], default="random")
    parser.add_argument("--max_trial_count", type=int, default=2)
    parser.add_argument("--max_failed_trial_count", type=int, default=2)
    parser.add_argument("--parallel_trial_count", type=int, default=2)

    parser.add_argument("--training_batchsize", type=int, default=32)
    parser.add_argument("--eval_batchsize", type=int, default=32)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    with open(args.cookie_file, 'r') as f:
        session_cookie = f.readline().strip()
    client = kfp.Client(host=args.api_endpoint, cookies=session_cookie)

    pipeline_conf = PipelineConf()
    pipeline_conf.ttl_seconds_after_finished = int(86400 * args.ttl_pod)

    client.create_run_from_pipeline_func(cifar10_pipeline,
                                                         arguments={
                                                             "git_pvc_mount": args.git_pvc_mount,
                                                             "git_pvc_name": args.git_pvc_name,
                                                             "pvc_data": args.pvc_data,
                                                             "pvc_data_mount": args.pvc_data_mount,
                                                             "git_repo": args.git_repo,
                                                             "rev": args.rev,
                                                             "branch": args.branch,
                                                             "git_secret": args.git_secret,
                                                             "namespace": args.namespace,
                                                             "experiment_name": args.experiment_name,
                                                             "hypertune_algorithm": args.hypertune_algorithm,
                                                             "max_trial_count": args.max_trial_count,
                                                             "max_failed_trial_count": args.max_failed_trial_count,
                                                             "parallel_trial_count": args.parallel_trial_count,
                                                             "training_batchsize": args.training_batchsize,
                                                             "eval_batchsize": args.eval_batchsize
                                                         },
                                                        experiment_name=args.experiment_name,
                                                        namespace=args.namespace,
                                                        pipeline_conf=pipeline_conf)

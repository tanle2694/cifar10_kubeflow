import kfp.dsl as dsl
import datetime
import os
from kubernetes import client as k8s_client

CPU_IMAGE = "envcpu"
GPU_IMAGE = "envgpu"
BEST_HP_FILE = "/tmp/best_hp/best_hp_file.txt"


def clean_gitpvc_op(gitpvc_name='cifar-git-src', git_pvc_mount='/tmp/src'):
    return dsl.ContainerOp(
        name="clean-git-pvc",
        image='library/bash:4.4.23',
        command=["sh"],
        arguments=["-c", f"rm -rf {git_pvc_mount}/*"],
        pvolumes={git_pvc_mount: dsl.PipelineVolume(pvc=gitpvc_name)}
    )


def gitsync_op(git_repo, rev, branch, gitpvc_name='cifar-git-src', git_pvc_mount='/tmp/src',
               dest_source="pipeline_source", git_secret="git-creds"):

    gitsync_containerop = dsl.ContainerOp(
        name="Git-sync",
        image="k8s.gcr.io/git-sync/git-sync:v3.3.0",
        arguments=["--ssh",
                   f"--repo={git_repo}",
                   f"--root={git_pvc_mount}",
                   f"--dest={dest_source}",
                   f"--rev={rev}",
                   f"--branch={branch}",
                   "--one-time",
                   "--max-sync-failures=3",
                   "--wait=3"
                   ],
        pvolumes={git_pvc_mount: dsl.PipelineVolume(pvc=gitpvc_name)}
    )

    # setup git authentication
    gitsync_containerop.add_volume(k8s_client.V1Volume(
        name='git-cred-volume',
        secret=k8s_client.V1SecretVolumeSource(secret_name=git_secret))
    ).add_volume_mount(k8s_client.V1VolumeMount(mount_path="/etc/git-secret",
                                                name="git-cred-volume"))

    gitsync_containerop.execution_options.caching_strategy.max_cache_staleness = "P0D"

    return gitsync_containerop


def datapreparation_op(root_source, input_data, pvc_gitsrc, pvc_gitsrc_mount,
                 pvc_data, pvc_data_mount):
    datapreparation_step = dsl.ContainerOp(
        name="Datapreparation step",
        image=CPU_IMAGE,
        command=["python"],
        arguments=[f"{root_source}/datapreparation_step/datapreparation_step.py",
                   "--save_folder", input_data
                   ],
        pvolumes={pvc_gitsrc_mount: dsl.PipelineVolume(pvc=pvc_gitsrc),
                  pvc_data_mount: dsl.PipelineVolume(pvc=pvc_data)}
    ).add_env_variable(k8s_client.V1EnvVar(name="PYTHONPATH", value=root_source))
    datapreparation_step.set_image_pull_policy("Never")
    return datapreparation_step


def hypertune_op(root_source, namespace, experiment_name, algorithm_name,
                 max_trial_count, max_failed_trial_count, parallel_trial_count, pvc_gitsrc, pvc_gitsrc_mount,
                 pvc_data, pvc_data_mount):

    hyper_tune_containerop = dsl.ContainerOp(
        name="Hyperparameters tunning step",
        image=CPU_IMAGE,
        command=["python"],
        arguments=[f"{root_source}/hyper_tunning_step/hypertune_step.py",
                   "--learning_config",
                   f"{root_source}/hyper_tunning_step/learning_config.json",
                   "--namespace", namespace,
                   "--experiment_name", experiment_name,
                   "--algorithm_name", algorithm_name,
                   "--best_hp_file", BEST_HP_FILE,
                   "--max_trial_count", str(max_trial_count),
                   "--max_failed_trial_count", str(max_failed_trial_count),
                   "--parallel_trial_count", str(parallel_trial_count),
                   "--pvc_gitsrc_name", pvc_gitsrc,
                   "--pvc_gitsrc_mount", pvc_gitsrc_mount,
                   "--pvc_data_name", pvc_data,
                   "--pvc_data_mount", pvc_data_mount,
                   ],
        pvolumes={pvc_gitsrc_mount: dsl.PipelineVolume(pvc=pvc_gitsrc),
                  pvc_data_mount: dsl.PipelineVolume(pvc=pvc_data)},

        file_outputs={"best_hp_file": BEST_HP_FILE}
    )
    hyper_tune_containerop.add_env_variable(k8s_client.V1EnvVar(name="PYTHONPATH", value=root_source))
    hyper_tune_containerop.set_image_pull_policy("Never")

    return hyper_tune_containerop


def train_op(root_source, input_data, best_hp, checkpoint_path, git_pvc, git_pvc_mount,
             data_pvc, data_pvc_mount):

    training_containerop = dsl.ContainerOp(
        name="training-step",
        image=GPU_IMAGE,
        command=["python"],
        arguments=[f"{root_source}/training_step/training_step.py",
                   "--input_data", input_data,
                   "--tune_hp", best_hp,
                   "--checkpoint_path", checkpoint_path],
        pvolumes={git_pvc_mount: dsl.PipelineVolume(pvc=git_pvc),
                  data_pvc_mount: dsl.PipelineVolume(pvc=data_pvc)},
        file_outputs={"best_model": f"{checkpoint_path}/model_best.pth"}
    ).add_env_variable(k8s_client.V1EnvVar(name="PYTHONPATH", value=root_source))

    training_containerop.execution_options.caching_strategy.max_cache_staleness = "P0D"
    training_containerop.set_image_pull_policy("Never")

    training_containerop.set_gpu_limit(1)
    training_containerop.set_cpu_limit('2')
    training_containerop.set_memory_limit('4Gi')
    training_containerop.set_cpu_request('2')
    training_containerop.set_memory_request('4Gi')

    return training_containerop


def evaluation_op(root_source, input_data, batch_size, model_path, loss_name, eval_result,
                  git_pvc, git_pvc_mount, data_pvc, data_pvc_mount):
    eval_containerop = dsl.ContainerOp(
        name="evaluation-step",
        image=GPU_IMAGE,
        command=["python"],
        arguments=[f"{root_source}/evaluation_step/evaluation_step.py",
                   "--input_data", input_data,
                   "--batch_size", batch_size,
                   "--model_path", model_path,
                   "--loss_name", loss_name,
                   "--eval_result", eval_result],
        pvolumes={git_pvc_mount: dsl.PipelineVolume(pvc=git_pvc),
                  data_pvc_mount: dsl.PipelineVolume(pvc=data_pvc)},
        file_outputs={"metrics_score": "/tmp/data/eval_result/eval_result.json"}
    ).add_env_variable(k8s_client.V1EnvVar(name="PYTHONPATH", value=root_source))

    eval_containerop.execution_options.caching_strategy.max_cache_staleness = "P0D"
    eval_containerop.set_image_pull_policy("Never")

    eval_containerop.set_gpu_limit(1)
    eval_containerop.set_cpu_limit('2')
    eval_containerop.set_memory_limit('4Gi')
    eval_containerop.set_cpu_request('2')
    eval_containerop.set_memory_request('4Gi')

    return eval_containerop



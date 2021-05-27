import kfp.dsl as dsl
from utils import use_k8s_secret
from kubernetes import client as k8s_client
import kfp
from kfp.dsl._pipeline import PipelineConf

pipeline_conf = PipelineConf()
pipeline_conf.ttl_seconds_after_finished = 86400 * 3


@dsl.pipeline(
    name="Cifar10 Pipeline",
    description=""
)
def pipeline_head(git_repo,
                  branch="master",
                  rev='HEAD',
                  git_secret="git-creds"):
    src_vol_op = dsl.VolumeOp(
        name="Git_source_pvc",
        resource_name="git-pvc",
        size='60Mi',
        modes=dsl.VOLUME_MODE_RWM
    )

    gitsync_step = dsl.ContainerOp(
        name="Git-sync",
        image="k8s.gcr.io/git-sync/git-sync:v3.3.0",
        arguments=["--ssh",
                   f"--repo={git_repo}",
                   "--root=/tmp/src",
                   "--dest=cifar_source",
                   f"--rev={rev}",
                   f"--branch={branch}",
                   "--one-time"
                   "--wait=30"
                   ],
        pvolumes={"/tmp/src": src_vol_op.volume}
    )

    gitsync_step.add_volume(k8s_client.V1Volume(
        name='git-cred-volume',
        secret=k8s_client.V1SecretVolumeSource(secret_name=git_secret))
    ).add_volume_mount(k8s_client.V1VolumeMount(mount_path="/etc/git-secret",
                                                name="git-cred-volume"))
    gitsync_step.execution_options.caching_strategy.max_cache_staleness = "P0D"

    step1 = dsl.ContainerOp(
        name="step1",
        image="python:3.8",
        command=["python"],
        arguments=["/tmp/src/pipeline_source/step1.py", "--arg1", "input_arg1", "--arg2", "input_arg2"],
        pvolumes={"/tmp/src": src_vol_op.volume.after(gitsync_step)}
    ).add_env_variable(k8s_client.V1EnvVar(name="PYTHONPATH", value="/tmp/src/pipeline_source"))
    step1.execution_options.caching_strategy.max_cache_staleness = "P0D"

    step2 = dsl.ContainerOp(
        name="step2",
        image="python:3.8",
        command=["python"],
        arguments=["/tmp/src/pipeline_source/step2.py", "--arg1", "input_arg1", "--arg2", "input_arg2"],
        pvolumes={"/tmp/src": src_vol_op.volume.after(step1)}
    ).add_env_variable(k8s_client.V1EnvVar(name="PYTHONPATH", value="/tmp/src/pipeline_source"))
    step2.execution_options.caching_strategy.max_cache_staleness = "P0D"


# if __name__ == "__main__":
#     # session_cookie="authservice_session=MTYyMTMwMzc4OXxOd3dBTkZVM1NWcEZXbEpVUmtFelV6Wk1UVFF5TlZkSVZFSkJUbEpMTWtsU1dqSTBSRWMxV2pWQ1JsZFBXRGRUVkZBMFdFUlZSRkU9fPPga611a2NuWEZb1_7WZmS9r2RoOM6yx2ye8gLsXoNY"
# api_endpoint = "http://10.0.19.82:8080/pipeline"
# client = kfp.Client(host=api_endpoint, cookies=session_cookie)
# namespace='kubeflow-user-example-com'
# client.create_run_from_pipeline_func(pipeline_head, arguments={
#                                                                "rev": "086727bbdf6d18b1cd4e4996abfb80d0850b4025",
#                                                                "git_repo": "git@github.com:tanle2694/head_det"\
#                                                                            "ection_pipeline.git"
#                                                                },
#                                                     experiment_name="tanlmtest",
#                                                     namespace="kubeflow-user-example-com",
#                                                     pipeline_conf=pipeline_conf)

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(pipeline_head, __file__ + '.yaml')

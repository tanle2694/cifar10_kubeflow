from typing import Dict


def use_k8s_secret(secret_name: str = 'k8s-secret', k8s_secret_key_to_env: Dict = {}):
    """An operator that configures the container to use k8s credentials.

    k8s_secret_key_to_env specifies a mapping from the name of the keys in the k8s secret to the name of the
    environment variables where the values will be added.

    The secret needs to be deployed manually a priori.

    Example:
        ::

            train = train_op(...)
            train.apply(use_k8s_secret(secret_name='s3-secret',
            k8s_secret_key_to_env={'secret_key': 'AWS_SECRET_ACCESS_KEY'}))

        This will load the value in secret 's3-secret' at key 'secret_key' and source it as the environment variable
        'AWS_SECRET_ACCESS_KEY'. I.e. it will produce the following section on the pod:
        env:
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: s3-secret
              key: secret_key
    """

    def _use_k8s_secret(task):
        from kubernetes import client as k8s_client
        for secret_key, env_var in k8s_secret_key_to_env.items():
            task.container \
                .add_env_variable(
                k8s_client.V1EnvVar(
                    name=env_var,
                    value_from=k8s_client.V1EnvVarSource(
                        secret_key_ref=k8s_client.V1SecretKeySelector(
                            name=secret_name,
                            key=secret_key
                        )
                    )
                )
            )
        return task

    return _use_k8s_secret
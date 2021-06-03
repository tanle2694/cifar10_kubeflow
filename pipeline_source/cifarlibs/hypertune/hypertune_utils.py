trial_spec_base = {

    "apiVersion": "batch/v1",
    "kind": "Job",
    "spec": {
        "template": {
            "metadata": {
                "annotations": {
                    "sidecar.istio.io/inject": "false"
                }
            },
            "spec": {
                # "volumes": [{
                #                 "name": "git-pvc",
                #                 "persistentVolumeClaim": {"claimName": ""}
                #             }
                #             ],
                "containers": [
                    {
                        "name": "",
                        "image": "",
                        "command": [],
                        # "volumeMounts":[
                        #                 {
                        #                     "name": "git-pvc",
                        #                     "mountPath": "/tmp/src/pipelinesource"
                        #                 }
                        #                ],
                        "resources":{
                            "requests": {
                                "memory": "4Gi",
                                "cpu": "2"
                            },
                            "limits": {
                                "memory": "4Gi",
                                "cpu": "2",
                                "nvidia.com/gpu": 1
                            }
                        },
                        "imagePullPolicy": "Never"
                    }
                ],
                "restartPolicy": "Never"
            }
        }
    }
}
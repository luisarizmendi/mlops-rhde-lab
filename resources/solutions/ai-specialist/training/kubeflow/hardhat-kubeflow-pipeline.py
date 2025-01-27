from kfp import dsl, compiler, kubernetes
import os
from typing import NamedTuple

# Component 1: Download Dataset
@dsl.component(
    base_image="quay.io/luisarizmendi/pytorch-custom:latest",
    packages_to_install=["roboflow", "pyyaml"]
)
def download_dataset(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    dataset_path: dsl.OutputPath(str)
) -> None:
    from roboflow import Roboflow
    import yaml
    import os

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    version = project.version(version)
    dataset = version.download("yolov11")

    # Update data.yaml paths
    dataset_yaml_path = f"{dataset.location}/data.yaml"
    with open(dataset_yaml_path, "r") as file:
        data_config = yaml.safe_load(file)

    data_config["train"] = f"{dataset.location}/train/images"
    data_config["val"] = f"{dataset.location}/valid/images"
    data_config["test"] = f"{dataset.location}/test/images"


    print(dataset)



    with open(dataset_path, "w") as f:
        f.write(dataset.location)


# Component 2: Train Model
@dsl.component(
    base_image="quay.io/luisarizmendi/pytorch-custom:latest",
    packages_to_install=["ultralytics", "torch", "pandas"]
)
def train_model(
    dataset_path: str,
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    name: str = "yolo",
) -> NamedTuple('Outputs', [
    ('train_dir', str),
    ('test_dir', str),
    ('metrics', dict)
]):
    import torch
    from ultralytics import YOLO
    import pandas as pd
    import os

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    CONFIG = {
        'name': name,
        'model': 'yolo11m.pt',
        'data': f"{dataset_path}/data.yaml",
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,
    }

    # Configure PyTorch
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Initialize and train model
    model = YOLO(CONFIG['model'])
    results_train = model.train(
        name=CONFIG['name'],
        data=CONFIG['data'],
        epochs=CONFIG['epochs'],
        batch=CONFIG['batch'],
        imgsz=CONFIG['imgsz'],
        device=CONFIG['device'],
    )

    # Evaluate model
    results_test = model.val(
        data=CONFIG['data'],
        split='test',
        device=CONFIG['device'],
        imgsz=CONFIG['imgsz']
    )

    # Compute metrics from CSV
    results_csv_path = os.path.join(results_train.save_dir, "results.csv")
    results_df = pd.read_csv(results_csv_path)

    # Extract metrics
    metrics = {
        "precision": results_df["metrics/precision(B)"].iloc[-1],
        "recall": results_df["metrics/recall(B)"].iloc[-1],
        "mAP50": results_df["metrics/mAP50(B)"].iloc[-1],
        "mAP50-95": results_df["metrics/mAP50-95(B)"].iloc[-1]
    }

    return NamedTuple('Outputs', [
        ('train_dir', str),
        ('test_dir', str),
        ('metrics', dict)
    ])(
        train_dir=str(results_train.save_dir),
        test_dir=str(results_test.save_dir),
        metrics=metrics
    )

    
# Component 3: Upload to MinIO
@dsl.component(
    base_image="quay.io/luisarizmendi/pytorch-custom:latest",
    packages_to_install=["minio"]
)
def upload_to_minio(
    train_dir: str,
    test_dir: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
    bucket: str,
    model_path: dsl.OutputPath(str)
) -> NamedTuple('Outputs', [
    ('model_artifact_s3_path', str)
]):
    from minio import Minio
    from minio.error import S3Error
    import os
    import datetime

    client = Minio(
        endpoint.replace('https://', '').replace('http://', ''),
        access_key=access_key,
        secret_key=secret_key,
        secure=True
    )

    # Get paths for files
    weights_path = os.path.join(train_dir, "weights")

    files_train = [os.path.join(train_dir, f) for f in os.listdir(train_dir)
                   if os.path.isfile(os.path.join(train_dir, f))]
    files_models = [os.path.join(weights_path, f) for f in os.listdir(weights_path)
                    if os.path.isfile(os.path.join(weights_path, f))]

    files_model_pt = os.path.join(train_dir, "weights") + "/best.pt"
    
    #files_model_onnx = os.path.join(train_dir, "weights") + "/best.onnx"
    #files_model_torchscript = os.path.join(train_dir, "weights") + "/best.torchscript"
    
    files_test = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                  if os.path.isfile(os.path.join(test_dir, f))]
    
    directory_name = os.path.basename(train_dir) + "-" + datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
    
    # Upload files
    for file_path in files_train:
        try:
            client.fput_object(bucket, f"models/{directory_name}/train-val/{os.path.basename(file_path)}", file_path)
        except S3Error as e:
            print(f"Error uploading {file_path}: {e}")
    
    for file_path in files_test:
        try:
            client.fput_object(bucket, f"models/{directory_name}/test/{os.path.basename(file_path)}", file_path)
        except S3Error as e:
            print(f"Error uploading {file_path}: {e}")

    with open(model_path, "w") as f:
        f.write("models/" + directory_name)

    try:
        client.fput_object(bucket, f"models/{directory_name}/model/pytorch/{os.path.basename(files_model_pt)}", files_model_pt)
    except S3Error as e:
        print(f"Error uploading {files_model_pt}: {e}")

    #try:
    #    client.fput_object(bucket, f"models/{directory_name}/model/onnx/1/{os.path.basename(files_model_onnx)}", files_model_onnx)
    #except S3Error as e:
    #    print(f"Error uploading {files_model_onnx}: {e}")

    #try:
    #    client.fput_object(bucket, f"models/{directory_name}/model/torchscript/1/model.pt", files_model_torchscript)
    #except S3Error as e:
    #    print(f"Error uploading {files_model_torchscript}: {e}")


    model_artifact_s3_path=f"models/{directory_name}/model/pytorch/{os.path.basename(files_model_pt)}"

    return NamedTuple('Outputs', [
        ('model_artifact_s3_path', str)
    ])(
        model_artifact_s3_path
    )



# Component 4: Push to Model Registry
@dsl.component(
    base_image='python:3.9',
    packages_to_install=['model-registry']
)
def push_to_model_registry(
    model_name: str,
    version: str,
    metrics: dict,
    model_registry_name: str,
    model_artifact_s3_path: str,
    s3_endpoint: str,
    roboflow_workspace: str,
    roboflow_project: str,
    roboflow_version: int,    
):
    from model_registry import ModelRegistry
    import os
    from datetime import datetime
    import json
    
    s3_endpoint_url=s3_endpoint.replace('https://', '').replace('http://', '')
    version = version if version else datetime.now().strftime('%y%m%d%H%M')
    model_object_prefix = model_name if model_name else "model"
    cluster_domain= ""    
    namespace_file_path = '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
    with open(namespace_file_path, 'r') as namespace_file:
        namespace = namespace_file.read().strip()


    # Get Cluster domain from MinIO s3_endpoint.
    cluster_domain = s3_endpoint.split("//")[-1].split(".", 2)[-1]

 
    os.environ["KF_PIPELINES_SA_TOKEN_PATH"] = "/var/run/secrets/kubernetes.io/serviceaccount/token"
      
   
    def _register_model(author_name , server_address, model_object_prefix, version):
        registry = ModelRegistry(server_address=server_address, port=443, author=author_name, is_secure=False)
        registered_model_name = model_object_prefix
        version_name = version
        metadata = {
            "dataset": f"https://universe.roboflow.com/{roboflow_workspace}/{roboflow_project}/dataset/{str(roboflow_version)}",
            "mAP50": str(metrics["mAP50"]),
            "mAP50-95": str(metrics["mAP50-95"]),
            "precision": str(metrics["precision"]),
            "recall": str(metrics["recall"])
        }
      
        rm = registry.register_model(
            registered_model_name,
            f"s3://{s3_endpoint_url}/{model_artifact_s3_path}",
            model_format_name="pt",
            model_format_version="1",
            version=version_name,
            description=f"{registered_model_name} is a dense neural network that detects Hardhats in images.",
            metadata=metadata
        )
        print("Model registered successfully")
    
    
    # Register the model
    server_address = f"https://{model_registry_name}-rest.apps.{cluster_domain}"
    print(f"s3://{s3_endpoint_url}/{model_artifact_s3_path}")

    _register_model(namespace, server_address, model_object_prefix, f"{model_object_prefix}-{version}")
    
    
# Define the pipeline
@dsl.pipeline(
    name='YOLO Training Pipeline',
    description='Pipeline to download data, train YOLO model, and upload results to MinIO'
)
def yolo_training_pipeline(

    roboflow_api_key: str,
    roboflow_workspace: str,
    roboflow_project: str,
    roboflow_version: int,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_bucket: str,
    pvc_storage_class: str = "gp3-csi",
    pvc_size: str = "5Gi",
    pvc_name_suffix: str = "-kubeflow-pvc",
    train_name: str = "user99-hardhat",
    train_epochs: int = 50,
    train_batch_size: int = 16,
    train_img_size: int = 640,
    model_registry_name: str = "object-detection-model-registry",
):
    # Create PV
    pvc = kubernetes.CreatePVC(
        pvc_name_suffix=pvc_name_suffix,
        access_modes=['ReadWriteOnce'],
        size=pvc_size,
        storage_class_name=pvc_storage_class,
    )

    # Download dataset
    download_task = download_dataset(
        api_key=roboflow_api_key,
        workspace=roboflow_workspace,
        project=roboflow_project,
        version=roboflow_version
    )
    download_task.set_caching_options(enable_caching=False)
    kubernetes.mount_pvc(
        download_task,
        pvc_name=pvc.outputs['name'],
        mount_path='/opt/app-root/src',
    )

    # Train model
    train_task = train_model(
        dataset_path=download_task.output,
        epochs=train_epochs,
        batch_size=train_batch_size,
        img_size=train_img_size,
        name=train_name
    ).after(download_task)
    train_task.set_caching_options(enable_caching=False)
    kubernetes.mount_pvc(
        train_task,
        pvc_name=pvc.outputs['name'],
        mount_path='/opt/app-root/src',
    )

    # Upload results
    upload_task = upload_to_minio(
        train_dir=train_task.outputs['train_dir'],
        test_dir=train_task.outputs['test_dir'],
        endpoint=minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        bucket=minio_bucket
    ).after(train_task)
    upload_task.set_caching_options(enable_caching=False)
    kubernetes.mount_pvc(
        upload_task,
        pvc_name=pvc.outputs['name'],
        mount_path='/opt/app-root/src',
    )
    
    # Push to model registry
    push_to_model_registry(
        model_name=train_name,
        version="",
        metrics=train_task.outputs['metrics'],
        model_registry_name=model_registry_name,
        model_artifact_s3_path=upload_task.outputs['model_artifact_s3_path'],
        s3_endpoint=minio_endpoint,
        roboflow_workspace=roboflow_workspace,
        roboflow_project=roboflow_project,
        roboflow_version=roboflow_version
    ).after(upload_task)

    delete_pvc = kubernetes.DeletePVC(
        pvc_name=pvc.outputs['name']
    ).after(upload_task)

if __name__ == "__main__":
    # Compile the pipeline
    compiler.Compiler().compile(
        pipeline_func=yolo_training_pipeline,
        package_path='yolo_training_pipeline.yaml'
    )
    print("Pipeline compiled successfully to yolo_training_pipeline.yaml")



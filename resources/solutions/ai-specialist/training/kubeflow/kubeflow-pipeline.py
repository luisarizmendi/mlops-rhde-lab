from kfp import dsl, compiler
from kfp.dsl import Dataset, Model, Input, Output, component
import os

# Define component for data preparation
@component(
    base_image="python:3.9",
    packages_to_install=[
        "roboflow",
        "torch",
        "ultralytics",
        "PyYAML"
    ]
)
def get_data(
    roboflow_key: str,
    roboflow_workspace: str,
    roboflow_project: str,
    roboflow_version: str,
    dataset_path: Output[Dataset]
):
    import os
    import yaml
    from roboflow import Roboflow

    # Download dataset from Roboflow
    rf = Roboflow(api_key=roboflow_key)
    project = rf.workspace(roboflow_workspace).project(roboflow_project)
    version = project.version(roboflow_version)
    dataset = version.download("yolov11")

    # Update data.yaml paths
    dataset_yaml_path = f"{dataset.location}/data.yaml"
    with open(dataset_yaml_path, "r") as file:
        data_config = yaml.safe_load(file)

    data_config["train"] = f"{dataset.location}/train/images"
    data_config["val"] = f"{dataset.location}/valid/images"
    data_config["test"] = f"{dataset.location}/test/images"

    with open(dataset_yaml_path, "w") as file:
        yaml.safe_dump(data_config, file)

    # Save dataset path
    with open(dataset_path.path, "w") as f:
        f.write(dataset.location)

# Define component for model training
@component(
    base_image="python:3.9",
    packages_to_install=[
        "torch",
        "ultralytics",
        "PyYAML"
    ]
)
def train_model(
    dataset_path: Input[Dataset],
    model_epochs: int,
    model_batch: int,
    trained_model: Output[Model]
):
    import os
    import torch
    from ultralytics import YOLO

    # Configure device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Read dataset path
    with open(dataset_path.path, "r") as f:
        dataset_location = f.read().strip()

    # Configure training parameters
    config = {
        'name': 'yolo_hardhat',
        'model': 'yolo11m.pt',
        'data': f"{dataset_location}/data.yaml",
        'epochs': model_epochs,
        'batch': model_batch,
        'imgsz': 640,
        'patience': 15,
        'device': device,
        'optimizer': 'SGD',
        'lr0': 0.001,
        'lrf': 0.005,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_bias_lr': 0.01,
        'warmup_momentum': 0.8,
        'amp': False,
        'augment': True,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10,
        'translate': 0.1,
        'scale': 0.3,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.1,
        'fliplr': 0.1,
        'mosaic': 1.0,
        'mixup': 0.0,
    }

    # Initialize and train model
    model = YOLO(config['model'])
    results = model.train(**config)

    # Export model
    model.export(format='onnx', imgsz=config['imgsz'])

    # Save model path
    model_path = os.path.join(results.save_dir, "weights")
    with open(trained_model.path, "w") as f:
        f.write(model_path)

# Define component for saving model to storage
@component(
    base_image="python:3.9",
    packages_to_install=["minio"]
)
def save_model(
    trained_model: Input[Model],
    aws_endpoint: str,
    aws_access_key: str,
    aws_secret_key: str,
    aws_bucket: str
):
    import os
    from minio import Minio

    # Read model path
    with open(trained_model.path, "r") as f:
        model_path = f.read().strip()

    # Initialize Minio client
    client = Minio(
        aws_endpoint,
        access_key=aws_access_key,
        secret_key=aws_secret_key,
        secure=True
    )

    # Upload model files
    directory_name = os.path.basename(os.path.dirname(model_path))
    for file_name in os.listdir(model_path):
        file_path = os.path.join(model_path, file_name)
        if os.path.isfile(file_path):
            object_name = f"models/{directory_name}/{file_name}"
            client.fput_object(aws_bucket, object_name, file_path)

# Define the pipeline
@dsl.pipeline(
    name='YOLOv11 Training Pipeline',
    description='Pipeline for training YOLOv11 model using Roboflow dataset'
)
def yolo_training_pipeline(
    roboflow_key: str,
    roboflow_workspace: str,
    roboflow_project: str,
    roboflow_version: str,
    model_epochs: int,
    model_batch: int,
    aws_endpoint: str,
    aws_access_key: str,
    aws_secret_key: str,
    aws_bucket: str
):
    # Data preparation step
    get_data_task = get_data(
        roboflow_key=roboflow_key,
        roboflow_workspace=roboflow_workspace,
        roboflow_project=roboflow_project,
        roboflow_version=roboflow_version
    )

    # Model training step
    train_model_task = train_model(
        dataset_path=get_data_task.outputs["dataset_path"],
        model_epochs=model_epochs,
        model_batch=model_batch
    )

    # Save model step
    save_model_task = save_model(
        trained_model=train_model_task.outputs["trained_model"],
        aws_endpoint=aws_endpoint,
        aws_access_key=aws_access_key,
        aws_secret_key=aws_secret_key,
        aws_bucket=aws_bucket
    )

# Compile the pipeline
compiler.Compiler().compile(
    pipeline_func=yolo_training_pipeline,
    package_path='training-kubeflow.yaml'
)

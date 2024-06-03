# model_inference

General class to run inference with AI models.

# Skypilot

## Prerequisites (Macos)

```bash
brew install socat
brew install netcat
```

## GCP

```bash
gcloud init
# Cgange default region
gcloud config set compute/region NAME
# Run this if you don't have a credentials file.
# This will generate ~/.config/gcloud/application_default_credentials.json.
gcloud auth application-default login
# Run `gcloud help config` to learn how to change individual settings

# To list all GCP projects:
gcloud projects list

```

## AWS (MACOS)

```bash
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /
# check that it worked
which aws
rm AWSCLIV2.pkg
aws configure --profile <your_profile> # AlexandrePasquiou
# Verify configuration
aws configure list
```

## Azure

```bash
# Login
az login
# Set the subscription to use
az account set -s <subscription_id>
```

## Launching jobs

```bash
curl <IP>:<port>/generate \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
    -H 'Content-Type: application/json'
```

```bash
#After a task’s execution, use rsync or scp to download files (e.g., checkpoints):
rsync -Pavz mycluster:/remote/source /local/dest  # copy from remote VM
#For uploading files to the cluster, see [Syncing Code and Artifacts](https://skypilot.readthedocs.io/en/latest/examples/syncing-code-artifacts.html#sync-code-artifacts).
```

# GPUs

## AWS

Amazon EC2 P3 Instances have up to 8 NVIDIA Tesla V100 GPUs.
Amazon EC2 P4 Instances have up to 8 NVIDIA Tesla A100 GPUs.
Amazon EC2 P5 Instances have up to 8 NVIDIA Tesla H100 GPUs.
Amazon EC2 G3 Instances have up to 4 NVIDIA Tesla M60 GPUs.
Amazon EC2 G4 Instances have up to 4 NVIDIA T4 GPUs.
Amazon EC2 G5 Instances have up to 8 NVIDIA A10G GPUs.
Amazon EC2 G6 Instances have up to 8 NVIDIA L4 GPUs.
Amazon EC2 G5g Instances have Arm64-based AWS Graviton2 processors.

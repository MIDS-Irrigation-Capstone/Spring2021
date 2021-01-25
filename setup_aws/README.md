# AWS Infrastructure Setup

## Prerequisites

- AWS account and API keys configured
- Terraform or docker installed

If you prefer not install terraform you can run with docker as follows

```
docker run --rm -w /tf -v "$HOME/.aws":/root/.aws -v "$PWD":/tf hashicorp/terraform
```

It would be easiest to alias this command
```
alias terraform='docker run --rm -w /tf -v "$HOME/.aws":/root/.aws -v "$PWD":/tf hashicorp/terraforma'

```
## Deployment

1. Set terraform variables in `terraform.tfvars`. If you create a key public key called `mids` in the `us-west-2` region then you will likely not have to modify any of the default values. The exception is for testing. If a `v100` is not needed for model training then it would be a beneficial cost-saving measure to launch a smaller instance and EBS volume.

2. If this is the first deployment terraform must be initialized

```
$ terraform init

```

3. Deploy
```
$ terraform apply

```

4. A new AWS VPC should be created and an instance should be launched with the an EBS volume attached and a mount to the imae data in s3.

  - S3 mount: `/mnt/irrigation_data`
  - EBS mount: `/data`


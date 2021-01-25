provider "aws" {
  region = "us-west-2"
}

module "vpc" {
  source = "./modules/vpc"

  availability_zone = var.availability_zone
  ssh_ingress_cidr  = var.ssh_ingress_cidr
}

module "instance" {
  source = "./modules/instance"

  subnet_id         = module.vpc.mids_subnet
  sg_ids            = [module.vpc.sg_ssh_access]
  availability_zone = var.availability_zone
  key_name          = var.key_name
  bucket_name       = var.bucket_name
  spot_price        = var.spot_price
  instance_type     = var.instance_type
  volume_size       = var.volume_size
}

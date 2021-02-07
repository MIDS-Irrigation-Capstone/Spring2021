variable "key_name" {
  type = string
  description = "SSH public key for EC2"
  default = "mids"
}
variable "bucket_name" {
  type = string
  description = "S3 bucket containing image data"
  default = "mids-capstone-irrigation-detection"
}
variable "ami_id" {
  type = string
  description = "AMI ID to use for instance. Should be debian based. Default is ubuntu 18.04"
  default = "ami-025102f49d03bec05"
}
variable "availability_zone" {
  type = string
  description = "AWS availability zone to launch instance"
  default = "us-west-2a"
}
variable "spot_price" {
  type    = string
  description = "Max spot price for EC2 instance"
  default = "1.50"
}
variable "instance_type" {
  type    = string
  description = "EC2 instance type"
  default = "p3.2xlarge"
}
variable "volume_size" {
  type    = number
  description = "Volume size in GB for EBS mount"
  default = 1000
}
variable "ssh_ingress_cidr" {
  type = string
  description = "The CIDR block to allow SSH access to the instance."
  default = "0.0.0.0/0"
}

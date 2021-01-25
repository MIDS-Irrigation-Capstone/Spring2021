variable "ssh_ingress_cidr" {
  type = string
  description = "The CIDR block to allow SSH access to the instance."
}
variable "availability_zone" {
  type = string
}

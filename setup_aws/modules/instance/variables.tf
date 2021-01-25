variable "key_name" {
  type = string
}
variable "bucket_name" {
  type = string
}
variable "availability_zone" {
  type = string
}
variable "subnet_id" {
  type = string
}
variable "sg_ids" {
  type = list(string)
}
variable "spot_price" {
  type    = string
}
variable "instance_type" {
  type    = string
}
variable "volume_size" {
  type    = number
}

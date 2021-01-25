output "ec2_public_ip" {
  value = aws_spot_instance_request.mids.public_ip
}

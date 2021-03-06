output "ec2_public_ip" {
  value = aws_instance.mids.public_ip
}

output "mids_subnet" {
  value = aws_subnet.mids.id
}

output "sg_ssh_access" {
  value = aws_security_group.ssh_access.id
}


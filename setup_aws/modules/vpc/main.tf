resource "aws_vpc" "mids" {
  cidr_block          = "10.0.0.0/16"
  instance_tenancy    = "default"

  tags = {
    Name = "mids"
  }
}

resource "aws_internet_gateway" "egress" {
  vpc_id = aws_vpc.mids.id
}

resource "aws_route" "r" {
  route_table_id         = aws_vpc.mids.main_route_table_id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.egress.id
}

resource "aws_subnet" "mids" {
  vpc_id                  = aws_vpc.mids.id
  availability_zone       = var.availability_zone
  cidr_block              = "10.0.1.0/24"
  map_public_ip_on_launch = true

  tags = {
    Name = "mids"
  }
}

resource "aws_security_group" "ssh_access" {
  name        = "ssh_access"
  description = "Allow SSH inbound traffic"
  vpc_id      = aws_vpc.mids.id

  ingress {
    description = "SSH access"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_ingress_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "sg_mids_ssh"
  }
}


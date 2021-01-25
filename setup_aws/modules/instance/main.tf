data "aws_ami" "ubuntu" {
  most_recent = true

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-bionic-18.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  owners = ["099720109477"] # Canonical
}

resource "aws_iam_instance_profile" "mids" {
  name = "mids_instance"
  role = aws_iam_role.mids.name
}

resource "aws_iam_role_policy" "mids" {
  name = "mids_s3_policy"
  role = aws_iam_role.mids.id

  policy = <<-EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": ["arn:aws:s3:::${var.bucket_name}"]
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject"
      ],
      "Resource": ["arn:aws:s3:::${var.bucket_name}/*"]
    }
  ]
}
EOF
}

resource "aws_iam_role" "mids" {
  name = "mids_s3_role"
  path = "/"

  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": "sts:AssumeRole",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Effect": "Allow",
      "Sid": ""
    }
  ]
}
EOF
}

resource "aws_spot_instance_request" "mids" {
  ami                    = data.aws_ami.ubuntu.id
  spot_price             = var.spot_price
  instance_type          = var.instance_type
  iam_instance_profile   = aws_iam_instance_profile.mids.name
  key_name               = var.key_name
  vpc_security_group_ids = var.sg_ids
  subnet_id              = var.subnet_id
  availability_zone      = var.availability_zone
  user_data              = file("modules/instance/bootstrap.sh")

  tags = {
    Name = "MIDS-Capstone"
  }

  ebs_block_device {
    device_name = "/dev/sdh"
    volume_size = var.volume_size
  }
}


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
    },
    {
      "Effect": "Allow",
      "Action": ["secretsmanager:GetSecretValue"],
      "Resource": ["arn:aws:secretsmanager:us-west-2:672750028551:secret:github-5LKXx9"]
    },
    {
      "Effect": "Allow",
      "Action": [
        "kms:Encrypt",
        "kms:Decrypt",
        "kms:ReEncrypt*",
        "kms:GenerateDataKey*",
        "kms:DescribeKey"
      ],
      "Resource": ["arn:aws:kms:us-west-2:672750028551:key/ed81d25b-fcbd-4b39-98b4-4986e03158bb"]
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

resource "aws_instance" "mids" {
  ami                    = var.ami_id
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

/* resource "aws_spot_instance_request" "mids" { */
/*   ami                    = var.ami_id */
/*   spot_price             = var.spot_price */
/*   instance_type          = var.instance_type */
/*   iam_instance_profile   = aws_iam_instance_profile.mids.name */
/*   key_name               = var.key_name */
/*   vpc_security_group_ids = var.sg_ids */
/*   subnet_id              = var.subnet_id */
/*   availability_zone      = var.availability_zone */
/*   user_data              = file("modules/instance/bootstrap.sh") */

/*   tags = { */
/*     Name = "MIDS-Capstone" */
/*   } */

/*   ebs_block_device { */
/*     device_name = "/dev/sdh" */
/*     volume_size = var.volume_size */
/*   } */
/* } */


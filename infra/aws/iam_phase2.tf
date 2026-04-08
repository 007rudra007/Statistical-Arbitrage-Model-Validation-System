# ── Aladdin – IAM Roles for Service Accounts (IRSA) ──────────────────────────
# Grants K8s pods scoped S3 permissions without static credentials.
#
# Prerequisites:
#   1. EKS cluster with OIDC provider (output from cluster Terraform)
#   2. Set TF_VAR_cluster_oidc_issuer from: aws eks describe-cluster ...
#
# Deploy:
#   cd infra/aws && terraform init && terraform apply -target=module.iam_phase2
#
# Reference the role ARNs in helm/minio/values.yaml or K8s ServiceAccount annotations.

terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# ── Variables ─────────────────────────────────────────────────────────────────
variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "ap-south-1"
}

variable "account_id" {
  description = "AWS account ID (12 digits)"
  type        = string
}

variable "cluster_oidc_issuer" {
  description = "EKS cluster OIDC provider URL (without https://)"
  type        = string
  # Example: oidc.eks.ap-south-1.amazonaws.com/id/EXAMPLED539D4633E53DE1B716D3041E
}

variable "namespace" {
  description = "Kubernetes namespace where Aladdin runs"
  type        = string
  default     = "aladdin"
}

# ── Locals ────────────────────────────────────────────────────────────────────
locals {
  oidc_provider_arn = "arn:aws:iam::${var.account_id}:oidc-provider/${var.cluster_oidc_issuer}"

  buckets = {
    raw        = "aladdin-raw"
    clean      = "aladdin-clean"
    delta      = "aladdin-delta"
    portfolios = "aladdin-portfolios"
    reports    = "aladdin-reports"
  }
}

# ── Helper: OIDC trust policy ─────────────────────────────────────────────────
data "aws_iam_policy_document" "oidc_assume" {
  for_each = {
    ingestion  = "aladdin-ingestion"
    spark      = "aladdin-spark"
    api        = "aladdin-api"
  }

  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [local.oidc_provider_arn]
    }

    condition {
      test     = "StringEquals"
      variable = "${var.cluster_oidc_issuer}:sub"
      values   = ["system:serviceaccount:${var.namespace}:${each.value}"]
    }

    condition {
      test     = "StringEquals"
      variable = "${var.cluster_oidc_issuer}:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

# ── IAM Role: Ingestion ───────────────────────────────────────────────────────
resource "aws_iam_role" "ingestion" {
  name               = "AladdinIngestionRole"
  assume_role_policy = data.aws_iam_policy_document.oidc_assume["ingestion"].json

  tags = { Component = "ingestion", Project = "aladdin" }
}

resource "aws_iam_role_policy" "ingestion_s3" {
  name = "IngestionS3Policy"
  role = aws_iam_role.ingestion.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadWriteRawClean"
        Effect = "Allow"
        Action = [
          "s3:PutObject", "s3:GetObject", "s3:DeleteObject",
          "s3:ListBucket", "s3:GetBucketLocation",
        ]
        Resource = [
          "arn:aws:s3:::${local.buckets.raw}",
          "arn:aws:s3:::${local.buckets.raw}/*",
          "arn:aws:s3:::${local.buckets.clean}",
          "arn:aws:s3:::${local.buckets.clean}/*",
        ]
      },
      {
        Sid    = "ListAllBuckets"
        Effect = "Allow"
        Action = ["s3:ListAllMyBuckets"]
        Resource = "*"
      }
    ]
  })
}

# ── IAM Role: Spark ───────────────────────────────────────────────────────────
resource "aws_iam_role" "spark" {
  name               = "AladdinSparkRole"
  assume_role_policy = data.aws_iam_policy_document.oidc_assume["spark"].json

  tags = { Component = "spark", Project = "aladdin" }
}

resource "aws_iam_role_policy" "spark_s3" {
  name = "SparkS3Policy"
  role = aws_iam_role.spark.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadRaw"
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:ListBucket", "s3:GetBucketLocation"]
        Resource = [
          "arn:aws:s3:::${local.buckets.raw}",
          "arn:aws:s3:::${local.buckets.raw}/*",
        ]
      },
      {
        Sid    = "WriteCleanDelta"
        Effect = "Allow"
        Action = [
          "s3:PutObject", "s3:GetObject", "s3:DeleteObject",
          "s3:ListBucket", "s3:GetBucketLocation",
        ]
        Resource = [
          "arn:aws:s3:::${local.buckets.clean}",
          "arn:aws:s3:::${local.buckets.clean}/*",
          "arn:aws:s3:::${local.buckets.delta}",
          "arn:aws:s3:::${local.buckets.delta}/*",
        ]
      }
    ]
  })
}

# ── IAM Role: API (read-only + portfolio write) ───────────────────────────────
resource "aws_iam_role" "api" {
  name               = "AladdinAPIRole"
  assume_role_policy = data.aws_iam_policy_document.oidc_assume["api"].json

  tags = { Component = "api", Project = "aladdin" }
}

resource "aws_iam_role_policy" "api_s3" {
  name = "APIS3Policy"
  role = aws_iam_role.api.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadDataLake"
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:ListBucket", "s3:GetBucketLocation"]
        Resource = [
          for b in [local.buckets.clean, local.buckets.delta] :
          "arn:aws:s3:::${b}"
        ] + [
          for b in [local.buckets.clean, local.buckets.delta] :
          "arn:aws:s3:::${b}/*"
        ]
      },
      {
        Sid    = "PortfolioReadWrite"
        Effect = "Allow"
        Action = [
          "s3:PutObject", "s3:GetObject", "s3:DeleteObject",
          "s3:ListBucket", "s3:GetBucketLocation",
        ]
        Resource = [
          "arn:aws:s3:::${local.buckets.portfolios}",
          "arn:aws:s3:::${local.buckets.portfolios}/*",
          "arn:aws:s3:::${local.buckets.reports}",
          "arn:aws:s3:::${local.buckets.reports}/*",
        ]
      }
    ]
  })
}

# ── S3 Buckets (create if not using MinIO on-cluster) ─────────────────────────
resource "aws_s3_bucket" "aladdin_buckets" {
  for_each = local.buckets

  bucket        = each.value
  force_destroy = false   # protect prod data

  tags = {
    Project   = "aladdin"
    ManagedBy = "terraform"
    Phase     = "2"
  }
}

resource "aws_s3_bucket_versioning" "aladdin_buckets" {
  for_each = local.buckets
  bucket   = aws_s3_bucket.aladdin_buckets[each.key].id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "aladdin_buckets" {
  for_each = local.buckets
  bucket   = aws_s3_bucket.aladdin_buckets[each.key].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "aladdin_buckets" {
  for_each = local.buckets
  bucket   = aws_s3_bucket.aladdin_buckets[each.key].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ── Outputs ───────────────────────────────────────────────────────────────────
output "ingestion_role_arn" {
  description = "IRSA Role ARN for NSE ingestion pods"
  value       = aws_iam_role.ingestion.arn
}

output "spark_role_arn" {
  description = "IRSA Role ARN for Spark pipeline pods"
  value       = aws_iam_role.spark.arn
}

output "api_role_arn" {
  description = "IRSA Role ARN for FastAPI pods"
  value       = aws_iam_role.api.arn
}

output "bucket_names" {
  description = "All created S3 bucket names"
  value       = { for k, b in aws_s3_bucket.aladdin_buckets : k => b.bucket }
}

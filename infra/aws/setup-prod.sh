#!/usr/bin/env bash
# =============================================================================
# Aladdin — AWS Free Tier Production Setup
# Uses EC2 t2.micro (free 12 months) + k3s (lightweight Kubernetes)
# NOT EKS — EKS costs $73/mo and is not free tier
# Run this ON the EC2 instance after SSH-ing in
# =============================================================================
set -euo pipefail

REGION="ap-south-1"          # Mumbai
KEY_NAME="aladdin-key"
INSTANCE_TYPE="t2.micro"     # Free tier eligible (1 vCPU, 1GB RAM)
AMI_ID="ami-0f5ee92e2d63afc18"  # Ubuntu 22.04 LTS ap-south-1 (verify latest)
SG_NAME="aladdin-sg"
S3_BUCKET="aladdin-data-lake-$(date +%s)"  # unique suffix

echo "=== [1/5] Creating security group ==="
SG_ID=$(aws ec2 create-security-group \
  --group-name "$SG_NAME" \
  --description "Aladdin trading platform" \
  --region "$REGION" \
  --query 'GroupId' --output text 2>/dev/null || \
  aws ec2 describe-security-groups \
    --group-names "$SG_NAME" \
    --region "$REGION" \
    --query 'SecurityGroups[0].GroupId' --output text)

# Allow SSH + K8s API + app ports
aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --region "$REGION" \
  --ip-permissions \
  '[{"IpProtocol":"tcp","FromPort":22,"ToPort":22,"IpRanges":[{"CidrIp":"0.0.0.0/0"}]},
    {"IpProtocol":"tcp","FromPort":6443,"ToPort":6443,"IpRanges":[{"CidrIp":"0.0.0.0/0"}]},
    {"IpProtocol":"tcp","FromPort":8000,"ToPort":8001,"IpRanges":[{"CidrIp":"0.0.0.0/0"}]},
    {"IpProtocol":"tcp","FromPort":80,"ToPort":80,"IpRanges":[{"CidrIp":"0.0.0.0/0"}]},
    {"IpProtocol":"tcp","FromPort":443,"ToPort":443,"IpRanges":[{"CidrIp":"0.0.0.0/0"}]}]' \
  2>/dev/null || echo "Rules already exist"

echo "=== [2/5] Creating key pair ==="
aws ec2 create-key-pair \
  --key-name "$KEY_NAME" \
  --region "$REGION" \
  --query 'KeyMaterial' \
  --output text > "${KEY_NAME}.pem" 2>/dev/null || echo "Key already exists"
chmod 400 "${KEY_NAME}.pem" 2>/dev/null || true

echo "=== [3/5] Launching EC2 t2.micro (free tier) ==="
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --region "$REGION" \
  --user-data file://infra/aws/k3s-userdata.sh \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=aladdin-prod},{Key=Project,Value=aladdin}]" \
  --count 1 \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Instance ID: $INSTANCE_ID"
echo "Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --region "$REGION" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)
echo "Public IP: $PUBLIC_IP"

echo "=== [4/5] Creating S3 bucket for data lake ==="
aws s3 mb "s3://$S3_BUCKET" --region "$REGION" 2>/dev/null || echo "Bucket exists"
aws s3api put-bucket-versioning \
  --bucket "$S3_BUCKET" \
  --versioning-configuration Status=Enabled

echo "=== [5/5] Output kubeconfig instructions ==="
echo ""
echo "▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄"
echo " NEXT STEPS (wait 3 min for k3s to boot)"
echo "▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄"
echo ""
echo "1. SSH into your instance:"
echo "   ssh -i ${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo ""
echo "2. Get kubeconfig (from local machine):"
echo "   ssh -i ${KEY_NAME}.pem ubuntu@${PUBLIC_IP} 'sudo cat /etc/rancher/k3s/k3s.yaml' | sed 's/127.0.0.1/${PUBLIC_IP}/g' > ~/.kube/aladdin-prod.yaml"
echo "   export KUBECONFIG=~/.kube/aladdin-prod.yaml"
echo ""
echo "3. Verify:"
echo "   kubectl get nodes"
echo ""
echo "4. Deploy auth service:"
echo "   kubectl create namespace aladdin"
echo "   helm upgrade --install aladdin-auth ./helm/auth --namespace aladdin --set jwtSecret=\$(openssl rand -hex 32)"
echo ""
echo "S3 bucket: $S3_BUCKET"
echo "Save this: echo '$S3_BUCKET' > infra/aws/.s3-bucket-name"
echo "$S3_BUCKET" > infra/aws/.s3-bucket-name

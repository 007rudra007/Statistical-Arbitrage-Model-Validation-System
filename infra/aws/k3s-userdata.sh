#!/usr/bin/env bash
# =============================================================================
# EC2 User Data — auto-runs on first boot
# Installs: Docker, k3s (lightweight Kubernetes), kubectl alias
# =============================================================================
set -euo pipefail

# Update and install Docker
apt-get update -y
apt-get install -y docker.io curl wget

# Enable Docker
systemctl enable docker
systemctl start docker
usermod -aG docker ubuntu

# Install k3s (lightweight certified Kubernetes — ~70MB vs 500MB for full K8s)
# k3s bundles: kubectl, containerd, flannel CNI, CoreDNS, Traefik ingress
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="server --write-kubeconfig-mode 644 --disable traefik" sh -

# Wait for k3s to be ready
sleep 30
k3s kubectl get nodes

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Create namespace
k3s kubectl create namespace aladdin || true

# Signal: ready
echo "k3s installation complete" > /tmp/k3s-ready
echo "[ALADDIN] Node ready: $(hostname) at $(date)" >> /var/log/aladdin-setup.log

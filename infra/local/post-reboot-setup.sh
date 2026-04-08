#!/usr/bin/env bash
# =============================================================================
# post-reboot-setup.sh
# Run this ONCE after rebooting (WSL2 needs reboot to activate)
# Sets up: WSL2 Ubuntu, Docker, Minikube, and deploys auth to local cluster
# =============================================================================
set -euo pipefail

echo "======================================"
echo " Aladdin Phase 1 - Post Reboot Setup"
echo "======================================"

# 1. Install Ubuntu in WSL2
echo "[1/6] Installing Ubuntu in WSL2..."
wsl --install -d Ubuntu --no-launch

# Wait for WSL to finish
sleep 5

# 2. Set WSL2 as default
wsl --set-default-version 2

echo "[2/6] Restarting Docker Desktop with WSL2 backend..."
# Docker Desktop should now auto-detect WSL2 and start the Linux engine
# Restart it if it was already running
taskkill /IM "Docker Desktop.exe" /F 2>/dev/null || true
sleep 2
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
echo "Waiting 60s for Docker Desktop to initialize with WSL2..."
sleep 60

# 3. Verify Docker
echo "[3/6] Verifying Docker..."
docker version

# 4. Start Minikube
echo "[4/6] Starting Minikube (docker driver)..."
minikube start --driver=docker --cpus=2 --memory=4096 --kubernetes-version=stable
minikube status

# 5. Create namespace and test
echo "[5/6] Creating namespace and smoke testing..."
kubectl apply -f infra/k8s/namespace.yaml
kubectl apply -f infra/k8s/nginx-test.yaml
kubectl rollout status deployment/nginx-test --timeout=120s
echo "Nginx test: $(minikube service nginx-test --url)"

# 6. Build and deploy auth
echo "[6/6] Building and deploying auth service..."
eval $(minikube docker-env)   # point Docker CLI at Minikube's daemon
docker build -t aladdin-auth:latest ./services/auth/

helm upgrade --install aladdin-auth ./helm/auth \
  --namespace aladdin \
  --create-namespace \
  --set image.repository=aladdin-auth \
  --set image.tag=latest \
  --set image.pullPolicy=Never \
  --set jwtSecret="$(python -c 'import secrets; print(secrets.token_hex(32))')" \
  --wait --timeout=120s

kubectl get pods -n aladdin
kubectl get svc -n aladdin

echo ""
echo "======================================"
echo " Phase 1 LOCAL COMPLETE!"
echo " Auth docs: http://$(minikube ip):30081/auth/docs"
echo " Test login:"
echo '  curl -X POST http://$(minikube ip):30081/auth/login \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"username":"admin","password":"aladdin123"}'"'"''
echo "======================================"

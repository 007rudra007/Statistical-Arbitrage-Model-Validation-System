# Aladdin Clone: End-to-End Trading System

A quantitative trading system deployable on OCI with NSE trading, based on a granular 10-week implementation plan.

## User Review Required

> [!WARNING]
> This plan will require installing Docker Desktop, Minikube, Helm, and Kubectl locally on Windows to begin Week 1. These tools typically require manual approval, reboots, and administrator access (e.g., using Chocolatey or Winget). 
> Please approve this plan so we can begin the installation of these required tools and process Week 1.

## Proposed Changes

We will strictly follow the provided 10-week plan with no changes to the steps. Below is the structured roadmap.

---

### Phase 1: Week 1: Foundation (Infra + Auth)
**Goal**: Running Kubernetes cluster with secure API gateway.

1. **Local K8s Setup:** Install minikube on Ubuntu/Windows. Provide exact commands for kubectl context switch. Test with nginx deployment.
2. **OCI Cluster (Prod):** Step-by-step: Create OKE Kubernetes cluster on OCI Free Tier. Generate kubeconfig. Budget <₹500/month.
3. **Docker Base:** Dockerfile for Python 3.12 FastAPI app with QuantLib, LangGraph. Multi-stage build <500MB.
4. **Auth Service:** FastAPI /auth endpoint with JWT (Keycloak OSS). Users: admin/trader. Dockerize + Helm chart.

*Deploy: `kubectl apply -f auth.yaml`; test login.*

---

### Phase 2: Week 2: Data Pipeline
**Goal**: Real-time NSE data ingestion + lake.

5. **NSE Ingestion:** Python script: yfinance + NSEpy fetch NIFTY50/BankNifty OHLCV 1min. Save Parquet to MinIO.
6. **Spark Pipeline:** Dockerized Apache Spark job: Process 1TB tick data daily. Delta Lake format. Kubernetes cronjob YAML.
7. **MinIO/S3:** Helm install MinIO on K8s. Python boto3 client for portfolio uploads. IAM roles.

*Test: Query 1M rows; latency <2s.*

---

### Phase 3: Week 3-4: Risk Engine (Core Aladdin)
**Goal**: VaR/Monte Carlo identical to Aladdin SDK.

8. **QuantLib Setup:** pip install QuantLib-Python. Example: Black-Scholes pricer for NIFTY calls.
9. **VaR Service:** FastAPI /risk/var: Input portfolio JSON, output historical/parametric VaR + ES. NumPy 10k sims.
10. **Stress Tests:** Aladdin-like shocks: COVID(-30%), Rate+200bps. Endpoint /risk/stress.
11. **GARCH Vol:** arch library GJR-GARCH fit on NIFTY returns. Forecast 1D/1W vol for risk.
12. **ORE Integration:** Docker QuantLib-ORE sim: XVA for derivatives portfolio.

*Deploy services; test portfolio JSON → VaR report.*

---

### Phase 4: Week 5-6: AI Multi-Agent Layer
**Goal**: LangGraph agents mirroring BlackRock's production AI.

13. **LangGraph Scaffold:** LangGraph StateGraph: 4 agents (Macro, Quant, Trade, Compliance). Command routing like Synthetix Alpha.
14. **Macro Agent (RAG)::** Offline multimodal RAG: Ingest NSE news/PDFs. FAISS + Llama3. Query 'NIFTY outlook' → thesis.
15. **Quant Agent:** /agent/quant: Input signals → GARCH/MC risk scores. Output: {risk_score: 0.7}.
16. **Consensus:** LLM debate: Agents vote on trade thesis. Softmax confidence.
17. **Offline Mode:** Extend my multimodal RAG: Add agent traces to Streamlit debug.

*Test: End-to-end query → agent decision.*

---

### Phase 5: Week 7-8: Portfolio & Trading
**Goal**: Optimizer + live execution.

18. **Optimizer:** CVXPY mean-variance: NSE universe, constraints (beta<1, sectors). /portfolio/optimize.
19. **FIX Engine:** QuickFIX/Python: Connect Zerodha Kite API. Place market/limit orders.
20. **Backtester:** Qlib workflow: Event-driven backtest pairs trading. Metrics: Sharpe, MDD.
21. **Compliance:** SEBI rules checker: Position limits, KYC flags.

*Live trade test: Paper → real ₹1000 NIFTY position.*

---

### Phase 6: Week 9: Dashboard + Monitoring
**Goal**: Aladdin UX clone.

22. **React Frontend:** Vite React app: Portfolio viewer, risk charts (Recharts), agent chat. API proxy.
23. **Streamlit MVP:** Streamlit dashboard: Upload portfolio → VaR heatmap + agent sim.
24. **Grafana Dash:** Prometheus scrape FastAPI metrics. Risk/vol dashboards.
25. **ELK Logs:** Fluentd → Elasticsearch → Kibana. Agent trace search.

---

### Phase 7: Week 10: Deploy & Scale

26. **CI/CD:** GitHub Actions: Lint/test/build/push Docker → OCI registry → ArgoCD sync.
27. **Helm Umbrella:** Single Helm chart: All microservices + Postgres + Redis.
28. **Load Test:** k6 script: 1000 concurrent /risk calls. Tune HPA autoscaling.
29. **Security:** Kyverno policies: Pod security, network policies. Cert-Manager TLS.
30. **Backup/Monitor:** Velero backups to OCI. PagerDuty alerts on Sharpe drop.

---

## Required Tools Installation

We need to install the following core tools on the Windows host to proceed with building the infrastructure. Once approved, I will initiate installation for:
- `docker` (may require manual install / Windows Docker Desktop)
- `minikube` via package manager (e.g., winget)
- `kubectl` 
- `helm`
- Python libraries

## Open Questions
- Do you have administrative access to this Windows system to install Docker Desktop and Minikube?
- Would you like me to attempt auto-installing these tools via `winget`, or would you prefer to install Docker manually first?

## Verification Plan

### Automated Tests
- `kubectl version --client` and `minikube status` should succeed.
- Python tests for `QuantLib` or any other dependencies.
- Infrastructure and microservices tested per the 10-week milestones defined above.

### Manual Verification
- Verify Docker Desktop and Minikube are running.
- Ensure Kubernetes context points correctly.

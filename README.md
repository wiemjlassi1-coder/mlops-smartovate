# MLOps Pipeline — Azure Machine Learning
**SMARTOVATE Internship Project | June–July 2026**

---

## Project overview

End-to-end MLOps pipeline on Azure ML covering automated training, CI/CD,
model versioning with MLflow, real-time deployment, and A/B testing between
a champion and challenger model.

---

## Folder structure

```
mlops-project/
│
├── src/
│   ├── train.py                  # Training script (US 1.2 + US 2.1)
│   └── score.py                  # Inference script for the endpoint (US 3.2)
│
├── tests/
│   └── test_utils.py             # Unit tests — run by CI pipeline (US 3.1)
│
├── deployment/
│   ├── endpoint.yaml             # Managed Online Endpoint definition (US 3.2)
│   ├── champion-deployment.yaml  # Champion deployment — 80% traffic (US 4.1)
│   ├── challenger-deployment.yaml# Challenger deployment — 20% traffic (US 4.1)
│   ├── ab-testing.yaml           # Traffic split configuration (US 4.1)
│   └── sample-input.json         # Test payload for endpoint verification
│
├── .github/
│   └── workflows/
│       ├── ci.yml                # CI: lint + test on every PR (US 3.1)
│       └── cd.yml                # CD: deploy model on merge to main (US 3.2)
│
├── job.yaml                      # Azure ML Command Job definition (US 2.2)
├── conda.yaml                    # Pinned dependencies (prevents Bug 1)
└── README.md
```

---

## Week-by-week execution guide

### Week 1–2 — Setup and baseline (Epic 1)
```bash
# 1. Clone this repo and install dependencies locally
pip install -r requirements-dev.txt

# 2. Run train.py locally to verify it works
python src/train.py --n_estimators 100 --max_depth 5

# 3. Run tests
pytest tests/ -v
```

### Week 3–4 — MLflow + Azure ML Job (Epic 2)
```bash
# Submit the training job to the remote compute cluster
az ml job create --file job.yaml \
  --workspace-name mlops-workspace-smartovate \
  --resource-group wiemjlassi

# Monitor in Azure ML Studio → Jobs
```

### Week 5–6 — CI/CD (Epic 3)
```bash
# Add AZURE_CREDENTIALS secret to your GitHub repo:
# Settings → Secrets → Actions → New repository secret
# Paste the output of:
az ad sp create-for-rbac \
  --name "mlops-github-actions" \
  --role contributor \
  --scopes /subscriptions/<your-subscription-id>/resourceGroups/wiemjlassi \
  --sdk-auth

# Then push to main — CI runs automatically on PRs
# CD runs automatically after CI passes on main
```

### Week 7–8 — A/B Testing (Epic 4)
```bash
# After champion is deployed, deploy the challenger
az ml online-deployment create \
  --file deployment/challenger-deployment.yaml \
  --endpoint-name breast-cancer-endpoint

# Apply the 80/20 traffic split
az ml online-endpoint update \
  --file deployment/ab-testing.yaml \
  --name breast-cancer-endpoint

# Test the endpoint (request header x-ms-deployment tells you which model answered)
az ml online-endpoint invoke \
  --name breast-cancer-endpoint \
  --request-file deployment/sample-input.json
```

---

## Azure resources used

| Resource | Name | Purpose |
|---|---|---|
| ML Workspace | mlops-workspace-smartovate | Central hub |
| Compute Cluster | cpu-cluster | Training jobs |
| Model Registry | breast-cancer-classifier | Model versioning |
| Container Registry | auto-created | Docker images |
| Managed Endpoint | breast-cancer-endpoint | Real-time inference |

---

## Known bugs and mitigations

| Bug | Cause | Fix |
|---|---|---|
| Dependency mismatch | Different scikit-learn versions locally vs Docker | Pinned in `conda.yaml` |
| CD timeout | Endpoint provisioning > 6 min | `timeout-minutes: 45` in `cd.yml` |
| HTTP 400 on inference | numpy types not JSON serialisable | `.tolist()` in `score.py` |

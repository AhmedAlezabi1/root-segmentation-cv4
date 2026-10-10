# Azure Cost Data

Raw Azure Cost Management exports for the CV4 project. These files record actual cloud spend and feed the cost analysis report at [`../cost_analysis.md`](../cost_analysis.md).

> **Note:** `*.csv` is globally git-ignored. This folder has an explicit `!docs/cost_data/*.csv` exception in `.gitignore` so exports committed here are tracked normally.

---

## Current export

**File:** [`azure_cost_export_2026-04-13_2026-06-13.csv`](azure_cost_export_2026-04-13_2026-06-13.csv)

**Columns:** `UsageDate`, `ServiceName`, `CostUSD`, `Cost`, `Currency`

Future exports follow the naming pattern: `azure_cost_export_<start>_<end>.csv`

---

## Provenance

| Field | Value |
|---|---|
| Source | Azure Portal -- Cost Management + Billing -- Cost analysis |
| Subscription | `0a94de80-6d3b-49f2-b3e9-ec5818862801` |
| Resource group | `buas-y2` (shared across all BUAS year-2 teams) |
| Azure ML workspace | `team-CV4-2026` |
| Scope filter | CV4 deployment resources (VMs, load balancer, storage, networking) |
| Date range | 2026-04-13 to 2026-06-13 |
| Granularity | Monthly (usage dates: `2026-05-01`, `2026-06-01`) |
| Grouped by | Service name |
| Currency | EUR (with USD column) |
| Exported by | Nazeer Idris |
| Export date | 2026-06-08 |

---

## Per-service totals

Aggregated across both usage months. Primary figures in EUR; USD shown for reference.

| Service | EUR | USD |
|---|---:|---:|
| Virtual Machines | 0.9103 | 1.0637 |
| Load Balancer | 0.2968 | 0.3467 |
| Storage | 0.1995 | 0.2332 |
| Virtual Network | 0.0400 | 0.0467 |
| Bandwidth | 0.0011 | 0.0013 |
| Logic Apps | 0.0000 | 0.0001 |
| Azure App Service | 0.0000 | 0.0000 |
| Functions | 0.0000 | 0.0000 |
| **Total** | **1.4477** | **1.6916** |

---

## Caveats

**Shared resource group:** `buas-y2` is shared across all BUAS year-2 teams. The export is filtered to CV4 resources only; figures from other teams are excluded.

**Compute is not reflected here:** Most ML training runs on the shared `lambda-0` Kubernetes cluster (namespace `cv4`), which is not billed as per-team Azure compute. This export covers deployment infrastructure only (VMs, Load Balancer, Storage, Networking) and does not include Azure ML, Container Registry, or Key Vault line items.

**Granularity:** Monthly only -- two rows per service across the export period. Suitable for a block-level total but not for daily trend analysis.

---

## Real cost estimation (standard pay-as-you-go pricing)

The figures in the export above are artificially low because the project runs on the university subscription: VMs are only powered on intermittently, training happens on the shared `lambda-0` Kubernetes cluster (not billed per team), and several services (Azure ML compute, Container Registry, Key Vault) do not appear as line items at all. To give a realistic picture of what this deployment would cost as a standalone production system, the table below estimates a full month at standard Azure pay-as-you-go list prices.

### Assumptions

- Backend, frontend, and database containers run on one **Standard B2s VM** (2 vCPU, 4 GiB RAM), running 24/7 (730 hours/month).
- One **Standard Load Balancer** with 5 or fewer rules, processing roughly 50 GB/month.
- One **standard static public IP** attached to the deployment.
- **100 GB Blob Storage (Hot, LRS)** for image data, model artifacts, and versioned data assets.
- **Weekly automated retraining** on a GPU compute instance (NC4as T4 v3, NVIDIA T4), roughly 2 hours per run, about 8.7 hours/month.
- **Container Registry (Basic tier)** for hosting backend and frontend images.
- Outbound bandwidth stays under the 100 GB/month free allowance, so it is billed at zero.
- Prices are US-region list prices (East US); West Europe rates are typically 5 to 15% higher, so treat the totals as a lower bound.
- USD to EUR conversion uses the same effective rate as the export above (1 USD = 0.856 EUR).

### Estimated monthly cost at list prices

| Service | Basis | USD/month | EUR/month |
|---|---|---:|---:|
| Virtual Machines | 1x B2s, $0.0416/hr x 730 hr | 30.37 | 26.00 |
| Load Balancer | Standard, $0.025/hr x 730 hr + $0.005/GB x 50 GB | 18.50 | 15.84 |
| Storage | 100 GB Hot LRS at $0.018/GB + transactions | 2.00 | 1.71 |
| Virtual Network | 1 standard static public IP, ~$0.005/hr x 730 hr | 3.65 | 3.12 |
| Bandwidth | ~50 GB egress, within 100 GB free tier | 0.00 | 0.00 |
| ML training compute | NC4as T4 v3 at ~$0.594/hr x 8.7 hr | 5.17 | 4.43 |
| Container Registry | Basic tier, ~$0.167/day x 30 days | 5.00 | 4.28 |
| **Total** | | **64.69** | **55.38** |

### Comparison with actual billed cost

| | Billed (2 months, export above) | Estimated (2 months, list prices) |
|---|---:|---:|
| EUR | 1.45 | ~110.76 |
| USD | 1.69 | ~129.38 |

The roughly 75x gap is explained by three things: the VM only runs during development and demo sessions rather than 24/7, training compute is absorbed by the shared university cluster instead of billed Azure ML compute, and supporting services (registry, public IP, monitoring) are pooled at the subscription level. A real production deployment of this system should budget around **55 to 70 EUR/month** at current list prices, before any reserved-instance or savings-plan discounts (which can reduce VM cost by 30 to 60% on 1 to 3 year commitments).

### Cost at scale (users and requests)

The estimate above is for the deployment as it currently runs: one always-on VM, one weekly retraining job, fixed supporting services. Real cost is driven mostly by *request volume*, not by user count directly, so the table below maps a range of usage levels to an estimated monthly bill. The user column assumes an average of **50 inference requests per user per month**; if your real usage differs, read the table by the requests/month column instead.

**Scaling assumptions (on top of the base assumptions above):**

- Each request is one image inference: ~2 MB uploaded, ~0.1 MB returned (results JSON plus a small annotated thumbnail).
- One Standard B2s VM sustains roughly **250,000 inference requests/month** on CPU before latency degrades; beyond that, capacity is added by horizontally scaling B2s instances (1 extra VM per additional 250k requests).
- Load Balancer data processed is billed on total traffic (in + out) at $0.005/GB; its hourly base charge stays fixed.
- Outbound egress is free up to 100 GB/month, then billed at ~$0.087/GB.
- Storage grows modestly as more user images and model artifacts accumulate.
- Weekly retraining, the static public IP, and the Container Registry are treated as fixed; retraining is bumped slightly at the largest tier to reflect a bigger training set.
- All figures are US-region pay-as-you-go list prices; USD to EUR uses the export's effective rate (1 USD = 0.856 EUR).

| Users | Requests/month | VMs needed | USD/month | EUR/month |
|---|---:|---:|---:|---:|
| 10 | 500 | 1 | 64.45 | 55.17 |
| 100 | 5,000 | 1 | 64.59 | 55.29 |
| 1,000 | 50,000 | 1 | 65.45 | 56.03 |
| 10,000 | 500,000 | 2 | 101.94 | 87.26 |
| 100,000 | 5,000,000 | 20 | 739.67 | 633.16 |

Up to roughly 250k requests/month the cost is essentially flat at the base estimate, because a single B2s and the fixed services absorb all the traffic and egress stays within the free tier; the only thing that moves is the small per-GB Load Balancer data charge. Cost only starts climbing once request volume forces additional VMs and egress crosses the 100 GB free allowance.

The largest tier assumes naive CPU horizontal scaling (20 B2s VMs), which is deliberately conservative and almost certainly *not* how this would be deployed in practice. At that volume a single GPU-backed inference endpoint or an autoscaling container service would handle the same load far more cheaply, and reserved-instance or savings-plan discounts (30 to 60% on VM cost) would apply, so treat the 100k-user row as an upper bound rather than a realistic target.

### Sources

Prices retrieved June 2026. All figures are pay-as-you-go list prices and subject to change; verify against the Azure Pricing Calculator before relying on them.

- Azure Virtual Machines pricing (B2s on-demand rate): https://instances.vantage.sh/azure/vm/b2s and https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/
- Azure Load Balancer pricing (Standard SKU, rules and data processed): https://azure.microsoft.com/en-us/pricing/details/load-balancer/
- Azure Blob Storage pricing (Hot tier, LRS): https://azure.microsoft.com/en-us/pricing/details/storage/blobs/
- Azure Bandwidth pricing (100 GB/month free egress allowance, then ~$0.087/GB used for the cost-at-scale table): https://azure.microsoft.com/en-us/pricing/details/bandwidth/
- Azure IP Addresses pricing (standard static public IP): https://azure.microsoft.com/en-us/pricing/details/ip-addresses/
- Azure Container Registry pricing (Basic tier): https://azure.microsoft.com/en-us/pricing/details/container-registry/
- GPU VM hourly rates (NC4as T4 v3 reference rate): https://acer.uic.edu/services/sce/azure and https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/
- Azure Pricing Calculator (for re-verifying any figure): https://azure.microsoft.com/en-us/pricing/calculator/
The cost-at-scale table reuses the per-service prices above. Its usage assumptions (50 requests/user/month, ~2 MB per image request, ~250k requests/month per B2s VM) are internal engineering estimates, not sourced figures, and should be re-tuned against real traffic data.

---

## How to produce a new export

1. Go to **Azure Portal** -> **Cost Management + Billing** -> **Cost analysis**.
2. Set the scope to subscription `buas-y2` (`0a94de80-...`). If subscription-level access is not available, fall back to the resource group `buas-y2` scope.
3. Set a **custom date range** covering the period you want to capture.
4. Add a **Resource filter** scoped to `team-CV4-2026` and its associated resources (storage account, container registry, key vault, application insights). Use a tag filter if team tags are available.
5. Set **Group by** to `Service name`.
6. Click **Download** -> **CSV** and save the file into this folder using the naming pattern above.
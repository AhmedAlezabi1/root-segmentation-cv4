# Cloud Cost Analysis — CV4 Block D

**Issue 4.10 (Task 4.10.2) · ILO 4.4 — design, planning, and cost-effective resource management**

This report compares the **estimated** cloud cost for the CV4 plant-root
segmentation system against the **actual** Azure spend recorded during Block D,
explains the variances, recommends optimisations, and projects an annual cost at
the current usage rate.

> **Time period:** 2026-04-13 – 2026-06-13 (billed usage months 2026-05 and 2026-06)
> **Source:** Azure Cost Management export at
> [`docs/cost_data/azure_cost_export_2026-04-13_2026-06-13.csv`](cost_data/azure_cost_export_2026-04-13_2026-06-13.csv)
> (provenance and per-service aggregation in [`docs/cost_data/README.md`](cost_data/README.md)).

**Headline finding.** Actual Azure spend for the block is **€1.45 (~$1.69)** — far
below estimate. This is an honest, expected result: CV4's ML training and inference
run on the shared **`lambda-0` Kubernetes cluster** (namespace `cv4`), which is
**not billed as per-team Azure compute**, and the subscription (`buas-y2`) uses
shared infrastructure under Azure-for-Students credits. The cost export therefore
captures only the Azure-billed **deployment infrastructure** (Virtual Machines,
Load Balancer, Storage, Virtual Network, Bandwidth) — it does **not** include Azure
ML compute, Container Registry, or Key Vault line items.

> **On the estimate.** No formal Sprint-1 cost estimate was recorded in the project
> repository. Per the task's fallback guidance, the **Estimated** column below is
> reconstructed from **Azure Pricing Calculator retail defaults** for the planned
> architecture (a dedicated managed GPU endpoint + supporting services running at
> retail rates for the ~2-month block). These figures are **assumptions for
> comparison only**, not a saved quote. The per-line basis is listed in §1.

---

## 1. Summary Table

| Service | Estimated Cost (€) | Actual Cost (€) | Variance (€) | Variance (%) |
|---|---:|---:|---:|---:|
| Azure ML Compute (training) | 40.00 | 0.00 | −40.00 | −100% |
| Compute / Inference VMs | 18.00 | 0.91 | −17.09 | −95% |
| Load Balancer | 6.00 | 0.30 | −5.70 | −95% |
| Blob Storage | 4.00 | 0.20 | −3.80 | −95% |
| Container Registry | 4.00 | 0.00 | −4.00 | −100% |
| Virtual Network | 1.00 | 0.04 | −0.96 | −96% |
| Networking / Egress (Bandwidth) | 2.00 | 0.00 | −2.00 | −100% |
| **Total** | **75.00** | **1.45** | **−73.55** | **−98%** |

> Actual total reconciles to the export aggregation: **€1.4477** (rounded €1.45).
> Variance is negative throughout (actual − estimate), i.e. **under budget**.

**Reconstructed estimate basis (Azure Pricing Calculator defaults, ~2-month block):**

| Service | Assumed basis for the estimate |
|---|---|
| Azure ML Compute (training) | ~40 GPU-hours on a dedicated NC-series VM @ ~€0.90/hr |
| Compute / Inference VMs | small GPU/CPU inference node, partial uptime over the block |
| Load Balancer | one Standard Load Balancer, 2 months |
| Blob Storage | ~50 GB hot tier + transactions |
| Container Registry | Basic tier, 2 months |
| Virtual Network | base VNet charges |
| Networking / Egress | modest outbound data transfer |

---

## 2. Variance Analysis

Actual spend came in **~98% below the reconstructed estimate**. The estimate assumed
CV4 would provision and pay retail for dedicated cloud resources; in practice almost
all compute was absorbed by shared, non-billed infrastructure. The three dominant
drivers:

**1. Azure ML Compute — €40.00 estimated → €0.00 actual (−100%).**
This is the single largest variance and the core story of the block. Model training
and the live `cv4-root-seg` endpoint run on the shared **`lambda-0` Kubernetes
cluster** (GPU: NVIDIA RTX 6000 Ada), which BUAS provides to year-2 teams and which
is **not billed back to the CV4 team's Azure meter**. The estimate, by contrast,
assumed a dedicated paid GPU VM for training (~40 GPU-hours). Because no such VM was
provisioned, the entire estimated compute cost evaporates — it did not disappear, it
simply sits on shared infrastructure outside this subscription's billing scope.

**2. Inference VMs + Load Balancer — €24.00 estimated → €1.21 actual (−95%).**
Virtual Machines (€0.91) and the Load Balancer (€0.30) are the only non-trivial
billed lines, together making up **~83% of all actual spend**. They reflect the
small Azure-billed footprint of the deployment (the managed endpoint's ingress and
control-plane VMs on the shared cluster), not a full dedicated inference fleet. Both
lines **dropped sharply from May to June** — Virtual Machines fell €0.71 → €0.20 and
Load Balancer €0.22 → €0.08 — consistent with the endpoint being scaled down or left
idle between demos in June. The estimate assumed a continuously-running paid node,
which never materialised.

**3. Container Registry & supporting services — €11.00 estimated → ~€0.20 actual.**
Container Registry shows **€0.00** in the export: the registry used for the project's
images is either a shared/free-tier resource or not scoped to the team-filtered
export, so no per-team charge appears. Storage (€0.20), Virtual Network (€0.04), and
Bandwidth (€0.001) are all near-zero — the project's data volumes and egress are tiny
relative to the retail-tier assumptions in the estimate.

**Caveat (carried from the export README).** The `buas-y2` resource group is **shared**
across BUAS year-2 teams, and figures here are scoped to CV4 deployment resources.
This export reflects only Azure-billed deployment infrastructure — **not** the
project's full compute/training cost, most of which is borne (un-metered) by the
shared `lambda-0` cluster. The near-zero actual is therefore a property of the shared,
credit-funded environment, **not** evidence that a production deployment would be free
(see §4).

---

## 3. Optimisation Recommendations

### Recommendation 1 — Scale the inference endpoint to zero when idle
- **Issue:** Virtual Machines + Load Balancer are the dominant billed lines (~83% of
  actual spend), driven by the `cv4-root-seg` endpoint holding **minimum 1 instance**
  continuously. The May→June drop shows the cost is almost entirely uptime-driven, so
  any time the endpoint is up but unused is pure waste.
- **Action:** Set the managed endpoint's minimum instances to **0** (scale-to-zero)
  outside active demo/assessment windows, or add a scheduled auto-shutdown so it only
  runs when needed. Re-warm on demand — the measured cold start (~17–20 s) is
  acceptable for a non-real-time research workload.
- **Estimated saving:** Up to **~€0.45–0.70/month** on the VM line and **~€0.15/month**
  on the Load Balancer — roughly **a further 50–80%** off the two largest actual lines.

### Recommendation 2 — Right-size or remove the Standard Load Balancer between demos
- **Issue:** A Standard Load Balancer is billed whenever provisioned, regardless of
  traffic. With endpoint min-instances at 1 and very low request volume (~10.6
  req/min single-client), the LB is over-provisioned for the actual demand.
- **Action:** Tear the endpoint (and its associated Load Balancer) **down between
  assessment milestones**, or front it with a lighter ingress when only single-client
  testing is needed. Recreate from the deployment scripts when a live demo is required.
- **Estimated saving:** Eliminates the **~€0.15/month** LB charge during idle periods
  (~€1.78/year at current rates), and avoids it scaling up in a production scenario.

### Recommendation 3 — Apply Blob Storage lifecycle tiering and tag CV4 resources
- **Issue:** Storage (€0.20) is small now but accumulates as masks, checkpoints, and
  intermediate artefacts pile up on hot-tier storage. Separately, because `buas-y2` is
  a **shared** resource group, costs cannot currently be attributed cleanly to CV4.
- **Action:** (a) Add a **lifecycle management policy** moving artefacts older than
  30–60 days to **Cool/Archive** tier and deleting throwaway intermediates; (b) apply a
  consistent **`team=CV4` tag** to all CV4 resources so future Cost Management exports
  can be filtered by tag instead of manual resource selection.
- **Estimated saving:** Keeps storage growth in check (Cool tier is ~50% cheaper than
  hot for cold data) and — more importantly for ILO 4.4 — gives **accurate, auditable
  per-team cost attribution** going forward.

### Recommendation 4 (bonus) — Keep training on the shared `lambda-0` cluster
- **Issue:** The reconstructed estimate's largest line (€40 Azure ML Compute) assumed a
  dedicated paid GPU VM. Provisioning one would multiply project cost for no benefit.
- **Action:** Continue running training and inference on the shared, un-metered
  `lambda-0` GPU cluster rather than spinning up dedicated Azure ML compute.
- **Estimated saving:** Avoids the entire **~€40/block** dedicated-GPU cost — the
  single biggest avoided expense in this analysis.

---

## 4. Projected Annual Cost

**Assumption:** usage continues at the **current rate observed during Block D** — i.e.
training/inference stays on the shared, un-metered `lambda-0` cluster, the subscription
remains under Azure-for-Students credits, and **no production scaling** occurs. The
block covers ~2 billed months (€1.4477 total), so the monthly run-rate is
**€1.4477 ÷ 2 ≈ €0.72/month**.

| Service | Monthly Estimate (€) | Annual Projection (€) |
|---|---:|---:|
| Virtual Machines | 0.46 | 5.46 |
| Load Balancer | 0.15 | 1.78 |
| Blob Storage | 0.10 | 1.20 |
| Virtual Network | 0.02 | 0.24 |
| Networking / Egress (Bandwidth) | 0.00 | 0.01 |
| **Total** | **≈0.72** | **≈8.69** |

> **Note — this projection is indicative only.** It assumes the *current shared-cluster
> usage rate* with no production scaling and no change in the credit/shared-infrastructure
> arrangement. It is also likely **slightly conservative on the high side**: the May→June
> trend was sharply downward (€1.12 → €0.33/month), so if the endpoint stays scaled down
> the true annual figure could be closer to **~€4/year**.
>
> A **production deployment would cost materially more**: a dedicated, always-on managed
> GPU endpoint, a paid Container Registry, and real user egress are not reflected here.
> Using the reconstructed retail estimate (§1, ~€75 per 2-month block) as a rough proxy,
> a dedicated production setup would land in the order of **~€450/year** before any
> reserved-instance or scale-to-zero optimisation. The near-zero actuals are a property
> of the shared, credit-funded research environment, not of the workload itself.

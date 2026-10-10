# Alert Runbooks – cv4 Operational Monitoring

A runbook is a short guide that says what to do when an alert fires. Each alert
below has: what it means, the likely causes, the steps to investigate, and how
to resolve it. These are written for the engineering team.

All alerts notify the `cv4-monitoring-alerts` action group by email. The
monitoring data lives in the `cv4-monitoring` Application Insights resource.

---

## Alert 1: cv4-high-error-rate

**Condition:** More than 5 failed requests in a 5-minute window.

**What it means:** The backend is returning errors (HTTP 4xx or 5xx) to more
requests than normal. Something is failing for users.

**Likely causes:**
- A bad deployment introduced a bug.
- The model failed to load, so `/predict` returns 503.
- A downstream dependency is down (the database, or the Azure ML endpoint if the
  backend is wired to it).
- A spike of malformed or unauthorised requests.

**How to investigate:**
1. Open `cv4-monitoring` in Azure, go to Failures (under Investigate).
2. Look at which endpoint and which status code is failing most.
3. If it is `/predict` returning 503, check `GET /health` to see if the model
   loaded.
4. Check the backend logs in Portainer for the matching time window.

**How to resolve:**
- If a recent deploy caused it, roll back to the previous image in Portainer.
- If the model did not load, restart the backend container so it reloads.
- If a dependency is down, restore that dependency.
- If it is junk traffic (lots of 401/404), confirm it is not an outage, then
  the alert can be acknowledged.

---

## Alert 2: cv4-high-latency

**Condition:** Average server response time over 10 seconds in a 5-minute window.

**What it means:** The backend is responding much more slowly than usual. Users
experience long waits.

**Likely causes:**
- The model endpoint is cold or under heavy load.
- The single backend instance is overloaded by too many requests at once.
- A slow database query or a resource limit (CPU, memory) on the container.
- A large or unusual input image taking longer to process.

**How to investigate:**
1. Open `cv4-monitoring`, go to Performance (under Investigate).
2. See which operation is slow (for example `/predict`).
3. Compare the current response time to the normal baseline (about 4 to 7
   seconds for a prediction).
4. Check the backend container resource use in Portainer.

**How to resolve:**
- If the container is resource-starved, give it more CPU or memory, or run more
  than one instance.
- If a burst of traffic caused it, the slowness should clear once the burst
  passes; consider scaling if it repeats.
- If one specific input is slow, check whether the image is unusually large.

---

## Alert 3: cv4-server-exceptions

**Condition:** More than 3 server exceptions in a 5-minute window.

**What it means:** The backend code is throwing unhandled exceptions, not just
returning error status codes. This usually points to a real bug.

**Likely causes:**
- A code bug introduced by a recent change.
- An unexpected input the code does not handle.
- A failed connection to the database or the model endpoint.

**How to investigate:**
1. Open `cv4-monitoring`, go to Failures, then look at the Exceptions tab.
2. Read the exception type and stack trace to find the line of code.
3. Check whether the exceptions started right after a deploy.

**How to resolve:**
- If a recent deploy caused it, roll back the image in Portainer.
- Fix the bug shown in the stack trace, add a test, and redeploy.
- If it is a transient dependency error, confirm the dependency recovered.

---

## Alert 4: Failure Anomalies - cv4-monitoring

**Condition:** Azure's smart detector sees an unusual rise in the failure rate
(this alert is created automatically by Application Insights and uses machine
learning rather than a fixed threshold).

**What it means:** The proportion of failing requests has risen in a way that is
abnormal compared to the recent pattern, even if the absolute count is below the
other thresholds.

**How to investigate and resolve:** Follow the same steps as
`cv4-high-error-rate`. This alert is an early-warning companion to the fixed
threshold alert; treat it the same way but expect it to fire earlier or on
smaller changes.

---

## General notes

- All four alerts are Severity 2 (Warning) except the smart detector, which is
  Severity 3 (Informational).
- Alerts notify by email through the `cv4-monitoring-alerts` action group. To
  add Teams or more recipients, edit that action group.
- After resolving an issue, the alert auto-resolves once the metric returns to
  normal. You can also change the user response in the Alerts page to mark it as
  acknowledged or closed.
- Thresholds are starting values and can be tuned as we learn the system's
  normal behaviour, to reduce false alarms or catch problems sooner.

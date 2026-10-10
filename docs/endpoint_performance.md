# Endpoint Performance: cv4-root-seg

These are the speed numbers for the live Azure ML endpoint (`cv4-root-seg`,
the "blue" deployment) running on the lambda-0 Kubernetes cluster with a GPU
instance (NVIDIA RTX 6000 Ada). They were measured with
`scripts/measure_endpoint.py`, sending the full-size validation image
(`val_Karna_240474_im1.png`, about 8.5 MB).

## Latency (how long one request takes)

| Metric | Value |
|--------|-------|
| First request (warm worker) | 7.39 s |
| Average over 5 requests | 5.65 s |
| Fastest | 4.97 s |
| Slowest | 7.39 s |

The model is loaded once when the endpoint starts (in `init()`) and stays in
memory, so most requests are fast. The very first request after a fresh
deploy is slower (about 17 to 20 s) because the worker is still warming up and
TensorFlow sets up the GPU on the first run. After that, requests settle to
around 5 s each.

## Throughput (how many requests per minute)

| Metric | Value |
|--------|-------|
| One client, one instance | 10.6 requests/min |

This is with a single client sending requests one after another to a single
instance. When the endpoint is busy, autoscaling adds more instances (up to 3),
so the total throughput goes up roughly in line with the number of instances,
to about 30 requests/min at full scale.

## Configuration

- Instance type: GPU (1 GPU per instance)
- Autoscaling: minimum 1 instance, maximum 3 instances, target 70% CPU usage
- Request timeout: 90 s
- GPU memory growth is turned on in the scoring script, so the endpoint only
  uses the GPU memory it needs instead of reserving the whole GPU on the
  shared cluster.

## How to reproduce

```bash
uv run --group azure python scripts/measure_endpoint.py \
    --image path/to/plant.png \
    --requests 5 \
    --from-workspace
```

# Deployment strategies

This section describes how cv4 rolls out new model versions on the Azure ML
online endpoint, and the two deployment strategies the team uses to do it
safely. It covers what each strategy is, when it is appropriate, and the exact
commands that carry it out.

## Background: the endpoint

The model is served by a single Kubernetes online endpoint, `cv4-root-seg`, on
the `lambda-0` cluster. The endpoint is an HTTPS URL; behind it sit one or more
*deployments*, each pairing a registered model version with the `score.py`
scoring script and the `cv4-inference` environment. Traffic to the endpoint is
split across its deployments by percentage, and that split is what every
deployment strategy below manipulates.

Two deployments are used:

  - `blue` serves model version 2 (`cv4_root_segmentation:2`, F1 0.8488), the
    current production model.
  - `green` serves model version 3 (`cv4_root_segmentation:3`, F1 0.8553), the
    sweep winner and the candidate being promoted.

Both deployments use identical serving stacks (same scoring code, environment,
GPU instance type, autoscaling, and 90s request timeout). The only difference
between them is the model version, so any change in behaviour between blue and
green comes from the model itself, not the surrounding infrastructure.

## Why a split-traffic endpoint enables both strategies

On an Azure ML online endpoint, blue/green and canary are not separate
mechanisms. Both are expressed by setting the traffic percentages across the
endpoint's deployments. Blue/green is the special case of an all-or-nothing
split (one deployment at 100, the other at 0, then flipped). Canary is the
general case of moving the split through intermediate values. Because canary
covers the in-between, a working canary process also demonstrates that a
blue/green flip is possible.

## Strategy 1: blue/green

Blue/green keeps two complete deployments live and switches traffic between
them. The new version is deployed to green and validated, then traffic is
flipped from blue to green. Rollback is the reverse flip, which is immediate.

It is the right choice when the new version has already been validated and the
goal is a clean, instant cutover with an equally instant way back. Its
trade-off is exposure: at the moment of the flip, all users move to the new
version at once, so a problem that only shows under real traffic affects
everyone immediately.

In cv4 this is demonstrated by `scripts/add_green_deployment.py`, which stands
up a second deployment alongside blue and sets a fixed split. Running it shows
two model versions coexisting behind one endpoint with traffic divided between
them, which is the foundation a blue/green flip builds on.

```
uv run --group azure python scripts/add_green_deployment.py
```

## Strategy 2: canary

Canary promotes the new version gradually. Traffic to green is increased in
stages (10%, 25%, 50%, 75%, 100%), and the endpoint's health is checked at each
stage before moving on. If the new version misbehaves at an early stage, only a
small fraction of traffic was exposed, and traffic is returned to blue.

It is the right choice when the new version has not been proven under
production traffic and the priority is limiting the blast radius of any
problem. Its trade-off is that the rollout is slower and runs a mixed fleet
during the shift.

Canary is the strategy cv4 uses to promote v3, because v3 is a freshly trained
candidate whose real-world behaviour has only been measured offline. Shifting
to it gradually means the offline F1 improvement is confirmed against live
requests before v3 carries all traffic.

### Running the canary rollout

The rollout is driven by `scripts/canary_rollout.py`. It first ensures green is
on the candidate version (v3), then moves the split through the stages. Between
stages, endpoint health is measured with `scripts/measure_endpoint.py`
(latency, throughput, and failure count), and the rollout only continues if
those numbers hold.

Advance one stage at a time, checking health between runs:

```
uv run --group azure python scripts/canary_rollout.py
uv run --group azure python scripts/measure_endpoint.py --image plant.png --requests 10 --from-workspace
uv run --group azure python scripts/canary_rollout.py
```

Or walk every stage in a single run, pausing for confirmation between each:

```
uv run --group azure python scripts/canary_rollout.py --auto
```

Jump to a specific split (for example, an even 50/50):

```
uv run --group azure python scripts/canary_rollout.py --green-traffic 50
```

### Rollback

At any stage, sending all traffic back to blue restores the proven version
without removing green, so the rollout can be retried after the candidate is
fixed:

```
uv run --group azure python scripts/canary_rollout.py --rollback
```

## What "healthy" means at each stage

Before advancing the canary, the metrics from `measure_endpoint.py` should hold
against the blue baseline:

  - failure count stays at zero (no new 5xx responses or timeouts),
  - average latency stays within the range blue already serves, since v3 uses
    the same architecture and GPU instance and should not be slower,
  - the responses are well formed (per-lane root tips and confidence scores).

A regression in any of these is the signal to roll back rather than advance.

## Cost note

The endpoint runs on the shared GPU cluster, so it is brought up only to run a
demonstration or serve a request load, then taken down to free the GPU. The
rollout scripts and this document are the durable artifacts; the live endpoint
is recreated from `scripts/create_endpoint.py` when needed.

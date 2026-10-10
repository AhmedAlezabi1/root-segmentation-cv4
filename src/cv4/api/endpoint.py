"""
api/endpoint.py
===============
Azure ML Managed Endpoint client for the cv4 inference pipeline.

This module provides ``AzureMLEndpointClient``, an HTTP client that delegates
the full segmentation pipeline to the Azure ML Managed Endpoint.  The endpoint
runs the complete cv4 pipeline — preprocessing, inference, postprocessing,
segmentation, and localisation — and returns per-lane root measurements
identical in structure to those produced by the local ``run_pipeline`` call.

The client accepts a raw image file path and returns a result dict shaped
exactly like the local ``run_pipeline`` output, so ``app.py``'s post-pipeline
code (database write, response mapping) is shared across both modes without
modification.

Wire-format
-----------
Request body (JSON)::

    {
        "image": "<base64-encoded raw image bytes>"
    }

The image is the original uploaded file — the same bytes written to the
temporary file in ``POST /predict``.  No preprocessing is performed on the
client side; the endpoint's scoring script handles it.

Response body (JSON, HTTP 200)::

    {
        "roots": [
            {"lane": 0, "has_root": true,
             "root_tip_px": {"x": 145, "y": 890}, "confidence": 0.91},
            ...
        ],
        "plate_bbox": [x, y, side],
        "processing_time_s": 4.21
    }

Error body returned by the scoring script (still HTTP 200)::

    {"error": "<human-readable description>"}

The client translates the successful response into the ``run_pipeline`` dict
shape before returning it to the caller::

    {
        "plants": [...],          # renamed from "roots"
        "plate_bbox": [...],      # passed through unchanged
        "processing_time_seconds": 4.21   # renamed from "processing_time_s"
    }

Deployment routing (blue/green)
--------------------------------
Setting ``AZURE_ML_DEPLOYMENT_NAME`` causes every request to carry an
``azureml-model-deployment`` header, routing traffic to a specific named
deployment on the endpoint.  When the variable is absent, Azure ML uses its
default traffic-split rules.  This is the mechanism used by Issue 4.2 to
implement blue/green version switching.

Environment variables consumed here
-------------------------------------
AZURE_ML_ENDPOINT_URL : str
    Full scoring URI, e.g.
    ``http://portainer.example.com:32452/api/v1/endpoint/cv4-root-seg/score``
AZURE_ML_ENDPOINT_KEY : str
    Bearer key sent in the ``Authorization`` header.
CV4_ENDPOINT_TIMEOUT_S : float, optional
    Per-request timeout in seconds.  Defaults to 30.
AZURE_ML_DEPLOYMENT_NAME : str, optional
    Named deployment for blue/green routing (Issue 4.2).
"""

from __future__ import annotations

import base64
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class AzureMLEndpointError(RuntimeError):
    """Base class for all Azure ML endpoint communication errors.

    Raised when a request to the managed endpoint fails for any reason that
    is not covered by a more specific subclass.  The message always contains
    enough detail to diagnose the failure from server logs without exposing
    internal state to API clients.
    """


class AzureMLAuthError(AzureMLEndpointError):
    """Raised when the endpoint returns HTTP 401 or 403.

    Indicates that ``AZURE_ML_ENDPOINT_KEY`` is wrong, has expired, or was
    not sent.  The operator should verify the key in the Azure Portal under
    the endpoint resource and rotate it if necessary.
    """


class AzureMLTimeoutError(AzureMLEndpointError):
    """Raised when the HTTP request exceeds the configured timeout.

    The timeout is controlled by ``CV4_ENDPOINT_TIMEOUT_S`` (default 30 s).
    A timeout usually means the endpoint pod is cold-starting or the image
    is unusually large.  Switching to local mode (``CV4_INFERENCE_MODE=local``)
    is the recommended mitigation during a demo.
    """


class AzureMLServerError(AzureMLEndpointError):
    """Raised when the endpoint returns a 5xx HTTP response.

    The ``status_code`` attribute carries the original status code so
    callers can distinguish 500 (scoring-script crash) from 503 (no healthy
    pods) for logging and alerting.

    Parameters
    ----------
    message : str
        Human-readable description including the status code and a truncated
        excerpt of the response body.
    status_code : int
        The HTTP status code returned by the endpoint (500–599).
    """

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class AzureMLEndpointClient:
    """HTTP client for the cv4 Azure ML Managed Endpoint.

    Accepts a raw image file path, sends the image bytes to the endpoint,
    and returns a result dict shaped identically to ``cv4.pipeline.run_pipeline``
    output.  This lets ``app.py``'s post-pipeline code (database write,
    response mapping) run unchanged regardless of whether inference happened
    locally or remotely.

    The endpoint runs the complete cv4 pipeline (preprocessing, UNet inference,
    postprocessing, segmentation, localisation).  The client is therefore not
    a drop-in for the local model — it replaces the entire ``run_pipeline``
    call, not just the inference step.

    All HTTP I/O is synchronous (``httpx`` in blocking mode) because the
    client is always invoked from a worker thread via ``asyncio.to_thread``
    in the FastAPI endpoint handler — the same pattern used for local
    ``run_pipeline`` calls.

    Parameters
    ----------
    url : str
        Full scoring URI of the Azure ML Managed Endpoint.
    key : str
        Bearer key sent in the ``Authorization`` header.
    timeout_s : float
        Per-request timeout in seconds.  Defaults to 30.
    deployment_name : str | None
        Optional named deployment for blue/green routing (Issue 4.2).
        When set, requests carry the ``azureml-model-deployment`` header.
    """

    def __init__(
        self,
        url: str,
        key: str,
        *,
        timeout_s: float = 30.0,
        deployment_name: str | None = None,
    ) -> None:
        self._url = url
        self._key = key
        self._timeout_s = timeout_s
        self._deployment_name = deployment_name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_headers(self) -> dict[str, str]:
        """Construct request headers including auth and optional deployment routing."""
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._key}",
        }
        if self._deployment_name:
            headers["azureml-model-deployment"] = self._deployment_name
        return headers

    @staticmethod
    def _translate_response(payload: dict[str, Any]) -> dict[str, Any]:
        """Translate the endpoint response into the run_pipeline output shape.

        The endpoint returns ``roots``, ``plate_bbox``, and
        ``processing_time_s``.  The local ``run_pipeline`` returns ``plants``,
        ``plate_bbox``, and ``processing_time_seconds``.  This method renames
        the two divergent fields so the rest of ``app.py`` needs no branching.

        Parameters
        ----------
        payload : dict
            Parsed JSON response body from the endpoint.

        Returns
        -------
        dict
            ``{"plants": [...], "plate_bbox": [...],
            "processing_time_seconds": float}``

        Raises
        ------
        AzureMLEndpointError
            If any expected key is missing from the response.
        """
        try:
            return {
                "plants": payload["roots"],
                "plate_bbox": payload["plate_bbox"],
                "processing_time_seconds": payload["processing_time_s"],
            }
        except KeyError as exc:
            raise AzureMLEndpointError(
                f"Endpoint response is missing expected key {exc}. "
                f"Received keys: {list(payload.keys())}. "
                "Check that score.py returns 'roots', 'plate_bbox', and "
                "'processing_time_s' in its success response."
            ) from exc

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, image_path: str) -> dict[str, Any]:
        """Send an image to the endpoint and return the pipeline result.

        Reads the raw image bytes from ``image_path``, base64-encodes them,
        POSTs to the endpoint, and returns a result dict with the same shape
        as ``cv4.pipeline.run_pipeline`` output so the caller requires no
        special-casing for remote vs local results.

        Parameters
        ----------
        image_path : str
            Path to the temporary image file written by ``POST /predict``.
            The file must exist and be readable; it is not deleted here.

        Returns
        -------
        dict
            ``{"plants": [...], "plate_bbox": [x, y, side],
            "processing_time_seconds": float}`` — identical structure to
            the local ``run_pipeline`` return value.

        Raises
        ------
        AzureMLAuthError
            Endpoint returned 401 or 403 — key is wrong or missing.
        AzureMLTimeoutError
            Request exceeded ``timeout_s``.
        AzureMLServerError
            Endpoint returned a 5xx status code.
        AzureMLEndpointError
            Any other failure: DNS, connection refused, scoring-script error
            (HTTP 200 with ``{"error": "..."}``), missing response keys, etc.
        """
        # Read the raw image bytes and base64-encode them for JSON transport.
        with open(image_path, "rb") as fh:
            image_b64 = base64.b64encode(fh.read()).decode("ascii")

        body = {"image": image_b64}

        logger.debug(
            "Sending inference request to endpoint (deployment=%r)",
            self._deployment_name,
        )

        # --- HTTP request ---------------------------------------------------
        try:
            response = httpx.post(
                self._url,
                json=body,
                headers=self._build_headers(),
                timeout=self._timeout_s,
            )
        except httpx.TimeoutException as exc:
            raise AzureMLTimeoutError(
                f"Endpoint request timed out after {self._timeout_s:.0f} s. "
                "Increase CV4_ENDPOINT_TIMEOUT_S or switch to local inference "
                "(CV4_INFERENCE_MODE=local) while the endpoint is unavailable."
            ) from exc
        except httpx.RequestError as exc:
            # Covers DNS failure, connection refused, SSL errors, etc.
            raise AzureMLEndpointError(f"Failed to reach endpoint at {self._url!r}: {exc}") from exc

        # --- Response status ------------------------------------------------
        if response.status_code in (401, 403):
            raise AzureMLAuthError(
                f"Endpoint rejected the request with {response.status_code}. "
                "Verify that AZURE_ML_ENDPOINT_KEY is correct and has not expired. "
                "Keys can be found under the endpoint resource in the Azure Portal."
            )

        if response.status_code >= 500:
            excerpt = response.text[:300].replace("\n", " ")
            raise AzureMLServerError(
                f"Endpoint returned {response.status_code}. Response excerpt: {excerpt!r}",
                status_code=response.status_code,
            )

        if not response.is_success:
            excerpt = response.text[:300].replace("\n", " ")
            raise AzureMLEndpointError(f"Endpoint returned unexpected status {response.status_code}. Response excerpt: {excerpt!r}")

        # --- Parse response -------------------------------------------------
        try:
            result = response.json()
            # Azure ML double-encodes when run() returns a str — unwrap once.
            if isinstance(result, str):
                import json as _json

                result = _json.loads(result)
        except Exception as exc:
            excerpt = response.text[:300].replace("\n", " ")
            raise AzureMLEndpointError(f"Endpoint returned non-JSON response (status {response.status_code}). Body excerpt: {excerpt!r}") from exc

        # The scoring script catches all exceptions internally and returns
        # {"error": "..."} as an HTTP 200 rather than raising.  Treat this
        # as a pipeline failure, not a client error, so the detail reaches
        # the API caller as a 502 with the scoring script's message.
        if "error" in result:
            raise AzureMLEndpointError(f"Endpoint scoring script returned an error: {result['error']}")

        # Translate field names to match local run_pipeline output.
        translated = self._translate_response(result)

        logger.debug(
            "Endpoint response received (%d plants, %.2fs)",
            len(translated["plants"]),
            translated["processing_time_seconds"],
        )

        return translated

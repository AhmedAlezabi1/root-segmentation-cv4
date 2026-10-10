"""
monitoring/telemetry.py
=======================
Application Insights / Azure Monitor setup for the cv4 backend.

This module wires the FastAPI backend up to Azure Application Insights so
that operational metrics (request count, latency, error rate) and traces
are collected automatically and sent to Azure.

It uses the Azure Monitor OpenTelemetry distro, which auto-instruments
FastAPI: once configured, every HTTP request is tracked (method, route,
status code, duration) with no per-endpoint code. Outgoing calls made with
the `requests` library (for example, a call to the Azure ML endpoint) are
also tracked as dependencies.

The connection string is read from the environment variable
APPLICATIONINSIGHTS_CONNECTION_STRING. It is never hardcoded, so the secret
stays out of source control. If the variable is not set (for example, when
running locally or in tests), monitoring is simply skipped with a warning,
and the application runs normally without it.

Usage (in app.py):

    from cv4.monitoring.telemetry import setup_monitoring

    app = FastAPI(...)
    setup_monitoring(app)

Custom events:

    from cv4.monitoring.telemetry import track_event

    track_event("prediction_succeeded", {"lanes_with_root": 4})
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Set to True once configure_azure_monitor has run, so custom-event helpers
# know whether telemetry is active.
_MONITORING_ENABLED = False

# Environment variable that holds the Application Insights connection string.
CONNECTION_STRING_ENV = "APPLICATIONINSIGHTS_CONNECTION_STRING"


def setup_monitoring(app) -> bool:
    """Turn on Azure Application Insights monitoring for the FastAPI app.

    Reads the connection string from the environment. If it is present,
    configures Azure Monitor and auto-instruments the FastAPI app so all
    HTTP requests, their latency, and any errors are sent to Application
    Insights. If the connection string is missing, logs a warning and does
    nothing, so local development and tests still run without monitoring.

    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance to instrument.

    Returns
    -------
    bool
        True if monitoring was turned on, False if it was skipped.
    """
    global _MONITORING_ENABLED

    connection_string = os.environ.get(CONNECTION_STRING_ENV, "").strip()

    if not connection_string:
        logger.warning(
            "%s is not set. Application Insights monitoring is disabled. Set this environment variable to enable monitoring.",
            CONNECTION_STRING_ENV,
        )
        return False

    try:
        # Imported here, not at module top, so the app does not hard-depend
        # on the azure monitor packages when monitoring is turned off.
        from azure.monitor.opentelemetry import configure_azure_monitor
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        # configure_azure_monitor reads the connection string from the
        # environment automatically, and sets up traces, logs, and metrics
        # that export to Application Insights.
        configure_azure_monitor(connection_string=connection_string)

        # Auto-instrument FastAPI: every request becomes a tracked span with
        # the route, status code, and duration.
        FastAPIInstrumentor.instrument_app(app)

        _MONITORING_ENABLED = True
        logger.info("Application Insights monitoring is enabled.")
        return True

    except Exception as exc:  # noqa: BLE001 - monitoring must never crash the app
        # If anything goes wrong setting up monitoring, the app should still
        # start and serve traffic. Monitoring is important but not worth
        # taking the whole service down for.
        logger.error("Failed to set up Application Insights monitoring: %s", exc)
        return False


def track_event(name: str, properties: dict | None = None) -> None:
    """Record a custom event in Application Insights.

    Use this to log meaningful business events such as a prediction
    succeeding or feedback being submitted. If monitoring is turned off,
    this is a no-op, so it is always safe to call.

    Parameters
    ----------
    name : str
        The event name, for example "prediction_succeeded".
    properties : dict, optional
        Extra key/value details to attach to the event.
    """
    if not _MONITORING_ENABLED:
        return

    try:
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        # A short span carries the event name and properties to App Insights,
        # where it shows up as a custom event/trace that can be searched.
        with tracer.start_as_current_span(name) as span:
            if properties:
                for key, value in properties.items():
                    span.set_attribute(key, value)
    except Exception as exc:  # noqa: BLE001 - telemetry must never break a request
        logger.warning("Failed to track event '%s': %s", name, exc)

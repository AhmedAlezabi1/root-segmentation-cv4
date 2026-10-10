# Password Reset

This document describes how to reset a forgotten password in the cv4 system.

## Overview

Password reset uses a short-lived signed JWT token (valid for **60 minutes**).
The token is tied to the user's account and cannot be used as a login token.
Requesting a reset for an unregistered email returns the same response as a
valid request — this is intentional and prevents leaking which addresses are
registered.

---

## Development (no email server)

In development (`CV4_ENV=development`) the reset token is returned directly
in the API response so the flow can be tested without an email server.

### Step 1 — Request a reset token

```
POST /auth/reset-password/request
Content-Type: application/json

{
  "email": "you@example.com"
}
```

**Response (development only):**
```json
{
  "message": "If that email is registered you will receive a reset token.",
  "reset_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

Copy the `reset_token` value.

### Step 2 — Set a new password

```
POST /auth/reset-password/confirm
Content-Type: application/json

{
  "token": "<paste reset_token here>",
  "new_password": "mynewpassword1",
  "new_password_confirm": "mynewpassword1"
}
```

**Response:**
```json
{
  "id": 1,
  "email": "you@example.com",
  "is_active": true
}
```

### Step 3 — Log in with the new password

```
POST /auth/login
Content-Type: application/json

{
  "email": "you@example.com",
  "password": "mynewpassword1"
}
```

---

## Via the web interface

1. Go to `http://localhost:3000/forgot-password`
2. Enter your email address and click **Send reset token**
3. In development you are redirected automatically to the reset form
4. Enter your new password and click **Set new password**
5. You are redirected to the login page

---

## Production (with email server)

In production (`CV4_ENV` not set to `development`) the `reset_token` field
in the response is `null`. The token would instead be delivered by email
(SMTP integration is out of scope for sprint 4.9 and can be added in a
future sprint).

To add email delivery:
1. Add an SMTP library such as `fastapi-mail` to `pyproject.toml`
2. In `POST /auth/reset-password/request`, send the token as a link:
   `https://yourdomain.com/reset-password?token=<token>`
3. Remove the `reset_token` field from the response entirely

---

## Token expiry

Reset tokens expire after **60 minutes** by default. This is controlled by
the `RESET_TOKEN_EXPIRE_MINUTES` environment variable.

After expiry, requesting a new token via `POST /auth/reset-password/request`
is required.

---

## Interactive API docs

The full request/response schema for both endpoints is available in the
auto-generated Swagger UI at:

```
http://localhost:8000/docs#/Auth
```

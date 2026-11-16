---
id: api-limits
title: API Rate Limits
---
# API Rate Limits

The Starter plan allows 60 API requests per minute.
The Pro plan allows 600 API requests per minute.
The Enterprise plan allows 3,000 API requests per minute.
The Free plan allows 10 API requests per minute.
When you exceed the limit, the API returns HTTP 429 with a Retry-After header.
Clients should retry with exponential backoff after a 429 response.
The maximum API request body size is 10 MB.

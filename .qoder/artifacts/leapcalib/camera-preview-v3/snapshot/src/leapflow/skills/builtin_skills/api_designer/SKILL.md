---
name: api_designer
description: "REST and GraphQL API design with OpenAPI spec generation and best practices"
version: 1.0.0
metadata:
  leapflow:
    category: "development"
    source: "builtin"
    confidence: 1.0
    quality_score: 1.0
  hermes:
    category: "development"
    tags: ["API", "REST", "GraphQL", "OpenAPI", "design", "specification"]
    requires_tools: ["file_read", "file_write"]
platforms: []
triggers:
  - "design API"
  - "OpenAPI"
  - "REST API"
  - "GraphQL"
  - "API设计"
  - "接口设计"
  - "API specification"
  - "endpoint design"
---

# API Designer

## Purpose

Design clean, consistent, and evolvable APIs — REST or GraphQL — and produce
machine-readable specifications (OpenAPI 3.1, GraphQL SDL).  This skill treats
API design as a contract negotiation between producer and consumer, prioritizing
developer experience, backward compatibility, and operational clarity.

## Guiding Principles

1. **Contract first** — Design the API before writing implementation code.
   The specification is the source of truth; code conforms to it.
2. **Resource-oriented thinking** — Model APIs around domain nouns (resources),
   not implementation verbs.  `/orders/{id}` not `/getOrderById`.
3. **Consistency is kindness** — Naming, pagination, error format, and
   authentication must be uniform across every endpoint.
4. **Evolvability over perfection** — Design for extension (additive changes)
   without breaking existing clients.  Avoid enums in responses; prefer
   open-ended strings with documented values.
5. **Operational transparency** — Every API must be traceable (request IDs),
   rate-limited, and versioned.

## Workflow

### Phase 1 — Understand Requirements

Before designing endpoints:

1. Identify the **domain model**: what are the core entities and their
   relationships?
2. Determine the **consumers**: frontend SPA, mobile app, third-party
   integrators, internal microservices.  Different consumers imply different
   granularity and authentication.
3. List the **use cases**: what actions do consumers need to perform?  Map each
   to a resource + operation.
4. Clarify **constraints**: auth method (OAuth2, API key, JWT), rate limits,
   data sensitivity, compliance requirements (GDPR, HIPAA).
5. Read existing API code or specs to maintain consistency.

### Phase 2 — Resource Modeling

Map domain entities to REST resources:

- **Naming**: plural nouns, lowercase, hyphen-separated
  (`/order-items`, not `/orderItems` or `/OrderItem`).
- **Hierarchy**: nest only for strong ownership (`/users/{id}/addresses`);
  use query parameters for loose associations.
- **Operations**:

  | Action | Method | Path | Status |
  |---|---|---|---|
  | List | GET | `/resources` | 200 |
  | Create | POST | `/resources` | 201 |
  | Read | GET | `/resources/{id}` | 200 |
  | Full update | PUT | `/resources/{id}` | 200 |
  | Partial update | PATCH | `/resources/{id}` | 200 |
  | Delete | DELETE | `/resources/{id}` | 204 |

- **Sub-resources vs query params**: use sub-resources for composition
  (`/orders/{id}/line-items`); use query params for filtering, sorting,
  and pagination on collection endpoints.

For **GraphQL**:
- Design a schema with clear type boundaries.
- Use `input` types for mutations; never reuse output types as inputs.
- Implement connections (Relay-style cursor pagination) for lists.
- Keep resolvers thin; business logic lives in service layer.

### Phase 3 — Standard Patterns

Apply consistent patterns across the API:

**Pagination** (choose one and use everywhere):
```json
{
  "data": [...],
  "pagination": {
    "cursor": "eyJpZCI6MTAwfQ==",
    "has_more": true,
    "total_count": 1523
  }
}
```
Prefer cursor-based for large/changing datasets; offset-based for small,
stable datasets.

**Filtering and sorting**:
```
GET /orders?status=shipped&sort=-created_at&fields=id,status,total
```
- Filter: `field=value` for equality; `field[gte]=10` for ranges.
- Sort: comma-separated fields; `-` prefix for descending.
- Sparse fields: `fields=` parameter to reduce payload.

**Error format** (RFC 7807 Problem Details):
```json
{
  "type": "https://api.example.com/errors/insufficient-funds",
  "title": "Insufficient Funds",
  "status": 422,
  "detail": "Account balance ($10.00) is below the required $25.00.",
  "instance": "/orders/42"
}
```

**Versioning**: prefer URL-path versioning (`/v1/resources`) for simplicity;
use header versioning (`Accept: application/vnd.api+json;version=2`) when
URL changes are unacceptable.

**Rate limiting headers**:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 998
X-RateLimit-Reset: 1625097600
Retry-After: 30
```

### Phase 4 — Generate OpenAPI Specification

Produce an OpenAPI 3.1 YAML document:

1. Define `info` (title, version, description, contact).
2. Define `servers` (dev, staging, production URLs).
3. Define `paths` with full request/response schemas.
4. Define `components/schemas` for all domain objects (use `$ref`).
5. Define `components/securitySchemes` and apply globally or per-operation.
6. Add `examples` for every endpoint — at least one success and one error.
7. Use `tags` to group related endpoints.

Validate the spec:
```
npx @redocly/cli lint openapi.yaml
```

### Phase 5 — Review Checklist

Before finalizing:

- [ ] Every endpoint has a description and at least one example.
- [ ] All 4xx/5xx responses are documented with error schemas.
- [ ] Pagination, filtering, and sorting are consistent across collections.
- [ ] Authentication is defined and applied to all non-public endpoints.
- [ ] No breaking changes if this is an API revision (additive only).
- [ ] Response bodies do not expose internal IDs, stack traces, or secrets.
- [ ] Enum values in responses use strings, not integers.

## Error Handling

| Situation | Action |
|---|---|
| Conflicting naming conventions in existing API | Document the conflict; follow the dominant pattern; suggest a migration plan for outliers. |
| Requirement implies RPC-style action (e.g., "send email") | Model as a resource creation: `POST /emails` with a body, not `POST /sendEmail`. |
| GraphQL schema grows too large | Split into modules/namespaces; use schema stitching or federation for microservices. |
| Consumer needs real-time updates | Recommend WebSocket or SSE alongside REST; define event schema. |

## Limitations

- This skill produces specifications and design documents, not running server
  code.  Implementation scaffolding requires the target framework's CLI.
- API security testing (penetration testing, fuzzing) is out of scope; the
  skill focuses on design-time security patterns.
- Mock server generation depends on external tools (Prism, WireMock).

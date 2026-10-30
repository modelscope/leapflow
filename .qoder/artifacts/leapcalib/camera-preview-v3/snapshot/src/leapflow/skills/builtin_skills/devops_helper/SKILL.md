---
name: devops_helper
description: "CI/CD pipeline configuration, Docker orchestration, and deployment strategy"
version: 1.0.0
metadata:
  leapflow:
    category: "operations"
    source: "builtin"
    confidence: 1.0
    quality_score: 1.0
  hermes:
    category: "operations"
    tags: ["devops", "CI/CD", "Docker", "deployment", "pipeline", "infrastructure"]
    requires_tools: ["file_read", "file_write", "shell_run"]
platforms: []
triggers:
  - "CI/CD"
  - "Docker"
  - "deploy"
  - "DevOps"
  - "pipeline"
  - "部署"
  - "持续集成"
  - "container orchestration"
---

# DevOps Helper

## Purpose

Generate and troubleshoot CI/CD pipelines, Docker configurations, and deployment
strategies.  This skill bridges the gap between application code and the
infrastructure that builds, tests, and ships it — producing configurations that
are secure, reproducible, and maintainable.

## Guiding Principles

1. **Infrastructure as code** — Every configuration must be version-controlled,
   reviewable, and reproducible.  No manual console clicks.
2. **Least privilege** — Containers run as non-root, CI tokens have minimal
   scopes, secrets never appear in logs or images.
3. **Fail fast, fail loud** — Pipelines should catch errors early and report
   them clearly.  A green build must mean the artifact is shippable.
4. **Immutable artifacts** — Build once, deploy many times.  The same image
   that passes staging goes to production.
5. **Incremental complexity** — Start with the simplest pipeline that works;
   add caching, parallelism, and matrix builds only when justified.

## Workflow

### Phase 1 — Assess the Project

1. Read project structure to understand:
   - Language and build system (package.json, pyproject.toml, go.mod, Makefile).
   - Existing CI config (.github/workflows/, .gitlab-ci.yml, Jenkinsfile,
     .circleci/).
   - Existing Docker files (Dockerfile, docker-compose.yml, .dockerignore).
   - Deployment targets (cloud provider, Kubernetes, bare metal, serverless).
2. Identify the **deployment pipeline stages** already in place vs missing:
   - Build → Test → Lint → Security scan → Package → Deploy → Smoke test.
3. Note environment-specific requirements: secrets, environment variables,
   database migrations, feature flags.

### Phase 2 — CI/CD Pipeline Design

Generate or improve CI/CD configuration:

**GitHub Actions** (default when `.github/` exists):
- Use reusable workflows for shared steps.
- Pin action versions to SHA, not tags.
- Separate jobs for lint, test, build, deploy — with explicit dependencies.
- Cache dependencies (`actions/cache`) keyed on lockfile hash.
- Use matrix strategy for multi-version testing.
- Set `concurrency` to cancel superseded runs on the same branch.

**Pipeline structure template**:
```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps: [checkout, setup, lint]

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        version: ["3.11", "3.12"]
    steps: [checkout, setup, install-deps, run-tests, upload-coverage]

  build:
    needs: [lint, test]
    steps: [checkout, build-artifact, upload-artifact]

  deploy:
    needs: [build]
    if: github.ref == 'refs/heads/main'
    environment: production
    steps: [download-artifact, deploy]
```

Adapt to the project's actual CI platform and requirements.

### Phase 3 — Docker Configuration

Generate production-grade Dockerfiles:

1. **Multi-stage builds**: separate build and runtime stages to minimize
   image size.
2. **Base image selection**: use official slim/alpine images; pin to specific
   digest or version tag.
3. **Layer ordering**: copy dependency manifests first, install dependencies,
   then copy source — maximizes cache reuse.
4. **Security**:
   - Run as non-root user (create and switch with `USER`).
   - No secrets in build args or layers.
   - Scan with `docker scout` or `trivy`.
5. **Health checks**: include `HEALTHCHECK` for orchestrated deployments.
6. **`.dockerignore`**: exclude `.git/`, `node_modules/`, `__pycache__/`,
   test fixtures, and documentation.

**docker-compose.yml** for local development:
- Define services with proper dependency ordering (`depends_on` with
  `condition: service_healthy`).
- Use named volumes for persistent data.
- Map ports explicitly; avoid `network_mode: host`.
- Provide `.env.example` for required environment variables.

### Phase 4 — Deployment Strategy

Recommend and implement a deployment strategy:

| Strategy | When to Use |
|---|---|
| **Rolling update** | Default for stateless services; zero downtime, gradual rollout. |
| **Blue/green** | When instant rollback is critical; requires 2× resources briefly. |
| **Canary** | For high-traffic services; route small percentage to new version first. |
| **Feature flags** | When deployment and release should be decoupled. |
| **Recreate** | For stateful services that cannot run two versions simultaneously. |

For each deployment:
- Define rollback triggers (error rate threshold, latency spike).
- Include smoke tests that run post-deploy.
- Ensure database migrations are backward-compatible for rolling deployments.

### Phase 5 — Validation

1. Lint all generated configs:
   - YAML: `yamllint` or platform-specific validators.
   - Dockerfile: `hadolint`.
   - docker-compose: `docker compose config`.
2. Dry-run where possible (`act` for GitHub Actions, `docker build --check`).
3. Verify secrets are not hardcoded anywhere in the generated files.
4. Test the pipeline locally before pushing.

## Error Handling

| Situation | Action |
|---|---|
| Unknown CI platform | Generate GitHub Actions config and note how to adapt for other platforms. |
| Secrets required but not configured | Generate placeholder `${{ secrets.NAME }}` references and list all required secrets with setup instructions. |
| Docker build fails | Read the build output; common causes: missing system deps, wrong base image arch, cache invalidation. |
| Pipeline is slow (>15 min) | Audit for missing caches, unnecessary steps, serial jobs that could parallelize. |

## Limitations

- This skill generates configuration files; it does not have direct access to
  CI/CD platforms or cloud consoles.
- Secret management setup (vault, cloud KMS) is guided but not automated.
- Kubernetes manifests are generated as static YAML; Helm chart generation
  is out of scope.

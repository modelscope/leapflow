---
name: security_audit
description: "Security vulnerability scanning strategy, threat modeling, and fix recommendations"
version: 1.0.0
metadata:
  leapflow:
    category: "security"
    source: "builtin"
    confidence: 1.0
    quality_score: 1.0
  hermes:
    category: "security"
    tags: ["security", "vulnerability", "audit", "OWASP", "threat-model", "hardening"]
    requires_tools: ["file_read", "shell_run"]
platforms: []
triggers:
  - "security audit"
  - "vulnerability scan"
  - "安全审计"
  - "漏洞扫描"
  - "security check"
  - "threat model"
  - "harden this code"
  - "OWASP check"
---

# Security Audit

## Purpose

Conduct a systematic security audit of a codebase or configuration, identifying
vulnerabilities, assessing their severity, and providing actionable remediation
guidance.  This skill follows a structured methodology inspired by OWASP and
industry threat-modeling frameworks — not a checklist run, but a risk-prioritized
analysis that focuses effort where exploitability is highest.

## Guiding Principles

1. **Attacker's perspective** — Think about how each finding could be exploited,
   not just whether a pattern looks suspicious.  A theoretical vulnerability
   with no practical attack path is low priority.
2. **Risk = Likelihood × Impact** — Prioritize findings by both exploitability
   and damage potential.  A SQL injection in a public endpoint outranks a
   minor info leak in an admin-only debug page.
3. **Evidence-based findings** — Every vulnerability report includes the exact
   code location, a concrete attack scenario, and a specific remediation.
   No vague warnings.
4. **Defense in depth** — Do not stop after finding one vulnerability.  Assess
   whether multiple layers of defense exist and where they are weakest.
5. **Fix the root cause** — Recommend fixes that address the underlying design
   flaw, not just the specific instance.

## Workflow

### Phase 1 — Scope and Reconnaissance

1. Map the **attack surface**:
   - Entry points: HTTP endpoints, CLI commands, message queues, file uploads,
     IPC sockets, deserialization points.
   - Authentication boundaries: which paths are public, authenticated, or
     require elevated privileges.
   - Data flows: where sensitive data (credentials, PII, tokens) enters,
     is processed, stored, and exits.
2. Identify the **technology stack** and its known vulnerability patterns:
   - Language-specific: Python (pickle, eval, SSTI), JS (prototype pollution,
     ReDoS), Java (deserialization, XXE), Go (integer overflow, goroutine leak).
   - Framework-specific: check for known CVEs in the framework version.
   - Dependency-specific: run `shell_run` with `pip audit`, `npm audit`,
     `cargo audit`, or `govulncheck` as appropriate.
3. Review **security configuration**:
   - TLS settings, CORS policy, CSP headers, cookie flags.
   - Secret management: how are API keys, DB passwords, and tokens stored?
   - Logging: are sensitive values redacted?

### Phase 2 — Automated Scanning

Run available automated tools:

| Tool | Scope | Command |
|---|---|---|
| `pip audit` / `npm audit` | Dependency CVEs | `pip audit --format json` |
| `bandit` (Python) | Source code patterns | `bandit -r src/ -f json` |
| `semgrep` | Multi-language patterns | `semgrep --config auto src/` |
| `trivy fs` | Filesystem + deps | `trivy fs --severity HIGH,CRITICAL .` |
| `gitleaks` | Secrets in history | `gitleaks detect --source .` |

Parse results and de-duplicate.  Automated findings are leads, not conclusions
— each must be validated manually in Phase 3.

### Phase 3 — Manual Analysis (OWASP Top 10 Focus)

Systematically examine code for each OWASP category:

**A01 — Broken Access Control**:
- Are authorization checks enforced at the handler level, not just the router?
- Can a user access another user's resources by changing IDs (IDOR)?
- Are admin endpoints protected by role checks, not just authentication?

**A02 — Cryptographic Failures**:
- Are passwords hashed with bcrypt/scrypt/argon2 (not MD5/SHA1)?
- Are secrets stored in environment variables or vaults (not code/config files)?
- Is data in transit encrypted (TLS 1.2+)?  Data at rest?

**A03 — Injection**:
- SQL: parameterized queries everywhere?  No string concatenation in queries?
- Command: is `subprocess` called with `shell=False`?  Is user input sanitized?
- Template: is user input escaped before rendering in templates?
- XSS: is output encoding applied in all HTML contexts?

**A04 — Insecure Design**:
- Are rate limits enforced on authentication endpoints?
- Is there account lockout or CAPTCHA after repeated failures?
- Are business logic constraints enforced server-side (not just client)?

**A05 — Security Misconfiguration**:
- Are debug modes, default credentials, or verbose error pages exposed?
- Are unnecessary services, ports, or features enabled?
- Are HTTP security headers set (X-Frame-Options, X-Content-Type-Options)?

**A06 — Vulnerable Components**:
- Cross-reference dependency scan results with NVD/OSV databases.
- Check for unmaintained dependencies (no commits in >2 years).

**A07 — Authentication Failures**:
- Are sessions invalidated on logout and password change?
- Are tokens short-lived with proper refresh mechanisms?
- Is MFA supported for sensitive operations?

**A08 — Data Integrity Failures**:
- Are software updates and CI/CD pipelines integrity-verified?
- Is deserialization of untrusted data avoided or validated?

**A09 — Logging and Monitoring**:
- Are authentication events (login, failure, lockout) logged?
- Are logs protected from injection and tampering?
- Is there alerting on anomalous patterns?

**A10 — SSRF**:
- Are outbound requests validated against an allowlist?
- Can user input influence internal URLs or DNS resolution?

### Phase 4 — Severity Assessment

Rate each confirmed finding using CVSS-like scoring:

| Severity | Criteria | Example |
|---|---|---|
| **Critical** | Remote exploitation, no auth required, data breach likely | Unauthenticated SQL injection on public endpoint |
| **High** | Exploitation requires low-privilege auth, significant impact | IDOR allowing access to other users' data |
| **Medium** | Requires specific conditions, moderate impact | XSS in admin panel requiring social engineering |
| **Low** | Theoretical risk, minimal real-world impact | Information disclosure of framework version |
| **Info** | Best-practice deviation, no direct risk | Missing security header on non-sensitive endpoint |

### Phase 5 — Report

Produce a structured security audit report:

```
## Executive Summary
<Overall risk posture: Critical/High/Medium/Low>
<N critical, N high, N medium, N low findings>

## Critical Findings
### [CRIT-001] <Title>
- **Location**: <file:line>
- **Description**: <what the vulnerability is>
- **Attack Scenario**: <step-by-step exploitation>
- **Impact**: <what an attacker gains>
- **Remediation**: <specific code change with example>
- **References**: <CWE/CVE/OWASP link>

## High Findings
### [HIGH-001] ...

## Medium / Low / Info Findings
...

## Positive Security Practices
<Things the project does well — reinforcement matters>

## Recommended Hardening Steps
1. <Prioritized action items>
```

## Error Handling

| Situation | Action |
|---|---|
| Scanning tools not installed | Report which tools are missing; provide install commands; proceed with manual analysis. |
| Codebase too large for full audit | Focus on the highest-risk areas: authentication, input processing, external interfaces. |
| Encrypted or obfuscated code | Report as a limitation; audit only readable portions. |
| Finding severity is ambiguous | Default to the higher severity and note the uncertainty. |

## Limitations

- This skill performs static analysis and code review; it does not execute
  dynamic tests (penetration testing, fuzzing) or interact with running services.
- Automated tool results depend on tool availability in the environment.
- Findings are based on code as read; runtime behavior may differ due to
  configuration, middleware, or infrastructure layers not visible in source.

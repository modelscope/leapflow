---
name: test_writer
description: "Generate unit and integration tests from source code with edge-case coverage"
version: 1.0.0
metadata:
  leapflow:
    category: "development"
    source: "builtin"
    confidence: 1.0
    quality_score: 1.0
  hermes:
    category: "development"
    tags: ["testing", "unit-test", "integration-test", "test-generation", "coverage"]
    requires_tools: ["file_read", "file_write", "shell_run"]
platforms: []
triggers:
  - "write tests"
  - "generate tests"
  - "create test cases"
  - "unit test"
  - "写测试"
  - "生成测试用例"
  - "add test coverage"
  - "integration test"
---

# Test Writer

## Purpose

Generate high-quality test cases — unit, integration, and boundary tests — by
analyzing source code, identifying testable units, determining edge cases, and
producing test code that follows the project's existing conventions.  Tests must
not only pass but must catch real regressions.

## Guiding Principles

1. **Tests document behavior** — A test file is the executable specification of
   what the code does.  Name tests after the behavior they verify, not the
   implementation they exercise.
2. **Arrange-Act-Assert** — Every test has three clear sections: set up state,
   perform the action, verify the outcome.  One action per test.
3. **Edge cases over happy paths** — Happy paths are obvious; value comes from
   testing boundaries, error paths, empty inputs, and concurrent access.
4. **Independence** — Tests must not depend on each other's execution order or
   shared mutable state.  Each test sets up and tears down its own context.
5. **Follow the project** — Use the project's existing test framework, directory
   layout, naming conventions, and fixture patterns.  Do not introduce new
   testing libraries without explicit approval.

## Workflow

### Phase 1 — Analyze the Target Code

1. Read the source file(s) with `file_read`.
2. Identify:
   - **Public API surface**: functions, methods, and classes intended for
     external use.
   - **Input types and constraints**: what each parameter accepts, valid
     ranges, required vs optional.
   - **Output types**: return values, raised exceptions, side effects
     (file writes, network calls, state mutations).
   - **Dependencies**: external services, databases, file system, time,
     randomness — these will need mocking or stubbing.
   - **Existing tests**: check if a test file already exists; understand
     what is already covered.

3. Determine the test framework by examining existing tests:
   - Python: `pytest`, `unittest`
   - JavaScript/TypeScript: `jest`, `vitest`, `mocha`
   - Go: standard `testing` package
   - Other: detect from import patterns or config files.

### Phase 2 — Design Test Cases

For each testable unit, enumerate cases across these categories:

**Normal behavior**:
- Typical valid inputs producing expected outputs.
- Multiple valid input combinations if the function is polymorphic.

**Boundary conditions**:
- Empty inputs (empty string, empty list, zero, None/null).
- Minimum and maximum valid values.
- Single-element collections.
- Boundary of numeric ranges (off-by-one).

**Error paths**:
- Invalid inputs that should raise exceptions or return error codes.
- Missing required parameters.
- Type mismatches (if the language is dynamically typed).

**State transitions** (for stateful code):
- Initial state → action → expected state.
- Invalid state transitions that should be rejected.

**Integration points** (for integration tests):
- Correct interaction with mocked dependencies.
- Behavior when dependencies fail (timeout, error, empty response).

Produce a test plan:
```
Target: calculate_discount(price, tier, coupon_code)
Cases:
  1. Normal: valid price + gold tier → 20% discount
  2. Normal: valid price + no tier → 0% discount
  3. Boundary: price = 0 → discount = 0
  4. Boundary: price = MAX_FLOAT → no overflow
  5. Error: negative price → raises ValueError
  6. Error: unknown tier → raises ValueError
  7. Error: expired coupon → returns original price + warning
  8. Integration: coupon service unreachable → graceful fallback
```

### Phase 3 — Generate Test Code

Write test code following these rules:

1. **File location**: place tests where the project expects them (e.g.,
   `tests/test_<module>.py`, `__tests__/<module>.test.ts`, `<module>_test.go`).
2. **Test naming**: `test_<function>_<scenario>_<expected_outcome>`.
   Examples: `test_calculate_discount_negative_price_raises_value_error`,
   `test_parse_config_empty_file_returns_defaults`.
3. **Fixtures and setup**: extract common setup into fixtures (`@pytest.fixture`,
   `beforeEach`, `TestMain`).  Keep fixtures close to the tests that use them.
4. **Mocking strategy**:
   - Mock at the boundary: external services, I/O, time, randomness.
   - Do not mock the unit under test or its core logic.
   - Use dependency injection where the code supports it; patch as last resort.
5. **Assertions**:
   - Assert specific values, not just truthiness.
   - For exceptions: assert both the exception type and message content.
   - For collections: assert length and key elements, not just non-empty.
6. **Readability**: each test should be understandable in isolation without
   reading other tests.  Inline small data; use descriptive variable names.

### Phase 4 — Verify Tests

After writing tests:

1. Run the test suite with `shell_run`:
   - Python: `python -m pytest <test_file> -v`
   - JS/TS: `npx jest <test_file>` or `npx vitest run <test_file>`
   - Go: `go test -v -run <TestName> ./<package>`
2. Verify all tests **pass**.  If any fail:
   - Read the failure message carefully.
   - Distinguish between a bug in the test (wrong expectation) and a bug
     in the source code (genuine regression).
   - Fix test bugs; report source code bugs to the user.
3. Check that tests **fail when they should**: temporarily break the source
   logic and confirm the test catches it (mutation testing principle).
4. Review test output for:
   - Flaky behavior (tests that pass/fail non-deterministically).
   - Slow tests (>1 second for unit tests indicates I/O leaking in).
   - Missing coverage for the identified edge cases.

### Phase 5 — Report

Summarize the test generation:

```
## Test Summary
- File: tests/test_discount.py
- Tests added: 8 (5 unit, 2 boundary, 1 integration)
- All passing: Yes
- Mocks used: coupon_service (httpx response stub)
- Coverage delta: +12% for discount.py (estimated)

## Notable Edge Cases Covered
1. Negative price rejection
2. Coupon service timeout fallback
3. Floating-point precision at MAX_FLOAT

## Gaps / Recommendations
- Concurrent discount calculations not tested (requires async fixtures)
- Property-based testing recommended for numeric inputs (hypothesis/fast-check)
```

## Error Handling

| Situation | Action |
|---|---|
| No existing test framework detected | Ask the user which framework to use; default to the language's standard (`pytest`, `jest`, `go test`). |
| Source code has no clear testable units (monolithic function) | Suggest refactoring; write tests for observable inputs/outputs of the monolith. |
| Tests pass but are tautological (assert True) | Flag as a quality issue; rewrite with meaningful assertions. |
| Cannot run tests (missing dependencies) | Generate the test code and provide the exact install/run commands the user needs. |

## Limitations

- Test generation requires access to the source code and an understanding of
  the project's test infrastructure.
- Integration tests that require live services (databases, APIs) will use mocks;
  the user must configure real service connections for end-to-end testing.
- Code coverage measurement requires project-specific tooling configuration
  that this skill does not modify.

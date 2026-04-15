# Risk Acceptance (Temporary)

- Project: AetherForecast
- Owner: Nguyễn
- Date: 2026-04-15
- Validity: 7 days (until 2026-04-22)

## Scope of Temporary Waiver

1. mypy strict errors
- Current status: 54 errors (16 files)
- Decision: Temporarily waived for this release window

2. bandit findings
- Current status: 10 findings (6 medium, 4 low)
- Decision: Temporarily waived for this release window

## Rationale

- These are code quality and static-analysis warnings.
- They do not indicate a confirmed runtime outage in the current production hotfix state.
- Production backend has been stabilized after the ARM64 hotfix rollout and functional verification.

## Runtime Verification Basis

- Public health check is OK.
- Predict endpoint is functioning with expected auth behavior.
- Chart data and WebSocket stream checks are stable.

## Expiration and Follow-up

- This waiver expires in 7 days.
- After expiration, issues will be fixed incrementally and waiver will be removed.

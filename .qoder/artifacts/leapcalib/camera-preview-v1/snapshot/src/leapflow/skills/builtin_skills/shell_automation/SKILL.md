---
name: shell_automation
description: "Natural language to shell commands with safety review and step-by-step execution"
version: 1.0.0
metadata:
  leapflow:
    category: "automation"
    source: "builtin"
    confidence: 1.0
    quality_score: 1.0
  hermes:
    category: "automation"
    tags: ["shell", "terminal", "automation", "commands", "scripting"]
    requires_tools: ["shell_run"]
platforms: []
triggers:
  - "run a shell command"
  - "automate this task"
  - "write a script to"
  - "shell automation"
  - "batch process"
  - "execute commands"
  - "terminal commands for"
  - "help me with the command line"
---

# Shell Automation

## Purpose

Translate a user's natural-language goal into a safe, step-by-step shell
command plan.  Each command is reviewed for safety before execution, and
intermediate results are verified before proceeding to the next step.  The
skill prioritizes safety and predictability over speed.

## Guiding Principles

1. **Safety first** — Every command is classified by risk level before
   execution.  Destructive or irreversible operations require explicit
   acknowledgement of consequences.
2. **Incremental execution** — Run one logical step at a time.  Verify the
   output before proceeding.  Never chain destructive commands with `&&`.
3. **Least privilege** — Use the minimum permissions necessary.  Avoid `sudo`
   unless the user explicitly requests it and the task genuinely requires it.
4. **Idempotent preference** — Prefer commands that are safe to re-run
   (e.g., `mkdir -p` over `mkdir`, `cp` with backup over `mv`).
5. **Transparency** — Show every command before running it.  Explain what it
   does and what side effects it has.  No hidden operations.

## Workflow

### Phase 1 — Understand the Goal

Parse the user's request into:

- **Objective**: What end state does the user want?
- **Scope**: Which files, directories, or services are involved?
- **Constraints**: OS, shell flavor (bash/zsh/fish), available tools,
  environment (local dev, CI, production server).
- **Risk tolerance**: Is the user experimenting or operating on production
  data?

If anything is ambiguous, ask a clarifying question before planning.

### Phase 2 — Plan the Command Sequence

Design an ordered list of commands.  For each command:

1. **Write the command** — use proper flags and quoting.
2. **Explain it** — one sentence on what it does.
3. **Classify the risk**:

| Risk Level | Criteria | Examples |
|---|---|---|
| **Safe** | Read-only, no side effects | `ls`, `cat`, `grep`, `find`, `wc`, `df`, `ps` |
| **Low** | Creates new files/dirs, non-destructive writes | `mkdir -p`, `touch`, `tee`, `cp` (no overwrite) |
| **Medium** | Modifies existing files, installs packages, changes config | `sed -i`, `pip install`, `chmod`, `git commit` |
| **High** | Deletes data, stops services, modifies system state | `rm`, `kill`, `systemctl stop`, `drop table` |
| **Critical** | Irreversible, wide blast radius | `rm -rf /`, `dd`, `mkfs`, `git push --force` |

4. **Add a verification step** after each non-trivial command — a read-only
   command that confirms the expected outcome (e.g., `ls` after `mv`,
   `cat` after `sed`).

Present the full plan before executing anything.

### Phase 3 — Safety Review

Before execution, perform a checklist:

- [ ] No command uses `sudo` unless explicitly justified.
- [ ] No `rm` without a narrow, explicit target (never `rm -rf` with a
      variable or glob that could expand dangerously).
- [ ] All file paths are absolute or explicitly scoped to the working
      directory.
- [ ] No secrets, passwords, or tokens appear in plaintext in any command.
- [ ] Pipes and redirects do not silently overwrite important files (prefer
      `>>` over `>` when appending; use `tee` for visibility).
- [ ] The plan has a **rollback path** for any medium-or-higher risk step
      (e.g., "if this fails, run X to restore state").

If a Critical-risk command is part of the plan, add a prominent warning block:

```
⚠️  CRITICAL: The following command is irreversible.
    Command: <cmd>
    Effect:  <what it destroys/changes>
    Verify:  <how to confirm this is correct before running>
```

### Phase 4 — Step-by-Step Execution

Execute the plan one command at a time:

1. **Show** the command and its explanation.
2. **Run** it via `shell_run`.
3. **Check** the exit code and output:
   - Exit 0 + expected output → proceed to the next step.
   - Non-zero exit or unexpected output → stop, diagnose, and report.
4. **Run the verification step** if one was planned.
5. **Log** a brief result: "✓ Created directory `/tmp/backup`" or
   "✗ Failed: permission denied on `/etc/hosts`".

Do NOT proceed past a failed step unless:
- The failure is explicitly expected (e.g., `grep` returning 1 for no match).
- The user confirms they want to skip and continue.

### Phase 5 — Summary

After all steps complete (or after a halt):

```
## Execution Summary
- Steps completed: N / M
- Status: <all succeeded / stopped at step K>

## Commands Executed
1. `<cmd>` — ✓ <result>
2. `<cmd>` — ✗ <error>

## Next Steps
- <Anything the user should do manually or verify>
```

## Error Handling

| Situation | Action |
|---|---|
| Command not found | Check if the tool is installed (`which <cmd>`); suggest installation if missing. |
| Permission denied | Do NOT auto-escalate to `sudo`.  Report the error and let the user decide. |
| Command hangs (timeout) | Report the timeout; suggest running with `timeout <N>s` wrapper or checking for interactive prompts. |
| Ambiguous user request | Ask for clarification rather than guessing — a wrong guess with `rm` is unrecoverable. |
| User requests a dangerous pattern (`rm -rf *`, `chmod 777`) | Explain the risk clearly; suggest a safer alternative; proceed only if the user explicitly confirms after understanding the consequences. |

## Forbidden Patterns

The following patterns must NEVER be executed without explicit, informed user
confirmation and a clear justification:

- `rm -rf /` or `rm -rf ~` or any `rm` targeting a root-level directory
- `:(){ :|:& };:` or any fork bomb variant
- `dd` writing to a block device
- `chmod -R 777` on system directories
- `> /dev/sda` or equivalent destructive redirects
- Any command that downloads and pipes directly to `sh`/`bash` without
  the user reviewing the script first (e.g., `curl ... | sh`)

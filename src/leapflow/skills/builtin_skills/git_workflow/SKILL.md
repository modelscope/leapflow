---
name: git_workflow
description: "Git operation orchestration: branching, commits, conflicts, and PR workflow"
version: 1.0.0
metadata:
  leapflow:
    category: "development"
    source: "builtin"
    confidence: 1.0
    quality_score: 1.0
  hermes:
    category: "development"
    tags: ["git", "version-control", "branching", "commits", "pull-request", "conflict-resolution"]
    requires_tools: ["shell_run"]
platforms: []
triggers:
  - "git workflow"
  - "branch management"
  - "commit convention"
  - "resolve conflict"
  - "git操作"
  - "分支管理"
  - "create pull request"
  - "git best practices"
---

# Git Workflow

## Purpose

Orchestrate Git operations with disciplined branching strategy, consistent commit
conventions, systematic conflict resolution, and streamlined PR workflow.  This
skill does not simply run git commands — it enforces a methodology that keeps the
repository history clean, bisectable, and reviewable.

## Guiding Principles

1. **History is documentation** — Every commit message is a permanent record read
   by future developers.  Treat it with the same care as code comments.
2. **Atomic commits** — Each commit captures exactly one logical change.  A commit
   that mixes a refactor with a feature is two commits.
3. **Branch hygiene** — Short-lived branches merged frequently beat long-lived
   branches merged painfully.  Delete merged branches immediately.
4. **Safety first** — Never rewrite published history.  Use `--force-with-lease`
   only on personal branches after explicit confirmation.
5. **Verify before sharing** — Every branch must build and pass tests locally
   before pushing.

## Workflow

### Phase 1 — Assess the Situation

Before running any git command, understand the current state:

1. Run `git status` and `git log --oneline -10` to see working tree state and
   recent history.
2. Identify the **branching model** in use:
   - **Trunk-based**: short-lived feature branches off `main`, merged via PR.
   - **GitFlow**: `develop` as integration branch, `release/*` and `hotfix/*`
     branches for releases and urgent fixes.
   - **Unknown**: inspect branch names and merge patterns to infer the model.
3. Check for uncommitted changes, stashed work, or in-progress rebases.
4. Confirm the **remote** topology (`git remote -v`).

### Phase 2 — Branch Management

Create or navigate branches following the project's model:

- **Naming convention**: `<type>/<ticket>-<short-description>`
  (e.g. `feat/PROJ-42-add-auth`, `fix/PROJ-99-null-pointer`).
- **Base branch**: always branch from the latest upstream target
  (`git fetch origin && git checkout -b <branch> origin/main`).
- **Rebase vs merge**: prefer `git rebase` to keep a linear history on feature
  branches; use `git merge --no-ff` when recording an explicit merge point.
- **Cleanup**: after merge, delete the local and remote branch
  (`git branch -d <branch> && git push origin --delete <branch>`).

### Phase 3 — Commit Conventions

Apply Conventional Commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `ci`, `chore`,
`style`, `build`.

Rules:
- **Subject**: imperative mood, ≤72 characters, no trailing period.
- **Body**: wrap at 72 characters.  Explain *why*, not *what* (the diff shows
  what).  Reference issue/ticket IDs.
- **Footer**: `BREAKING CHANGE:` for incompatible changes; `Refs:` or
  `Closes:` for issue links.
- **Scope**: optional; matches the module or area affected.

When staging changes:
1. Use `git add -p` for interactive staging to keep commits atomic.
2. Review the staged diff (`git diff --cached`) before committing.
3. Run `git commit` (not `git commit -m`) for multi-line messages when a body
   is warranted.

### Phase 4 — Conflict Resolution

When merge or rebase conflicts arise:

1. **Identify scope**: run `git diff --name-only --diff-filter=U` to list
   conflicting files.
2. **Understand both sides**: for each conflict marker, read the surrounding
   context to understand the intent of both changes.
3. **Resolution strategy**:
   - **Ours-then-theirs**: when both changes are needed but ours should come
     first (common in additive changes).
   - **Theirs-wins**: when upstream refactored and our branch should adopt.
   - **Manual merge**: when changes overlap semantically and require a new
     combined implementation.
4. After resolving each file, run `git add <file>`.
5. Verify the resolution: run tests or at minimum a build check.
6. Complete with `git rebase --continue` or `git merge --continue`.

Never blindly accept `--ours` or `--theirs` on the entire repository.

### Phase 5 — PR Workflow

Prepare and manage pull requests:

1. **Pre-push checklist**:
   - All tests pass locally.
   - Linter/formatter has been run.
   - Commit history is clean (squash fixups with `git rebase -i`).
   - Branch is rebased on latest target.
2. **PR description template**:
   ```
   ## What
   <concise summary of the change>

   ## Why
   <motivation, link to issue/ticket>

   ## How
   <implementation approach, key decisions>

   ## Testing
   <what was tested and how>
   ```
3. **Review cycle**: address feedback with fixup commits; squash before final
   merge to keep the target branch clean.
4. **Merge method**: prefer "squash and merge" for single-purpose PRs; use
   "merge commit" when preserving intermediate history matters.

## Error Handling

| Situation | Action |
|---|---|
| Detached HEAD state | Identify the intended branch; `git checkout <branch>` or create a new branch from current HEAD. |
| Accidental commit on wrong branch | `git cherry-pick` the commit to the correct branch, then `git reset` on the wrong one. |
| Force-push request | Refuse on shared branches. On personal branches, use `--force-with-lease` and confirm with the user first. |
| Large binary accidentally committed | Use `git filter-branch` or `git-filter-repo` to remove; advise `.gitignore` and LFS for future binaries. |
| Merge conflict during rebase | Resolve file-by-file as described in Phase 4; abort with `git rebase --abort` if the user requests. |

## Limitations

- This skill executes git commands via `shell_run`.  Destructive operations
  (force push, history rewrite, branch deletion) always require explicit user
  confirmation before execution.
- Repository hosting platform APIs (GitHub, GitLab) are not directly accessible;
  PR creation guidance is command-line oriented.

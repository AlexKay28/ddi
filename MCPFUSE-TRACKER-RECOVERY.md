# mcpfuse Tracker recovery (2026-09-08)

## Symptom and cause

Tracker was enabled but disconnected; other mcpfuse backends worked. Repeated
backend restarts returned only `Failed to connect to tracker`.

Tracker stderr under `browser/infra/projects/ai/mcpfuse/.logs/` exposed:

```text
getcwd() failed: Transport endpoint is not connected
FileNotFoundError: [Errno 2] No such file or directory
```

The long-running mcpfuse supervisor retained a stale Arcadia cwd after a mount
disruption. New stdio children inherited it; `ya` failed while resolving cwd,
before Tracker initialization/authentication. Running `ya --help` from the
supervisor's `/proc/<pid>/cwd` reproduced it, while a stable directory worked.
This was a local launcher failure, not evidence of a Tracker service outage.

## Applied fix

Active config: `browser/infra/projects/ai/mcpfuse/mcpfuse.yaml`, selected by
`~/.config/opencode/opencode.jsonc`.

Added `tracker-local.yaml` as the LAST include after `harness/mcpfuse.yaml`.
The local override starts `/bin/sh`, changes to `$HOME`, and uses `exec "$@"`
to run the unchanged `ya tool mcpfuse bootstrap mcp connect tracker_mcp` command.
No credentials, auth scopes or shared harness definitions were changed.

A root-level override placed before include processing did not take effect:
reload reported Tracker unchanged. A separate trailing include did take effect.
The backend schema inspected did not expose a cwd setting; adding an unsupported
`cwd:` field would not fix inheritance.

## Verification

`mcpfuse_ctl__reload_config` reported only `tracker` restarted, no errors, and
all other backends unchanged. `tracker__GetIssue(VIBEQUEUE-58)` then returned
its summary and status successfully. This verifies an authenticated read,
not merely successful process startup.

## Future recovery and rollback

Check ctl status, then the affected backend's stderr before retrying repeatedly.
If getcwd fails, verify the inherited cwd and use a stable directory before
launching `ya`. The current override protects Tracker only; other newly restarted
backends could encounter the same stale cwd. A future supervisor-wide stable cwd
would require a separately managed supervisor restart.

The absolute ya path in this local override is machine-specific. Do not publish
it as a shared portable default. Remove the trailing include and local override
to roll back, then reload; without fixing the supervisor cwd, rollback can restore
the startup failure. No commits or pushes were made.

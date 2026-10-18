# Chaos persist runbook (homelab)

Run only after unit tests are green. Destructive to the live Ray cluster.

## Prep

- Meta-supervisor CT up (`ray-hive-supervisor` on home-media).
- Markets (or a test project) registered; DurableJobQueue path on durable storage.
- Note ergos VM IDs on Aitherios / Sterianos / Phidios (see Homelab k3s notes).

## Steps

1. **Baseline** — `CronBoard.list_entries`, queue counts, sticky models list.
2. **Kill one worker VM** — `qm stop <vmid>` on a CPU worker. Confirm stale running jobs requeue; gates recreate; board still readable from meta CT.
3. **Power off all Ray VMs** — stop head + workers. Meta + Markets CTs stay up; queue/board SQLite intact; projects go `suspect` then recover on reconnect.
4. **Start VMs** — wait for k3s/Ray ready. Meta heal; project supervisors re-register; drain pending; smoke BatchFanout + one gated fetch.

## Automate

`pytest -m chaos tests/live/test_chaos_persist.py` (skipped unless `HIVE_CHAOS=1` and SSH/Proxmox creds available).

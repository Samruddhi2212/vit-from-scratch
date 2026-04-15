# Security notes for contributors and reviewers

## Credentials and SSH

- **Never commit** private keys, PEM files, or outputs from `ssh-copy-id` (temporary key material sometimes appears as files named like `ssh-copy-id <user>@<host>` in the project directory).
- Use the system **SSH agent** (`ssh-add`) and **`~/.ssh/config`** for Explorer or other hosts; keep secrets outside the repository.
- This repository’s **`.gitignore`** ignores common accidental patterns (`*ssh-copy-id*`, `*.pem`, etc.); ignored files must still be **deleted locally** if they were created by mistake.

## Dependency supply chain

- Install pinned dependencies from **`requirements.txt`** or **`pyproject.toml`** via trusted indexes (`pip` defaults). For reproducible course submissions, record the exact environment (Python version, `pip freeze` if required by the course).

## Large outputs

- Training checkpoints (`*.pth`, `*.pt`) and datasets are **gitignored** by default; do not force-add large binaries unless the course explicitly requires it.

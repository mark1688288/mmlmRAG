# Security Policy

mmlmRAG is a local command-line tool. Version 0.1.x on the `main` branch is the supported line. There is no older release line.

## Reporting a vulnerability

Report an unfixed vulnerability through GitHub private vulnerability reporting:

[https://github.com/mark1688288/mmlmRAG/security/advisories/new](https://github.com/mark1688288/mmlmRAG/security/advisories/new)

That form stays off the public issue list. Use it once this repository is public and private vulnerability reporting is turned on (Settings, Code security, Private vulnerability reporting).

Include:

- The commit or version you ran, plus the macOS and Python versions.
- What an attacker gains: code execution, a write outside `data/`, a read of another file, document text leaving the machine, or a bypass of a control described in the README.
- Steps to reproduce. A minimal document or config is welcome. Skip live secrets and anyone else's files.
- Whether you want to be credited, and the name to use.

## Scope

A report is in scope when a document, a `rag.toml`, or a command-line argument can make this program:

- Run code or start a process other than `/usr/bin/vm_stat` and `/usr/sbin/sysctl -n hw.pagesize`.
- Write outside the data directory, or follow a symbolic link the README says is rejected.
- Send document text, page images, embeddings, or prompts to a host other than loopback Ollama.
- Turn Chroma telemetry on, or ignore the loopback, redirect, and proxy rules.
- Expose the corpus to another local user beyond the `0700` directory and `0600` file modes the program sets on files it creates.
- Bypass the size, page, image-count, or pixel limits while still parsing the input.

A defect that lives only in Ollama, a model, PyMuPDF, Pillow, pillow-heif, Chroma, or another dependency belongs with that project. Report it here as well when this program's defaults let a document or a config file trigger it.

These are outside this policy:

- A person who already has the operator's macOS account. That account can already read `data/`.
- Ollama, model weights, and macOS themselves.
- A scanner finding with no effect on the behavior above.
- Availability bugs that the documented limits already stop.

## Response

This is a single-maintainer project. There is no bug bounty.

- You get an acknowledgement within 7 days.
- A fix, or a written mitigation, is the target within 90 days of the acknowledgement.
- A report that is still open after 30 days gets a status note on the private advisory.
- Please leave the report private until a fix is on `main`, or until 90 days after the acknowledgement, whichever comes first.

A confirmed issue is published as a GitHub Security Advisory when the fix lands. The advisory names you unless you ask to stay anonymous. Good-faith research that follows this policy, on a machine and corpus you are allowed to use, is welcome.

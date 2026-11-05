## Usage

```bash
rag ingest ./notes.pdf
rag ask "What does the note say about the budget?"
rag status
rag config show
```

| Command | Purpose |
| --- | --- |
| `rag ingest FILE` | Parse, OCR, caption, and embed one file |
| `rag ask QUESTION` | Answer from the active collection |
| `rag reembed` | Rebuild vectors for another embedding model or width |
| `rag use --collection NAME` | Switch the active collection |
| `rag delete --path PATH` or `--doc-id ID` | Remove one document |
| `rag status` | Show the local corpus and whether Ollama answers |
| `rag models pull` | Pull the configured models |
| `rag models unload` | Unload models Ollama has loaded |
| `rag config show` | Print the resolved configuration |

`--config` selects a TOML file. The default search walks upward from the working directory for `rag.toml`. `--data-dir` overrides `paths.data_dir`. `--verbose` turns on debug logs. `--lock-timeout` is how many seconds to wait for the Ollama lock. `0` means do not wait.

`rag ingest` accepts `--force`, `--resume`, `--no-ocr`, `--no-vision`, `--no-embed`, `--strict`, `--max-pages`, and `--max-bytes`. `rag ask` accepts `--k` (1 to 8), `--model`, `--no-fallback`, `--show-chunks`, `--json`, `--verify`, and `--no-verify`.

`--verify` checks this draft against the excerpts with the same local model. `--no-verify` turns the check off for this question. The default is off. `[retrieve] verify` in `rag.toml` sets the default, and either flag overrides the file. The model must answer exactly `PASS` or exactly `FAIL`. `PASS` keeps the draft. `FAIL` prints `The documents do not contain the answer.` The check does not retrieve again.

`--lang` chooses the interface language, `en` or `zh`. `en` is English and is the default. `zh` is Traditional Chinese. The flag overrides `[ui] language` in `rag.toml`. When both are absent, the interface is English. ragcli does not read `LANG`, `LC_ALL`, or the macOS preferred language. The value must be exactly `en` or `zh`. `[ocr] languages` chooses Vision recognition and does not follow the interface language. The usage line, section headings, the help-action sentence, and argparse error text follow `--lang` and `ui.language`.

```toml
[ui]
language = "zh"
```

## Security

Document text, page images, and embeddings stay on this Mac. `rag.toml` has no secret fields.

- **Loopback only.** The Ollama URL must be `http` or `https` on `127.0.0.1`, `localhost`, or `::1`. Usernames, passwords, paths, and query strings are rejected. The HTTP client stops when it receives a redirect, and it ignores proxy environment variables. Model pulls use Ollama's HTTP API. ragcli does not start a shell to run the `ollama` binary.
- **Weights versus documents.** `rag models pull` asks the local daemon to download model weights. That transfer is Ollama's. Ingest and ask send prompts only to that loopback daemon.
- **Telemetry.** The Chroma client is opened with anonymous telemetry off. ragcli refuses to open Chroma when that switch is on.
- **Files.** A data directory ragcli creates is mode `0700`. The SQLite database and its sidecar files are mode `0600`. A directory that already exists keeps its mode. The data directory and an ingest path must be real files, not symbolic links. `data/` is gitignored.
- **Logs.** Debug logging omits document text.
- **Untrusted input.** PDFs are parsed with PyMuPDF, and images with Pillow. Treat both as parsers of untrusted bytes. Defaults stop a file over 200 MiB, a document over 500 pages, a file with more than 300 images, and an image over 20 million pixels. Password-protected PDFs are rejected without a password prompt. An image whose header exceeds the pixel cap is not decoded.
- **Processes.** Before loading a generator, ragcli reads `/usr/bin/vm_stat` and, when it needs the page size, `/usr/sbin/sysctl -n hw.pagesize`. Those two argument lists are the only subprocesses.
- **License.** mmlmRAG is AGPL-3.0-only. Redistributing the program, or serving a modified version over a network, requires the corresponding source under that license. See [License](#license).
- **Reporting.** Report an unfixed vulnerability through [SECURITY.md](SECURITY.md). That policy uses GitHub private vulnerability reporting, so the report stays off the public issue list.
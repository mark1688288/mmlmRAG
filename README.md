<div align="center">
<img src="imgs/mmlmRAG_logo.jpeg" alt="mmlmRAG" width="720">

mmlmRAG (`ragcli`) is a local mixed-language retrieval-augmented generation command-line tool. The `rag` command reads documents on the local device and uses local models through [Ollama](https://ollama.com) to embed them and answer questions from that corpus. Text and vectors stay in a SQLite database and a Chroma index under `data/`.

[![Version](https://img.shields.io/badge/version-0.1.2-2F6F8F)](https://github.com/mark1688288/mmlmRAG)
[![Python](https://img.shields.io/badge/python-3.13%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![macOS](https://img.shields.io/badge/macOS-000000?logo=apple&logoColor=white)](#prerequisites)
[![Ollama](https://img.shields.io/badge/Ollama-local-000000?logo=ollama&logoColor=white)](https://ollama.com)
[![License](https://img.shields.io/badge/license-AGPL--3.0--only-1F6FEB)](LICENSE)

[中文](README.zh-Hant.md)

</div>

## Prerequisites

- macOS. OCR goes through the Vision framework. Available-memory checks use `vm_stat`.
- Python 3.13 or newer.
- Ollama listening on a loopback address. The default in `rag.toml` is `http://127.0.0.1:11434`. Allowed hosts are `127.0.0.1`, `localhost`, and `::1`.
- Free memory for the generator you load. The default `qwen3.5:9b` needs 9 GiB available. The fallback `qwen3.5:4b` needs 6 GiB.
- At least 2 GiB of free disk before ingest, ask, or reembed.

`rag models pull` downloads these models through the local Ollama daemon:

| Role | Model |
| --- | --- |
| Embeddings | `qwen3-embedding:0.6b` (1024 dimensions) |
| Answers | `qwen3.5:9b` |
| Fallback when memory is short | `qwen3.5:4b` |

## Dependencies

Runtime dependencies declared in `pyproject.toml`:

| Package | Requirement | Role |
| --- | --- | --- |
| `chromadb` | `==1.5.9` | Persistent vector index |
| `httpx` | unpinned | Loopback HTTP client for Ollama |
| `pymupdf` | unpinned | PDF parser |
| `python-docx` | unpinned | Word documents |
| `python-pptx` | unpinned | PowerPoint documents |
| `ocrmac` | unpinned | macOS Vision OCR |
| `pillow` | unpinned | Image decoding |
| `pillow-heif` | unpinned | HEIC and HEIF |
| `charset-normalizer` | unpinned | Text encoding detection |

The development extra is `pytest`. `chromadb` stays at 1.5.9. Changing that pin is a deliberate upgrade.

SQLite comes from the Python standard library. It is in the public domain.

Accepted ingest files are `.pdf`, `.docx`, `.pptx`, `.txt`, `.md`, `.png`, `.jpg`, `.jpeg`, `.webp`, `.tif`, `.tiff`, `.gif`, `.bmp`, `.heic`, and `.heif`. Each `rag ingest` call takes one regular file.

## Install

### Homebrew

```bash
brew install mark1688288/mmlmrag/mmlmrag
```

This installs the `rag` command from the [homebrew-mmlmrag](https://github.com/mark1688288/homebrew-mmlmrag) tap. The formula depends on macOS and `python@3.13`, and it pins the `v0.1.1` tag archive. Its Python resources are the Apple Silicon wheels for that tag. The first `rag` command creates `~/.mmlmrag`, including a private environment and, when absent, `rag.toml`. A `rag.toml` in the working directory or a parent is used instead. Otherwise `rag` uses `~/.mmlmrag/rag.toml`.

Install Ollama, then pull the default models.

```bash
brew install ollama
brew services start ollama
rag models pull
```

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

`rag ingest` accepts `--force`, `--resume`, `--no-ocr`, `--no-vision`, `--no-embed`, `--strict`, `--max-pages`, and `--max-bytes`. `rag ask` accepts `--k` (1 to 8), `--model`, `--no-fallback`, `--show-chunks`, and `--json`.

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

## Demo
### Pull models

<img src="imgs/mmlmRAG_models-pull.jpeg" alt="rag models pull" width="720">

### Ingest

<img src="imgs/mmlmRAG_ingest.jpeg" alt="rag ingest" width="720">

### Ask

<img src="imgs/mmlmRAG_ask.jpeg" alt="rag ask" width="720">

### Live demo
#### PDF to text
source: empf.org.hk - Mandatory Provident Fund Schemes (Specification of Dates for Purposes of Section 19M(2)(a)) (Amendment) Notice 2026 

<img src="imgs/mmlmRAG_LiveDemo.jpeg" alt="Ingest a PDF, then ask" width="720">

source: hb.gov.hk/eng/policy/housing/policy/light_public_housing/pamphlet.html - "Light Public Housing — Full Rollout"

<img src="imgs/mmlmRAG_LiveDemo-ocr-pdf.jpeg" alt="Ingest a PDF with OCR, then ask" width="720">


#### OCR Image to text

source: [samples-live/demo.gif](samples-live/demo.gif)

<img src="imgs/mmlmRAG_LiveDemo-ocr-ingest.jpeg" alt="Ingest a GIF" width="720">
<img src="imgs/mmlmRAG_LiveDemo-ocr-ask.jpeg" alt="Ask a GIF" width="720">

#### OCR-Vision Image find objects

source: [samples-live/demo.jpeg](samples-live/demo.jpeg)

<img src="imgs/mmlmRAG_LiveDemo-ocr-find-objects.jpeg" alt="Find objects in images 1" width="720">

<img src="imgs/mmlmRAG_LiveDemo-ocr-vision-find-obj.jpg" alt="Find objects in images 2" width="720">

#### Query corpus
<img src="imgs/mmlmRAG_LiveDemo-corpus-query.jpeg" alt="Query the corpus" width="720">

## License

Copyright (c) 2026 mark1688288.

mmlmRAG is free software. You can redistribute it and modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, version 3 only. The full text is in [LICENSE](LICENSE).

Anyone who distributes this program, or who runs a modified version for users over a network, must offer the corresponding source under AGPL-3.0-only. Documents ingested into `data/` are not part of the program.

PyMuPDF is available under those same AGPL-3.0 terms, or under an Artifex commercial license. The pillow-heif binary wheel bundles x265 under GPL version 2 or any later version, so a combined distribution stays under AGPL-3.0-only. See [Credits](#credits).

## Credits

These are the direct runtime libraries whose licenses require a copyright or attribution notice. The notes at the bottom of this file are that notice. Each installed package also carries its full license text.

- [Chroma](https://github.com/chroma-core/chroma)[^chroma] stores vectors.
- [HTTPX](https://github.com/encode/httpx)[^httpx] talks to Ollama.
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF)[^pymupdf] reads PDFs.
- [python-docx](https://github.com/python-openxml/python-docx)[^docx] reads Word documents.
- [python-pptx](https://github.com/scanny/python-pptx)[^pptx] reads PowerPoint documents.
- [ocrmac](https://github.com/straussmaximilian/ocrmac)[^ocrmac] calls macOS Vision for OCR.
- [Pillow](https://github.com/python-pillow/Pillow)[^pillow] decodes images.
- [pillow-heif](https://github.com/bigcat88/pillow_heif)[^heif] opens HEIC and HEIF.
- [Charset Normalizer](https://github.com/jawah/charset_normalizer)[^charset] detects text encodings.

[^chroma]: Chroma (`chromadb` 1.5.9) is licensed under the Apache License, Version 2.0. http://www.apache.org/licenses/LICENSE-2.0

[^httpx]: HTTPX is Copyright © 2019 Encode OSS Ltd. All rights reserved. It is licensed under the BSD 3-Clause License. Redistributions must retain that copyright notice, the license conditions, and the disclaimer. The name of the copyright holder and the names of its contributors may not be used to endorse or promote products derived from this software without specific prior written permission. The full text is `LICENSE.md` in the installed `httpx` distribution. https://github.com/encode/httpx/blob/master/LICENSE.md

[^pymupdf]: PyMuPDF is developed by Artifex and is dual-licensed under the GNU Affero General Public License v3.0 and the Artifex Commercial License. This project follows the AGPL path and is AGPL-3.0-only. The installed distribution carries the PyMuPDF notice in `COPYING`. https://pymupdf.readthedocs.io/

[^docx]: python-docx is Copyright (c) 2013 Steve Canny, https://github.com/scanny. It is licensed under the MIT License. That copyright notice and the MIT permission notice must stay with copies or substantial portions of the software. The full text is `LICENSE` in the installed `python-docx` distribution.

[^pptx]: python-pptx is Copyright (c) 2013 Steve Canny, https://github.com/scanny. It is licensed under the MIT License. That copyright notice and the MIT permission notice must stay with copies or substantial portions of the software. The full text is `LICENSE` in the installed `python-pptx` distribution.

[^ocrmac]: ocrmac is Copyright (c) 2022 Maximilian Strauss. It is licensed under the MIT License. That copyright notice and the MIT permission notice must stay with copies or substantial portions of the software. The full text is `LICENSE` in the installed `ocrmac` distribution. https://github.com/straussmaximilian/ocrmac

[^pillow]: Pillow is Copyright © 2010 by Jeffrey 'Alex' Clark and contributors. The Python Imaging Library (PIL) is Copyright © 1997-2011 by Secret Labs AB and Copyright © 1995-2011 by Fredrik Lundh and contributors. Pillow is licensed under the MIT-CMU License: permission to use, copy, modify, and distribute this software and its documentation for any purpose and without fee is granted, provided that the above copyright notice appears in all copies, and that both that copyright notice and this permission notice appear in supporting documentation, and that the name of Secret Labs AB or the author not be used in advertising or publicity pertaining to distribution of the software without specific, written prior permission. Pillow wheels also include third-party image codecs. Their notices are in the `LICENSE` file of the installed `pillow` distribution. https://github.com/python-pillow/Pillow

[^heif]: pillow-heif is Copyright (c) 2021-2023, Pillow-Heif contributors. All rights reserved. The source is licensed under the BSD 3-Clause License. Redistributions must retain that copyright notice, the license conditions, and the disclaimer. The name of Pillow-Heif and the names of its contributors may not be used to endorse or promote products derived from this software without specific prior written permission. The full text is `LICENSE.txt` in the installed distribution. Binary wheels are GPLv2 because of the libraries they bundle: libheif (LGPLv3, https://github.com/strukturag/libheif), libde265 (LGPLv3, https://github.com/strukturag/libde265), and x265 (GPLv2, https://github.com/Multicorewareinc/x265). Those notices are in `LICENSES_bundled.txt` in the installed `pillow-heif` distribution. https://github.com/bigcat88/pillow_heif

[^charset]: Charset Normalizer is Copyright (c) 2025 TAHRI Ahmed R. It is licensed under the MIT License. That copyright notice and the MIT permission notice must stay with copies or substantial portions of the software. The full text is `LICENSE` in the installed `charset-normalizer` distribution. https://github.com/jawah/charset_normalizer

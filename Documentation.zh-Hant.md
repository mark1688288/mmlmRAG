## 用法

```bash
rag ingest ./notes.pdf
rag ask "What does the note say about the budget?"
rag status
rag config show
```

| 指令 | 用途 |
| --- | --- |
| `rag ingest FILE` | 解析、OCR、描述圖像，並嵌入一個檔案 |
| `rag ask QUESTION` | 依作用中集合回答 |
| `rag reembed` | 為另一個嵌入模型或寬度重建向量 |
| `rag use --collection NAME` | 切換作用中集合 |
| `rag delete --path PATH` 或 `--doc-id ID` | 刪除一份文件 |
| `rag status` | 顯示本機語料，以及 Ollama 能否回答 |
| `rag models pull` | 拉取已設定的模型 |
| `rag models unload` | 卸載 Ollama 已載入的模型 |
| `rag config show` | 印出解析後的設定 |

`--config` 指定一個 TOML 檔。預設由工作目錄向上尋找 `rag.toml`。`--data-dir` 覆蓋 `paths.data_dir`。`--verbose` 打開除錯日誌。`--lock-timeout` 是等待 Ollama 鎖的秒數。`0` 表示不等待。

`rag ingest` 接受 `--force`、`--resume`、`--no-ocr`、`--no-vision`、`--no-embed`、`--strict`、`--max-pages` 同 `--max-bytes`。`rag ask` 接受 `--k`（1 到 8）、`--model`、`--no-fallback`、`--show-chunks`、`--json`、`--verify` 同 `--no-verify`。

`--verify` 用同一個本機模型對照摘錄核對這次草稿。`--no-verify` 關閉這次核對。預設是關閉。`rag.toml` 的 `[retrieve] verify` 設定預設，兩個旗標都會覆蓋設定檔。模型只能回答精確的 `PASS` 或精確的 `FAIL`。`PASS` 保留草稿。`FAIL` 印出「資料中找不到答案。」核對不會再檢索。

`--lang` 選擇介面語言，`en` 或 `zh`。`en` 是英文，也是預設。`zh` 是繁體中文。這個旗標覆蓋 `rag.toml` 的 `[ui] language`。兩者都沒有時，介面是英文。ragcli 不讀 `LANG`、`LC_ALL`，也不讀 macOS 偏好語言。值必須恰好是 `en` 或 `zh`。`[ocr] languages` 選擇 Vision 辨識語言，不跟介面語言連動。用法列、區段標題、說明動作那一句，以及 argparse 的錯誤外框，跟 `--lang` 與 `ui.language`。

```toml
[ui]
language = "zh"
```

## 安全

文件正文、頁面圖像同嵌入留在這部 Mac。`rag.toml` 沒有秘密欄位。

- **只走迴路。** Ollama URL 必須是 `127.0.0.1`、`localhost` 或 `::1` 上的 `http` 或 `https`。帳號、密碼、路徑同查詢字串都會被拒絕。HTTP 客戶端遇到重新導向就停止，而且忽略代理環境變數。拉取模型使用 Ollama 的 HTTP API。ragcli 不會啟動 shell 去執行 `ollama` 二進位檔。
- **權重同文件。** `rag models pull` 要求本機常駐程式下載模型權重。該傳輸屬於 Ollama。入庫同提問只把提示送到該迴路常駐程式。
- **遙測。** Chroma 客戶端在開啟時關閉匿名遙測。該開關若打開，ragcli 拒絕開啟 Chroma。
- **檔案。** ragcli 建立的資料目錄模式是 `0700`。SQLite 資料庫同其附帶檔的模式是 `0600`。已經存在的目錄保持原來的模式。資料目錄同入庫路徑必須是真實路徑，不能是符號連結。`data/` 列在 gitignore。
- **日誌。** 除錯日誌不印文件正文。
- **不受信任的輸入。** PDF 用 PyMuPDF 解析，圖像用 Pillow。兩者都應視為解析不受信任位元組的程式。預設會停下超過 200 MiB 的檔案、超過 500 頁的文件、超過 300 張圖像的檔案，以及超過 2,000 萬像素的圖像。有密碼保護的 PDF 會被拒絕，而且不會提示輸入密碼。檔頭已超過像素上限的圖像不會解碼。
- **程序。** 載入生成模型之前，ragcli 讀取 `/usr/bin/vm_stat`，需要頁面大小時再讀 `/usr/sbin/sysctl -n hw.pagesize`。這兩組引數是僅有的子程序。
- **授權。** mmlmRAG 是 AGPL-3.0-only。再散布本程式，或透過網路提供修改版，都必須依該授權提供對應原始碼。見[授權](#授權)。
- **回報。** 尚未修復的漏洞請依 [SECURITY.md](SECURITY.md) 回報。該政策使用 GitHub 的非公開漏洞回報，因此回報不會出現在公開議題清單。

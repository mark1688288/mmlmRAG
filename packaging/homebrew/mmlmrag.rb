class Mmlmrag < Formula
  include Language::Python::Virtualenv

  # Resources are the macOS wheels resolved on Apple Silicon.
  # Regenerate them before building an Intel bottle.

  desc "Local mixed-language RAG command-line tool"
  homepage "https://github.com/mark1688288/mmlmRAG"
  url "https://github.com/mark1688288/mmlmRAG/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "8d6c2f938fdb841cdc9139fd32ebe8fea5026e0c264a4f0d40c28273d432eeb2"
  license "AGPL-3.0-only"

  depends_on :macos
  depends_on "python@3.13"

  resource "aiohappyeyeballs" do
    url "https://files.pythonhosted.org/packages/71/43/1947f06babed6b3f1d7f38b0c767f52df66bfb2bc10b468c4a7de9eceff2/aiohappyeyeballs-2.7.1-py3-none-any.whl"
    sha256 "9243213661e29250eb41368e5daa826fc017156c3b8a11440826b2e3ed376472"
  end

  resource "aiohttp" do
    url "https://files.pythonhosted.org/packages/15/e5/b57e58695a757fd4c02497c033fced96a69c631b13866c43d336530c9670/aiohttp-3.14.5-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "e1cc2bfaee8c214f06080a7c7d5772419b8a1108e8e5349236189811823fb02a"
  end

  resource "aiohttp-retry" do
    url "https://files.pythonhosted.org/packages/1a/99/84ba7273339d0f3dfa57901b846489d2e5c2cd731470167757f1935fffbd/aiohttp_retry-2.9.1-py3-none-any.whl"
    sha256 "66d2759d1921838256a05a3f80ad7e724936f083e35be5abb5e16eed6be6dc54"
  end

  resource "aiosignal" do
    url "https://files.pythonhosted.org/packages/fb/76/641ae371508676492379f16e2fa48f4e2c11741bd63c48be4b12a6b09cba/aiosignal-1.4.0-py3-none-any.whl"
    sha256 "053243f8b92b990551949e63930a839ff0cf0b0ebbe0597b0f3fb19e1a0fe82e"
  end

  resource "annotated-doc" do
    url "https://files.pythonhosted.org/packages/3e/30/e900b21425a860e195f32e37657aa1f7c7f2b1bfb26f03ca209b90933c06/annotated_doc-0.0.5-py3-none-any.whl"
    sha256 "117bac03a25ede5df5440e855b32d556049ca169ead221505badf432fed4b101"
  end

  resource "annotated-types" do
    url "https://files.pythonhosted.org/packages/99/91/8acff4f5e50511b911bbccb72b8628a49c68ce14148cd9f6431094859a90/annotated_types-0.8.0-py3-none-any.whl"
    sha256 "f072f4d804ea359e4eaf198b1af7a8b0943881a87f31bb764f8bf219bb9419e0"
  end

  resource "anyio" do
    url "https://files.pythonhosted.org/packages/12/b8/4bd346e22b28902df4d651910f5242c28d84e4a5c2435ca5c3f797ed7e2e/anyio-4.15.1-py3-none-any.whl"
    sha256 "6152fdbbf9a77fdec97731721bebf7c4c44f7c29b424b0065826173efc7ed101"
  end

  resource "attrs" do
    url "https://files.pythonhosted.org/packages/64/b4/17d4b0b2a2dc85a6df63d1157e028ed19f90d4cd97c36717afef2bc2f395/attrs-26.1.0-py3-none-any.whl"
    sha256 "c647aa4a12dfbad9333ca4e71fe62ddc36f4e63b2d260a37a8b83d2f043ac309"
  end

  resource "bcrypt" do
    url "https://files.pythonhosted.org/packages/5d/ba/2af136406e1c3839aea9ecadc2f6be2bcd1eff255bd451dd39bcf302c47a/bcrypt-5.0.0-cp39-abi3-macosx_10_12_universal2.whl"
    sha256 "0c418ca99fd47e9c59a301744d63328f17798b5947b0f791e9af3c1c499c2d0a"
  end

  resource "build" do
    url "https://files.pythonhosted.org/packages/ad/9b/9fb3585dabcd73a1b2a6267f63f62649347c9e6d072c9fde365b105abb2c/build-1.6.1-py3-none-any.whl"
    sha256 "ecd351a4be9d35a9eaaba244a7687143c9c7d4aea6ac964e7e7ddab20cbcf4e7"
  end

  resource "certifi" do
    url "https://files.pythonhosted.org/packages/32/d1/a3652fe2c19134d0d15922177821d863cbd7e842d53a21bbe7e8e3a1b5ac/certifi-2026.10.23-py3-none-any.whl"
    sha256 "e9b54de76e530bb6cf0ccd036787103acb58fd42ba930ea8bf948ddbaa2f61d1"
  end

  resource "charset-normalizer" do
    url "https://files.pythonhosted.org/packages/91/da/3c5a7798c046df7d2d68ad653cf5b6c5a8bfee225055a843c6f2f42aac1a/charset_normalizer-3.5.2-cp313-cp313-macosx_10_13_universal2.whl"
    sha256 "6045373d5a89a5ec71afde535db987ca28e76dfa276c2d4c818265b375d4b055"
  end

  resource "chromadb" do
    url "https://files.pythonhosted.org/packages/34/4c/adcef1f4e82a2ef69ccd3711d55fc289193d54c4c0ff7a0292a3631db46f/chromadb-1.5.9-cp39-abi3-macosx_11_0_arm64.whl"
    sha256 "814b9c95617377f6501e5757d63dfddb554a283a7739c87b9fa573850174e6f3"
  end

  resource "click" do
    url "https://files.pythonhosted.org/packages/58/50/6c0d534c5f134586a8e1ba4e330569e32f057e33372ae556463212fb4cd3/click-8.5.0-py3-none-any.whl"
    sha256 "255bc9599cf7748b4b1a446ccc735421bd08a2ae529a8b88597d3de5664ee360"
  end

  resource "durationpy" do
    url "https://files.pythonhosted.org/packages/e6/c4/ebdf7837bc4ef6fd98cfb013c28855bb358467bf86c1af011bbc21e21df0/durationpy-0.11-py3-none-any.whl"
    sha256 "a739fe2b8972c250ff72f8e2c488d18cf25f7b852f49ee76048775d5171df30c"
  end

  resource "filelock" do
    url "https://files.pythonhosted.org/packages/e0/1b/5e74289a1878e666bb4b65fe5805a04cafcac2aa5f519ef1ead67ef91c56/filelock-4.2.4-py3-none-any.whl"
    sha256 "301241438c57936ef93f4243a94f2559022268bfcf9b375a9a72eb0f8db8358e"
  end

  resource "flatbuffers" do
    url "https://files.pythonhosted.org/packages/e8/2d/d2a548598be01649e2d46231d151a6c56d10b964d94043a335ae56ea2d92/flatbuffers-25.12.19-py2.py3-none-any.whl"
    sha256 "7634f50c427838bb021c2d66a3d1168e9d199b0607e6329399f04846d42e20b4"
  end

  resource "frozenlist" do
    url "https://files.pythonhosted.org/packages/0c/ab/6e5080ee374f875296c4243c381bbdef97a9ac39c6e3ce1d5f7d42cb78d6/frozenlist-1.8.0-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "f21f00a91358803399890ab167098c131ec2ddd5f8f5fd5fe9c9f2c6fcd91e40"
  end

  resource "fsspec" do
    url "https://files.pythonhosted.org/packages/6c/c0/a98505f18594f1bce828bb159cec0fcf9860562f1a2c85913409fc8f3d9e/fsspec-2026.9.0-py3-none-any.whl"
    sha256 "8dd6e646e99ea382bd85f97a45e6b526a442d79423a7dc673f1e2756d05fcb5f"
  end

  resource "googleapis-common-protos" do
    url "https://files.pythonhosted.org/packages/05/6d/78dc67a9e779e594b49bf62604034d3a6aa428a55afc729cafda4b7333e1/googleapis_common_protos-1.76.0-py3-none-any.whl"
    sha256 "59e13344a9f3651aa201a03e2bee815a3bf43a4baf1f9ff8a9b62a840ce25fac"
  end

  resource "grpcio" do
    url "https://files.pythonhosted.org/packages/f0/b5/bd9b3d4c148566dc797aca206b0db255f23df40a53c64dfb6c6410560a21/grpcio-1.85.0-cp313-cp313-macosx_11_0_universal2.whl"
    sha256 "874af68625a2ff2585cb112e7583c19a7cb463ab638a7fe5b8a58f76e6d4e7c2"
  end

  resource "h11" do
    url "https://files.pythonhosted.org/packages/04/4b/29cac41a4d98d144bf5f6d33995617b185d14b22401f75ca86f384e87ff1/h11-0.16.0-py3-none-any.whl"
    sha256 "63cf8bbe7522de3bf65932fda1d9c2772064ffb3dae62d55932da54b31cb6c86"
  end

  resource "hf-xet" do
    url "https://files.pythonhosted.org/packages/c3/47/a26ebdce7056a61e931f228439bc0ab08cbec239d1690f965e5e637cba79/hf_xet-1.7.0-cp38-abi3-macosx_11_0_arm64.whl"
    sha256 "59fba37039233c7fcbe196817d6cdcf1b40dfb17b410f229d85b0cf0a1848da4"
  end

  resource "httpcore" do
    url "https://files.pythonhosted.org/packages/7e/f5/f66802a942d491edb555dd61e3a9961140fd64c90bce1eafd741609d334d/httpcore-1.0.9-py3-none-any.whl"
    sha256 "2d400746a40668fc9dec9810239072b40b4484b640a8c38fd654a024c7a1bf55"
  end

  resource "httpcore2" do
    url "https://files.pythonhosted.org/packages/af/c2/ce417c397afa5181f0e3c6ff01a4d01639414e1adef08730d97571297151/httpcore2-2.16.0-py3-none-any.whl"
    sha256 "5c4358319f2602d984ad93182c9fee5160f22e9da7babdda870a5e7c4e2dc34f"
  end

  resource "httptools" do
    url "https://files.pythonhosted.org/packages/9c/04/223994f8589750d2a36ceb43203e739cf75bd9e12c226680d73567766908/httptools-0.9.0-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "4fb995082fe41ec410b33c48b54fb1d44abb8a6ee762c31e8c42519e8c3a30a9"
  end

  resource "httpx" do
    url "https://files.pythonhosted.org/packages/2a/39/e50c7c3a983047577ee07d2a9e53faf5a69493943ec3f6a384bdc792deb2/httpx-0.28.1-py3-none-any.whl"
    sha256 "d909fcccc110f8c7faf814ca82a9a4d816bc5a6dbfea25d6591d6985b8ba59ad"
  end

  resource "httpx2" do
    url "https://files.pythonhosted.org/packages/7b/3d/2e4bc4fd1ef9770871669adb348d612a6e4786b303a23f9935b81c48dd60/httpx2-2.16.0-py3-none-any.whl"
    sha256 "fd2c5d7909e8d86fe92d53020c310ca0e4e431b838c207c5754b9bbf0308d49a"
  end

  resource "huggingface_hub" do
    url "https://files.pythonhosted.org/packages/c7/89/5f6ef1162d0323f167b6baab04d1b4bd6a946161a8af6d86f622d9d79aa1/huggingface_hub-2.3.0-py3-none-any.whl"
    sha256 "5da369600233568a126eb5578754935b2add9446b42d9e2fc0927d56840ac1a5"
  end

  resource "idna" do
    url "https://files.pythonhosted.org/packages/58/a2/bb081bab032533a855d44de1d56f8e8426114ff1ba5d1f07a438a0a654f8/idna-3.20-py3-none-any.whl"
    sha256 "ab7ae7122974553370f0bdb919e1a960b2cd1bc1ef0276416d896db81c14582c"
  end

  resource "importlib_resources" do
    url "https://files.pythonhosted.org/packages/8a/db/55a262f3606bebcae07cc14095338471ad7c0bbcaa37707e6f0ee49725b7/importlib_resources-7.1.0-py3-none-any.whl"
    sha256 "1bd7b48b4088eddb2cd16382150bb515af0bd2c70128194392725f82ad2c96a1"
  end

  resource "jsonschema" do
    url "https://files.pythonhosted.org/packages/69/90/f63fb5873511e014207a475e2bb4e8b2e570d655b00ac19a9a0ca0a385ee/jsonschema-4.26.0-py3-none-any.whl"
    sha256 "d489f15263b8d200f8387e64b4c3a75f06629559fb73deb8fdfb525f2dab50ce"
  end

  resource "jsonschema-specifications" do
    url "https://files.pythonhosted.org/packages/41/45/1a4ed80516f02155c51f51e8cedb3c1902296743db0bbc66608a0db2814f/jsonschema_specifications-2025.9.1-py3-none-any.whl"
    sha256 "98802fee3a11ee76ecaca44429fda8a41bff98b00a0f2838151b113f210cc6fe"
  end

  resource "kubernetes" do
    url "https://files.pythonhosted.org/packages/8a/de/ac1904ac7fb651426e5698f56a268669a1ca3d63378629bacbf057b25e75/kubernetes-37.0.1-py2.py3-none-any.whl"
    sha256 "a313a482361506340b5972cabaf20a7ee16ec8713e73325bf699b783a40464a9"
  end

  resource "lxml" do
    url "https://files.pythonhosted.org/packages/52/05/3ef45db776baea068044c799bbba68f3ca00a440c0e930a17c572f3d9639/lxml-6.1.3-cp313-cp313-macosx_10_13_universal2.whl"
    sha256 "3a48093cdb058a93af842ede9703520e810b05dcd0fc6d7190a06376c3bfb6bd"
  end

  resource "markdown-it-py" do
    url "https://files.pythonhosted.org/packages/b3/81/4da04ced5a082363ecfa159c010d200ecbd959ae410c10c0264a38cac0f5/markdown_it_py-4.2.0-py3-none-any.whl"
    sha256 "9f7ebbcd14fe59494226453aed97c1070d83f8d24b6fc3a3bcf9a38092641c4a"
  end

  resource "mdurl" do
    url "https://files.pythonhosted.org/packages/b3/38/89ba8ad64ae25be8de66a6d463314cf1eb366222074cfda9ee839c56a4b4/mdurl-0.1.2-py3-none-any.whl"
    sha256 "84008a41e51615a49fc9966191ff91509e3c40b939176e643fd50a5c2196b8f8"
  end

  resource "mmh3" do
    url "https://files.pythonhosted.org/packages/5a/0a/67d5082ad1fe184c4c928ca6be61d775947590863d52de0a9d9aa7d525b2/mmh3-5.3.1-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "c25a6d4b6ff31d801ff6f1ad5ce003271bceabf21c3e9ffcf04a47774354e956"
  end

  resource "multidict" do
    url "https://files.pythonhosted.org/packages/06/18/29874e132d78ce182305e818a1553c4ddae238471ba55903a1ea38b5c4ee/multidict-7.2.1-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "4bebc9587cc29359e3b045518a7a09264fa882a7d165b0d143fdac8eeb7ea948"
  end

  resource "networkx" do
    url "https://files.pythonhosted.org/packages/7e/cd/fe58041e9011f307c490e3e17dd48cc516448f7c698a3f2d9d9d65d7e6a8/networkx-3.7-py3-none-any.whl"
    sha256 "e3fd2c13a7814cee3746340d8d7f8598a67f16a58bf47fb7f8793fab6efca1b0"
  end

  resource "numpy" do
    url "https://files.pythonhosted.org/packages/e2/84/6a3d75b3ba3dfe84ac0053450753d1e6d250a8bf80f66474cc46d1fb643f/numpy-2.5.4-cp313-cp313-macosx_14_0_arm64.whl"
    sha256 "6d6a71b9d9a97c03633aa12565ef2825ffa036cc1d99cfd50dacf0f128af4fe2"
  end

  resource "oauthlib" do
    url "https://files.pythonhosted.org/packages/d9/f4/78229a1066068ca14fc60fb26cf7381cabe4382261392b90e5f9552722d4/oauthlib-4.0.0-py3-none-any.whl"
    sha256 "624c28c13a0a59cabf9747dfa52af63be3e512a7f2714df16e91b5b3a145e6cd"
  end

  resource "ocrmac" do
    url "https://files.pythonhosted.org/packages/37/15/7cc16507a2aca927abe395f1c545f17ae76b1f8ed44f43ebe4e8670ee203/ocrmac-1.0.1-py3-none-any.whl"
    sha256 "1cef25426f7ae6bbd57fe3dc5553b25461ae8ad0d2b428a9bbadbf5907349024"
  end

  resource "onnxruntime" do
    url "https://files.pythonhosted.org/packages/e0/2b/117f94d73a3bac4276c285c47e384e1b3ea67b191aa4c7592df9d3f4a136/onnxruntime-1.31.0-cp313-cp313-macosx_14_0_arm64.whl"
    sha256 "0ba02a44acb6203040354d9a1f160e3f37a43feac7bb05caa3e0ea545efed505"
  end

  resource "opentelemetry-api" do
    url "https://files.pythonhosted.org/packages/1e/41/f7dcf80b81ee8e71c1a2b59f14208bc723edbd89ed027a73b175abf6348e/opentelemetry_api-1.45.1-py3-none-any.whl"
    sha256 "b31553efa588ae44bc306f863c785c5333a9ecc091248c6ee68b4b6c87fdedfb"
  end

  resource "opentelemetry-exporter-otlp-common" do
    url "https://files.pythonhosted.org/packages/fc/39/8c23d67665c762aa51840fa06f86e902e8f6f1693bc8d7e3d98cd6e2f753/opentelemetry_exporter_otlp_common-0.66b1-py3-none-any.whl"
    sha256 "00ff8592c3a7cb729ff3fdc7ffa12372c243bdf2163e80c180994d0c7bd83ee9"
  end

  resource "opentelemetry-exporter-otlp-proto-common" do
    url "https://files.pythonhosted.org/packages/84/aa/92f225d353904e7f70b8b3e3c1b02db0cf56f744c2e83c581dc372e78873/opentelemetry_exporter_otlp_proto_common-1.45.1-py3-none-any.whl"
    sha256 "2f446183ae7047b036226f1d846c41a834b0e8755ad13b51a51dd38952eb466c"
  end

  resource "opentelemetry-exporter-otlp-proto-grpc" do
    url "https://files.pythonhosted.org/packages/4d/46/2d1da202f1e17c81aae7efcf702898d524b46709e4d3e2bf1f7f8ca8fbc6/opentelemetry_exporter_otlp_proto_grpc-1.45.1-py3-none-any.whl"
    sha256 "e42ecb789d2fc5d8145e3dadc3e2991c9f18cd166d7c7514e234702540274b76"
  end

  resource "opentelemetry-proto" do
    url "https://files.pythonhosted.org/packages/ab/9a/42ec8180a769516ae757e893b69736826efceac7332553915b4528a91c6d/opentelemetry_proto-1.45.1-py3-none-any.whl"
    sha256 "f38e2a8413053c180cd3d2637fbb279673ec2f6a6e09c995aafa2f452c52b46e"
  end

  resource "opentelemetry-sdk" do
    url "https://files.pythonhosted.org/packages/95/3c/87c42b4bd6dd297536f04cd9383d212ac557ecd49f2cbdcd46da1c9ef5c8/opentelemetry_sdk-1.45.1-py3-none-any.whl"
    sha256 "c604c11dc429810812348989115fa44bd558772a3d7442afc43d024f2c250ca4"
  end

  resource "opentelemetry-semantic-conventions" do
    url "https://files.pythonhosted.org/packages/bc/14/67f8aa798857f8cf686f515bf93d9bb877ce952ddc8efae0fa25b45ce0d6/opentelemetry_semantic_conventions-0.66b1-py3-none-any.whl"
    sha256 "d4cddeb4315490b35213f55e2bdc9ac54bb1e4d318927475bed62b35545e581b"
  end

  resource "orjson" do
    url "https://files.pythonhosted.org/packages/1f/19/725d23160b2471a3f27026c55bb79af34687652d8be8f5f583cee5dcd42f/orjson-3.13.0-cp313-cp313-macosx_15_0_arm64.whl"
    sha256 "ded33b972cffdaf4ca0ac917338ab61d2bb10d68987dbcae641c313fbfdbf499"
  end

  resource "overrides" do
    url "https://files.pythonhosted.org/packages/2c/ab/fc8290c6a4c722e5514d80f62b2dc4c4df1a68a41d1364e625c35990fcf3/overrides-7.7.0-py3-none-any.whl"
    sha256 "c7ed9d062f78b8e4c1a7b70bd8796b35ead4d9f510227ef9c5dc7626c60d7e49"
  end

  resource "packaging" do
    url "https://files.pythonhosted.org/packages/63/34/ba1c580383c9eada3711951fef0795c80b829a078d72188184bcab9dd527/packaging-26.3-py3-none-any.whl"
    sha256 "d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c"
  end

  resource "pillow" do
    url "https://files.pythonhosted.org/packages/98/c8/ccede3dcc4f274b80fb49199eed1ef34ba05566c8b71976b9d1f111da2e3/pillow-13.0.0-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "f11a3c1358bb3c68012a7b1f41b223c3db9f7bfde90b481df028cfe1c3363313"
  end

  resource "pillow_heif" do
    url "https://files.pythonhosted.org/packages/29/96/4170d91441cbb3336dbe02155b57c0004b2516a40538f7aae8c0b8af497d/pillow_heif-1.8.1-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "98c500475f3add0d2ac4a6686b925c22fd0cf05def1ce977fec8ec753dabd66a"
  end

  resource "propcache" do
    url "https://files.pythonhosted.org/packages/25/7d/c1ab1ef09e9d4d835be5d58c0a32a1e1de8397abaa4e502a9d4141328cad/propcache-0.5.4-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "44149f46500a0a41b95b4d99c2e586a77319539730607b9892974a092788b111"
  end

  resource "protobuf" do
    url "https://files.pythonhosted.org/packages/32/72/98342feb672507c8f3a69e34b4fa8961f608edba5c1a48a6f47156d92cb5/protobuf-7.36.2-cp310-abi3-macosx_10_9_universal2.whl"
    sha256 "cbc70b17ee27e28894c7fee8bb04be1abead49e936bc70eb60052531eee2079e"
  end

  resource "psutil" do
    url "https://files.pythonhosted.org/packages/80/c4/f5af4c1ca8c1eeb2e92ccca14ce8effdeec651d5ab6053c589b074eda6e1/psutil-7.2.2-cp36-abi3-macosx_11_0_arm64.whl"
    sha256 "1a7b04c10f32cc88ab39cbf606e117fd74721c831c98a27dc04578deb0c16979"
  end

  resource "pybase64" do
    url "https://files.pythonhosted.org/packages/50/b4/034e1fb26abc4e6c78812b53f25c589931ebb79b91c837173a1917ac2b53/pybase64-1.5.1-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "fd656d6e8c48acacd9048285936fa2b3d6010764563fc5593a32458c3feaa167"
  end

  resource "pydantic" do
    url "https://files.pythonhosted.org/packages/35/ea/a56b9fe5066f3537b7882f77e9c5dfb26d8c2eefdaed9b5fc73d57b4dc22/pydantic-2.14.1-py3-none-any.whl"
    sha256 "9195d967ec791692a04438115466764fb8b9a27b31f14a760437694f40d6b454"
  end

  resource "pydantic-settings" do
    url "https://files.pythonhosted.org/packages/3c/74/fdf926afe145d8629c4b2ff57f4ea1d64ed7abc4d84d97463b2517a2547e/pydantic_settings-2.16.1-py3-none-any.whl"
    sha256 "30f5313340da4e8f1b440d1910b92a5bc3ccbb6b4ba70ec2b8395bc921f418f5"
  end

  resource "pydantic_core" do
    url "https://files.pythonhosted.org/packages/f7/b9/c720e56858d4e1539503297ed37063e0c08e0f3541c41b777f6a800f75fa/pydantic_core-2.50.1-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "0c003c3b7f49debb893d2d85ae099ac5959c9839e2f330fadb1fcdf7a6594482"
  end

  resource "Pygments" do
    url "https://files.pythonhosted.org/packages/71/46/17f022dd3e953bf20a04a028a21ec746d942f8d2af30fa0f124fa0e6a684/pygments-2.21.0-py3-none-any.whl"
    sha256 "2363c69b61c4a97c838da3b130dcd6468f4848992b21a82f2a63ec34377137d9"
  end

  resource "pymupdf" do
    url "https://files.pythonhosted.org/packages/bf/0d/f72a28bb74af3e6ced3766000381f75eea119d59dbcce376ef3fbc886796/pymupdf-2.0-cp311-abi3-macosx_11_0_arm64.whl"
    sha256 "707594e8de4946445489ca3274943911a505a5cb665adf8751a4edfc4c0fff58"
  end

  resource "pymupdf4llm" do
    url "https://files.pythonhosted.org/packages/b1/86/91671857614bbf931c04532c11572c70fa2031862b1332befcd3bcde1f51/pymupdf4llm-2.0-py3-none-any.whl"
    sha256 "43d4b4a2d0bf8b9b4d7f21a48875ef0016018dbad6677ef6b01ef2b0f6e392dd"
  end

  resource "pymupdf_layout" do
    url "https://files.pythonhosted.org/packages/37/4e/d8a1eca5fda53a7a288d4f1e0f8d625344bbc84d5d9d4e5b95c73c8789d2/pymupdf_layout-2.0-py3-none-any.whl"
    sha256 "03ac486c796839eae5c46e910364ad524fb1e83c7dc34553e8e704939d0236a7"
  end

  resource "pyobjc-core" do
    url "https://files.pythonhosted.org/packages/1b/ed/a8bf040caf3704023d74086b7fb96cf4ed2e844e24bd94e5248ba214b700/pyobjc_core-12.2.2-cp313-cp313-macosx_10_13_universal2.whl"
    sha256 "950bd2d9c74634398c4e3d24ef2f213d4e23d705083697464fa67afedc53c1ad"
  end

  resource "pyobjc-framework-Cocoa" do
    url "https://files.pythonhosted.org/packages/db/e1/5d9b04ebb60042b9cb49adc2d33115e2f2c2e4ff7d548017bfaff8b7f536/pyobjc_framework_cocoa-12.2.2-cp313-cp313-macosx_10_13_universal2.whl"
    sha256 "600b1723184ca094931330e79355274949965460e23de38628d601b5a967baf9"
  end

  resource "pyobjc-framework-CoreML" do
    url "https://files.pythonhosted.org/packages/21/7a/46cd02ceb310f2f20e4a5fb78b4f2e456ea0c54799c773343ca575ccfb04/pyobjc_framework_coreml-12.2.2-cp313-cp313-macosx_10_13_universal2.whl"
    sha256 "70ae22934f2432677bbbcee04a21db2b0301056cd8390e61101fb59e1d374174"
  end

  resource "pyobjc-framework-Quartz" do
    url "https://files.pythonhosted.org/packages/bb/ae/b515852dbe491171f2f2e2eb7739588a5eb7f36720a739545337b8c0d706/pyobjc_framework_quartz-12.2.2-cp313-cp313-macosx_10_13_universal2.whl"
    sha256 "0ec9751904ef975bf0789d760dc4fadcb400edc4ffe4a736eb54971968babe5c"
  end

  resource "pyobjc-framework-Vision" do
    url "https://files.pythonhosted.org/packages/c9/f6/e3cb58a12ab51ff9ac159988fc3c6410a60234dc7b448223fc090cd3d1f7/pyobjc_framework_vision-12.2.2-cp313-cp313-macosx_10_13_universal2.whl"
    sha256 "306e536e12d3565e53649818e3324e801fbb1696ea75ba751852dffbacba8b14"
  end

  resource "PyPika" do
    url "https://files.pythonhosted.org/packages/57/83/c77dfeed04022e8930b08eedca2b6e5efed256ab3321396fde90066efb65/pypika-0.51.1-py2.py3-none-any.whl"
    sha256 "77985b4d7ce71b9905255bf12468cf598349e98837c037541cfc240e528aec46"
  end

  resource "pyproject_hooks" do
    url "https://files.pythonhosted.org/packages/85/11/044d1ae1b4ec0d7af88ee5bc91e081be1022533032b906a7bdabdbb60977/pyproject_hooks-1.3.3-py3-none-any.whl"
    sha256 "5fc53fdac9f7bd63fbcdc868fb5f90b4784d78a53a3d3388cd738b807441a20b"
  end

  resource "python-dateutil" do
    url "https://files.pythonhosted.org/packages/5a/17/cebee01d1c0c1441174cf0456ddc91b280d79acb65340c05e41d77866ee8/python_dateutil-3.0.0-py2.py3-none-any.whl"
    sha256 "3c79630202e4152c6add678446984236efe4f57e84d8e892aab0637bab82e9fd"
  end

  resource "python-docx" do
    url "https://files.pythonhosted.org/packages/d0/00/1e03a4989fa5795da308cd774f05b704ace555a70f9bf9d3be057b680bcf/python_docx-1.2.0-py3-none-any.whl"
    sha256 "3fd478f3250fbbbfd3b94fe1e985955737c145627498896a8a6bf81f4baf66c7"
  end

  resource "python-dotenv" do
    url "https://files.pythonhosted.org/packages/60/d1/38f3a3405989a89ac18390803e70c6ad7c7760da4f9b83cbeca0c44a0c72/python_dotenv-1.2.4-py3-none-any.whl"
    sha256 "42269a8a5b3fd54ffa6f3d84b18abed50064717576b4ecf03dc4a55d8aa04fdc"
  end

  resource "python-pptx" do
    url "https://files.pythonhosted.org/packages/d9/4f/00be2196329ebbff56ce564aa94efb0fbc828d00de250b1980de1a34ab49/python_pptx-1.0.2-py3-none-any.whl"
    sha256 "160838e0b8565a8b1f67947675886e9fea18aa5e795db7ae531606d68e785cba"
  end

  resource "PyYAML" do
    url "https://files.pythonhosted.org/packages/b1/16/95309993f1d3748cd644e02e38b75d50cbc0d9561d21f390a76242ce073f/pyyaml-6.0.3-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "2283a07e2c21a2aa78d9c4442724ec1eb15f5e42a723b99cb3d822d48f5f7ad1"
  end

  resource "referencing" do
    url "https://files.pythonhosted.org/packages/2c/58/ca301544e1fa93ed4f80d724bf5b194f6e4b945841c5bfd555878eea9fcb/referencing-0.37.0-py3-none-any.whl"
    sha256 "381329a9f99628c9069361716891d34ad94af76e461dcb0335825aecc7692231"
  end

  resource "requests" do
    url "https://files.pythonhosted.org/packages/a0/f4/c67b0b3f1b9245e8d266f0f112c500d50e5b4e83cb6f3b71b6528104182a/requests-2.34.2-py3-none-any.whl"
    sha256 "2a0d60c172f83ac6ab31e4554906c0f3b3588d37b5cb939b1c061f4907e278e0"
  end

  resource "requests-oauthlib" do
    url "https://files.pythonhosted.org/packages/3b/5d/63d4ae3b9daea098d5d6f5da83984853c1bbacd5dc826764b249fe119d24/requests_oauthlib-2.0.0-py2.py3-none-any.whl"
    sha256 "7dd8a5c40426b779b0868c404bdef9768deccf22749cde15852df527e6269b36"
  end

  resource "rich" do
    url "https://files.pythonhosted.org/packages/82/3b/64d4899d73f91ba49a8c18a8ff3f0ea8f1c1d75481760df8c68ef5235bf5/rich-15.0.0-py3-none-any.whl"
    sha256 "33bd4ef74232fb73fe9279a257718407f169c09b78a87ad3d296f548e27de0bb"
  end

  resource "rpds-py" do
    url "https://files.pythonhosted.org/packages/57/71/a097d6552f837500fc36e6b23d09cfb9890c3cc47531f9ca64e149799615/rpds_py-2026.9.1-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "eba5d173f7d5708b22a93815017a4611873ed54db9f268077c0dd1ed99cfc858"
  end

  resource "shellingham" do
    url "https://files.pythonhosted.org/packages/e0/f9/0595336914c5619e5f28a1fb793285925a8cd4b432c9da0a987836c7f822/shellingham-1.5.4-py2.py3-none-any.whl"
    sha256 "7ecfff8f2fd72616f7481040475a65b2bf8af90a56c89140852d1120324e8686"
  end

  resource "six" do
    url "https://files.pythonhosted.org/packages/b7/ce/149a00dd41f10bc29e5921b496af8b574d8413afcd5e30dfa0ed46c2cc5e/six-1.17.0-py2.py3-none-any.whl"
    sha256 "4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274"
  end

  resource "tabulate" do
    url "https://files.pythonhosted.org/packages/99/55/db07de81b5c630da5cbf5c7df646580ca26dfaefa593667fc6f2fe016d2e/tabulate-0.10.0-py3-none-any.whl"
    sha256 "f0b0622e567335c8fabaaa659f1b33bcb6ddfe2e496071b743aa113f8774f2d3"
  end

  resource "tenacity" do
    url "https://files.pythonhosted.org/packages/d6/26/1ff2b0721ac66a3ec5b1402b333110b352ab0a8724052ac279a7b82d40c4/tenacity-9.2.1-py3-none-any.whl"
    sha256 "9e56f17539296baab7beabb08b92f6ee3d7be92d8be72d763360677c2ad6580e"
  end

  resource "tokenizers" do
    url "https://files.pythonhosted.org/packages/b2/72/01e49f032bb346e5aaf06c10c74fe8aeec847173adbadd66eb7c53054bf2/tokenizers-0.23.3-cp310-abi3-macosx_11_0_arm64.whl"
    sha256 "68649e97d5b43c44c031d8d848874a6eecae8f8fe40ea989aa777a5a83aca716"
  end

  resource "tqdm" do
    url "https://files.pythonhosted.org/packages/a7/03/921a3d3c75785aca9ebfbfcabfbc3a1be12e2ab5265deb026d55a5a3f83e/tqdm-4.70.1-py3-none-any.whl"
    sha256 "c293e525e6fef9c20e8728fd4612df02a0aa31bb5fe91ecd93e123b1b7bffa73"
  end

  resource "truststore" do
    url "https://files.pythonhosted.org/packages/51/e9/3a7820be2bb0fe53b6bc9c3be26d3d1158004e4c3ab953aa6840b955b1e9/truststore-0.10.5-py3-none-any.whl"
    sha256 "9aaaedaefaf06d8b206278cf8b5012bc897f485a874503501e12d776df78951c"
  end

  resource "typer" do
    url "https://files.pythonhosted.org/packages/07/ea/2e31b67051e91a133189e9c000c222502ddc6969856416de0d095de4c0b0/typer-0.27.3-py3-none-any.whl"
    sha256 "e50022f28b82a86313e54501317a1db64bf8f8d036ff8cfe5ca7e47675454aff"
  end

  resource "typing-inspection" do
    url "https://files.pythonhosted.org/packages/67/81/4add07e5172b7ac40d8ed5ff580409a7801a4fe26d529bdd915401dabfbe/typing_inspection-0.4.4-py3-none-any.whl"
    sha256 "65b8397ba37ccbce054456aaccddfc91e6e3083c92824df348d96ca832f3f147"
  end

  resource "typing_extensions" do
    url "https://files.pythonhosted.org/packages/49/d3/b8441a820a491ddfc024b0b0cf0393375b75ea13866d9c66727e54c2fc80/typing_extensions-4.16.0-py3-none-any.whl"
    sha256 "481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8"
  end

  resource "tzdata" do
    url "https://files.pythonhosted.org/packages/94/21/1e5995a1c920cce14e4bffae20c665ec10e7ed03ab25e006cd741092b718/tzdata-2026.5-py2.py3-none-any.whl"
    sha256 "b683bd1b6659ddcd810ff02ad09ba821d4bf1065072805063eb35c49617905ac"
  end

  resource "urllib3" do
    url "https://files.pythonhosted.org/packages/92/9d/c4e665119135114480843e7ab388fa94d8480650450e6f8e26b70d323a4c/urllib3-2.8.0-py3-none-any.whl"
    sha256 "0cf3cae568d36aa9576b28dfb35f11328f1cb974ca7647d9475ebb86c75ac6e3"
  end

  resource "uvicorn" do
    url "https://files.pythonhosted.org/packages/38/0c/b54a4fdd7f90a3af8b02ebc9ce6712c2c208b7926a2f7bad95c33ebbe943/uvicorn-0.54.0-py3-none-any.whl"
    sha256 "505bdb0f318731d45f1f712071fc781a8981f6847a31c902c9f5e652d4f67faf"
  end

  resource "uvloop" do
    url "https://files.pythonhosted.org/packages/5f/83/eb980d64e6dd5da46d4dc35755fa6afd6b5b47141437cf89615f1117c5a6/uvloop-0.23.0-cp313-cp313-macosx_10_13_universal2.whl"
    sha256 "2dcff2d69be43e6559e5dad2c5a7a2dbfb60e05a77311b6c4b7a4a8123d86c65"
  end

  resource "watchfiles" do
    url "https://files.pythonhosted.org/packages/c7/8a/894799b485fe9473ad10422a0d9668e53fb78e3a2b5cc6061159572844ad/watchfiles-1.3.0-cp310-abi3-macosx_11_0_arm64.whl"
    sha256 "bbc1198edfdc90fda0600f825aa94150f428dfcbf8138746f55998e0e660d64c"
  end

  resource "websocket-client" do
    url "https://files.pythonhosted.org/packages/d5/d2/cc4dc1271e464942db7ee278baae2daa99ee77cb2af744025c04da585a3e/websocket_client-1.9.2-py3-none-any.whl"
    sha256 "e1a673830a9c7bfa47b1cd3d5e4178f4c9651d80a4eab02c9c23a1c3ec6250ce"
  end

  resource "websockets" do
    url "https://files.pythonhosted.org/packages/16/df/ab0ed0445d8d81a133c8fed760f3b4f743a2f563804fdbecb2c32fe05d94/websockets-17.3-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "fe7a168d805c8a7bf1c2de55b7ab2bc958586bfd88d0e87235a92cac66b590a7"
  end

  resource "xlsxwriter" do
    url "https://files.pythonhosted.org/packages/ce/9d/3d30eaaa7e06bdb6172f938b2661d64f8d899bdca54120384ecf3759488a/xlsxwriter-3.3.0-py3-none-any.whl"
    sha256 "5f04772d43281da5109af66f39b65153359a203e896847c416476e709b024469"
  end

  resource "yarl" do
    url "https://files.pythonhosted.org/packages/a8/aa/50acc5c3e5da04172ae3c281c75405af4d2ca911e16120ab0563f4dffb66/yarl-1.25.1-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "0f12afda4eea8c8994a76d4df1875c765194f5fbe8a9d197929ea303caee29ec"
  end

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match "ingest", shell_output("#{bin}/rag --help")
  end
end

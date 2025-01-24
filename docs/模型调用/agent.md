<!-- markdownlint-disable MD033 -->
<!-- markdownlint-disable MD001 -->
# 后端开发文档-模型调用模块 `src.agent`

## 文件结构

<details>

<summary>文件结构树</summary>

```plaintext
src/agent
├── __init__.py
├── agents.py
├── core.py
├── settings.py
├── data
│   ├── db
│   │   ├── chroma.sqlite3
│   │   ├── comic.csv
│   │   ├── data.json
│   │   └── wiki.db
│   ├── pattern
│   │   ├── cf.yaml
│   │   ├── comic.yaml
│   │   ├── test.yaml
│   │   └── three.yaml
│   ├── prompt
│   │   ├── comic.yaml
│   │   ├── nolm.yaml
│   │   ├── prompt.yaml
│   │   └── utils.yaml
│   └── saves
├── llmchain
│   ├── __init__.py
│   ├── memory.py
│   ├── models.py
│   ├── parser.py
│   ├── prompt.py
│   └── utils.py
├── load_tools.py
├── pllm
│   ├── __init__.py
│   ├── adapter.py
│   ├── module.py
│   ├── predict
│   │   └── predict.py
│   ├── signature
│   │   ├── __init__.py
│   │   ├── field.py
│   │   └── signature.py
│   └── t.py
└── retriever
    ├── __init__.py
    ├── core.py
    └── fileparse.py
```

</details>

<details>

<summary>结构图</summary>

[![agent](https://mermaid.ink/img/pako:eNpllMlu2zAQQH_F4DnRpTcXKKDEu50uSdeMhYKVxhZbLio5cmEE-fcyouyIpg6E-B45Q4pDPbHSVMjGbG95U48292-3euSfHPgeNRWj6-t3o5vQcVlzLHrd8VsojcVLOgFpePWTjJHJjClUnHiEZiClKmsudITn0HgeoQVYJCvwgLbnoZ12dgnVr2IIVtBwIrQ6omtorFENRXADjh_QRVGXnbmDsvbjeeb-SkH4phjK937_SpRZ6Q4R_9DtMvvtjI74R_gn_ojsvM7Qrjr3CcpdduRKFkN632dIxAMQOkr5Z6DaIg5FaNed_pLGC-IraCNVyr_1nys136El4U84yTTr9A9Aa1-PP8BHUKiMPV7yPAfly3BYL7248YdoHSaR8tvTws4itPOgJ8Ar3tBwYm-mL6laiYmY-YhYiZJiPAcn9ppTazEWC0iS5_3ilqdQg_rvZ61gJ1BWqVi_5rkMuwgjNhfXrcd3PqLE7jMFx66YQqu4qPy9fnoZu2VUo8ItG_vXCne8lbRlW_3sh_KWzMNRl2xMtsUrZk27r9l4x6XzvbbxpYwTwf3_QZ1pw_WjMaf-83-mTkDp?type=png)](https://mermaid-js.github.io/mermaid-live-editor/edit#pako:eNpllMlu2zAQQH_F4DnRpTcXKKDEu50uSdeMhYKVxhZbLio5cmEE-fcyouyIpg6E-B45Q4pDPbHSVMjGbG95U48292-3euSfHPgeNRWj6-t3o5vQcVlzLHrd8VsojcVLOgFpePWTjJHJjClUnHiEZiClKmsudITn0HgeoQVYJCvwgLbnoZ12dgnVr2IIVtBwIrQ6omtorFENRXADjh_QRVGXnbmDsvbjeeb-SkH4phjK937_SpRZ6Q4R_9DtMvvtjI74R_gn_ojsvM7Qrjr3CcpdduRKFkN632dIxAMQOkr5Z6DaIg5FaNed_pLGC-IraCNVyr_1nys136El4U84yTTr9A9Aa1-PP8BHUKiMPV7yPAfly3BYL7248YdoHSaR8tvTws4itPOgJ8Ar3tBwYm-mL6laiYmY-YhYiZJiPAcn9ppTazEWC0iS5_3ilqdQg_rvZ61gJ1BWqVi_5rkMuwgjNhfXrcd3PqLE7jMFx66YQqu4qPy9fnoZu2VUo8ItG_vXCne8lbRlW_3sh_KWzMNRl2xMtsUrZk27r9l4x6XzvbbxpYwTwf3_QZ1pw_WjMaf-83-mTkDp)
</details>

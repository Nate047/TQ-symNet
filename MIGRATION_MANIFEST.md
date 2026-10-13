# 独立项目迁移清单

迁移整个外层TQ_SymNet目录即可；不要只迁移内层同名Python包。
不再依赖或覆盖原项目的mmdet、tools及NAM单文件。

| 路径 | 用途 |
|---|---|
| mmdet/ | 当前训练、数据处理、评测、Sym/TransNeXt源码 |
| TQ_SymNet/ | 拆分模型、质量网、checkpoint审计 |
| tools/ | 训练/测试/debug/online导出入口 |
| configs/ | 自包含配置链及消融开关 |
| requirements/、setup.py、setup.cfg、LICENSE.md | 打包、环境说明、许可证 |
| PROVENANCE.json、OFFICIAL_SYM_TEST.json、STANDALONE_SOURCE_MANIFEST.json | 保留实现与核心源码哈希 |
| RUN_CORE_SINGLE_SEED.md | 当前实验命令 |

不要迁移__pycache__、Windows扩展、conda_env和实验输出。
外部图片、JSON和权重仍使用cfg内绝对路径；沿用服务器mmcv-full及swattention扩展。
不读取肺mask。

```bash
cd /实际目录/TQ_SymNet
python tools/check_structure.py
python tools/check_standalone.py --runtime
python tools/preflight.py --server-files --build
```

历史NAM源码仅作可选数值对照，不在TQ运行链自动导入；本次未更改原项目源码。

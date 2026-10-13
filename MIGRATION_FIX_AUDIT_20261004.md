# 简洁独立版故障核对与修复

## 已复现的移植缺陷

1. `TQ_SymNet/models/selection.py::hard_select`：初始循环上限按原候选数确定，IoU抑制可以删掉全部剩余候选，但下一轮未判断空集合便argmax。三个相同框即可稳定复现服务器的IndexError。旧检测器的原head NMS不走这段新控制组实现。
2. `tools/check_structure.py`：本机3.12的ast.dump哈希含type_params字段，不能直接与服务器3.8的dump比较。并非源码字节哈希。当前改为校验归档源文件固定SHA，然后在当前解释器下对比16个保留方法。没有删除完整性检查，也不执行归档模块。
3. 邻近边界：Hard/Soft-NMS的topk=0原先会返回不一致或非空结果。已统一为空索引、空分数，并加入框/分数形状检查。

## 未改动与已验证

- 没改分类分、NMS IoU阈值、融合公式、候选预算、坐标、主干或CrossScale；NMS只增加边界保护。
- 当前主检测器16个保留方法与归档源AST一致；训练API、数据处理、EMA、COCO评测、Sym注意力、位置编码、TransNeXt及确定性工具的SHA检查通过。
- CrossScale、FDR解码和质量网组件的CPU/CUDA初始化、前向/梯度数值检查通过；组件开关、空框及checkpoint隔离检查通过。
- Hard-NMS与torchvision原生NMS的随机用例比较；独立回归覆盖输入为空、抑制后为空、仅单框、多个簇、topk=0/1及形状错误。
- 不代表在服务器PyTorch1.10/MMCV1.7环境完成了全模型/真实checkpoint/完整val验证。本机未安装MMCV；仍需服务器跑完整验证。

## 额外审查边界

- 新M2的纯TB hard-NMS与旧head的分类别NMS不是同一推理模式，这是现有对照设计而非字节级旧版重放；不要预设两者能得到相同历史AP。
- 两阶段A组若不使用命令行覆盖初始化hook，默认仍会取历史D3底座。完整复现须用本次M2权重。
- 原COCO评测器仍有全数据集预测为空时loadRes([])的遗留风险，不是本次Hard-NMS堆栈的原因。本次未修改它，也没有用造框或topK回填掩盖空候选。
- 主网络EMA断点恢复应同时指定runner的resume和EMA hook的resume_from，保证注册EMA buffers后再次恢复完整权重。无需改EMA更新公式。

## 迁移文件

必需：`TQ_SymNet/models/selection.py`、`tools/check_structure.py`。
回归工具：`tools/check_nms_regression.py`、`tools/check_structure_regression.py`。
来源文件及PROVENANCE不变；继续保留独立项目内已打包的归档源。

```bash
python tools/check_structure.py
python tools/check_structure_regression.py
python tools/check_nms_regression.py
```

# STATUS

## 当前目标

在 `D:\DataPrep` 中完成两条线：

1. 先把现有项目文档体系补齐，让非机器学习背景也能看懂代码、功能、环境和使用方式
2. 再把 `papers/FATE_SIGIR25` 和 `papers/DARN_VLDB25` 梳理清楚，并设计它们整合进 `tabular/imputation` 的路线

## 当前进度

已经完成：

1. 阅读并遵循 `AGENTS.md` 和 `C:\Users\Administrator\.codex\RTK.md`
2. 梳理仓库结构、现有 `imputation/detection/correction` 模块、`main.py`、`index.html`
3. 确认当前 `imputation` 已接入算法为 `GAIN`、`VAEGAIN`、`SCIS`
4. 确认 `FATE` 原始代码主任务是“缺失数据上的公平分类”，不是纯补全器
5. 确认 `DARN` 原始代码主任务是“联邦不完整表格预测”，不是纯单机补全器
6. 已补完整套项目文档，并接入 README 导航

## 已新增文档

位于 `docs/`：

1. `REQUIREMENTS_SPEC.md`
2. `DESIGN_SPEC.md`
3. `CODE_ANALYSIS.md`
4. `INSTALLATION_GUIDE.md`
5. `USER_GUIDE.md`
6. `DOCUMENT_INDEX.md`
7. `FATE_DARN_INTEGRATION_PLAN.md`

其中最关键的两份：

1. `docs/CODE_ANALYSIS.md`
   说明现有代码结构、功能、关键入口、重要文件，并补了类图/时序图
2. `docs/FATE_DARN_INTEGRATION_PLAN.md`
   说明 FATE/DARN 原始代码在做什么、和当前 `imputation` 的差异、推荐整合路线

## 当前关键结论

1. 不能把 `papers/FATE_SIGIR25/code/run.py` 原样塞进 `imputation`
2. 不能把 `papers/DARN_VLDB25/DARN-main/train_DARN.py` 原样塞进 `imputation`
3. 正确路线是抽取它们内部的“缺失感知编码 + 重构/补全能力”，重写成 `BaseImputer` 风格
4. 推荐先做 `FATE` 的最小补全版，再做 `DARN` 的最小补全版

## 推荐下一步

下一窗口优先做：

1. 先决定 `FATE` 第一版是否只支持数值特征
2. 然后开始落 `tabular/imputation/FATE.py`
3. 再落 `tabular/imputation/FATE_modules.py`
4. 第一版先以最小可运行为目标，不要先碰 `DARN`

## 一句话恢复现场

可直接用这句话恢复：

“继续 `DataPrep`：文档体系和 FATE/DARN 整合专项已完成，下一步开始实现 `FATE` 的最小补全版，按 `BaseImputer` 接口接到 `tabular/imputation`。”


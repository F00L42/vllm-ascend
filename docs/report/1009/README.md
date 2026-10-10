# Jev-like 工程进展材料

- [HTML 演示](index.html)：七个实质页面，含 4B / 27B 性能规格对照，支持左右键翻页、全屏、D/G 批次阶段切换与原图查看。可双击离线打开，也可单独复制该文件演示。
- [Markdown 事实源](jev_like_engineering_progress.md)：完整报告、验证范围和提交链接。
- [事实源索引](source_index.json)：源文件 SHA-256、章节行号与引用链接。

HTML 主画面采用提炼后的短句与示意图；每页的“依据与补充”关联对应 Markdown 章节。原始图片、完整 Markdown 和索引数据均内嵌于 HTML。

## 更新材料

事实变更先更新 Markdown，再同步 [演示模板](assets/demo.template.html) 中的提炼文案与图示，最后重新构建：

```powershell
python build_demo.py
```

[构建脚本](build_demo.py) 刷新内嵌事实源、章节索引和原图，输出 `index.html` 与 `source_index.json`；演示文案需要人工随事实更新。

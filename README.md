# KnowledgePilot 企业知识库 Agent

KnowledgePilot是一个以Python为主开发的可执行RAG Agent。系统支持PDF、Markdown和TXT文档导入、本地向量检索、DeepSeek回答生成、模型意图路由、知识覆盖判断、联网补充、引用溯源、Agent执行轨迹和历史会话持久化。

项目内置员工手册和费用报销制度，因此安装完成后无需额外准备知识库文件即可演示。

## 功能

- 上传和删除PDF、Markdown、TXT知识库文件
- 自动解析、切片并建立本地TF-IDF字符向量索引
- 每个问题先由DeepSeek判断闲聊、项目身份、澄清或知识查询
- DeepSeek根据检索片段判断知识库能否完整回答
- 知识库不足时分别展示“企业知识库内容”和“互联网补充内容”
- 企业文档来源与互联网来源严格隔离
- 展示引用文档、页码、相关度和原文
- 展示Agent路由与工具执行轨迹
- 使用SQLite持久化历史会话


## 知识查询顺序：

```text
DeepSeek意图识别
  → 本地知识库向量检索
  → DeepSeek判断知识覆盖度并生成知识库部分
  → 知识库不足时由DeepSeek Tool Calls生成结构化搜索计划
  → Tavily按查询词、地域、权威域名和相关度阈值联网搜索
  → 首轮无合格结果时由DeepSeek改写搜索词并重试一次
  → 独立生成互联网补充部分
  → 分区展示两组内容和来源
```

## 项目结构

```text
RAG/
├── app.py
├── requirements.txt
├── .env.example
├── data/knowledge_base/
├── scripts/smoke_test.py
├── src/knowledge_pilot/
└── tests/
```

## 环境要求

- Windows 10/11
- Python 3.11 或更高版本
- DeepSeek API Key（可被替换）
- Tavily Search API Key（互联网搜索）

## 安装

安装虚拟环境以及相关依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.venv/Scripts/python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 启动

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

浏览器访问：

```text
http://localhost:8501
```

## 演示问题

- 正式员工每年有多少天年假？
- 一线城市的住宿报销上限是多少？
- 出差结束后应该在什么时候提交报销？
- 账号密码长度有什么要求？
- 发现办公电脑丢失后应该怎么处理？

## 数据和隐私

- 上传文件保存在 `data/knowledge_base/uploads/`
- 本地向量索引保存在 `data/index/`
- 历史会话保存在 `data/app.db`
- 运行日志保存在 `logs/app.log`
- DeepSeek 与 Tavily Key 只从服务器端 `.env` 读取，不会发送到前端组件

文档内容在使用DeepSeek生成回答时会作为上下文发送到 DeepSeek API。
处理敏感企业资料前，应确认组织的数据安全政策。

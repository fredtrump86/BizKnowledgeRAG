# KnowledgePilot 企业知识库 Agent

KnowledgePilot 是一个以 Python 为主开发的可执行 RAG Agent。系统支持
PDF、Markdown 和 TXT 文档导入、本地向量检索、DeepSeek 回答生成、
模型意图路由、知识覆盖判断、联网补充、引用溯源、Agent 执行轨迹和
历史会话持久化。

项目内置员工手册和费用报销制度，因此安装完成后无需
额外准备知识库文件即可演示。

## 功能

- 上传和删除 PDF、Markdown、TXT 知识库文件
- 自动解析、切片并建立本地 TF-IDF 字符向量索引
- 每个问题先由 DeepSeek 判断闲聊、项目身份、澄清或知识查询
- DeepSeek 根据检索片段判断知识库能否完整回答
- 知识库不足时分别展示“企业知识库内容”和“互联网补充内容”
- 企业文档来源与互联网来源严格隔离
- 通过 DeepSeek API 生成简体中文回答
- DeepSeek 不可用时自动降级为本地抽取式回答
- 展示引用文档、页码、相关度和原文
- 展示 Agent 路由与工具执行轨迹
- 使用 SQLite 持久化历史会话
- 包含单元测试和离线冒烟测试

## Agent 路由

正常业务路由包括：

- `smalltalk`：问候、感谢和简单寒暄
- `identity`：介绍 KnowledgePilot 的身份和能力
- `clarification`：问题为空或需要用户补充
- `knowledge_base`：知识库足以完整回答
- `knowledge_plus_web`：知识库不足，使用互联网补充
- `knowledge_base_limited`：知识库不足且联网关闭或失败

知识查询的执行顺序：

```text
DeepSeek 意图识别
  → 本地知识库向量检索
  → DeepSeek 判断知识覆盖度并生成知识库部分
  → 知识库不足时由 DeepSeek Tool Calls 生成结构化搜索计划
  → Tavily 按查询词、地域、权威域名和相关度阈值联网搜索
  → 首轮无合格结果时由 DeepSeek 改写搜索词并重试一次
  → 独立生成互联网补充部分
  → 分区展示两组内容和来源
```

## 项目结构

```text
RAG/
├── app.py
├── requirements.txt
├── DEPENDENCIES.md
├── .env.example
├── data/knowledge_base/
├── scripts/smoke_test.py
├── src/knowledge_pilot/
└── tests/
```

## 环境要求

- Windows 10/11
- Python 3.11 或更高版本
- DeepSeek API Key（可选；不填写也可以运行）
- Tavily Search API Key（启用互联网搜索时需要）

## 安装

在项目目录打开终端：

```bash
python -m venv .venv
```

PowerShell 激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

安装相关依赖

```powershell
.venv/Scripts/python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

各依赖包的用途、安装方式和外部服务要求见

## 配置 DeepSeek

复制配置模板：

```powershell
Copy-Item .env.example .env
```

编辑 `.env`：

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=你的DeepSeek_API_Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

不要将 `.env` 文件或 API Key 提交到 Git 仓库。

DeepSeek API Key 仅从服务器端 `.env` 读取，不会在网页中显示或提供输入框。

## 配置联网搜索

联网检索使用 Tavily Search API。把 Tavily Key 写入 `.env`：

```env
TAVILY_API_KEY=你的Tavily_API_Key
TAVILY_BASE_URL=https://api.tavily.com
TAVILY_SEARCH_DEPTH=basic
```

Tavily API Key 仅从服务器端 `.env` 读取，不会在网页中显示或提供输入框。
DeepSeek Key 负责模型生成，Tavily Key 负责网页搜索，两者不能互相替代。

## 启动

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

浏览器访问：

```text
http://localhost:8501
```

安装完成后，也可以双击 `run.bat` 启动。

## 演示问题

- 正式员工每年有多少天年假？
- 一线城市的住宿报销上限是多少？
- 出差结束后应该在什么时候提交报销？
- 账号密码长度有什么要求？
- 发现办公电脑丢失后应该怎么处理？

## 检索设计

默认检索器使用字符 2～4 gram 的 TF-IDF 向量，能够在不下载额外模型的
情况下处理中英文混合内容。它保证项目开箱可运行，但不等同于神经网络
语义向量。

后续如需增强语义召回能力，可以保持 `LocalVectorRetriever` 的调用接口
不变，将内部实现替换为 BGE Embedding、ChromaDB、pgvector 或 Milvus。

## 联网搜索

知识库信息不足时，Agent 调用 Tavily Search API 获取真实网页结果。
在调用 Tavily 前，DeepSeek 通过 Tool Calls 把用户原问题和知识库缺失点
转换为结构化搜索计划，包括具体查询词、搜索深度、国家、可信域名和最低
相关度。法律、政策和合规问题会优先检索政府权威来源。低于相关度阈值的
页面不会进入回答；首轮没有合格结果时，DeepSeek 会改写查询并重试一次。

如果没有配置 Tavily Key、DeepSeek 搜索计划失败或接口调用异常，系统会
使用保留原问题上下文的安全查询降级；仍无合格结果时会明确说明搜索失败，
不再使用 Wikipedia 等低质量结果填充。用户可以在页面侧边栏关闭联网搜索。

互联网搜索结果与企业知识库引用会在界面中明确区分。

## 数据和隐私

- 上传文件保存在 `data/knowledge_base/uploads/`
- 本地向量索引保存在 `data/index/`
- 历史会话保存在 `data/app.db`
- 运行日志保存在 `logs/app.log`
- DeepSeek 与 Tavily Key 只从服务器端 `.env` 读取，不会发送到前端组件

文档内容在使用 DeepSeek 生成回答时会作为上下文发送到 DeepSeek API。
处理敏感企业资料前，应确认组织的数据安全政策。

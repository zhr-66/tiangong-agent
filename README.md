# chenguang-agent-

#### 介绍
《天宫医疗-智能体》集中于解决医疗领域的 Agent 业务构建。包括：
1. 智慧问诊 Agent
2. 报告解读 Agent
3. 药物 Agent
4. 知识文档 Agent
5. 运营数据 Agent

#### 软件架构
软件架构说明


#### 安装教程

```sh
# 创建环境
conda create -n tiangong python=3.13
# 激活环境
conda activate tiangong

# 安装依赖
pip install -r requirements.txt
```


## 指定端口启动
```
uvicorn src.main:app --port 8080 --reload
```

## env配置
把 .env.example 复制一份 叫 .env ，修改为自己的信息即可

## 启动docker
docker compose -f docker-compose.yml up -d

## 连接postgreSQL
Host：localhost  
Port：5432  
User：medical  
pwd：medical123  
Database：medical_db

## 连接Neo4j
浏览器：http://localhost:7474  
Connect URL：bolt://localhost:7687  
User：neo4j  
Password：medical123

## Redis 连接信息
Host：localhost  
Port：6379  
Password：空（无密码）  
Database：0
访问地址：http://localhost:8001


## 完整启动步骤

本项目是 FastAPI 单体架构，前端 HTML 由后端托管，**只需启动一个服务**，无需前后端分别启动。

### 步骤 1：启动基础设施服务

确认 docker-compose 里的服务都在运行（PostgreSQL / Redis / Milvus / Neo4j / MinIO）：

```bash
# 查看服务状态
docker compose ps

# 未启动时启动
docker compose -f docker-compose.yml up -d
```

### 步骤 2：配置环境变量

把 `.env.example` 复制为 `.env`，填入自己的 API Key 等信息：

```bash
cp .env.example .env
```

关键字段：
- `DEEPSEEK_API_KEY` — DeepSeek 大模型 API Key
- `DASHSCOPE_API_KEY` — 阿里云 DashScope 向量化模型 API Key
- `BASE_URL_CHAT` — 大模型 API 地址（默认 https://api.deepseek.com）

### 步骤 3：安装 Python 依赖（首次运行）

```bash
# 创建并激活 conda 环境
conda create -n tiangong python=3.13
conda activate tiangong

# 安装依赖
pip install -r requirements.txt
```

### 步骤 4：初始化 Milvus 症状向量索引（首次运行）

智慧问诊依赖 `symptom_index` 向量索引，首次部署时执行一次：

```bash
python scripts/init_symptom_index.py
```

> 之后 Neo4j 新增症状节点时，可增量执行此脚本更新索引。

### 步骤 5：启动 FastAPI 服务（后端 + 前端托管）

在项目根目录执行：

```bash
# 默认端口 8000
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# 或指定其他端口（如 8080）
uvicorn src.main:app --port 8080 --reload

E:\conda\miniConda\envs\tiangong\python.exe -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

参数说明：
- `src.main:app` — 指向 `src/main.py` 里的 `app` 对象
- `--reload` — 代码改动自动重启（开发模式）
- `--host 0.0.0.0` — 允许外部访问
- `--port 8000` — 服务端口

### 步骤 6：访问前端聊天界面

浏览器打开：

```
http://localhost:8000/
```

前端会通过相对路径调用后端接口：
- `/` — 返回聊天界面首页（`index.html`）
- `/static/*` — 静态资源
- `/api/v1/chat` — 非流式对话接口
- `/api/v1/chat/stream` — 流式对话接口（SSE）

### 架构说明

```
浏览器  ──HTTP──>  FastAPI(:8000)
                    ├── /                  → 返回 index.html（前端页面）
                    ├── /static/*          → 静态资源
                    ├── /api/v1/chat       → 对话接口
                    └── /api/v1/chat/stream → 流式对话接口
```

前端 HTML 文件由 FastAPI 通过 `StaticFiles` 和 `FileResponse` 直接托管，属于同一个服务，**不需要单独启动前端**。

### 接口测试（可选）

不打开浏览器，用 curl 测试：

```bash
# 非流式
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user_001","session_id":"session_001","message":"我最近头很疼，还有点发烧"}'

# 流式（SSE）
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user_001","session_id":"session_001","message":"我最近头很疼，还有点发烧"}'
```

### 常见问题

**Q: 为什么不能直接双击 `index.html` 打开？**
A: 双击是 `file://` 协议，前端用相对路径调 `/api/v1/chat/stream` 会变成 `file:///api/v1/chat/stream`，找不到后端。必须从 `http://localhost:8000/` 访问。

**Q: 启动报错 `ModuleNotFoundError: No module named 'langgraph'`？**
A: 确认已激活 `tiangong` conda 环境（`conda activate tiangong`），且已执行 `pip install -r requirements.txt`。base 环境没装项目依赖。

**Q: 端口 8000 被占用？**
A: 换个端口启动：`uvicorn src.main:app --port 8080 --reload`，然后访问 `http://localhost:8080/`。

**Q: 前端改动不生效？**
A: 前端是纯静态文件，**刷新浏览器即可**（Ctrl+F5 强制刷新）。后端 Python 改动会触发 `--reload` 自动重启。

**Q: 启动后 Milvus / Neo4j 连接失败？**
A: 检查 docker 服务是否正常运行：`docker compose ps`。访问 `http://localhost:8000/health/deps` 查看各依赖状态。




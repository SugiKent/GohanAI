# AIエージェントプロジェクト初期化ガイド

このガイドでは、LangChainとLangGraphを使ったAIエージェントをFastAPIでWeb API化する方法について、初期化から実装までを解説します。

## 目次

1. プロジェクト概要
2. 開発環境のセットアップ
3. プロジェクト構造の作成
4. 各ファイルの実装
5. Docker Composeの設定
6. プロジェクトの実行と動作確認

## 1. プロジェクト概要

このプロジェクトでは、LangManusの構造を参考にした自律型AIエージェントをFastAPIを使ってWeb API化します。エージェントは以下の要素で構成されます：

- **Coordinator**: 初期インタラクションを処理し、タスクをルーティング
- **Planner**: タスクを分析し、実行戦略を作成
- **Supervisor**: 他のエージェントの実行を監督・管理
- **Researcher**: 情報を収集・分析
- **Coder**: コードの生成・修正
- **Reporter**: ワークフロー結果のレポートや要約を生成

## 2. 開発環境のセットアップ

### 必要なツール

- Docker と Docker Compose
- エディタ（VS Code推奨）
- Git（バージョン管理用）

### Dockerのインストール

まだインストールしていない場合は、[Docker公式サイト](https://www.docker.com/products/docker-desktop/)からDocker Desktopをインストールしてください。Docker Desktopには通常Docker Composeも含まれています。

## 3. プロジェクト構造の作成

以下のコマンドを実行して、プロジェクトの基本構造を作成します：

```bash
# プロジェクトディレクトリを作成
mkdir langmanus-fastapi
cd langmanus-fastapi

# 必要なディレクトリを作成
mkdir -p app/agents app/graph app/prompts app/utils
mkdir -p tests
```

次に、必要なファイルを作成します：

```bash
# 主要なファイルを作成
touch docker-compose.yml Dockerfile requirements.txt .env
touch app/__init__.py app/main.py
touch app/agents/__init__.py
touch app/graph/__init__.py
touch app/utils/__init__.py
```

各エージェント用のファイルを作成：

```bash
touch app/agents/coordinator.py
touch app/agents/planner.py
touch app/agents/supervisor.py
touch app/agents/researcher.py
touch app/agents/coder.py
touch app/agents/reporter.py
```

プロンプトファイルを作成：

```bash
touch app/prompts/coordinator.md
touch app/prompts/planner.md
touch app/prompts/supervisor.md
touch app/prompts/researcher.md
touch app/prompts/coder.md
touch app/prompts/reporter.md
```

グラフ定義ファイルを作成：

```bash
touch app/graph/workflow.py
```

## 4. 各ファイルの実装

### requirements.txt

```
fastapi==0.110.0
uvicorn==0.27.1
langchain==0.1.12
langchain_core==0.1.32
langchain_community==0.0.26
langgraph==0.0.27
python-dotenv==1.0.1
pydantic==2.6.1
```

### .env

```
# LLMの設定
LLM_MODEL=gpt-3.5-turbo
OPENAI_API_KEY=your_openai_api_key_here
```

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
version: '3'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - .:/app
    env_file:
      - .env
```

### app/main.py

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

from app.graph.workflow import build_graph

# 環境変数を読み込む
load_dotenv()

app = FastAPI(title="LangManus API", description="LangGraph AIエージェントAPI")

class Message(BaseModel):
    role: str
    content: str

class AgentRequest(BaseModel):
    messages: List[Message]

class AgentResponse(BaseModel):
    result: Dict[str, Any]

@app.post("/agent", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    try:
        # グラフの構築
        graph = build_graph()
        
        # リクエストメッセージをdict形式に変換
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # グラフを実行
        result = graph.invoke({
            "messages": messages
        })
        
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "LangManus AIエージェントAPIへようこそ！"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
```

### app/graph/workflow.py

```python
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from typing import Dict, List, Literal, TypedDict, Annotated, Union
import os
from jinja2 import Environment, FileSystemLoader
from datetime import datetime

# プロンプトを読み込むための設定
env = Environment(loader=FileSystemLoader("app/prompts"))

# エージェントの状態を表現するクラス
class AgentState(TypedDict):
    messages: List[Dict[str, str]]
    next: str
    full_plan: Optional[str]

def load_prompt(template_name: str, **kwargs):
    """テンプレートからプロンプトを生成する"""
    template = env.get_template(f"{template_name}.md")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return template.render(current_time=current_time, **kwargs)

def get_llm():
    """LLMモデルを取得する"""
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
        temperature=0.2,
    )

# 各エージェントノードの定義
def coordinator_node(state: AgentState):
    """初期のインタラクションを処理し、タスクをルーティングする"""
    llm = get_llm()
    
    # 最後のメッセージを取得
    last_message = state["messages"][-1]["content"]
    
    # プロンプトの読み込み
    prompt = load_prompt("coordinator", query=last_message)
    
    # LLMに問い合わせ
    response = llm.invoke(prompt)
    
    # 簡単な会話かタスクかを判断
    if "TASK" in response.content:
        # タスクとして処理
        return {"messages": state["messages"] + [{"role": "assistant", "content": response.content}], "next": "planner"}
    else:
        # 簡単な会話として処理
        return {"messages": state["messages"] + [{"role": "assistant", "content": response.content}], "next": END}

def planner_node(state: AgentState):
    """タスクを分析し、実行戦略を作成する"""
    llm = get_llm()
    
    # メッセージ履歴を取得
    messages = state["messages"]
    
    # プロンプトの読み込み
    prompt = load_prompt("planner", messages=messages)
    
    # LLMに問い合わせ
    response = llm.invoke(prompt)
    
    # 計画を保存
    full_plan = response.content
    
    return {
        "messages": state["messages"] + [{"role": "assistant", "content": full_plan}],
        "full_plan": full_plan,
        "next": "supervisor"
    }

def supervisor_node(state: AgentState):
    """他のエージェントの実行を監督・管理する"""
    llm = get_llm()
    
    # 実行計画と現在の状態を取得
    full_plan = state.get("full_plan", "")
    messages = state["messages"]
    
    # プロンプトの読み込み
    prompt = load_prompt("supervisor", full_plan=full_plan, messages=messages)
    
    # LLMに問い合わせ
    response = llm.invoke(prompt)
    
    # 次のエージェントを決定
    content = response.content
    
    if "RESEARCHER" in content:
        next_agent = "researcher"
    elif "CODER" in content:
        next_agent = "coder"
    elif "REPORTER" in content:
        next_agent = "reporter"
    else:
        next_agent = END
    
    return {
        "messages": state["messages"] + [{"role": "assistant", "content": content}],
        "next": next_agent
    }

def researcher_node(state: AgentState):
    """情報を収集・分析する"""
    llm = get_llm()
    
    # メッセージ履歴を取得
    messages = state["messages"]
    
    # プロンプトの読み込み
    prompt = load_prompt("researcher", messages=messages)
    
    # LLMに問い合わせ
    response = llm.invoke(prompt)
    
    return {
        "messages": state["messages"] + [{"role": "assistant", "content": response.content}],
        "next": "supervisor"
    }

def coder_node(state: AgentState):
    """コードの生成・修正を行う"""
    llm = get_llm()
    
    # メッセージ履歴を取得
    messages = state["messages"]
    
    # プロンプトの読み込み
    prompt = load_prompt("coder", messages=messages)
    
    # LLMに問い合わせ
    response = llm.invoke(prompt)
    
    return {
        "messages": state["messages"] + [{"role": "assistant", "content": response.content}],
        "next": "supervisor"
    }

def reporter_node(state: AgentState):
    """ワークフローの結果をレポートや要約として生成する"""
    llm = get_llm()
    
    # メッセージ履歴を取得
    messages = state["messages"]
    
    # プロンプトの読み込み
    prompt = load_prompt("reporter", messages=messages)
    
    # LLMに問い合わせ
    response = llm.invoke(prompt)
    
    return {
        "messages": state["messages"] + [{"role": "assistant", "content": response.content}],
        "next": END
    }

def build_graph():
    """エージェントワークフローのグラフを構築する"""
    # グラフの初期化
    builder = StateGraph(AgentState)
    
    # ノードの追加
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("planner", planner_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("coder", coder_node)
    builder.add_node("reporter", reporter_node)
    
    # エッジの設定
    builder.add_edge("coordinator", "planner")
    builder.add_edge("planner", "supervisor")
    builder.add_edge("supervisor", "researcher")
    builder.add_edge("supervisor", "coder")
    builder.add_edge("supervisor", "reporter")
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("coder", "supervisor")
    builder.add_edge("reporter", END)
    
    # 条件付きエッジの設定
    builder.add_conditional_edges(
        "coordinator",
        lambda state: state["next"],
        {
            "planner": "planner",
            END: END
        }
    )
    
    builder.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "researcher": "researcher",
            "coder": "coder",
            "reporter": "reporter",
            END: END
        }
    )
    
    # グラフをコンパイル
    return builder.compile()
```

### プロンプトファイル (app/prompts/)

#### coordinator.md

```markdown
# Coordinatorエージェント

現在時刻: {{ current_time }}

あなたはCoordinatorエージェントです。ユーザーからの入力を最初に受け取り、それが対話的な質問であるか、より複雑なタスクであるかを判断します。

- 簡単な挨拶や簡単な質問の場合は、直接回答してください。
- より複雑なタスクや調査、分析、コード生成が必要な場合は、「TASK」というキーワードを含めて、Plannerに渡すべきことを示してください。

ユーザーの入力:
{{ query }}

判断して適切に応答してください。
```

#### planner.md

```markdown
# Plannerエージェント

現在時刻: {{ current_time }}

あなたはPlannerエージェントです。ユーザーからのタスクを分析し、実行戦略を作成します。タスクを小さなステップに分解し、どのエージェント（Researcher、Coder、Reporter）がどのステップを担当するかを決定してください。

これまでの会話履歴:
{% for message in messages %}
{{ message.role }}: {{ message.content }}
{% endfor %}

計画を立ててください。以下の形式で応答してください:

```
# タスク分析
[タスクの分析と理解]

# 実行計画
1. [ステップ1] - [担当エージェント]
2. [ステップ2] - [担当エージェント]
...
```

利用可能なエージェント:
- Researcher: 情報収集と分析を行います
- Coder: コード生成と修正を行います
- Reporter: 最終的な結果をまとめてレポートを作成します
```

#### supervisor.md

```markdown
# Supervisorエージェント

現在時刻: {{ current_time }}

あなたはSupervisorエージェントです。他のエージェントの実行を監督し、次に実行すべきエージェントを決定します。現在の状況を分析し、どのエージェントに次のタスクを割り当てるかを判断してください。

実行計画:
{{ full_plan }}

これまでの会話履歴:
{% for message in messages %}
{{ message.role }}: {{ message.content }}
{% endfor %}

次に実行すべきエージェントを決定してください。以下のキーワードのいずれかを必ず含めてください:
- RESEARCHER: 情報収集・分析が必要な場合
- CODER: コード生成・修正が必要な場合
- REPORTER: タスクが完了し、最終レポートを生成する場合
```

#### researcher.md

```markdown
# Researcherエージェント

現在時刻: {{ current_time }}

あなたはResearcherエージェントです。情報収集と分析を担当します。与えられたトピックについて、事実に基づいた詳細な情報を提供してください。

これまでの会話履歴:
{% for message in messages %}
{{ message.role }}: {{ message.content }}
{% endfor %}

以下の形式で情報を収集・分析した結果を提供してください:

```
# 調査結果
[詳細な情報と分析結果]

# 主要なポイント
- [ポイント1]
- [ポイント2]
...

# 追加調査が必要な項目（もしあれば）
- [項目1]
- [項目2]
...
```
```

#### coder.md

```markdown
# Coderエージェント

現在時刻: {{ current_time }}

あなたはCoderエージェントです。コードの生成や修正を担当します。ユーザーの要求に基づいて、適切なコードを生成してください。

これまでの会話履歴:
{% for message in messages %}
{{ message.role }}: {{ message.content }}
{% endfor %}

以下の形式でコードを提供してください:

```
# コード説明
[コードの目的と機能の説明]

# コード
```[言語]
[コード本体]
```

# 使用方法
[コードの使用方法や注意点]
```
```

#### reporter.md

```markdown
# Reporterエージェント

現在時刻: {{ current_time }}

あなたはReporterエージェントです。これまでの情報を整理し、最終的なレポートや要約を生成します。わかりやすく構造化された形式で情報をまとめてください。

これまでの会話履歴:
{% for message in messages %}
{{ message.role }}: {{ message.content }}
{% endfor %}

以下の形式で最終レポートを作成してください:

```
# 最終レポート

## 概要
[全体的な概要]

## 詳細
[詳細情報をセクションに分けて説明]

## 結論
[最終的な結論や推奨事項]
```
```

## 5. Docker Composeの設定

プロジェクトのセットアップが完了したら、Docker Composeを使って環境を起動します。

```bash
# Docker Composeでコンテナをビルドして起動
docker-compose up --build
```

これにより、FastAPIアプリケーションが起動し、ポート8000でリクエストを待ち受けるようになります。

## 6. プロジェクトの実行と動作確認

### APIの動作確認

ブラウザで http://localhost:8000/docs にアクセスすると、Swagger UIが表示され、APIのドキュメントが確認できます。

### APIへのリクエスト例

curlコマンドを使用してAPIにリクエストを送信する例：

```bash
curl -X 'POST' \
  'http://localhost:8000/agent' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "messages": [
    {
      "role": "user",
      "content": "Pythonを使って簡単なWebスクレイピングツールを作成する方法を教えてください"
    }
  ]
}'
```

または、Pythonを使用する場合：

```python
import requests
import json

url = "http://localhost:8000/agent"
payload = {
    "messages": [
        {
            "role": "user",
            "content": "Pythonを使って簡単なWebスクレイピングツールを作成する方法を教えてください"
        }
    ]
}
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}

response = requests.post(url, headers=headers, data=json.dumps(payload))
print(response.json())
```

## まとめ

これで、LangChainとLangGraphを使ったAIエージェントのFastAPIによるWeb API化の初期セットアップが完了しました。このプロジェクトでは、CoordinatorからReporterまでの複数のエージェントが連携して動作するシステムを構築しました。

プロジェクトの理解が深まったら、以下の拡張も検討してみてください：

1. エージェントの機能拡張
2. UI層の追加（StreamlitやReactなど）
3. エージェント間の通信の最適化
4. 認証機能の追加

Happy coding!
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
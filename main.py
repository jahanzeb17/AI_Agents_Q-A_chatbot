from dotenv import load_dotenv
import os
from uuid import uuid4
import streamlit as st
from IPython.display import Image, display
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore


load_dotenv()


llm = ChatGroq(model="gemma2-9b-it")

class State(TypedDict):
    question: str
    answer: str
    context: Annotated[list, operator.add]


def search_web(state: State):

    """Retrieve docs from web search"""
    question = state["question"]

    # search
    tavily_search = TavilySearchResults(max_results=3)
    search_docs = tavily_search.invoke(question)

    # format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context":[formatted_search_docs]}


def search_wikipedia(state):

    """Retrieve docs from wikipedia"""

    question = state["question"]

    # search
    search_docs = WikipediaLoader(query=question,load_max_docs=2).load()

    # format
    formatted_searh_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context":[formatted_searh_docs]}


def generate_answer(state):

    """Node to answer a question"""

    context = state["context"]
    question = state["question"]

    # Template
    answer_template = """Answer the question {question} using this context {context}: """
    answer_instruction = answer_template.format(question=question,context=context)

    answer = llm.invoke([SystemMessage(content=answer_instruction)] + [HumanMessage(content="Answer the question")])

    return {"answer": answer}



builder = StateGraph(State)

# Add Nodes
builder.add_node("search_web",search_web)
builder.add_node("search_wikipedia",search_wikipedia)
builder.add_node("generate_answer",generate_answer)

# adges
builder.add_edge(START,"search_web")
builder.add_edge(START,'search_wikipedia')
builder.add_edge("search_web","generate_answer")
builder.add_edge("search_wikipedia","generate_answer")
builder.add_edge("generate_answer",END)

memory = MemorySaver()

graph = builder.compile(checkpointer=memory)


def main():

    st.sidebar.image(image="download.png",caption="Multi AI Agents")

    st.header("Multi AI Agents Q&A chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid4())


    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message,AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message,HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
    

    user_input = st.chat_input("Enrter query here")
    
    if user_input is not None and user_input != "":
        st.session_state.chat_history.append(HumanMessage(content=user_input))

        with st.chat_message("Human"):
            st.markdown(user_input)
        
        with st.chat_message("AI"):
            with st.spinner("Generating response..."):
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                result = graph.invoke({"question": user_input}, config=config)
                answer = result["answer"].content

                st.write(answer)

        st.session_state.chat_history.append(AIMessage(content=answer))
if __name__=="__main__":
    main()

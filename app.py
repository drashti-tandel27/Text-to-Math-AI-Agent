import ast
import operator as op

import streamlit as st
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_groq import ChatGroq


# 1. Calculator Tool

allowed_operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def calculate_expression(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value

    if isinstance(node, ast.BinOp):
        operator_type = type(node.op)

        if operator_type in allowed_operators:
            left = calculate_expression(node.left)
            right = calculate_expression(node.right)
            return allowed_operators[operator_type](left, right)

    if isinstance(node, ast.UnaryOp):
        operator_type = type(node.op)

        if operator_type in allowed_operators:
            return allowed_operators[operator_type](
                calculate_expression(node.operand)
            )

    raise ValueError("Invalid mathematical expression")


@tool
def calculator(expression: str) -> str:
    """Calculate arithmetic expressions such as 10 * 5 or 100 / 4."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(calculate_expression(tree.body))
    except Exception as error:
        return f"Calculation error: {error}"


# 2. Create the Math Agent

@st.cache_resource
def create_math_agent():
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
    )

    agent = create_agent(
        model=llm,
        tools=[calculator],
        system_prompt=(
            "You are a helpful math tutor. "
            "Explain solutions step by step. "
            "Use the calculator tool for arithmetic calculations. "
            "Give a clear final answer. "
            "Use previous conversation messages to answer follow-up questions. "
            "If a question is unclear, ask for clarification."
        ),
    )

    return agent


# 3. Configure the Streamlit Page

st.set_page_config(
    page_title="Text-to-Math Agent",
    page_icon="🧮",
    layout="centered",
)

st.title("Text-to-Math Agent")
st.write("Ask a math question or a real-life word problem.")


# 4. Initialize Conversation History

if "messages" not in st.session_state:
    st.session_state.messages = []


# 5. Clear Chat Button

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()


# 6. Display Previous Messages

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# 7. Accept a New Question

if question := st.chat_input("Enter your math question..."):

    # Save the user's question
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Send conversation history to the agent
    history = [
        {
            "role": message["role"],
            "content": message["content"],
        }
        for message in st.session_state.messages
    ]

    # Generate the answer
    with st.chat_message("assistant"):
        with st.spinner("Solving your question..."):
            try:
                agent = create_math_agent()

                response = agent.invoke(
                    {"messages": history}
                )

                answer = response["messages"][-1].content

                if not isinstance(answer, str):
                    answer = str(answer)

                st.markdown(answer)

            except Exception as error:
                answer = f"Error: {error}"
                st.error(answer)

    # Save the assistant's answer
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

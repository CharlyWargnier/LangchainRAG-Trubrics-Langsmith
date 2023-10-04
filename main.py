"""Example Streamlit chat UI that exposes a Feedback button and link to LangSmith traces."""

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import streamlit as st
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.memory import StreamlitChatMessageHistory, ConversationBufferMemory
from langchain.schema.runnable import RunnableConfig
from langsmith import Client
from streamlit_feedback import streamlit_feedback
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from vanilla_chain import get_llm_chain


from essential_chain import initialize_chain

import os

st.set_page_config(
    page_title="Chat with the Streamlit docs via LangChain, Collect user feedback via Trubrics and LangSmith!",
    page_icon="🦜",
)

# Set LangSmith environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"

os.environ["LANGCHAIN_API_KEY"] = "ls__ea506496a9f54be786205187a0bd84ed"

os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = st.sidebar.text_input(
    "LangSmith Project", value="default"
)

langchain_api_key = st.sidebar.text_input(
    "LangChain API Key", value="ls__ea506496a9f54be786205187a0bd84ed", type='password')

if "last_run" not in st.session_state:
    st.session_state["last_run"] = "some_initial_value"

langchain_endpoint = "https://api.smith.langchain.com"
client = Client(api_url=langchain_endpoint, api_key=langchain_api_key)
# client = Client(api_url=LANGCHAIN_ENDPOINT, api_key=LANGCHAIN_API_KEY)


# Set environment variables using the values from st.secrets
os.environ['OPENAI_API_KEY'] = st.secrets["api_keys"]["OPENAI_API_KEY"]


col1, col2, col3 = st.columns([0.6, 3, 1])

with col2:
    st.image("images/logo.png", width=500)
    

st.markdown('___')

st.write('👇 Ask a question about the Streamlit Docs below - Check our blog post here')
#col1, col2, col3 = st.columns([0.11, 1, 1])
#with col1:
#    arrow = "images/red_arrow.png"
#    st.image(arrow, width=110)

# The rest of your script...

# client = Client(api_url=langchain_endpoint, api_key=langchain_api_key)

# st.subheader("Chat with the Streamlit docs, powered by LangChain 💬🦜")

# Initialize State
if "trace_link" not in st.session_state:
    st.session_state.trace_link = None
if "run_id" not in st.session_state:
    st.session_state.run_id = None

# Add a button to choose between llmchain and expression chain
_DEFAULT_SYSTEM_PROMPT = ""

# _DEFAULT_SYSTEM_PROMPT = (
#     "You are a funky parrot pal. You are not an AI. You are a parrot."
#     " You love poetry, reading, funk music, and friendship!"
# )

system_prompt = _DEFAULT_SYSTEM_PROMPT = ""


#system_prompt = st.sidebar.text_area(
#    "Custom Instructions",
#    _DEFAULT_SYSTEM_PROMPT,
#    help="Custom instructions to provide the language model to determine style, personality, etc.",
#)
system_prompt = system_prompt.strip().replace("{", "{{").replace("}", "}}")

chain_type = st.sidebar.radio(
    "Choose a chain type",
    ("GPT 3.5 Chain", "RAG Chain for Streamlit Docs"),  # Added "RAG Chain for Streamlit Docs" option
    help="Choose the chain type.",
)

memory = ConversationBufferMemory(
    chat_memory=StreamlitChatMessageHistory(key="langchain_messages"),
    return_messages=True,
    memory_key="chat_history",
)

# Create Chain
#if chain_type == "GPT 3.5 Chain":
#    chain = get_llm_chain(system_prompt, memory)
#else:
#    chain = get_expression_chain(system_prompt, memory)

if chain_type == "GPT 3.5 Chain":
    chain = get_llm_chain(system_prompt, memory)
else:  # This will be triggered when "RAG Chain for Streamlit Docs" is selected
    # chain = initialize_chain(system_prompt, memory)
    chain = initialize_chain(system_prompt, _memory=memory)


if st.sidebar.button("Clear message history"):
    print("Clearing message history")
    memory.clear()
    st.session_state.trace_link = None
    st.session_state.run_id = None


# Display chat messages from history on app rerun
# NOTE: This won't be necessary for Streamlit 1.26+, you can just pass the type directly
# https://github.com/streamlit/streamlit/pull/7094
def _get_openai_type(msg):
    if msg.type == "human":
        return "user"
    if msg.type == "ai":
        return "assistant"
    if msg.type == "chat":
        return msg.role
    return msg.type


for msg in st.session_state.langchain_messages:
    streamlit_type = _get_openai_type(msg)
    avatar = "🦜" if streamlit_type == "assistant" else None
    with st.chat_message(streamlit_type, avatar=avatar):
        st.markdown(msg.content)

run_collector = RunCollectorCallbackHandler()
runnable_config = RunnableConfig(
    callbacks=[run_collector],
    tags=["Streamlit Chat"],
)
if st.session_state.trace_link:
    st.sidebar.markdown(
        f'<a href="{st.session_state.trace_link}" target="_blank"><button>Latest Trace: 🛠️</button></a>',
        unsafe_allow_html=True,
    )


def _reset_feedback():
    st.session_state.feedback_update = None
    st.session_state.feedback = None


if prompt := st.chat_input(placeholder="Ask me a question!"):
    st.chat_message("user").write(prompt)
    _reset_feedback()
    with st.chat_message("assistant", avatar="🦜"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Define the basic input structure for the chains
        input_structure = {"input": prompt}
        
        # Modify the input structure for the RAG Chain for Streamlit Docs
        if chain_type == "RAG Chain for Streamlit Docs":
            input_structure = {
                "question": prompt, 
                "chat_history": [(msg.type, msg.content) for msg in st.session_state.langchain_messages]
            }
        
        # Handle LLMChain separately as it uses the invoke method
        if chain_type == "GPT 3.5 Chain":
            message_placeholder.markdown("thinking...")
            full_response = chain.invoke(input_structure, config=runnable_config)["text"]



        else:
            for chunk in chain.stream(input_structure, config=runnable_config):
                full_response += chunk['answer']  # Updated to use the 'answer' key
                message_placeholder.markdown(full_response + "▌")
            memory.save_context({"input": prompt}, {"output": full_response})


        message_placeholder.markdown(full_response)
        
        # ... (rest of your existing code)

        # The run collector will store all the runs in order. We'll just take the root and then
        # reset the list for next interaction.
        run = run_collector.traced_runs[0]
        run_collector.traced_runs = []
        st.session_state.run_id = run.id
        wait_for_all_tracers()
        # Requires langsmith >= 0.0.19
        url = client.share_run(run.id)
        # Or if you just want to use this internally
        # without sharing
        # url = client.read_run(run.id).url
        st.session_state.trace_link = url

# feedback_option = "faces" if st.sidebar.checkbox("Use faces feedback system", value=False) else "thumbs"
feedback_option = "faces" if st.toggle(label="`Thumbs` ⇄ `Faces`", value=False) else "thumbs"

if st.session_state.get("run_id"):
    feedback = streamlit_feedback(
        feedback_type=feedback_option,  # Use the selected feedback option
        optional_text_label="[Optional] Please provide an explanation",  # Adding a label for optional text input
        key=f"feedback_{st.session_state.run_id}",
    )
    
    # Define score mappings for both "thumbs" and "faces" feedback systems
    score_mappings = {
        "thumbs": {"👍": 1, "👎": 0},
        "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
    }
    
    # Get the score mapping based on the selected feedback option
    scores = score_mappings[feedback_option]
    
    if feedback:
        # Get the score from the selected feedback option's score mapping
        score = scores.get(feedback["score"])
        
        if score is not None:
            # Formulate feedback type string incorporating the feedback option and score value
            feedback_type_str = f"{feedback_option} {feedback['score']}"
            
            # Record the feedback with the formulated feedback type string and optional comment
            feedback_record = client.create_feedback(
                st.session_state.run_id, 
                feedback_type_str,  # Updated feedback type
                score=score, 
                comment=feedback.get("text")
            )
            st.session_state.feedback = {"feedback_id": str(feedback_record.id), "score": score}
        else:
            st.warning("Invalid feedback score.")

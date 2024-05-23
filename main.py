import streamlit as st
from vocab_learner import get_learner

models = [
    'mixtral-8x7b-instruct',
    "llama3-70b",
    "llama3-8b",
    "codellama-7b-instruct",
    "codellama-13b-instruct",
    "codellama-34b-instruct",
    "mixtral-8x22b-instruct",
    "mistral-7b-instruct",
    "mistral-7b",
    "mixtral-8x22b",
    "gemma-7b",
    "gemma-2b",
    "alpaca-7b",
    "vicuna-7b",
    "vicuna-13b",
    "vicuna-13b-16k",
    "falcon-7b-instruct",
    "falcon-40b-instruct",
    "openassistant-llama2-70b",
    "Nous-Hermes-Llama2-13b",
    "Nous-Hermes-llama-2-7b",
    "Nous-Hermes-2-Mistral-7B-DPO",
    "Nous-Hermes-2-Mixtral-8x7B-SFT",
    "Nous-Hermes-2-Mixtral-8x7B-DPO",
    "Nous-Hermes-2-Yi-34B",
    "Nous-Capybara-7B-V1p9",
    "OpenHermes-2p5-Mistral-7B",
    "OpenHermes-2-Mistral-7B",
    "Qwen1.5-72B-Chat"
]


with st.sidebar:
    st.header("Settings", divider = 'rainbow')
    model = st.selectbox(
    "Which model do you want to choose?",
    ("local", "remote"))
    
    model_name = st.selectbox(
        "Select a model",
        models,
    )
    
    learning_mode = st.selectbox(
        "Select learning mode",
        ("random", "sequential"),
    )



if "learner" not in st.session_state:
    st.session_state.learner = get_learner(jupyter = False, type = model,remote_model_name=model_name, mode = 'random')
    st.session_state.display_explaintion = ""

st.session_state.display_word = st.session_state.learner.get_word()

st.header("Vocabulary Book", divider = 'rainbow')

col1, col2 = st.columns(2)
with col1:
    explain_button = st.button("Explain", use_container_width=True, key = "explain")
with col2:
    next_button = st.button("Next", use_container_width=True, key = 'next')
   
if next_button:
    _ = st.session_state.learner.next()
    st.session_state.display_word = st.session_state.learner.get_word()
        
st.subheader(st.session_state.display_word)

with st.container(border = True,height = 500):
    if explain_button:
        with st.spinner("Explaining..."):
            st.session_state.display_explaintion = st.session_state.learner.explain()
            
    st.markdown(st.session_state.display_explaintion)
        
st.subheader("Create sentence challenge", divider = 'rainbow')
sentence = st.text_area(
    "Text to analyze",
    "Create a sentence with this word."
    )

st.write(f"You wrote {len(sentence)} characters.")

analyze = st.button("Aanalyze", use_container_width=True, key = 'analyze_sentence')
if analyze:
    with st.container(border = True, height = 500):
        with st.spinner("Analyzing..."):
            prompt = "The following is a sentence that uses the word, try to figure out if there are any grammar or other kinds of mistakes. \
                Be sure we are focusing on learning the current word. \
                The sentence is :"
            response = st.session_state.learner.ask(prompt+sentence)
            st.markdown(response)


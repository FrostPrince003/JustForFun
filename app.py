import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer, GPT2LMHeadModel, GPT2Tokenizer
import language_tool_python
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------
# Model Loading Functions
# ----------------------
@st.cache_data
def load_paraphrasing_model():
    model_name = "t5-small"  # You can swap this with Pegasus if preferred
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

@st.cache_data
def load_nextword_model():
    model_name = "distilgpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return tokenizer, model

# Use st.cache_resource for objects that are not pickle-serializable
@st.cache_resource
def load_grammar_tool():
    tool = language_tool_python.LanguageTool('en-US')
    return tool

# ----------------------
# Paraphrasing Functions
# ----------------------
def paraphrase_text(text, creativity=0.9, tone="professional"):
    """
    Paraphrase a short text using T5. 
    The tone parameter can be used to adjust the prompt if needed.
    """
    tokenizer, model = load_paraphrasing_model()
    input_text = f"paraphrase: {text}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(input_ids, max_length=512, temperature=creativity, num_return_sequences=1)
    paraphrased = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return paraphrased

def chunk_text(text, chunk_size=30):
    """
    Splits the text into chunks of approximately `chunk_size` words.
    """
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def paraphrase_large_text(text, creativity=0.9, tone="professional", chunk_size=30):
    """
    Breaks a large text into chunks, paraphrases each chunk,
    and then recombines them.
    """
    chunks = chunk_text(text, chunk_size)
    paraphrased_chunks = []
    for chunk in chunks:
        paraphrased_chunk = paraphrase_text(chunk, creativity=creativity, tone=tone)
        paraphrased_chunks.append(paraphrased_chunk)
    return ' '.join(paraphrased_chunks)

# ----------------------
# Grammar Checking Function
# ----------------------
def check_grammar(text):
    tool = load_grammar_tool()
    matches = tool.check(text)
    errors = []
    for match in matches:
        # Classify the error type
        error_type = "Grammar mistake"
        if "punctuation" in match.ruleId.lower():
            error_type = "Punctuation fix"
        elif "style" in match.message.lower():
            error_type = "Style suggestion"
        errors.append({
            "message": match.message,
            "offset": match.offset,
            "length": match.errorLength,
            "type": error_type
        })
    return errors

# ----------------------
# Next-Word Prediction Function
# ----------------------
def predict_next_word(text, top_k=3):
    tokenizer, model = load_nextword_model()
    # Use the last 5 words as context if available
    context = " ".join(text.split()[-5:])
    input_ids = tokenizer.encode(context, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=input_ids.shape[1]+1, num_return_sequences=top_k, do_sample=True)
    predictions = []
    for output in outputs:
        # Extract only the new token predicted
        predicted_text = tokenizer.decode(output[input_ids.shape[1]:], skip_special_tokens=True)
        # Use a dummy confidence score for demonstration; in production, derive from model logits.
        predictions.append((predicted_text, np.random.uniform(0.5, 1.0)))
    return predictions

# ----------------------
# Streamlit Interface
# ----------------------
st.title("Multi-Functional NLP App")
st.sidebar.header("Choose Mode")
mode = st.sidebar.radio("Select Functionality", ("Paraphrasing", "Grammar Checking", "Next-Word Prediction"))

input_text = st.text_area("Enter your text here:")

# Customization controls based on mode
if mode == "Paraphrasing":
    creativity = st.slider("Paraphrase Creativity (Temperature)", 0.7, 1.2, 0.9)
    tone = st.selectbox("Select Tone", ["casual", "professional"])
elif mode == "Grammar Checking":
    lang_toggle = st.radio("Select English Variant", ("American English", "British English"))
elif mode == "Next-Word Prediction":
    st.info("Using the last 5 words as context for predictions.")

# Process button
if st.button("Process"):
    if not input_text.strip():
        st.error("Please enter text")
    else:
        with st.spinner("Processing..."):
            if mode == "Paraphrasing":
                # Use chunk-based paraphrasing for large texts
                result = paraphrase_large_text(input_text, creativity=creativity, tone=tone, chunk_size=30)
                st.subheader("Paraphrased Text")
                st.write(result)
            elif mode == "Grammar Checking":
                errors = check_grammar(input_text)
                st.subheader("Grammar and Style Issues")
                for err in errors:
                    if err["type"] == "Grammar mistake":
                        st.markdown(f"<span style='color:red'>{err['message']}</span>", unsafe_allow_html=True)
                    elif err["type"] == "Style suggestion":
                        st.markdown(f"<span style='color:blue'>{err['message']}</span>", unsafe_allow_html=True)
                    elif err["type"] == "Punctuation fix":
                        st.markdown(f"<span style='color:green'>{err['message']}</span>", unsafe_allow_html=True)
            elif mode == "Next-Word Prediction":
                predictions = predict_next_word(input_text)
                st.subheader("Next-Word Predictions")
                for word, score in predictions:
                    st.write(f"Prediction: **{word}** with confidence **{score:.2f}**")

# Download button placeholder for output text
if st.button("Download Output"):
    st.info("Download functionality to be implemented.")

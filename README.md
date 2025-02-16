# JustForFun
# 📝 Streamlit Grammar Checker  

This is a **Streamlit-based grammar checker** that uses **LanguageTool** to detect and correct grammatical errors in English text.  

## 🚀 Features  
- Checks text for **grammar, spelling, and style mistakes**.  
- Provides **suggestions and corrections**.  
- Uses **Streamlit UI** for easy interaction.  

## 📦 Installation  

Make sure you have Python installed, then install the required dependencies:  

```bash
pip install streamlit language-tool-python
```

## ▶️ Usage  

Run the Streamlit app:  

```bash
streamlit run app.py
```

## 🛠 Code Overview  

```python
import streamlit as st
import language_tool_python

@st.cache_resource
def load_grammar_tool():
    return language_tool_python.LanguageTool('en-US')

# Load the grammar tool
tool = load_grammar_tool()

# User input
text = st.text_area("Enter your text:", "This is an example sentence with mistake.")

# Check grammar
if st.button("Check Grammar"):
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    st.write("**Corrected Text:**", corrected_text)
```

## 📝 How It Works  
1. The user enters text in the Streamlit UI.  
2. When **"Check Grammar"** is clicked, `language_tool_python` checks for grammar mistakes.  
3. Suggestions are applied, and the corrected text is displayed.  

## ❗ Notes  
- The function `load_grammar_tool()` is **cached** using `st.cache_resource` to avoid reloading the tool on every run.  
- LanguageTool supports multiple languages; change `'en-US'` to another language if needed.  

## 💡 Future Improvements  
- Add support for multiple languages.  
- Display **detailed grammar suggestions** instead of just corrections.  
- Implement a **real-time grammar checker** using Streamlit's session state.  

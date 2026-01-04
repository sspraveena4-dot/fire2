import streamlit as st
import google.generativeai as genai

# --- Page Configuration ---
st.set_page_config(page_title="Gemini Chatbot", layout="wide")
st.title("💬 Gemini Conversational Assistant")

# --- Sidebar: Configuration ---
st.sidebar.header("Configuration")

api_key = st.sidebar.text_input(
    "Google Gemini API Key",
    type="password"
)

model_option = st.sidebar.selectbox(
    "Model",
    [
        "gemini-1.5-flash",
        "gemini-1.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-3.0-flash",
        "gemini-3.0-flash-lite",
    ]
)

# --- Helper: Convert chat history to Gemini format ---
def build_gemini_history(messages):
    history = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        history.append({
            "role": role,
            "parts": [msg["content"]]
        })
    return history

# --- Main App ---
def main():
    if not api_key:
        st.warning("Please enter your Google Gemini API Key.")
        return

    # Initialize Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_option)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if user_input := st.chat_input("Ask me anything..."):
        # Store user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("Thinking…")

            try:
                # Build conversation context
                history = build_gemini_history(st.session_state.messages[:-1])

                chat = model.start_chat(history=history)
                response = chat.send_message(user_input)

                assistant_reply = response.text.strip()

            except Exception as e:
                assistant_reply = f"Something went wrong: {e}"

            placeholder.markdown(assistant_reply)

            # Store assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_reply
            })

if __name__ == "__main__":
    main()
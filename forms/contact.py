import streamlit as st
import re
import requests


def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def show_contact_form():
    # Add popup styling with larger dimensions
    st.markdown("""
        <style>
            div[data-testid="stForm"] {
                background-color: #1E2630;
                padding: 3rem;  /* Increased padding */
                border-radius: 10px;
                max-width: 600px;  /* Increased width */
                width: 100%;
                margin: 0 auto;
                position: relative;
            }

            /* Make text inputs bigger */
            .stTextInput input, .stTextArea textarea {
                padding: 12px !important;  /* Increased input padding */
                font-size: 16px !important;  /* Larger font size */
            }

            /* Make text area bigger */
            .stTextArea textarea {
                min-height: 150px !important;  /* Taller text area */
            }

            /* Success message styling */
            div.stSuccess {
                padding: 1rem;
                border-radius: 5px;
                background-color: #2F855A;
                color: white;
            }

            /* Header styling */
            h1 {
                margin-bottom: 2rem;
                font-size: 2rem !important;
            }
        </style>
    """, unsafe_allow_html=True)

    st.header("Contact Me")

    # Close button
    if st.button("×", key="close_button"):
        st.session_state.show_form = False
        st.rerun()

    # Contact form
    with st.form("contact_form", clear_on_submit=False):
        first_name = st.text_input("First Name")
        email = st.text_input("Email Address")
        message = st.text_area("Your Message")

        submitted = st.form_submit_button("Submit")

        if submitted:
            if first_name and email and message and is_valid_email(email):
                st.success("Message successfully sent!")
                # st.session_state.show_form = False
                # st.rerun()
            else:
                if not first_name:
                    st.error("Please enter your name")
                if not email or not is_valid_email(email):
                    st.error("Please enter a valid email address")
                if not message:
                    st.error("Please enter your message")


if __name__ == "__main__":
    show_contact_form()
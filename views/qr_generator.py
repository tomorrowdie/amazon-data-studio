import streamlit as st
import qrcode
from PIL import Image
import io
import base64


def show_page():
    st.title("QR Code Generator")
    st.write("Generate QR codes for websites, text, or contact information")

    # Create tabs for different QR code types
    tab1, tab2, tab3 = st.tabs(["Website/URL", "Text", "Contact (vCard)"])

    with tab1:
        show_url_qr()

    with tab2:
        show_text_qr()

    with tab3:
        show_vcard_qr()


def show_url_qr():
    st.subheader("Generate QR Code for Website/URL")

    # Input for URL
    url = st.text_input("Enter website URL", "https://")

    # QR code customization
    col1, col2 = st.columns(2)
    with col1:
        qr_color = st.color_picker("QR Code Color", "#000000")
    with col2:
        bg_color = st.color_picker("Background Color", "#FFFFFF")

    # Generate button
    if st.button("Generate URL QR Code"):
        if url and url != "https://":
            qr_image = generate_qr_code(url, qr_color, bg_color)
            st.image(qr_image, caption=f"QR Code for: {url}")

            # Download button
            img_str = base64.b64encode(qr_image).decode()
            href = f'<a href="data:file/png;base64,{img_str}" download="qr_code.png">Download QR Code</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.error("Please enter a valid URL")


def show_text_qr():
    st.subheader("Generate QR Code for Text")

    # Input for text
    text = st.text_area("Enter your text", "")

    # QR code customization
    col1, col2 = st.columns(2)
    with col1:
        qr_color = st.color_picker("QR Code Color", "#000000", key="text_qr_color")
    with col2:
        bg_color = st.color_picker("Background Color", "#FFFFFF", key="text_bg_color")

    # Generate button
    if st.button("Generate Text QR Code"):
        if text:
            qr_image = generate_qr_code(text, qr_color, bg_color)
            st.image(qr_image, caption="QR Code for text")

            # Download button
            img_str = base64.b64encode(qr_image).decode()
            href = f'<a href="data:file/png;base64,{img_str}" download="qr_code.png">Download QR Code</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.error("Please enter some text")


def show_vcard_qr():
    st.subheader("Generate QR Code for Contact Information")

    # Input for contact information
    name = st.text_input("Name")
    phone = st.text_input("Phone Number")
    email = st.text_input("Email")
    company = st.text_input("Company (optional)")
    website = st.text_input("Website (optional)")

    # QR code customization
    col1, col2 = st.columns(2)
    with col1:
        qr_color = st.color_picker("QR Code Color", "#000000", key="vcard_qr_color")
    with col2:
        bg_color = st.color_picker("Background Color", "#FFFFFF", key="vcard_bg_color")

    # Generate button
    if st.button("Generate Contact QR Code"):
        if name and (phone or email):
            # Create vCard format
            vcard = f"""BEGIN:VCARD
VERSION:3.0
FN:{name}
"""
            if phone:
                vcard += f"TEL:{phone}\n"
            if email:
                vcard += f"EMAIL:{email}\n"
            if company:
                vcard += f"ORG:{company}\n"
            if website:
                vcard += f"URL:{website}\n"
            vcard += "END:VCARD"

            qr_image = generate_qr_code(vcard, qr_color, bg_color)
            st.image(qr_image, caption=f"QR Code for: {name}")

            # Download button
            img_str = base64.b64encode(qr_image).decode()
            href = f'<a href="data:file/png;base64,{img_str}" download="contact_qr_code.png">Download QR Code</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.error("Please enter at least a name and either a phone number or email")


def generate_qr_code(data, qr_color, bg_color):
    """Generate a QR code with the given data and colors"""
    # Convert hex color to RGB tuple
    qr_color_rgb = tuple(int(qr_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
    bg_color_rgb = tuple(int(bg_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)

    img = qr.make_image(fill_color=qr_color_rgb, back_color=bg_color_rgb)

    # Convert PIL image to bytes for Streamlit
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return buffered.getvalue()


if __name__ == "__main__":
    show_page()
import streamlit as st

st.set_page_config(
    page_title="Amazon Data Studio",
    page_icon="📊",
    layout="wide"
)

# Navigation structure - Amazon Data Studio focused
navigation = {
    "Amazon Analytics": ["Data Process", "Catalog Insight"],
    "Tools": ["QR Generator"]
}

# Create navigation in sidebar
st.sidebar.title("📊 Amazon Data Studio")
st.sidebar.markdown("---")

# Initialize selected_page if not in session state
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "Data Process"

# Create sections in sidebar
for section, pages in navigation.items():
    st.sidebar.subheader(section)
    for page in pages:
        if st.sidebar.button(
            page,
            key=page,
            use_container_width=True,
            type="primary" if st.session_state.selected_page == page else "secondary"
        ):
            st.session_state.selected_page = page
            st.rerun()

# Add info in sidebar
st.sidebar.markdown("---")
st.sidebar.info("""
### About This Tool
Process and analyze Helium 10 Xray data for Amazon product research.

**Features:**
- 📁 Upload & process H10 CSV files
- 📊 Comprehensive catalog analysis
- 🎯 Seller analytics
- 📈 Performance metrics
- 🔍 QR code generation

**Created by:** John Chin  
**Contact:** chinhotak@gmail.com
""")

# Page routing based on selection
if st.session_state.selected_page == "Data Process":
    from views.amazon_data_process_h10 import show_page
    show_page()
elif st.session_state.selected_page == "Catalog Insight":
    from views.amazon_catalog_insight_h10 import show_page
    show_page()
elif st.session_state.selected_page == "QR Generator":
    from views.qr_generator import show_page
    show_page()
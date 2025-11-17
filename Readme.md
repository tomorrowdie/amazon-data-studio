# 📊 Amazon Data Studio

Process and analyze Helium 10 Xray data for Amazon product research.

## 🚀 Quick Start

### Local Development
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Usage
1. Go to "Data Process"
2. Upload your Helium 10 Xray CSV files (Ads & Organic)
3. Click "Process Files"
4. View analytics in "Catalog Insight"

## 📁 Required Files

Place these files in the `views/` folder:
- qr_generator.py
- amazon_data_process_h10.py
- amazon_catalog_insight_h10.py
- catalog_summary_01_data_overview.py
- catalog_summary_02_summary.py
- catalog_summary_03_sales_revenue.py
- catalog_summary_04_pricing.py
- catalog_summary_05_product_performance.py
- catalog_summary_06_product_characteristics.py
- catalog_summary_07_seller_analytics.py
- catalog_summary_07a_global_seller_metrics.py
- catalog_summary_08_detailed_analysis.py
- catalog_visualizations.py

## 👨‍💻 Author
**John Chin**  
Email: chinhotak@gmail.com

## 🌐 Deploy to Streamlit Cloud
1. Push to GitHub
2. Go to https://share.streamlit.io/
3. Connect your repo
4. Deploy!
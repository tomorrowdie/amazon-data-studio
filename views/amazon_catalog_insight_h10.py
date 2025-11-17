import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import io
from datetime import datetime

# Import new modular functions
from views.catalog_summary_01_data_overview import display_data_overview
from views.catalog_summary_02_summary import display_summary_overview
from views.catalog_summary_03_sales_revenue import display_sales_revenue
from views.catalog_summary_04_pricing import display_pricing
from views.catalog_summary_05_product_performance import display_product_performance
from views.catalog_summary_06_product_characteristics import display_product_characteristics
from views.catalog_summary_07_seller_analytics import display_seller_analytics
from views.catalog_summary_07a_global_seller_metrics import display_global_seller_metrics
from views.catalog_summary_08_detailed_analysis import display_detail_analysis

# For backward compatibility during transition, keep these until fully migrated
from views.catalog_visualizations import create_price_analysis, create_brand_analysis, create_ranking_analysis


def show_page():
    st.title('Amazon Catalog Insight / 亞馬遜目錄分析')

    # Check if data is available in session state
    if 'df_ads' not in st.session_state or 'df_organic' not in st.session_state or 'df_combined' not in st.session_state:
        show_data_needed_message()
        return

    # Display data that's been transferred from the other page
    show_analysis_dashboard()


def show_data_needed_message():
    """Show a message when no data is available yet"""
    st.warning("""
    ### No data available for analysis

    Please first process your Helium Xray files in the **Amazon Data Process H10** page.

    ### 沒有可用於分析的數據

    請先在 **Amazon Data Process H10** 頁面處理您的 Helium Xray 文件。
    """)

    # Show sample image or explain the process
    st.info("""
    Process flow:
    1. Go to **Amazon Data Process H10**
    2. Upload your Helium Xray CSV files
    3. Process the files
    4. Return to this page for analysis

    處理流程：
    1. 前往 **Amazon Data Process H10**
    2. 上傳您的 Helium Xray CSV 文件
    3. 處理文件
    4. 返回此頁面進行分析
    """)


def show_analysis_dashboard():
    """Display the main analysis dashboard when data is available"""
    # Get data from session state
    df_ads = st.session_state['df_ads']
    df_organic = st.session_state['df_organic']
    df_combined = st.session_state['df_combined']
    file_date = st.session_state.get('file_date', 'Unknown Date')
    week_number = st.session_state.get('week_number', 'Unknown Week')

    # Data overview section
    display_data_overview(df_combined, df_ads, df_organic)

    # 1. Summary Section
    st.markdown("---")
    st.header("Summary")
    display_summary_overview(df_combined, df_ads, df_organic)

    # 2. Sales & Revenue Metrics
    st.markdown("---")
    st.header("Sales & Revenue Metrics")
    display_sales_revenue(df_combined, df_ads, df_organic)

    # 3. Pricing & Profitability
    st.markdown("---")
    st.header("Pricing & Profitability")
    display_pricing(df_combined, df_ads, df_organic)

    # 4. Product Performance
    st.markdown("---")
    st.header("Product Performance")
    display_product_performance(df_combined, df_ads, df_organic)

    # 5. Product Characteristics
    st.markdown("---")
    st.header("Product Characteristics")
    display_product_characteristics(df_combined, df_ads, df_organic)

    # 6. Seller Analytics
    st.markdown("---")
    st.header("Seller Analytics")
    display_seller_analytics(df_combined, df_ads, df_organic)

    # 6a. Global Seller Metrics
    st.markdown("---")
    st.header("Global Seller Metrics")
    display_global_seller_metrics(df_combined, df_ads, df_organic)

    # 7. Detailed Analysis
    st.markdown("---")
    st.header("Detailed Analysis")
    display_detail_analysis(df_combined, df_ads, df_organic)


def preprocess_df(df):
    """Clean and preprocess dataframe for analysis"""
    if df is None or df.empty:
        return df

    # Make a copy to avoid modifying the original
    df_copy = df.copy()

    # Columns to convert to numeric
    numeric_cols = [
        'Price US', 'BSR', 'Review Count', 'ASIN Sales',
        'Parent Level Sales', 'Recent Purchases', 'Parent Level Revenue',
        'ASIN Revenue', 'Average Month Sales'
    ]

    # Convert columns to numeric if they exist
    for col in numeric_cols:
        if col in df_copy.columns:
            try:
                df_copy[col] = pd.to_numeric(
                    df_copy[col].astype(str).str.replace('[\$,]', '', regex=True),
                    errors='coerce'
                )
            except:
                pass

    return df_copy


def select_display_columns(df):
    """Select the most important columns for display to avoid overflow"""
    priority_columns = [
        'Display Order', 'Product Details', 'ASIN', 'Brand', 'Price US',
        'BSR', 'Organic VS Ads', 'Sales Rank (ALL)',
        'Organic Rank', 'Ad Rank', 'Source Files'
    ]

    # Return only columns that exist in the dataframe
    return [col for col in priority_columns if col in df.columns]


def convert_df_to_csv(df):
    """Convert dataframe to CSV for download"""
    return df.to_csv(index=False).encode('utf-8')


if __name__ == "__main__":
    show_page()
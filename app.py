import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai
import json
import time
import re
from datetime import datetime

# ==========================================
# 1. CONFIGURATION & PAGE SETUP
# ==========================================
st.set_page_config(
    page_title="AI Financial Analyst Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# ==========================================
# 2. CONSTANTS & DATASETS
# ==========================================

# 2.1 Benchmark Data (3 Tiers: Average, Leaders, Failure Patterns)
INDUSTRY_BENCHMARKS = {
    "SaaS / Technology": {
        "average": {"Gross Margin": (70, 80), "Net Margin": (5, 15), "Revenue Growth": (20, 40), "Current Ratio": (1.5, 2.5), "Debt-to-Assets": (10, 30)},
        "leaders": {"Gross Margin": (82, 95), "Net Margin": (25, 40), "Revenue Growth": (50, 100), "Current Ratio": (2.5, 4.0), "Debt-to-Assets": (0, 10)},
        "failure_patterns": {"Gross Margin": (0, 50), "Net Margin": (-100, -20), "Revenue Growth": (-50, 5), "Current Ratio": (0, 0.8), "Debt-to-Assets": (60, 100)}
    },
    "Retail / E-commerce": {
        "average": {"Gross Margin": (25, 45), "Net Margin": (2, 5), "Revenue Growth": (5, 15), "Current Ratio": (1.2, 2.0), "Debt-to-Assets": (30, 50)},
        "leaders": {"Gross Margin": (50, 65), "Net Margin": (7, 12), "Revenue Growth": (20, 35), "Current Ratio": (2.0, 3.0), "Debt-to-Assets": (10, 25)},
        "failure_patterns": {"Gross Margin": (0, 15), "Net Margin": (-20, 0), "Revenue Growth": (-20, 0), "Current Ratio": (0, 0.9), "Debt-to-Assets": (70, 100)}
    },
    "Manufacturing": {
        "average": {"Gross Margin": (20, 35), "Net Margin": (3, 8), "Revenue Growth": (3, 8), "Current Ratio": (1.2, 2.0), "Debt-to-Assets": (30, 50)},
        "leaders": {"Gross Margin": (35, 50), "Net Margin": (10, 18), "Revenue Growth": (10, 20), "Current Ratio": (2.0, 3.5), "Debt-to-Assets": (10, 25)},
        "failure_patterns": {"Gross Margin": (0, 15), "Net Margin": (-15, 0), "Revenue Growth": (-15, -2), "Current Ratio": (0, 0.8), "Debt-to-Assets": (65, 100)}
    },
    "Healthcare / Biotech": {
        "average": {"Gross Margin": (50, 70), "Net Margin": (5, 15), "Revenue Growth": (10, 25), "Current Ratio": (1.5, 3.0), "Debt-to-Assets": (20, 40)},
        "leaders": {"Gross Margin": (75, 90), "Net Margin": (20, 35), "Revenue Growth": (30, 60), "Current Ratio": (3.5, 6.0), "Debt-to-Assets": (0, 15)},
        "failure_patterns": {"Gross Margin": (0, 40), "Net Margin": (-50, -10), "Revenue Growth": (-20, 5), "Current Ratio": (0, 1.0), "Debt-to-Assets": (60, 100)}
    },
    "Restaurants / Food Service": {
        "average": {"Gross Margin": (60, 70), "Net Margin": (3, 6), "Revenue Growth": (2, 8), "Current Ratio": (0.8, 1.5), "Debt-to-Assets": (40, 60)},
        "leaders": {"Gross Margin": (70, 80), "Net Margin": (8, 15), "Revenue Growth": (10, 20), "Current Ratio": (1.5, 2.5), "Debt-to-Assets": (20, 35)},
        "failure_patterns": {"Gross Margin": (0, 55), "Net Margin": (-10, 0), "Revenue Growth": (-10, 0), "Current Ratio": (0, 0.6), "Debt-to-Assets": (70, 100)}
    },
    "Construction / Real Estate": {
        "average": {"Gross Margin": (15, 25), "Net Margin": (2, 6), "Revenue Growth": (5, 12), "Current Ratio": (1.1, 1.8), "Debt-to-Assets": (40, 60)},
        "leaders": {"Gross Margin": (25, 35), "Net Margin": (8, 15), "Revenue Growth": (15, 25), "Current Ratio": (2.0, 3.0), "Debt-to-Assets": (20, 35)},
        "failure_patterns": {"Gross Margin": (0, 10), "Net Margin": (-15, 0), "Revenue Growth": (-20, 0), "Current Ratio": (0, 0.9), "Debt-to-Assets": (75, 100)}
    },
    "Energy / Utilities": {
        "average": {"Gross Margin": (30, 45), "Net Margin": (8, 12), "Revenue Growth": (2, 6), "Current Ratio": (0.9, 1.3), "Debt-to-Assets": (50, 70)},
        "leaders": {"Gross Margin": (45, 60), "Net Margin": (15, 22), "Revenue Growth": (8, 15), "Current Ratio": (1.4, 2.0), "Debt-to-Assets": (30, 45)},
        "failure_patterns": {"Gross Margin": (0, 20), "Net Margin": (-10, 2), "Revenue Growth": (-15, 0), "Current Ratio": (0, 0.7), "Debt-to-Assets": (80, 100)}
    },
     "Financial Services": {
        "average": {"Gross Margin": (90, 100), "Net Margin": (15, 25), "Revenue Growth": (5, 10), "Current Ratio": (1.0, 1.5), "Debt-to-Assets": (60, 80)},
        "leaders": {"Gross Margin": (90, 100), "Net Margin": (30, 50), "Revenue Growth": (15, 30), "Current Ratio": (1.5, 3.0), "Debt-to-Assets": (40, 60)},
        "failure_patterns": {"Gross Margin": (0, 70), "Net Margin": (-20, 5), "Revenue Growth": (-10, 2), "Current Ratio": (0, 0.9), "Debt-to-Assets": (90, 100)}
    },
     "Professional Services": {
        "average": {"Gross Margin": (30, 50), "Net Margin": (10, 15), "Revenue Growth": (5, 12), "Current Ratio": (1.5, 2.5), "Debt-to-Assets": (20, 40)},
        "leaders": {"Gross Margin": (50, 70), "Net Margin": (20, 30), "Revenue Growth": (15, 25), "Current Ratio": (2.5, 4.0), "Debt-to-Assets": (0, 15)},
        "failure_patterns": {"Gross Margin": (0, 25), "Net Margin": (-10, 0), "Revenue Growth": (-15, 0), "Current Ratio": (0, 1.0), "Debt-to-Assets": (50, 100)}
    }
}

# 2.2 Column Mapping (Normalization)
COLUMN_MAPPING = {
    "Year": ["year", "fiscal year", "period", "fy", "date"],
    "Revenue": ["revenue", "total revenue", "sales", "turnover", "net sales"],
    "COGS": ["cogs", "cost of goods sold", "cost of sales", "cost of revenue"],
    "Operating Expenses": ["operating expenses", "opex", "sg&a", "selling, general and administrative"],
    "Operating Income": ["operating income", "operating profit", "ebit"],
    "Net Income": ["net income", "net profit", "earnings", "net earnings"],
    "Total Assets": ["total assets", "assets"],
    "Current Assets": ["current assets"],
    "Total Liabilities": ["total liabilities", "liabilities"],
    "Current Liabilities": ["current liabilities"],
    "Equity": ["shareholders' equity", "equity", "total equity", "stockholders' equity"],
    "Cash": ["cash", "cash and cash equivalents", "cash & equivalents"]
}

# 2.3 Sample Data
SAMPLE_DATA_SAAS = [
    {"Year": 2021, "Revenue": 1000000, "COGS": 200000, "Operating Expenses": 750000, "Net Income": 50000, "Total Assets": 800000, "Current Assets": 500000, "Total Liabilities": 300000, "Current Liabilities": 200000, "Equity": 500000, "Cash": 300000},
    {"Year": 2022, "Revenue": 1500000, "COGS": 280000, "Operating Expenses": 900000, "Net Income": 320000, "Total Assets": 1200000, "Current Assets": 800000, "Total Liabilities": 400000, "Current Liabilities": 250000, "Equity": 800000, "Cash": 550000},
    {"Year": 2023, "Revenue": 2400000, "COGS": 400000, "Operating Expenses": 1200000, "Net Income": 800000, "Total Assets": 2000000, "Current Assets": 1500000, "Total Liabilities": 500000, "Current Liabilities": 300000, "Equity": 1500000, "Cash": 1000000}
]

SAMPLE_DATA_RETAIL = [
    {"Year": 2020, "Revenue": 5000000, "COGS": 3000000, "Operating Expenses": 1800000, "Net Income": 200000, "Total Assets": 4000000, "Current Assets": 2000000, "Total Liabilities": 3000000, "Current Liabilities": 1500000, "Equity": 1000000, "Cash": 400000},
    {"Year": 2021, "Revenue": 4800000, "COGS": 3100000, "Operating Expenses": 1850000, "Net Income": -150000, "Total Assets": 3900000, "Current Assets": 1800000, "Total Liabilities": 3200000, "Current Liabilities": 1900000, "Equity": 700000, "Cash": 200000},
    {"Year": 2022, "Revenue": 4200000, "COGS": 2900000, "Operating Expenses": 1900000, "Net Income": -600000, "Total Assets": 3500000, "Current Assets": 1200000, "Total Liabilities": 3400000, "Current Liabilities": 2000000, "Equity": 100000, "Cash": 50000}
]

# ==========================================
# 3. UTILITY FUNCTIONS
# ==========================================

def normalize_columns(df):
    df.columns = [c.strip().lower() for c in df.columns]
    renamed = {}
    for standard, variations in COLUMN_MAPPING.items():
        for col in df.columns:
            if col in variations:
                renamed[col] = standard
                break
    for standard, variations in COLUMN_MAPPING.items():
        if standard not in renamed.values():
            for col in df.columns:
                if col not in renamed:
                    if any(v in col for v in variations):
                        renamed[col] = standard
                        break
    return df.rename(columns=renamed)

def safe_div(n, d):
    if d == 0 or pd.isna(d) or pd.isna(n):
        return None
    return n / d

def format_metric(val, is_percent=True):
    if val is None: return "N/A"
    if is_percent: return f"{val:.1f}%"
    return f"{val:.2f}"

def get_status_color(metric_name, value, benchmarks):
    if value is None: return "gray"
    leaders = benchmarks['leaders'].get(metric_name)
    failures = benchmarks['failure_patterns'].get(metric_name)
    if failures and failures[0] <= value <= failures[1]: return "red"
    if leaders and value >= leaders[0]: return "green"
    return "orange"

# ==========================================
# 4. FINANCIAL CALCULATION ENGINE
# ==========================================

def calculate_metrics(df):
    df = df.sort_values('Year')
    results = df.copy()
    if 'Revenue' in df and 'COGS' in df:
        results['Gross Margin'] = results.apply(lambda x: safe_div(x['Revenue'] - x['COGS'], x['Revenue']) * 100, axis=1)
    if 'Revenue' in df and 'Net Income' in df:
        results['Net Margin'] = results.apply(lambda x: safe_div(x['Net Income'], x['Revenue']) * 100, axis=1)
    if 'Total Assets' in df and 'Net Income' in df:
        results['ROA'] = results.apply(lambda x: safe_div(x['Net Income'], x['Total Assets']) * 100, axis=1)
    for col in ['Revenue', 'Net Income', 'Total Assets']:
        if col in df: results[f'{col} Growth'] = results[col].pct_change() * 100
    if 'Current Assets' in df and 'Current Liabilities' in df:
        results['Current Ratio'] = results.apply(lambda x: safe_div(x['Current Assets'], x['Current Liabilities']), axis=1)
    if 'Total Liabilities' in df and 'Total Assets' in df:
        results['Debt-to-Assets'] = results.apply(lambda x: safe_div(x['Total Liabilities'], x['Total Assets']) * 100, axis=1)
    return results

# ==========================================
# 5. AI INTEGRATION (GEMINI + MOCK)
# ==========================================

def get_gemini_analysis(api_key, industry, data_summary, comparison_text):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        You are a senior financial analyst. Analyze the following company in the {industry} industry.
        FINANCIAL DATA SUMMARY:
        {data_summary}
        BENCHMARK COMPARISON:
        {comparison_text}
        TASK:
        1. Executive Summary (2-3 sentences).
        2. Top 3 Strengths.
        3. Top 3 Weaknesses.
        4. Failure Pattern Check.
        5. 3-5 Strategic Recommendations.
        FORMAT: Markdown with bold headers.
        """
        with st.spinner('🤖 Gemini is analyzing...'):
            response = model.generate_content(prompt)
            return response.text
    except Exception as e:
        return f"API_ERROR: {str(e)}"

def get_mock_analysis(industry, df_latest):
    time.sleep(2)
    gm = df_latest.get('Gross Margin', 0)
    nm = df_latest.get('Net Margin', 0)
    cr = df_latest.get('Current Ratio', 0)
    return f"""
    **[MOCK ANALYSIS - API UNAVAILABLE]**
    ### 📊 Executive Summary
    The company shows mixed performance in {industry}.
    ### ✅ Strengths
    1. **Margin**: {f"Strong Gross Margin ({gm:.1f}%)" if gm > 50 else "Stable revenue base."}
    2. **Assets**: Significant asset base retained.
    ### ⚠️ Weaknesses
    1. **Liquidity**: {'CRITICAL' if cr < 1 else 'Watch'} Current Ratio of {cr:.2f}.
    2. **Profit**: Net Margin is {nm:.1f}%.
    ### 🚀 Recommendations
    1. Optimize COGS immediately.
    2. Renegotiate supplier terms to boost liquidity.
    """

def chat_agent(user_query, context_data):
    if "gross margin" in user_query.lower(): return "Gross Margin = (Revenue - COGS) / Revenue. It shows pricing power."
    return "Based on the analysis, focus on stabilizing cash flow and matching industry leaders."

# ==========================================
# 6. MAIN APPLICATION
# ==========================================

def main():
    with st.sidebar:
        st.header("⚙️ Configuration")
        if "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
        else:
            api_key = st.text_input("Gemini API Key (Optional)", type="password")
        st.divider()
        st.subheader("1. Upload Data")
        uploaded_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
        st.write("OR")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Load SaaS"):
                st.session_state.processed_data = pd.DataFrame(SAMPLE_DATA_SAAS)
                st.rerun()
        with col2:
            if st.button("Load Retail"):
                st.session_state.processed_data = pd.DataFrame(SAMPLE_DATA_RETAIL)
                st.rerun()
        st.divider()
        st.subheader("2. Select Industry")
        selected_industry = st.selectbox("Industry", list(INDUSTRY_BENCHMARKS.keys()))
        if st.session_state.processed_data is not None:
            if st.button("🔍 Analyze", type="primary"):
                with st.spinner("Analyzing..."):
                    df = normalize_columns(st.session_state.processed_data)
                    st.session_state.analysis_results = calculate_metrics(df)
                    st.session_state.chat_history = []
                    st.rerun()

    st.title("📊 AI Financial Analyst")
    
    if uploaded_file and st.session_state.processed_data is None:
        try:
            if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
            else: df = pd.read_excel(uploaded_file)
            st.session_state.processed_data = df
            st.success("Uploaded! Click 'Analyze' in sidebar.")
        except Exception as e: st.error(f"Error: {e}")

    if st

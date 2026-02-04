import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai
import time

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
    # (Other industries kept as is...)
}

COLUMN_MAPPING = {
    "Year": ["year", "fiscal year", "period", "fy", "date"],
    "Revenue": ["revenue", "total revenue", "sales", "turnover", "net sales"],
    "COGS": ["cogs", "cost of goods sold", "cost of sales", "cost of revenue"],
    "Operating Expenses": ["operating expenses", "opex", "sg&a", "selling, general and administrative"],
    "Operating Income": ["operating income", "operating profit", "ebit"],
    "Net Income": ["net income", "net profit", "earnings", "net earnings"],
    "Total Assets": ["total assets", "assets"],
    "Current Assets": ["current assets", "total current assets"],
    "Total Liabilities": ["total liabilities", "liabilities"],
    "Current Liabilities": ["current liabilities", "total current liabilities"],
    "Equity": ["shareholders' equity", "equity", "total equity", "stockholders' equity"],
    "Cash": ["cash", "cash and cash equivalents", "cash & equivalents"]
}

# ==========================================
# 3. UTILITY FUNCTIONS (UPDATED FOR YOUR DATA)
# ==========================================

def clean_currency(x):
    """Helper to remove commas and symbols from strings."""
    if isinstance(x, str):
        # Remove commas, dollar signs, spaces
        clean = x.replace('$', '').replace(',', '').replace(' ', '')
        # Handle accounting format negative numbers: (5000) -> -5000
        if '(' in clean and ')' in clean:
            clean = '-' + clean.replace('(', '').replace(')', '')
        try:
            return float(clean)
        except ValueError:
            return None
    return x

def normalize_columns(df):
    """
    UPDATED: Now auto-detects if years are headers (Horizontal) vs rows (Vertical)
    and cleans comma-separated numbers.
    """
    # 1. Check if the file is Horizontal (Years as headers)
    # We check the first column to see if it contains metric names like "Revenue"
    first_col_values = df.iloc[:, 0].astype(str).str.lower().tolist()
    is_horizontal = any('revenue' in v for v in first_col_values) or any('net income' in v for v in first_col_values)
    
    if is_horizontal:
        # Transpose the data: Swap rows and columns
        df = df.set_index(df.columns[0]).T
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Year'}, inplace=True)

    # 2. Standardize Column Names
    df.columns = [str(c).strip().lower() for c in df.columns]
    renamed = {}
    
    for standard, variations in COLUMN_MAPPING.items():
        if standard not in renamed.values():
            for col in df.columns:
                if col not in renamed:
                    for v in variations:
                        if v in col:
                            renamed[col] = standard
                            break
                    if col in renamed: break
    
    df = df.rename(columns=renamed)
    
    # 3. Ensure Year column exists
    if 'Year' not in df.columns:
        # If no explicit "Year" column, look for the column that holds 2019, 2020 etc
        for col in df.columns:
            try:
                if 1900 < float(str(df[col].iloc[0])) < 2100:
                    df.rename(columns={col: 'Year'}, inplace=True)
                    break
            except: pass

    # 4. Clean Numbers (Remove commas from "2,500,000")
    for col in df.columns:
        if col != 'Year':
            df[col] = df[col].apply(clean_currency)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def safe_div(n, d):
    if d == 0 or pd.isna(d) or pd.isna(n): return None
    return n / d

def format_metric(val, is_percent=True):
    if val is None or pd.isna(val): return "N/A"
    if is_percent: return f"{val:.1f}%"
    return f"{val:.2f}"

def get_status_color(metric_name, value, benchmarks):
    if value is None or pd.isna(value): return "gray"
    
    leaders = benchmarks.get('leaders', {}).get(metric_name)
    failures = benchmarks.get('failure_patterns', {}).get(metric_name)
    
    # Check Failure (Red)
    if failures and failures[0] <= value <= failures[1]: return "red"
    # Check Leaders (Green)
    if leaders and value >= leaders[0]: return "green"
    
    return "orange"

# ==========================================
# 4. FINANCIAL CALCULATION ENGINE
# ==========================================

def calculate_metrics(df):
    df = df.sort_values('Year')
    results = df.copy()
    
    # Profitability
    if 'Revenue' in df and 'COGS' in df:
        results['Gross Margin'] = results.apply(lambda x: safe_div(x['Revenue'] - x['COGS'], x['Revenue']) * 100, axis=1)
    if 'Revenue' in df and 'Net Income' in df:
        results['Net Margin'] = results.apply(lambda x: safe_div(x['Net Income'], x['Revenue']) * 100, axis=1)
    if 'Total Assets' in df and 'Net Income' in df:
        results['ROA'] = results.apply(lambda x: safe_div(x['Net Income'], x['Total Assets']) * 100, axis=1)

    # Growth
    for col in ['Revenue', 'Net Income', 'Total Assets']:
        if col in df: results[f'{col} Growth'] = results[col].pct_change() * 100

    # Liquidity
    if 'Current Assets' in df and 'Current Liabilities' in df:
        results['Current Ratio'] = results.apply(lambda x: safe_div(x['Current Assets'], x['Current Liabilities']), axis=1)
    
    # Leverage
    if 'Total Liabilities' in df and 'Total Assets' in df:
        results['Debt-to-Assets'] = results.apply(lambda x: safe_div(x['Total Liabilities'], x['Total Assets']) * 100, axis=1)
        
    return results

# ==========================================
# 5. AI INTEGRATION
# ==========================================

def get_gemini_analysis(api_key, industry, data_summary, comparison_text):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        Analyze this {industry} company.
        DATA: {data_summary}
        BENCHMARKS: {comparison_text}
        
        Provide:
        1. Executive Summary
        2. Top 3 Strengths & Weaknesses
        3. Strategic Recommendations
        """
        with st.spinner('🤖 Analyzing...'):
            return model.generate_content(prompt).text
    except Exception as e: return f"API Error: {str(e)}"

def get_mock_analysis(industry, df_latest):
    # Fallback mock analysis
    cr = df_latest.get('Current Ratio', 0)
    nm = df_latest.get('Net Margin', 0)
    
    status = "healthy" if nm > 0 else "struggling"
    liquidity = "stable" if cr > 1.5 else "critical"
    
    return f"""
    ### 📊 Executive Summary (Mock)
    This {industry} company appears to be **{status}**.
    Liquidity is currently **{liquidity}** (Current Ratio: {cr:.2f}).
    
    **Recommendations:**
    1. If Net Margin is negative ({nm:.1f}%), focus on cost reduction.
    2. Monitor cash reserves closely.
    """

def chat_agent(user_query, context):
    return "Based on the data, I recommend focusing on cash flow management and cost reduction."

# ==========================================
# 6. MAIN APPLICATION UI
# ==========================================

def main():
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("Gemini API Key (Optional)", type="password")
        
        st.divider()
        st.subheader("1. Upload Data")
        
        def clear_submit():
            st.session_state.processed_data = None
            st.session_state.analysis_results = None
            
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'], on_change=clear_submit)
        
        st.divider()
        st.subheader("2. Select Industry")
        # Default to SaaS/Technology based on the data provided
        default_idx = list(INDUSTRY_BENCHMARKS.keys()).index("SaaS / Technology") if "SaaS / Technology" in INDUSTRY_BENCHMARKS else 0
        selected_industry = st.selectbox("Industry Benchmark", list(INDUSTRY_BENCHMARKS.keys()), index=default_idx)
        
        if st.session_state.processed_data is not None:
            if st.button("🔍 Analyze Financials", type="primary"):
                with st.spinner("Processing..."):
                    df = normalize_columns(st.session_state.processed_data)
                    st.session_state.analysis_results = calculate_metrics(df)
                    st.session_state.chat_history = []
                    st.rerun()

    st.title("📊 AI Financial Analyst Pro")
    
    if uploaded_file and st.session_state.processed_data is None:
        try:
            if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
            else: df = pd.read_excel(uploaded_file)
            st.session_state.processed_data = df
            st.success("File uploaded! Click 'Analyze Financials'.")
        except Exception as e: st.error(f"Error: {e}")

    if st.session_state.analysis_results is not None:
        results = st.session_state.analysis_results
        latest_year = results.iloc[-1]
        benchmarks = INDUSTRY_BENCHMARKS.get(selected_industry, list(INDUSTRY_BENCHMARKS.values())[0])
        
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "🎯 Benchmarks", "🤖 AI Insights", "💬 Chat"])
        
        with tab1:
            st.dataframe(results.set_index('Year').T.style.format("{:.2f}"))
            
            c1, c2 = st.columns(2)
            with c1:
                if 'Revenue' in results:
                    fig = go.Figure(go.Bar(x=results['Year'], y=results['Revenue'], name='Revenue'))
                    fig.update_layout(title="Revenue")
                    st.plotly_chart(fig, use_container_width=True)
            with c2:
                if 'Net Income' in results:
                    fig = go.Figure(go.Scatter(x=results['Year'], y=results['Net Income'], name='Net Income'))
                    fig.update_layout(title="Net Income")
                    st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader(f"Vs {selected_industry}")
            bench_data = []
            for m in ['Gross Margin', 'Net Margin', 'Revenue Growth', 'Current Ratio', 'Debt-to-Assets']:
                val = latest_year.get(m)
                status = get_status_color(m, val, benchmarks)
                icon = "🟢" if status == "green" else "🔴" if status == "red" else "🟡"
                val_fmt = format_metric(val, is_percent=(m!='Current Ratio'))
                bench_data.append({"Metric": m, "Value": val_fmt, "Status": icon})
            st.table(pd.DataFrame(bench_data))
            
        with tab3:
            if st.button("Generate AI Analysis"):
                if api_key:
                    st.markdown(get_gemini_analysis(api_key, selected_industry, results.to_string(), pd.DataFrame(bench_data).to_string()))
                else:
                    st.markdown(get_mock_analysis(selected_industry, latest_year))

        with tab4:
            for msg in st.session_state.chat_history:
                st.chat_message(msg["role"]).write(msg["content"])
            if prompt := st.chat_input("Ask a question..."):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)
                resp = chat_agent(prompt, results)
                st.chat_message("assistant").write(resp)
                st.session_state.chat_history.append({"role": "assistant", "content": resp})

if __name__ == "__main__":
    main()

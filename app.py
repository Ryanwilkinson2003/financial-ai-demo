import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai

# ==========================================
# 1. CONFIGURATION & PAGE SETUP
# ==========================================
st.set_page_config(
    page_title="AI Financial Analyst Pro V2.1",
    page_icon="📈",
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
if 'common_size' not in st.session_state:
    st.session_state.common_size = None

# ==========================================
# 2. CONSTANTS & MAPPING
# ==========================================

INDUSTRY_BENCHMARKS = {
    "Manufacturing": {"average": {"Gross Margin": (20, 35), "Net Margin": (3, 8), "Current Ratio": (1.2, 2.0)}},
    "SaaS / Technology": {"average": {"Gross Margin": (70, 80), "Net Margin": (5, 15), "Current Ratio": (1.5, 2.5)}},
    "Retail / E-commerce": {"average": {"Gross Margin": (25, 45), "Net Margin": (2, 5), "Current Ratio": (1.2, 2.0)}}
}

COLUMN_MAPPING = {
    "Year": ["year", "fiscal year", "fy", "date"],
    "Revenue": ["revenue", "sales", "total revenue", "turnover"],
    "COGS": ["cogs", "cost of goods sold", "cost of sales"],
    "Operating Expenses": ["operating expenses", "opex", "sg&a"],
    "Operating Income": ["operating income", "ebit", "operating profit"],
    "Net Income": ["net income", "net profit", "earnings"],
    "Total Assets": ["total assets", "assets"],
    "Current Assets": ["current assets"],
    "Total Liabilities": ["total liabilities", "liabilities"],
    "Current Liabilities": ["current liabilities"],
    "Inventory": ["inventory", "stock"],
    "Accounts Receivable": ["accounts receivable", "ar", "receivables"],
    "Accounts Payable": ["accounts payable", "ap", "payables"],
    "Equity": ["shareholders' equity", "equity", "total equity"],
    "Cash": ["cash", "cash and cash equivalents"],
    "Operating Cash Flow": ["operating cash flow", "cash from operations", "ocf"],
    "Interest Expense": ["interest expense", "finance costs"]
}

# ==========================================
# 3. UTILITY FUNCTIONS
# ==========================================

def normalize_columns(df):
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
    return df.rename(columns=renamed)

def safe_div(n, d):
    if d is None or n is None or pd.isna(d) or pd.isna(n) or d == 0:
        return None
    return n / d

def format_metric(val, is_percent=True):
    if val is None or pd.isna(val): return "N/A"
    if is_percent: return f"{val:.1f}%"
    return f"{val:.2f}x"

# ==========================================
# 4. CALCULATION ENGINE
# ==========================================

def calculate_metrics(df):
    df = df.sort_values('Year')
    results = df.copy()
    
    # PROFITABILITY
    if 'Revenue' in df and 'COGS' in df:
        results['Gross Margin'] = results.apply(lambda x: safe_div(x['Revenue'] - x['COGS'], x['Revenue']) * 100, axis=1)
    if 'Net Income' in df and 'Revenue' in df:
        results['Net Margin'] = results.apply(lambda x: safe_div(x['Net Income'], x['Revenue']) * 100, axis=1)
    if 'Net Income' in df and 'Total Assets' in df:
        results['ROA'] = results.apply(lambda x: safe_div(x['Net Income'], x['Total Assets']) * 100, axis=1)
    if 'Net Income' in df and 'Equity' in df:
        results['ROE'] = results.apply(lambda x: safe_div(x['Net Income'], x['Equity']) * 100, axis=1)

    # LIQUIDITY
    if 'Current Assets' in df and 'Current Liabilities' in df:
        results['Current Ratio'] = results.apply(lambda x: safe_div(x['Current Assets'], x['Current Liabilities']), axis=1)
    if 'Current Liabilities' in df:
        results['Quick Ratio'] = results.apply(lambda x: safe_div(
            (x.get('Cash', 0) or 0) + (x.get('Accounts Receivable', 0) or 0), 
            x['Current Liabilities']), axis=1)

    # EFFICIENCY
    if 'Revenue' in df and 'Total Assets' in df:
        results['Asset Turnover'] = results.apply(lambda x: safe_div(x['Revenue'], x['Total Assets']), axis=1)
    if 'COGS' in df and 'Inventory' in df:
        results['Inventory Turnover'] = results.apply(lambda x: safe_div(x['COGS'], x['Inventory']), axis=1)
        
    # LEVERAGE
    if 'Total Liabilities' in df and 'Total Assets' in df:
        results['Debt-to-Assets'] = results.apply(lambda x: safe_div(x['Total Liabilities'], x['Total Assets']) * 100, axis=1)

    # GROWTH
    for col in ['Revenue', 'Net Income', 'Total Assets']:
        if col in df:
            results[f'{col} Growth'] = results[col].pct_change() * 100

    return results

def calculate_common_size(df):
    common = df.copy()
    if 'Revenue' in df:
        for col in ['COGS', 'Gross Profit', 'Operating Expenses', 'Net Income']:
            if col in df: common[f'{col} (%)'] = common.apply(lambda x: safe_div(x[col], x['Revenue']) * 100, axis=1)
    return common

# ==========================================
# 5. AI ENGINE (GEMINI + MOCK FALLBACK)
# ==========================================

def get_mock_analysis(df, industry):
    """Generates realistic looking analysis without an API Key."""
    latest = df.iloc[-1]
    
    # Extract key metrics safely
    gm = latest.get('Gross Margin'); gm_str = f"{gm:.1f}%" if gm else "N/A"
    nm = latest.get('Net Margin'); nm_str = f"{nm:.1f}%" if nm else "N/A"
    cr = latest.get('Current Ratio'); cr_str = f"{cr:.2f}" if cr else "N/A"
    growth = latest.get('Revenue Growth'); growth_str = f"{growth:.1f}%" if growth else "N/A"
    
    # Logic for Strengths/Weaknesses
    strengths = []
    weaknesses = []
    
    # Profitability Logic
    if gm and gm > 40: strengths.append(f"Strong pricing power indicated by {gm_str} Gross Margin.")
    elif gm: weaknesses.append(f"Low Gross Margin ({gm_str}) suggests high production costs.")
    
    # Liquidity Logic
    if cr and cr > 1.5: strengths.append(f"Healthy liquidity position (Current Ratio: {cr_str}).")
    elif cr and cr < 1.0: weaknesses.append(f"CRITICAL: Liquidity crisis likely (Current Ratio {cr_str} is < 1.0).")
    
    # Growth Logic
    if growth and growth > 0: strengths.append(f"Positive revenue trajectory of {growth_str} YoY.")
    elif growth: weaknesses.append(f"Revenue contraction of {growth_str} is a major concern.")

    return f"""
    ### 🤖 AI Executive Summary (Mock Engine)
    **Status: Analysis Generated without API Key**
    
    The company shows a mixed financial profile within the **{industry}** sector. 
    Based on the latest data, the primary focus should be on **{'stabilizing liquidity' if (cr and cr < 1) else 'improving margins'}**.
    
    #### ✅ Key Strengths
    * {strengths[0] if len(strengths) > 0 else "Data insufficient for strength analysis."}
    * {strengths[1] if len(strengths) > 1 else "Stable asset base."}
    
    #### ⚠️ Risks & Concerns
    * {weaknesses[0] if len(weaknesses) > 0 else "Data insufficient for risk analysis."}
    * {weaknesses[1] if len(weaknesses) > 1 else "Operational efficiency could be improved."}
    
    #### 🚀 Strategic Recommendations
    1. **Cost Optimization:** Review COGS structure to improve the {gm_str} Gross Margin.
    2. **Working Capital:** Focus on inventory turnover to free up cash.
    3. **Debt Management:** Monitor leverage ratios closely in the coming fiscal year.
    """

def get_gemini_analysis(api_key, industry, data_summary, common_summary):
    if not api_key: 
        return None # Signal to use mock
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"Analyze this {industry} company based on these ratios:\n{data_summary}\nAnd common size:\n{common_summary}\nProvide executive summary, strengths, weaknesses, and recommendations."
        return model.generate_content(prompt).text
    except:
        return None # Fallback on error

# ==========================================
# 6. VISUALS & UI
# ==========================================

def main():
    with st.sidebar:
        st.header("⚙️ Configuration")
        # Modified: API Key is completely optional
        api_key = st.text_input("Gemini API Key (Optional)", type="password", help="Leave blank to use the built-in Mock Analyst")
        
        st.divider()
        st.subheader("1. Input Data")
        
        def clear_submit():
            st.session_state.processed_data = None
            st.session_state.analysis_results = None
            
        uploaded_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'], on_change=clear_submit)
        selected_industry = st.selectbox("Industry", list(INDUSTRY_BENCHMARKS.keys()))
        
        if st.session_state.processed_data is not None:
            if st.button("🚀 Run Analysis", type="primary"):
                df = normalize_columns(st.session_state.processed_data)
                st.session_state.analysis_results = calculate_metrics(df)
                st.session_state.common_size = calculate_common_size(df)
                st.rerun()

    st.title("💰 AI Financial Analyst V2.1")
    
    if uploaded_file and st.session_state.processed_data is None:
        try:
            if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
            else: df = pd.read_excel(uploaded_file)
            st.session_state.processed_data = df
            st.success("✅ Data Loaded")
        except: st.error("File Error")

    if st.session_state.analysis_results is not None:
        results = st.session_state.analysis_results
        common = st.session_state.common_size
        
        tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔬 Ratios", "🤖 Insights"])
        
        with tab1:
            st.subheader("Strategic Visuals")
            c1, c2 = st.columns(2)
            with c1:
                if 'Current Ratio' in results:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=results['Year'], y=results['Current Ratio'], name='Current Ratio'))
                    if 'Quick Ratio' in results: fig.add_trace(go.Scatter(x=results['Year'], y=results['Quick Ratio'], name='Quick Ratio'))
                    st.plotly_chart(fig, use_container_width=True)
            with c2:
                if 'Net Margin' in results and 'Asset Turnover' in results:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=results['Year'], y=results['Net Margin'], name='Net Margin %'))
                    fig.add_trace(go.Scatter(x=results['Year'], y=results['Asset Turnover'], name='Asset Turnover', yaxis='y2'))
                    fig.update_layout(yaxis2=dict(overlaying='y', side='right'))
                    st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.dataframe(results.set_index('Year').T.style.format(lambda x: "N/A" if pd.isna(x) else "{:.2f}".format(x)))

        with tab3:
            st.subheader("💡 Strategic Analysis")
            if st.button("Generate Insights"):
                # AUTOMATIC LOGIC: Try Gemini -> If fails/empty -> Use Mock
                with st.spinner("Analyzing..."):
                    ai_response = get_gemini_analysis(api_key, selected_industry, results.to_string(), common.to_string())
                    
                    if ai_response:
                        st.markdown(ai_response)
                    else:
                        if api_key: st.warning("Gemini API failed. Showing Mock Analysis instead.")
                        st.markdown(get_mock_analysis(results, selected_industry))

if __name__ == "__main__":
    main()

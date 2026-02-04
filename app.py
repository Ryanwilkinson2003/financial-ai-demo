import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai

# ==========================================
# 1. CONFIGURATION & PAGE SETUP
# ==========================================
st.set_page_config(
    page_title="Financial Statement Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-left: 5px solid #4F46E5;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1F2937;
    }
    .sub-text {
        color: #6B7280;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
if 'processed_data' not in st.session_state: st.session_state.processed_data = None
if 'common_size' not in st.session_state: st.session_state.common_size = None

# ==========================================
# 2. BENCHMARKS & MAPPING
# ==========================================

INDUSTRY_BENCHMARKS = {
    "Manufacturing": {"Gross Margin": (25, 35), "Net Margin": (5, 10), "Current Ratio": (1.2, 2.0), "Quick Ratio": (0.8, 1.2), "Debt-to-Equity": (0.5, 1.5), "ROE": (10, 15)},
    "SaaS / Software": {"Gross Margin": (70, 85), "Net Margin": (10, 25), "Current Ratio": (1.5, 3.0), "Quick Ratio": (1.5, 3.0), "Debt-to-Equity": (0.0, 0.5), "ROE": (15, 30)},
    "Retail / E-commerce": {"Gross Margin": (30, 50), "Net Margin": (3, 7), "Current Ratio": (1.2, 1.8), "Quick Ratio": (0.2, 0.6), "Debt-to-Equity": (0.5, 1.2), "ROE": (15, 25)},
    "Healthcare": {"Gross Margin": (55, 75), "Net Margin": (10, 20), "Current Ratio": (2.0, 4.0), "Quick Ratio": (1.8, 3.5), "Debt-to-Equity": (0.2, 0.8), "ROE": (10, 20)},
    "Construction": {"Gross Margin": (10, 20), "Net Margin": (2, 5), "Current Ratio": (1.1, 1.5), "Quick Ratio": (0.9, 1.2), "Debt-to-Equity": (0.8, 2.0), "ROE": (12, 18)},
    "Energy": {"Gross Margin": (35, 50), "Net Margin": (8, 15), "Current Ratio": (1.0, 1.5), "Quick Ratio": (0.8, 1.2), "Debt-to-Equity": (0.5, 1.0), "ROE": (8, 15)},
    "Real Estate": {"Gross Margin": (60, 70), "Net Margin": (20, 40), "Current Ratio": (0.5, 1.5), "Quick Ratio": (0.5, 1.5), "Debt-to-Equity": (1.5, 3.0), "ROE": (5, 10)},
    "Financial Services": {"Gross Margin": (90, 100), "Net Margin": (20, 35), "Current Ratio": (1.0, 1.2), "Quick Ratio": (1.0, 1.2), "Debt-to-Equity": (2.0, 5.0), "ROE": (8, 12)},
    "Consumer Goods": {"Gross Margin": (35, 50), "Net Margin": (5, 10), "Current Ratio": (1.2, 2.0), "Quick Ratio": (0.6, 1.0), "Debt-to-Equity": (0.5, 1.5), "ROE": (15, 25)},
    "Telecom": {"Gross Margin": (45, 60), "Net Margin": (8, 15), "Current Ratio": (0.8, 1.2), "Quick Ratio": (0.6, 1.0), "Debt-to-Equity": (1.0, 2.5), "ROE": (10, 20)},
    "Automotive": {"Gross Margin": (10, 20), "Net Margin": (3, 6), "Current Ratio": (1.1, 1.4), "Quick Ratio": (0.8, 1.0), "Debt-to-Equity": (1.0, 2.0), "ROE": (10, 18)},
    "Hospitality": {"Gross Margin": (60, 75), "Net Margin": (4, 9), "Current Ratio": (0.8, 1.5), "Quick Ratio": (0.4, 0.8), "Debt-to-Equity": (1.5, 3.0), "ROE": (20, 40)}
}

COLUMN_MAPPING = {
    "Year": ["year", "fiscal year", "fy"],
    "Revenue": ["revenue", "total revenue", "sales", "net sales"],
    "COGS": ["cogs", "cost of goods sold", "cost of sales"],
    "Operating Expenses": ["operating expenses", "opex", "sg&a"],
    "Operating Income": ["operating income", "operating profit", "ebit"],
    "Interest Expense": ["interest expense", "interest"],
    "Net Income": ["net income", "net profit", "net earnings"],
    "Cash": ["cash", "cash and cash equivalents"],
    "Short Term Investments": ["short term investments", "marketable securities"],
    "Accounts Receivable": ["accounts receivable", "receivables"],
    "Inventory": ["inventory", "inventories"],
    "Total Assets": ["total assets", "assets"],
    "Current Assets": ["current assets", "total current assets"],
    "Accounts Payable": ["accounts payable", "payables"],
    "Current Liabilities": ["current liabilities", "total current liabilities"],
    "Total Liabilities": ["total liabilities", "liabilities"],
    "Long Term Debt": ["long term debt", "non-current debt"],
    "Equity": ["shareholders' equity", "equity", "total equity"],
    "Operating Cash Flow": ["operating cash flow", "cash from operations"],
    "CapEx": ["capital expenditures", "capex"]
}

# ==========================================
# 3. DATA PROCESSING
# ==========================================

def clean_currency(x):
    if isinstance(x, str):
        clean = x.replace('$', '').replace(',', '').replace(' ', '')
        if '(' in clean and ')' in clean: clean = '-' + clean.replace('(', '').replace(')', '')
        try: return float(clean)
        except: return None
    return x

def normalize_and_clean(df):
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
    df = df.loc[:, ~df.columns.duplicated()]
    for col in df.columns:
        if col != 'Year':
            df[col] = df[col].apply(clean_currency)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def safe_div(n, d):
    try:
        if d is None or n is None or pd.isna(d) or pd.isna(n) or d == 0: return None
        return float(n) / float(d)
    except: return None

def safe_pct(n, d):
    val = safe_div(n, d)
    return val * 100 if val is not None else None

# ==========================================
# 4. CALCULATION ENGINE
# ==========================================

def calculate_metrics(df):
    results = df.copy()
    results = results.sort_values('Year')
    def get(col): return results[col] if col in results else pd.Series([None]*len(results))

    results['Gross Margin'] = results.apply(lambda x: safe_pct(x.get('Revenue',0) - x.get('COGS',0), x.get('Revenue')), axis=1)
    results['Net Margin'] = results.apply(lambda x: safe_pct(x.get('Net Income'), x.get('Revenue')), axis=1)
    results['ROA'] = results.apply(lambda x: safe_pct(x.get('Net Income'), x.get('Total Assets')), axis=1)
    results['ROE'] = results.apply(lambda x: safe_pct(x.get('Net Income'), x.get('Equity')), axis=1)
    results['Current Ratio'] = results.apply(lambda x: safe_div(x.get('Current Assets'), x.get('Current Liabilities')), axis=1)
    results['Quick Ratio'] = results.apply(lambda x: safe_div(
        (x.get('Cash', 0) or 0) + (x.get('Accounts Receivable', 0) or 0) + (x.get('Short Term Investments', 0) or 0),
        x.get('Current Liabilities')
    ), axis=1)
    results['Debt-to-Assets'] = results.apply(lambda x: safe_pct(x.get('Total Liabilities'), x.get('Total Assets')), axis=1)
    results['Debt-to-Equity'] = results.apply(lambda x: safe_div(x.get('Total Liabilities'), x.get('Equity')), axis=1)

    for m in ['Revenue', 'Net Income', 'Total Assets']:
        if m in results: results[f'{m} Growth'] = results[m].pct_change() * 100
    return results

def calculate_common_size(df):
    common = df.copy()
    if 'Revenue' in df:
        for c in ['COGS', 'Gross Profit', 'Operating Expenses', 'Net Income']:
            if c in df: common[f'{c} (%)'] = common.apply(lambda x: safe_pct(x[c], x['Revenue']), axis=1)
    if 'Total Assets' in df:
        for c in ['Current Assets', 'Inventory', 'Accounts Receivable', 'Cash', 'PP&E', 'Total Liabilities', 'Equity']:
            if c in df: common[f'{c} (%)'] = common.apply(lambda x: safe_pct(x[c], x['Total Assets']), axis=1)
    return common

# ==========================================
# 5. VISUALS & AI LOGIC
# ==========================================

def get_traffic_light(val, metric, industry_data):
    if val is None or pd.isna(val): return "⚪", "No Data", ""
    ranges = industry_data.get(metric)
    if not ranges: return "⚪", "No Benchmark", ""
    low, high = ranges
    is_inverse = metric in ['Debt-to-Equity', 'Debt-to-Assets']
    if not is_inverse:
        if val >= high: return "🟢", "Leader", f"+{val-high:.1f} pts"
        elif val >= low: return "🟡", "Average", "In Range"
        else: return "🔴", "Weak/Risk", f"{val-low:.1f} pts"
    else:
        if val <= low: return "🟢", "Leader", "Low Risk"
        elif val <= high: return "🟡", "Average", "In Range"
        else: return "🔴", "Weak/Risk", "High Risk"

def get_gemini_analysis(api_key, industry, data_str):
    if not api_key: return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        # --- NEW STRUCTURED PROMPT ---
        prompt = f"""
        Act as a Chief Financial Officer (CFO). Analyze the following company in the {industry} industry.
        
        DATA PROVIDED:
        {data_str}
        
        INSTRUCTIONS:
        1. **Executive Summary:** Start with 3 bullet points summarizing the "Main Story" of the financials.
        2. **Detailed Deep Dive:**
           - **Profitability:** Analyze margins and ROE.
           - **Liquidity & Solvency:** Analyze cash position and debt levels.
           - **Growth:** Analyze the trajectory of revenue vs profit.
        3. **Strategic Recommendations:** Provide 3 specific, high-level business recommendations.
        
        FORMAT: Use Markdown. Bold key terms. Keep it professional but clear.
        """
        return model.generate_content(prompt).text
    except: return None

def get_mock_analysis(industry, df):
    return f"""
    ### 📊 Executive Summary (Mock Analysis)
    * **Profitability:** Margins are currently stable but trail industry leaders.
    * **Liquidity:** The company maintains a defensive cash position.
    * **Growth:** Revenue shows moderate volatility over the 5-year period.
    
    ---
    ### 🔍 Detailed Deep Dive
    
    **1. Profitability & Efficiency**
    The Gross Margin indicates standard production efficiency for the {industry} sector. However, operating expenses should be audited to improve the bottom line.
    
    **2. Liquidity & Solvency**
    Current and Quick ratios suggest the company can meet short-term obligations, though inventory turnover may be tying up working capital.
    
    **3. Growth Trajectory**
    While top-line revenue is present, net income growth has not perfectly correlated, suggesting "empty calories" revenue growth in certain years.
    
    ---
    ### 🚀 Strategic Recommendations
    1. **Optimize COGS:** Renegotiate with primary suppliers to boost Gross Margin by 1-2%.
    2. **Debt Structuring:** Consider refinancing long-term debt if interest rates are favorable.
    3. **Cash Flow:** Focus on reducing Days Sales Outstanding (DSO) to improve cash conversion.
    """

def chat_with_data(user_query, df_context, api_key):
    """Simple chat handler. In a real app, you would pass history to Gemini."""
    if not api_key:
        # Mock Chat Logic
        return f"**[Mock AI]:** Based on the data, I see that your latest Net Margin is {df_context.iloc[-1].get('Net Margin', 'N/A'):.1f}%. (To get real AI answers, please add an API Key)."
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        Context: The user is asking about this financial data:
        {df_context.to_string()}
        
        User Question: {user_query}
        
        Answer professionally in 2-3 sentences.
        """
        return model.generate_content(prompt).text
    except:
        return "I'm having trouble connecting to the AI right now."

# ==========================================
# 6. MAIN APPLICATION
# ==========================================

def main():
    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("Gemini API Key (Optional)", type="password")
        st.divider()
        
        def clear_cache():
            st.session_state.processed_data = None
            st.session_state.analysis_results = None
            st.session_state.chat_history = []
            
        uploaded_file = st.file_uploader("Upload Financials", type=['csv', 'xlsx'], on_change=clear_cache)
        selected_industry = st.selectbox("Select Industry", list(INDUSTRY_BENCHMARKS.keys()))
        
        if st.session_state.processed_data is not None:
            if st.button("🚀 Analyze Financials", type="primary"):
                with st.spinner("Crunching Numbers..."):
                    df_clean = normalize_and_clean(st.session_state.processed_data)
                    st.session_state.analysis_results = calculate_metrics(df_clean)
                    st.session_state.common_size = calculate_common_size(df_clean)
                    st.rerun()

    # --- Main Content ---
    st.markdown('<div class="main-header">Financial Statement Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Institutional-grade benchmarking and AI insights</div>', unsafe_allow_html=True)
    st.markdown("---")

    if not uploaded_file and st.session_state.analysis_results is None:
        st.info("👋 Welcome! Upload your balance sheet, income statement, or cash flow to begin.")
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown('<div class="metric-card"><h3>📥 1. Upload</h3><p>Upload any CSV or Excel file.</p></div>', unsafe_allow_html=True)
        with c2: st.markdown('<div class="metric-card"><h3>⚖️ 2. Benchmark</h3><p>Compare against 12+ industries.</p></div>', unsafe_allow_html=True)
        with c3: st.markdown('<div class="metric-card"><h3>🤖 3. Insights</h3><p>Get AI-powered executive summaries.</p></div>', unsafe_allow_html=True)

    # File Loader
    if uploaded_file and st.session_state.processed_data is None:
        try:
            if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
            else: df = pd.read_excel(uploaded_file)
            st.session_state.processed_data = df
            st.success("✅ File Loaded Successfully!")
        except Exception as e: st.error(f"Error: {e}")

    # Results
    if st.session_state.analysis_results is not None:
        results = st.session_state.analysis_results
        common = st.session_state.common_size
        latest = results.iloc[-1]
        
        # Tabs
        t1, t2, t3, t4, t5 = st.tabs(["🚦 Benchmarks", "📊 Visual Trends", "🔬 Deep Data", "🤖 AI Report", "💬 Assistant"])
        
        with t1:
            st.subheader(f"Performance vs. {selected_industry}")
            bench_data = []
            metrics = ["Gross Margin", "Net Margin", "Current Ratio", "Quick Ratio", "Debt-to-Equity", "ROE"]
            for m in metrics:
                val = latest.get(m)
                icon, status, dist = get_traffic_light(val, m, INDUSTRY_BENCHMARKS[selected_industry])
                if val is None or pd.isna(val): val_fmt = "N/A"
                else: val_fmt = f"{val:.1f}%" if "Margin" in m or "ROE" in m else f"{val:.2f}"
                bench_data.append({"Metric": m, "Value": val_fmt, "Status": icon, "Rating": status, "Variance": dist})
            st.dataframe(pd.DataFrame(bench_data), use_container_width=True, hide_index=True)
            
        with t2:
            c1, c2 = st.columns(2)
            with c1:
                if 'Revenue' in results and 'Net Income' in results:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=results['Year'], y=results['Revenue'], name='Revenue', marker_color='#6366F1'))
                    fig.add_trace(go.Scatter(x=results['Year'], y=results['Net Income'], name='Net Income', line=dict(color='#10B981', width=3)))
                    fig.update_layout(title="Revenue vs Profit", legend=dict(orientation="h"))
                    st.plotly_chart(fig, use_container_width=True)
            with c2:
                if 'Current Ratio' in results:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=results['Year'], y=results['Current Ratio'], name='Current Ratio', line=dict(color='#F59E0B', width=3)))
                    if 'Quick Ratio' in results: fig.add_trace(go.Scatter(x=results['Year'], y=results['Quick Ratio'], name='Quick Ratio', line=dict(dash='dot')))
                    fig.update_layout(title="Liquidity Trend", legend=dict(orientation="h"))
                    st.plotly_chart(fig, use_container_width=True)

        with t3:
            st.dataframe(results.set_index('Year').T.style.format(lambda x: "N/A" if pd.isna(x) else "{:.2f}".format(x)))
            st.write("Common Size (% of Rev/Assets)")
            st.dataframe(common.set_index('Year').T.style.format(lambda x: "N/A" if pd.isna(x) else "{:.1f}%".format(x)))

        with t4:
            st.subheader("CEO-Level Financial Report")
            if st.button("Generate Full Report"):
                with st.spinner("Writing Executive Brief..."):
                    if api_key:
                        analysis = get_gemini_analysis(api_key, selected_industry, results.to_string())
                        if analysis: st.markdown(analysis)
                        else: st.markdown(get_mock_analysis(selected_industry, results))
                    else: st.markdown(get_mock_analysis(selected_industry, results))

        with t5:
            st.subheader("💬 AI Financial Assistant")
            st.caption("Ask questions about your data (e.g., 'Why is my ROE declining?')")
            
            for msg in st.session_state.chat_history:
                st.chat_message(msg['role']).write(msg['content'])
            
            if prompt := st.chat_input("Ask a question..."):
                st.session_state.chat_history.append({'role':'user', 'content':prompt})
                st.chat_message('user').write(prompt)
                
                with st.spinner("Thinking..."):
                    response = chat_with_data(prompt, results, api_key)
                    st.session_state.chat_history.append({'role':'assistant', 'content':response})
                    st.chat_message('assistant').write(response)

if __name__ == "__main__":
    main()

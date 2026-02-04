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
    page_title="AI Financial Analyst Pro V2",
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
# 2. CONSTANTS & BENCHMARKS (EXPANDED)
# ==========================================

# Expanded to cover 20+ metrics. 
# NOTE: In a real app, these would be in a database. Here we use realistic placeholders.
INDUSTRY_BENCHMARKS = {
    "Manufacturing": {
        "average": {"Gross Margin": (20, 35), "Net Margin": (3, 8), "Current Ratio": (1.2, 2.0), "Quick Ratio": (0.8, 1.2), "ROE": (10, 15), "Debt-to-Assets": (30, 50), "Inventory Turnover": (4, 8)},
        "leaders": {"Gross Margin": (35, 50), "Net Margin": (10, 18), "Current Ratio": (2.0, 3.5), "Quick Ratio": (1.5, 2.5), "ROE": (20, 30), "Debt-to-Assets": (10, 25), "Inventory Turnover": (8, 12)},
        "failure_patterns": {"Gross Margin": (0, 15), "Net Margin": (-15, 0), "Current Ratio": (0, 0.8), "Quick Ratio": (0, 0.5), "ROE": (-50, 0), "Debt-to-Assets": (65, 100), "Inventory Turnover": (0, 2)}
    },
    "SaaS / Technology": {
        "average": {"Gross Margin": (70, 80), "Net Margin": (5, 15), "Current Ratio": (1.5, 2.5), "Quick Ratio": (1.5, 2.5), "ROE": (5, 15), "Debt-to-Assets": (10, 30), "Inventory Turnover": (0, 0)}, # N/A for pure SaaS
        "leaders": {"Gross Margin": (82, 95), "Net Margin": (25, 40), "Current Ratio": (2.5, 4.0), "Quick Ratio": (2.5, 4.0), "ROE": (20, 35), "Debt-to-Assets": (0, 10), "Inventory Turnover": (0, 0)},
        "failure_patterns": {"Gross Margin": (0, 50), "Net Margin": (-100, -20), "Current Ratio": (0, 0.8), "Quick Ratio": (0, 0.8), "ROE": (-100, -10), "Debt-to-Assets": (60, 100), "Inventory Turnover": (0, 0)}
    },
    "Retail / E-commerce": {
        "average": {"Gross Margin": (25, 45), "Net Margin": (2, 5), "Current Ratio": (1.2, 2.0), "Quick Ratio": (0.2, 0.6), "ROE": (10, 20), "Debt-to-Assets": (30, 50), "Inventory Turnover": (6, 12)},
        "leaders": {"Gross Margin": (50, 65), "Net Margin": (7, 12), "Current Ratio": (2.0, 3.0), "Quick Ratio": (0.8, 1.5), "ROE": (25, 40), "Debt-to-Assets": (10, 25), "Inventory Turnover": (12, 20)},
        "failure_patterns": {"Gross Margin": (0, 15), "Net Margin": (-20, 0), "Current Ratio": (0, 0.9), "Quick Ratio": (0, 0.1), "ROE": (-50, 0), "Debt-to-Assets": (70, 100), "Inventory Turnover": (0, 3)}
    }
    # (Add other industries as needed)
}

# The "Super-Mapper" for messy excel files
COLUMN_MAPPING = {
    # --- Income Statement ---
    "Year": ["year", "fiscal year", "period", "fy", "date"],
    "Revenue": ["revenue", "total revenue", "sales", "net sales", "turnover", "gross revenue"],
    "COGS": ["cogs", "cost of goods sold", "cost of sales", "cost of revenue", "direct costs"],
    "Gross Profit": ["gross profit", "gross margin dollar"],
    "Operating Expenses": ["operating expenses", "opex", "sg&a", "selling, general and administrative", "operating costs"],
    "EBITDA": ["ebitda"],
    "Operating Income": ["operating income", "operating profit", "ebit"],
    "Interest Expense": ["interest expense", "interest", "finance costs"],
    "Net Income": ["net income", "net profit", "earnings", "net earnings", "profit after tax"],
    "EPS": ["earnings per share", "eps", "basic eps"],
    
    # --- Balance Sheet Assets ---
    "Total Assets": ["total assets", "assets"],
    "Current Assets": ["current assets", "total current assets"],
    "Cash": ["cash", "cash and cash equivalents", "cash & equivalents", "cash & bank"],
    "Short Term Investments": ["short term investments", "marketable securities"],
    "Accounts Receivable": ["accounts receivable", "receivables", "ar", "debtors"],
    "Inventory": ["inventory", "inventories", "stock"],
    "Prepaid Expenses": ["prepaid expenses", "prepaids"],
    "Long Term Investments": ["long term investments", "non-current investments"],
    "PP&E": ["property, plant & equipment", "pp&e", "fixed assets", "net property plant and equipment"],
    "Goodwill": ["goodwill", "goodwill and intangibles"],
    "Intangible Assets": ["intangible assets", "intangibles"],
    
    # --- Balance Sheet Liabilities ---
    "Total Liabilities": ["total liabilities", "liabilities"],
    "Current Liabilities": ["current liabilities", "total current liabilities"],
    "Accounts Payable": ["accounts payable", "payables", "ap", "creditors"],
    "Short Term Debt": ["short term debt", "current portion of long term debt", "notes payable", "short-term borrowings"],
    "Long Term Debt": ["long term debt", "non-current debt", "bonds payable", "mortgages payable"],
    "Deferred Revenue": ["deferred revenue", "unearned revenue"],
    
    # --- Equity ---
    "Equity": ["shareholders' equity", "equity", "total equity", "stockholders' equity", "net worth"],
    "Retained Earnings": ["retained earnings", "accumulated deficit"],
    "Common Stock": ["common stock", "share capital"],
    
    # --- Cash Flow ---
    "Operating Cash Flow": ["cash flow from operations", "operating cash flow", "net cash from operating activities", "ocf"],
    "CapEx": ["capital expenditures", "capex", "purchases of property and equipment", "additions to property plant and equipment"],
    "Free Cash Flow": ["free cash flow", "fcf"]
}

# ==========================================
# 3. UTILITY FUNCTIONS
# ==========================================

def normalize_columns(df):
    """Deep search normalization for messy headers."""
    df.columns = [str(c).strip().lower() for c in df.columns]
    renamed = {}
    
    # 1. Exact Match
    for standard, variations in COLUMN_MAPPING.items():
        for col in df.columns:
            if col in variations:
                renamed[col] = standard
                
    # 2. Fuzzy Match (if not found)
    for standard, variations in COLUMN_MAPPING.items():
        if standard not in renamed.values():
            for col in df.columns:
                if col not in renamed:
                    for v in variations:
                        if v in col: # contains logic
                            renamed[col] = standard
                            break
    
    return df.rename(columns=renamed)

def safe_div(n, d):
    """Safe division returning None (not 0) if data is missing or div by zero."""
    if d is None or n is None or pd.isna(d) or pd.isna(n) or d == 0:
        return None
    return n / d

def format_metric(val, is_percent=True, is_currency=False):
    if val is None: return "N/A"
    if is_currency: return f"${val:,.0f}"
    if is_percent: return f"{val:.1f}%"
    return f"{val:.2f}x"

# ==========================================
# 4. ADVANCED CALCULATION ENGINE
# ==========================================

def calculate_metrics(df):
    df = df.sort_values('Year')
    results = df.copy()
    
    # ---------------- PROFITABILITY ----------------
    if 'Revenue' in df and 'COGS' in df:
        results['Gross Margin'] = results.apply(lambda x: safe_div(x['Revenue'] - x['COGS'], x['Revenue']) * 100, axis=1)
    
    if 'Operating Income' in df and 'Revenue' in df:
        results['Operating Margin'] = results.apply(lambda x: safe_div(x['Operating Income'], x['Revenue']) * 100, axis=1)
    elif 'EBITDA' in df and 'Revenue' in df: # Proxy if OpInc missing
        results['Operating Margin'] = results.apply(lambda x: safe_div(x['EBITDA'], x['Revenue']) * 100, axis=1)

    if 'Net Income' in df and 'Revenue' in df:
        results['Net Margin'] = results.apply(lambda x: safe_div(x['Net Income'], x['Revenue']) * 100, axis=1)
    
    if 'Net Income' in df and 'Total Assets' in df:
        results['ROA'] = results.apply(lambda x: safe_div(x['Net Income'], x['Total Assets']) * 100, axis=1)
        
    if 'Net Income' in df and 'Equity' in df:
        results['ROE'] = results.apply(lambda x: safe_div(x['Net Income'], x['Equity']) * 100, axis=1)

    # ---------------- LIQUIDITY ----------------
    if 'Current Assets' in df and 'Current Liabilities' in df:
        results['Current Ratio'] = results.apply(lambda x: safe_div(x['Current Assets'], x['Current Liabilities']), axis=1)
    
    # Quick Ratio (Acid Test): (Cash + AR + ShortTermInvest) / CL
    if 'Current Liabilities' in df:
        results['Quick Ratio'] = results.apply(
            lambda x: safe_div(
                (x.get('Cash', 0) or 0) + (x.get('Accounts Receivable', 0) or 0) + (x.get('Short Term Investments', 0) or 0), 
                x['Current Liabilities']
            ), axis=1
        )

    # ---------------- EFFICIENCY & CASH GAP ----------------
    # Asset Turnover
    if 'Revenue' in df and 'Total Assets' in df:
        results['Asset Turnover'] = results.apply(lambda x: safe_div(x['Revenue'], x['Total Assets']), axis=1)
    
    # Inventory Turnover & DSI (Days Sales Inventory)
    if 'COGS' in df and 'Inventory' in df:
        results['Inventory Turnover'] = results.apply(lambda x: safe_div(x['COGS'], x['Inventory']), axis=1)
        results['Days Sales Inventory'] = results.apply(lambda x: safe_div(365, x.get('Inventory Turnover')), axis=1)

    # Receivables Turnover & DSO (Days Sales Outstanding)
    if 'Revenue' in df and 'Accounts Receivable' in df:
        results['AR Turnover'] = results.apply(lambda x: safe_div(x['Revenue'], x['Accounts Receivable']), axis=1)
        results['Days Sales Outstanding'] = results.apply(lambda x: safe_div(365, x.get('AR Turnover')), axis=1)

    # Payables Turnover & DPO (Days Payable Outstanding) - Need COGS & AP
    if 'COGS' in df and 'Accounts Payable' in df:
        results['AP Turnover'] = results.apply(lambda x: safe_div(x['COGS'], x['Accounts Payable']), axis=1)
        results['Days Payable Outstanding'] = results.apply(lambda x: safe_div(365, x.get('AP Turnover')), axis=1)

    # Cash Conversion Cycle (Cash Gap) = DSI + DSO - DPO
    if 'Days Sales Inventory' in results and 'Days Sales Outstanding' in results and 'Days Payable Outstanding' in results:
        results['Cash Conversion Cycle'] = results['Days Sales Inventory'] + results['Days Sales Outstanding'] - results.get('Days Payable Outstanding', 0)

    # ---------------- LEVERAGE ----------------
    if 'Total Liabilities' in df and 'Total Assets' in df:
        results['Debt-to-Assets'] = results.apply(lambda x: safe_div(x['Total Liabilities'], x['Total Assets']) * 100, axis=1)
        
    if 'Total Liabilities' in df and 'Equity' in df:
        results['Debt-to-Equity'] = results.apply(lambda x: safe_div(x['Total Liabilities'], x['Equity']), axis=1)
        
    if 'Operating Income' in df and 'Interest Expense' in df:
        results['Interest Coverage'] = results.apply(lambda x: safe_div(x['Operating Income'], x['Interest Expense']), axis=1)

    # ---------------- GROWTH (YoY) ----------------
    metrics_to_grow = ['Revenue', 'Net Income', 'Total Assets', 'Operating Cash Flow', 'EPS']
    for m in metrics_to_grow:
        if m in df:
            results[f'{m} Growth'] = results[m].pct_change() * 100

    return results

def calculate_common_size(df):
    """Converts absolute numbers to percentages of Revenue (IS) or Assets (BS)."""
    common = df.copy()
    
    # 1. Income Statement items (as % of Revenue)
    is_cols = ['COGS', 'Gross Profit', 'Operating Expenses', 'Operating Income', 'Net Income', 'Interest Expense', 'EBITDA']
    if 'Revenue' in df:
        for col in is_cols:
            if col in df:
                common[f'{col} (%)'] = common.apply(lambda x: safe_div(x[col], x['Revenue']) * 100, axis=1)
                
    # 2. Balance Sheet items (as % of Total Assets)
    bs_cols = ['Current Assets', 'Cash', 'Accounts Receivable', 'Inventory', 'PP&E', 'Total Liabilities', 'Current Liabilities', 'Long Term Debt', 'Equity', 'Goodwill']
    if 'Total Assets' in df:
        for col in bs_cols:
            if col in df:
                common[f'{col} (%)'] = common.apply(lambda x: safe_div(x[col], x['Total Assets']) * 100, axis=1)
                
    return common

# ==========================================
# 5. AI INTEGRATION
# ==========================================

def get_gemini_analysis(api_key, industry, data_summary, common_size_summary):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        Act as a Senior Financial Analyst. Analyze this company in the {industry} industry.
        
        DATA PROVIDED:
        1. Financial Ratios & Trends (Last 3-5 Years):
        {data_summary}
        
        2. Common Size Analysis (Margins & Asset Structure):
        {common_size_summary}
        
        CRITICAL INSTRUCTIONS:
        1. **IGNORE MISSING DATA:** If a metric is "N/A", "None", or blank, DO NOT mention it. Assume it was not provided. DO NOT say "Cash flow is concerning" if cash flow data is missing.
        2. **CONTEXT:** Compare against standard {industry} benchmarks.
        3. **OUTPUT:**
           - **Executive Summary:** The "big picture" health.
           - **Strengths:** 3 specific areas performing well.
           - **Weaknesses:** 3 specific risks (Liquidity, Solvency, or Profitability).
           - **Actionable Recommendations:** 3 strategic moves.
        """
        
        with st.spinner('🤖 Gemini is analyzing complex patterns...'):
            response = model.generate_content(prompt)
            return response.text
    except Exception as e:
        return f"API ERROR: {str(e)}"

# ==========================================
# 6. VISUALIZATION ENGINE
# ==========================================

def render_charts(df):
    st.subheader("📊 Strategic Visuals")
    
    col1, col2 = st.columns(2)
    
    # Chart 1: Liquidity Trap (Current vs Quick)
    with col1:
        if 'Current Ratio' in df and 'Quick Ratio' in df:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Year'], y=df['Current Ratio'], name='Current Ratio', line=dict(color='#10B981', width=3)))
            fig.add_trace(go.Scatter(x=df['Year'], y=df['Quick Ratio'], name='Quick Ratio', line=dict(color='#EF4444', width=3, dash='dot')))
            fig.update_layout(title="Liquidity Quality (Current vs. Quick)", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
    # Chart 2: Profitability Drivers (DuPontish)
    with col2:
        if 'Net Margin' in df and 'Asset Turnover' in df:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df['Year'], y=df['Net Margin'], name='Net Margin %', yaxis='y1', marker_color='#6366F1'))
            fig.add_trace(go.Scatter(x=df['Year'], y=df['Asset Turnover'], name='Asset Turnover', yaxis='y2', line=dict(color='#F59E0B', width=3)))
            fig.update_layout(
                title="Profit Drivers (Margin vs. Efficiency)",
                yaxis=dict(title="Net Margin %"),
                yaxis2=dict(title="Asset Turnover (x)", overlaying='y', side='right')
            )
            st.plotly_chart(fig, use_container_width=True)
            
    col3, col4 = st.columns(2)
    
    # Chart 3: Top Line vs Bottom Line
    with col3:
        if 'Revenue Growth' in df and 'Net Income Growth' in df:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df['Year'], y=df['Revenue Growth'], name='Revenue Growth', marker_color='#3B82F6'))
            fig.add_trace(go.Scatter(x=df['Year'], y=df['Net Income Growth'], name='Net Income Growth', line=dict(color='#10B981', width=3)))
            fig.update_layout(title="Growth Trajectory (Revenue vs. Profit)", yaxis_title="Growth %")
            st.plotly_chart(fig, use_container_width=True)

    # Chart 4: Cash vs Debt (Solvency)
    with col4:
        if 'Cash' in df and 'Total Liabilities' in df:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df['Year'], y=df['Cash'], name='Cash Reserves', marker_color='#10B981'))
            fig.add_trace(go.Bar(x=df['Year'], y=df['Total Liabilities'], name='Total Debt', marker_color='#EF4444'))
            fig.update_layout(title="Solvency Check (Cash vs. Total Debt)", barmode='group')
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 7. MAIN APP UI
# ==========================================

def main():
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Key Logic
        if "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
            st.success("API Key Loaded from Secrets")
        else:
            api_key = st.text_input("Gemini API Key", type="password")

        st.divider()
        st.subheader("1. Data Input")
        
        # CLEAR FUNCTION for new uploads
        def clear_submit():
            st.session_state.processed_data = None
            st.session_state.analysis_results = None
            st.session_state.common_size = None
            
        uploaded_file = st.file_uploader(
            "Upload Financials (CSV/Excel)", 
            type=['csv', 'xlsx'], 
            on_change=clear_submit
        )
        
        st.divider()
        st.subheader("2. Industry Context")
        selected_industry = st.selectbox("Select Industry", list(INDUSTRY_BENCHMARKS.keys()))
        
        if st.session_state.processed_data is not None:
            if st.button("🚀 Run Deep Analysis", type="primary"):
                with st.spinner("Calculating 20+ Ratios & Common Size Analysis..."):
                    df = normalize_columns(st.session_state.processed_data)
                    st.session_state.analysis_results = calculate_metrics(df)
                    st.session_state.common_size = calculate_common_size(df)
                    st.session_state.chat_history = [] # Reset chat
                    st.rerun()

    # --- Main Page ---
    st.title("💰 AI Financial Analyst Pro V2.0")
    
    # File Processor
    if uploaded_file and st.session_state.processed_data is None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.processed_data = df
            st.success(f"✅ Loaded {len(df)} years of data successfully.")
        except Exception as e:
            st.error(f"File Error: {e}")

    # Results Display
    if st.session_state.analysis_results is not None:
        results = st.session_state.analysis_results
        common_size = st.session_state.common_size
        latest = results.iloc[-1]
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dashboard", 
            "📐 Common Size", 
            "🔬 Ratio Analysis", 
            "🤖 AI Insights", 
            "💬 AI Assistant"
        ])
        
        # TAB 1: VISUAL DASHBOARD
        with tab1:
            render_charts(results)
            st.markdown("### 📋 Key Metrics Summary (Latest Year)")
            
            # Smart KPI Cards
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Revenue Growth", format_metric(latest.get('Revenue Growth'), True), delta_color="normal")
            kpi2.metric("Net Margin", format_metric(latest.get('Net Margin'), True))
            kpi3.metric("Current Ratio", format_metric(latest.get('Current Ratio'), False))
            kpi4.metric("ROE", format_metric(latest.get('ROE'), True))

        # TAB 2: COMMON SIZE ANALYSIS
        with tab2:
            st.subheader("Common Size Income Statement (% of Revenue)")
            # Filter columns that end with (%) and are IS related
            is_cols = [c for c in common_size.columns if '(%)' in c and c.replace(' (%)','') in ['COGS','Gross Profit','Net Income','Operating Expenses']]
            st.dataframe(common_size[['Year'] + is_cols].set_index('Year').style.format("{:.1f}%"))
            
            st.subheader("Common Size Balance Sheet (% of Assets)")
            bs_cols = [c for c in common_size.columns if '(%)' in c and c not in is_cols]
            st.dataframe(common_size[['Year'] + bs_cols].set_index('Year').style.format("{:.1f}%"))

        # TAB 3: RATIO ANALYSIS (ALL DATA)
        with tab3:
            st.subheader("Comprehensive Ratio Table")
            st.dataframe(results.set_index('Year').T.style.format(lambda x: "N/A" if pd.isna(x) else "{:.2f}".format(x)))

        # TAB 4: AI ANALYSIS
        with tab4:
            st.subheader("🤖 Generative AI Strategic Assessment")
            if st.button("Generate Report"):
                if not api_key:
                    st.warning("Please provide an API Key in settings.")
                else:
                    # Prepare Data Context
                    ratios_str = results.to_string()
                    common_str = common_size.to_string()
                    analysis = get_gemini_analysis(api_key, selected_industry, ratios_str, common_str)
                    st.markdown(analysis)

        # TAB 5: CHAT
        with tab5:
            st.subheader("Ask the Analyst")
            for msg in st.session_state.chat_history:
                st.chat_message(msg['role']).write(msg['content'])
            
            if prompt := st.chat_input("Ex: Why is the Quick Ratio so low?"):
                st.session_state.chat_history.append({'role':'user', 'content':prompt})
                st.chat_message('user').write(prompt)
                
                # Simple Mock Chat Logic (Upgrade to Gemini if needed)
                response = f"Based on the data, the {prompt}..." 
                # Ideally pass this to Gemini as well, simplified here for length
                
                st.session_state.chat_history.append({'role':'assistant', 'content':"I can analyze that. (Note: Connect Chat to Gemini API for full Q&A)."})
                st.chat_message('assistant').write("I can analyze that. (Connect Chat to Gemini API for full Q&A).")

if __name__ == "__main__":
    main()

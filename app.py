import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="FinSight AI | Master Edition",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional UI Styling
st.markdown("""
<style>
    /* Global Font & Padding */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    /* Card Design */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-card h4 {
        color: #4B5563;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .metric-card p {
        color: #1F2937;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Dark Mode Overrides */
    @media (prefers-color-scheme: dark) {
        .metric-card {
            background-color: #262730;
            border: 1px solid #41424C;
        }
        .metric-card h4 { color: #9CA3AF; }
        .metric-card p { color: #F3F4F6; }
    }
    
    /* Headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #6366F1; /* Indigo */
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E0E7FF;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
if 'processed_data' not in st.session_state: st.session_state.processed_data = None
if 'common_size' not in st.session_state: st.session_state.common_size = None

# ==========================================
# 2. INTELLIGENCE LAYERS
# ==========================================

INDUSTRY_BENCHMARKS = {
    "Manufacturing": {"Gross Margin": (25, 35), "Net Margin": (5, 10), "Current Ratio": (1.2, 2.0), "Quick Ratio": (0.8, 1.2), "Debt-to-Equity": (0.5, 1.5), "ROE": (10, 15)},
    "SaaS / Software": {"Gross Margin": (70, 85), "Net Margin": (10, 25), "Current Ratio": (1.5, 3.0), "Quick Ratio": (1.5, 3.0), "Debt-to-Equity": (0.0, 0.5), "ROE": (15, 30)},
    "Retail / E-commerce": {"Gross Margin": (30, 50), "Net Margin": (3, 7), "Current Ratio": (1.2, 1.8), "Quick Ratio": (0.2, 0.6), "Debt-to-Equity": (0.5, 1.2), "ROE": (15, 25)},
    "General": {"Gross Margin": (30, 60), "Net Margin": (5, 15), "Current Ratio": (1.0, 2.0), "Quick Ratio": (1.0, 1.5), "Debt-to-Equity": (0.5, 2.0), "ROE": (10, 20)}
}
# Map others to General for safety
for ind in ["Healthcare", "Construction", "Energy", "Real Estate", "Financial Services", "Consumer Goods", "Telecom", "Automotive", "Hospitality"]:
    if ind not in INDUSTRY_BENCHMARKS: INDUSTRY_BENCHMARKS[ind] = INDUSTRY_BENCHMARKS["General"]

INDUSTRY_PROMPTS = {
    "SaaS / Software": "Focus heavily on Revenue Growth and Gross Retention. Net Losses are acceptable if growth > 20%. Ignore Inventory.",
    "Manufacturing": "Focus on Asset Turnover, Inventory Management (DSI), and ROA. High CAPEX is normal.",
    "Retail / E-commerce": "Focus on Inventory Turnover, Cash Conversion Cycle, and Thin Margins.",
    "General": "Provide a balanced analysis of Profitability, Liquidity, and Solvency."
}

# Extensive Column Mapping (The "Rosetta Stone" for Financials)
COLUMN_MAPPING = {
    "Year": ["year", "fiscal year", "fy"],
    # P&L
    "Revenue": ["revenue", "total revenue", "sales", "net sales", "turnover"],
    "COGS": ["cogs", "cost of goods sold", "cost of sales", "cost of revenue"],
    "Gross Profit": ["gross profit", "gross margin"],
    "Operating Expenses": ["operating expenses", "opex", "sg&a"],
    "Operating Income": ["operating income", "operating profit", "ebit"],
    "Interest Expense": ["interest expense", "interest", "finance costs"],
    "Net Income": ["net income", "net profit", "net earnings"],
    "EPS": ["eps", "earnings per share"],
    # Balance Sheet - Assets
    "Cash": ["cash", "cash and cash equivalents"],
    "Short Term Investments": ["short term investments", "marketable securities"],
    "Accounts Receivable": ["accounts receivable", "receivables", "ar"],
    "Inventory": ["inventory", "stock"],
    "Current Assets": ["current assets", "total current assets"],
    "PP&E": ["property, plant & equipment", "pp&e", "fixed assets"],
    "Goodwill": ["goodwill", "intangible assets"],
    "Total Assets": ["total assets", "assets"],
    # Balance Sheet - Liabilities/Equity
    "Accounts Payable": ["accounts payable", "payables", "ap"],
    "Current Liabilities": ["current liabilities", "total current liabilities"],
    "Long Term Debt": ["long term debt", "non-current debt", "bonds payable"],
    "Total Liabilities": ["total liabilities", "liabilities"],
    "Equity": ["shareholders' equity", "equity", "total equity", "stockholders' equity"],
    # Cash Flow
    "Operating Cash Flow": ["operating cash flow", "cash from operations", "ocf", "net cash from operating activities"],
    "CapEx": ["capital expenditures", "capex", "purchases of property"]
}

# ==========================================
# 3. DATA ENGINE (CLEANING & LOGIC)
# ==========================================

def clean_currency(x):
    """Robust cleaner for strings like '$1,000.00', '(500)', etc."""
    if isinstance(x, str):
        clean = x.replace('$', '').replace(',', '').replace(' ', '').replace('%', '')
        if '(' in clean and ')' in clean: 
            clean = '-' + clean.replace('(', '').replace(')', '')
        try: return float(clean)
        except: return None
    return x

def normalize_and_clean(df):
    """Auto-detects orientation and standardizes column names."""
    # 1. Orientation Check (Are years columns or rows?)
    first_col = df.iloc[:, 0].astype(str).str.lower().tolist()
    metric_matches = sum(1 for v in first_col if 'revenue' in v or 'net income' in v or 'assets' in v)
    
    # If first column has metrics, FLIP IT.
    if metric_matches > 0:
        df = df.set_index(df.columns[0]).T
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Year'}, inplace=True)
    
    # 2. Standardize Headers
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
    df = df.loc[:, ~df.columns.duplicated()] # Kill duplicates
    
    # 3. Year Extraction
    if 'Year' not in df.columns:
        for col in df.columns:
            try:
                sample = float(df[col].iloc[0])
                if 1900 < sample < 2100:
                    df.rename(columns={col: 'Year'}, inplace=True)
                    break
            except: pass
            
    # 4. Numeric Enforcement
    for col in df.columns:
        if col != 'Year':
            df[col] = df[col].apply(clean_currency)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

# --- Math Helpers ---
def safe_div(n, d):
    try:
        if d is None or n is None or pd.isna(d) or pd.isna(n) or d == 0: return None
        return float(n) / float(d)
    except: return None

def safe_pct(n, d):
    """Returns a percentage (e.g., 0.2 -> 20.0)"""
    val = safe_div(n, d)
    return val * 100 if val is not None else None

def calculate_metrics(df):
    """Calculates all 19 Requested Ratios."""
    results = df.copy()
    if 'Year' in results.columns: results = results.sort_values('Year')
    
    # Helper to avoid KeyErrors
    def get(col): return results[col] if col in results else pd.Series([None]*len(results))

    # 1. Profitability
    results['Gross Margin'] = results.apply(lambda x: safe_pct(x.get('Revenue',0) - x.get('COGS',0), x.get('Revenue')), axis=1)
    results['Operating Margin'] = results.apply(lambda x: safe_pct(x.get('Operating Income'), x.get('Revenue')), axis=1)
    results['Net Margin'] = results.apply(lambda x: safe_pct(x.get('Net Income'), x.get('Revenue')), axis=1)
    results['ROA'] = results.apply(lambda x: safe_pct(x.get('Net Income'), x.get('Total Assets')), axis=1)
    results['ROE'] = results.apply(lambda x: safe_pct(x.get('Net Income'), x.get('Equity')), axis=1)
    
    # 2. Liquidity
    results['Current Ratio'] = results.apply(lambda x: safe_div(x.get('Current Assets'), x.get('Current Liabilities')), axis=1)
    results['Quick Ratio'] = results.apply(lambda x: safe_div(
        (x.get('Cash', 0) or 0) + (x.get('Accounts Receivable', 0) or 0) + (x.get('Short Term Investments', 0) or 0),
        x.get('Current Liabilities')
    ), axis=1)

    # 3. Efficiency & Cash Gap
    results['Asset Turnover'] = results.apply(lambda x: safe_div(x.get('Revenue'), x.get('Total Assets')), axis=1)
    results['Inventory Turnover'] = results.apply(lambda x: safe_div(x.get('COGS'), x.get('Inventory')), axis=1)
    results['AR Turnover'] = results.apply(lambda x: safe_div(x.get('Revenue'), x.get('Accounts Receivable')), axis=1)
    
    # Days Calculation (365 / Turnover)
    results['DSI (Days Inventory)'] = results.apply(lambda x: safe_div(365, x.get('Inventory Turnover')), axis=1)
    results['DSO (Days Sales)'] = results.apply(lambda x: safe_div(365, x.get('AR Turnover')), axis=1)
    
    # Cash Gap Estimate (DSI + DSO - DPO) *Assuming DPO is 0 if AP missing
    # Note: We need AP Turnover for DPO.
    results['AP Turnover'] = results.apply(lambda x: safe_div(x.get('COGS'), x.get('Accounts Payable')), axis=1)
    results['DPO (Days Payable)'] = results.apply(lambda x: safe_div(365, x.get('AP Turnover')), axis=1)
    
    results['Cash Conversion Cycle'] = results.apply(
        lambda x: (x.get('DSI (Days Inventory)') or 0) + (x.get('DSO (Days Sales)') or 0) - (x.get('DPO (Days Payable)') or 0), 
        axis=1
    )

    # 4. Leverage
    results['Debt-to-Assets'] = results.apply(lambda x: safe_pct(x.get('Total Liabilities'), x.get('Total Assets')), axis=1)
    results['Debt-to-Equity'] = results.apply(lambda x: safe_div(x.get('Total Liabilities'), x.get('Equity')), axis=1)
    results['Interest Coverage'] = results.apply(lambda x: safe_div(x.get('Operating Income'), x.get('Interest Expense')), axis=1)

    # 5. Growth Rates
    for m in ['Revenue', 'Net Income', 'Total Assets', 'Operating Cash Flow']:
        if m in results: results[f'{m} Growth'] = results[m].pct_change() * 100

    return results

def calculate_common_size(df):
    """
    FIX: Only applies common size to Dollar-based columns.
    Excludes calculated Ratios to prevent the 5000% error.
    """
    common = df.copy()
    
    # Identify "Ratio" columns vs "Dollar" columns
    # We assume Dollar columns are the ones mapped in COLUMN_MAPPING + any unknowns that aren't ratios
    excluded_keywords = ['ratio', 'margin', 'turnover', 'growth', 'days', 'cycle', 'coverage', '%', 'year']
    
    potential_dollar_cols = [c for c in df.columns if not any(k in c.lower() for k in excluded_keywords)]

    # IS % of Revenue
    if 'Revenue' in df:
        for c in potential_dollar_cols:
            if c in df: common[f'{c} (% Rev)'] = common.apply(lambda x: safe_pct(x[c], x['Revenue']), axis=1)
            
    # BS % of Assets
    if 'Total Assets' in df:
        # We selectively pick BS items to avoid confusion with IS items
        bs_targets = ['Current Assets', 'Inventory', 'Accounts Receivable', 'Cash', 'PP&E', 'Total Liabilities', 'Equity', 'Long Term Debt', 'Current Liabilities']
        for c in bs_targets:
            if c in df: common[f'{c} (% Asset)'] = common.apply(lambda x: safe_pct(x[c], x['Total Assets']), axis=1)
            
    return common

# ==========================================
# 4. AI & VISUALS
# ==========================================

def get_traffic_light(val, metric, industry_data):
    if val is None or pd.isna(val): return "⚪", "No Data"
    ranges = industry_data.get(metric)
    if not ranges: return "⚪", "No Benchmark"
    
    low, high = ranges
    # Inverse Logic (Lower is better)
    if metric in ['Debt-to-Equity', 'Debt-to-Assets', 'Cash Conversion Cycle']:
        if val <= low: return "🟢", "Strong"
        elif val <= high: return "🟡", "Average"
        else: return "🔴", "Weak"
    # Standard Logic (Higher is better)
    if val >= high: return "🟢", "Strong"
    elif val >= low: return "🟡", "Average"
    else: return "🔴", "Weak"

def get_gemini_analysis(api_key, industry, data_str, prompt_instruction):
    if not api_key: return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        Act as a Chief Financial Officer.
        INDUSTRY: {industry}
        GUIDANCE: {prompt_instruction}
        
        DATA:
        {data_str}
        
        OUTPUT FORMAT (Markdown):
        1. **Executive Summary:** 3 critical takeaways.
        2. **Deep Dive:**
           - Profitability (Margins, ROE)
           - Liquidity (Ratios, Cash Gap)
           - Solvency (Debt loads)
        3. **Strategic Recommendations:** 3 specific actions to take immediately.
        """
        return model.generate_content(prompt).text
    except: return None

def chat_with_data(query, df, api_key):
    if not api_key: return "Please enter an API Key in the sidebar to chat."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"Data:\n{df.to_string()}\nUser: {query}\nAnswer as a CFO:"
        return model.generate_content(prompt).text
    except: return "Error connecting to AI."

# ==========================================
# 5. MAIN APP UI
# ==========================================

def main():
    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = st.text_input("Gemini API Key (Optional)", type="password")
        
        def clear_cache():
            st.session_state.processed_data = None
            st.session_state.analysis_results = None
            st.session_state.chat_history = []
            
        uploaded_file = st.file_uploader("📂 Upload Financials", type=['csv', 'xlsx'], on_change=clear_cache)
        
        # Immediate File Loading
        if uploaded_file and st.session_state.processed_data is None:
            try:
                if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
                else: df = pd.read_excel(uploaded_file)
                st.session_state.processed_data = df
                st.success("✅ File Loaded")
            except Exception as e: st.error(f"Error: {e}")
            
        selected_industry = st.selectbox("🏭 Industry", list(INDUSTRY_BENCHMARKS.keys()))
        
        if st.session_state.processed_data is not None:
            if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                with st.spinner("Calculating 19 Ratios..."):
                    df_clean = normalize_and_clean(st.session_state.processed_data)
                    st.session_state.analysis_results = calculate_metrics(df_clean)
                    st.session_state.common_size = calculate_common_size(df_clean)
                    st.rerun()

    # --- Main Content ---
    if not uploaded_file and st.session_state.analysis_results is None:
        st.title("FinSight AI | Master Edition")
        st.markdown("### Institutional Financial Statement Analysis")
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown('<div class="metric-card"><h4>📥 1. Universal Import</h4><p>Auto-detects format.</p></div>', unsafe_allow_html=True)
        with c2: st.markdown('<div class="metric-card"><h4>🧮 2. 19+ Ratios</h4><p>Margins, Liquidity, Cash Gap.</p></div>', unsafe_allow_html=True)
        with c3: st.markdown('<div class="metric-card"><h4>🧠 3. AI Insights</h4><p>CEO-Level Reports.</p></div>', unsafe_allow_html=True)
        return

    if st.session_state.analysis_results is not None:
        results = st.session_state.analysis_results
        common = st.session_state.common_size
        latest = results.iloc[-1]
        
        st.title(f"📊 Analysis: {selected_industry}")
        
        # Hero KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1: 
            st.markdown(f'<div class="metric-card"><h4>Revenue Growth</h4><p>{latest.get("Revenue Growth", 0):.1f}%</p></div>', unsafe_allow_html=True)
        with k2: 
            st.markdown(f'<div class="metric-card"><h4>Net Margin</h4><p>{latest.get("Net Margin", 0):.1f}%</p></div>', unsafe_allow_html=True)
        with k3: 
            st.markdown(f'<div class="metric-card"><h4>Current Ratio</h4><p>{latest.get("Current Ratio", 0):.2f}x</p></div>', unsafe_allow_html=True)
        with k4: 
            st.markdown(f'<div class="metric-card"><h4>ROE</h4><p>{latest.get("ROE", 0):.1f}%</p></div>', unsafe_allow_html=True)

        tabs = st.tabs(["🚦 Scorecard", "📈 Visual Trends", "🔬 Deep Data", "🤖 AI Report", "💬 Chat"])
        
        with tabs[0]:
            st.markdown('<div class="section-header">Industry Benchmarks</div>', unsafe_allow_html=True)
            bench_data = []
            metrics = ["Gross Margin", "Net Margin", "Current Ratio", "Quick Ratio", "Debt-to-Equity", "ROE"]
            for m in metrics:
                val = latest.get(m)
                icon, status = get_traffic_light(val, m, INDUSTRY_BENCHMARKS.get(selected_industry))
                val_fmt = f"{val:.1f}%" if "Margin" in m or "ROE" in m else f"{val:.2f}"
                if pd.isna(val): val_fmt = "N/A"
                bench_data.append({"Metric": m, "Your Value": val_fmt, "Indicator": icon, "Status": status})
            st.dataframe(pd.DataFrame(bench_data), hide_index=True, use_container_width=True)

        with tabs[1]:
            st.markdown('<div class="section-header">Financial Trajectory</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                if 'Revenue' in results and 'Net Income' in results:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=results['Year'], y=results['Revenue'], name='Revenue', marker_color='#6366F1'))
                    fig.add_trace(go.Scatter(x=results['Year'], y=results['Net Income'], name='Net Income', line=dict(color='#10B981', width=3)))
                    fig.update_layout(title="Revenue vs Net Income", height=400, template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
            with c2:
                if 'Current Ratio' in results:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=results['Year'], y=results['Current Ratio'], name='Current Ratio', line=dict(color='#F59E0B', width=3)))
                    fig.add_trace(go.Scatter(x=results['Year'], y=[1.0]*len(results), name='Risk Threshold', line=dict(color='red', dash='dot')))
                    fig.update_layout(title="Liquidity (Target > 1.0)", height=400, template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

        with tabs[2]:
            st.markdown('<div class="section-header">Detailed Financials</div>', unsafe_allow_html=True)
            st.write("**Ratio Analysis**")
            st.dataframe(results.set_index('Year').T.style.format("{:.2f}"), use_container_width=True)
            st.write("**Common Size Analysis** (Fixed)")
            st.dataframe(common.set_index('Year').T.style.format("{:.1f}%"), use_container_width=True)

        with tabs[3]:
            st.markdown('<div class="section-header">CEO Strategic Report</div>', unsafe_allow_html=True)
            if st.button("Generate Report", type="primary"):
                with st.spinner("Analyzing..."):
                    context = INDUSTRY_PROMPTS.get(selected_industry, INDUSTRY_PROMPTS["General"])
                    report = get_gemini_analysis(api_key, selected_industry, results.to_string(), context)
                    if report: st.markdown(report)
                    else: st.info("Mock Mode: Please add API Key for full AI analysis.")

        with tabs[4]:
            st.markdown('<div class="section-header">CFO Assistant</div>', unsafe_allow_html=True)
            for msg in st.session_state.chat_history: st.chat_message(msg['role']).write(msg['content'])
            if prompt := st.chat_input("Ask about the data..."):
                st.session_state.chat_history.append({'role':'user', 'content':prompt})
                st.chat_message('user').write(prompt)
                with st.spinner("Thinking..."):
                    resp = chat_with_data(prompt, results, api_key)
                    st.session_state.chat_history.append({'role':'assistant', 'content':resp})
                    st.chat_message('assistant').write(resp)

if __name__ == "__main__":
    main()

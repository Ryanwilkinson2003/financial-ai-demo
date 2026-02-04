import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="FinSight AI | Pro Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling
st.markdown("""
<style>
    /* Main Container Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Card Styling */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    /* Dark Mode Adjustment for Cards */
    @media (prefers-color-scheme: dark) {
        .metric-card {
            background-color: #262730;
            border: 1px solid #41424C;
        }
    }

    /* Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #4F46E5;
    }
    
    /* Tables */
    .dataframe {
        font-size: 0.9rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
if 'processed_data' not in st.session_state: st.session_state.processed_data = None
if 'common_size' not in st.session_state: st.session_state.common_size = None

# ==========================================
# 2. INDUSTRY INTELLIGENCE (CONTEXT LAYER)
# ==========================================

INDUSTRY_CONTEXT = {
    "SaaS / Software": {
        "focus": "Recurring revenue, burn rate, and customer acquisition efficiency.",
        "good": "High Gross Margins (70%+), High Growth (>20%).",
        "bad": "Declining revenue, high churn, low gross margins.",
        "prompt_instruction": "Focus heavily on Growth and Gross Margins. Net losses are acceptable IF revenue growth is >20%."
    },
    "Manufacturing": {
        "focus": "Asset utilization, inventory management, and cost control.",
        "good": "Stable Gross Margins (25-35%), Inventory Turnover > 6x.",
        "bad": "Bloated inventory (low turnover), high debt-to-assets.",
        "prompt_instruction": "Focus strictly on Efficiency (Inventory/Asset Turnover) and Solvency. Asset-heavy businesses must have stable cash flow."
    },
    "Retail / E-commerce": {
        "focus": "Inventory velocity, margins, and cash conversion cycle.",
        "good": "Quick inventory turnover, low cash conversion cycle.",
        "bad": "Inventory buildup, razor-thin net margins (<2%).",
        "prompt_instruction": "Analyze Liquidity deeply. Retailers die when they run out of cash to buy inventory."
    },
    # Default fallback
    "General": {
        "focus": "Profitability, liquidity, and solvency.",
        "good": "Positive trends in revenue and income.",
        "bad": "Declining margins, high debt.",
        "prompt_instruction": "Provide a balanced analysis of all financial statements."
    }
}

# Add the rest of the industries pointing to 'General' or specific if needed
for ind in ["Healthcare", "Construction", "Energy", "Real Estate", "Financial Services", "Consumer Goods", "Telecom", "Automotive", "Hospitality"]:
    if ind not in INDUSTRY_CONTEXT: INDUSTRY_CONTEXT[ind] = INDUSTRY_CONTEXT["General"]

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
# 3. ROBUST DATA PROCESSING
# ==========================================

def clean_currency(x):
    if isinstance(x, str):
        clean = x.replace('$', '').replace(',', '').replace(' ', '')
        if '(' in clean and ')' in clean: clean = '-' + clean.replace('(', '').replace(')', '')
        try: return float(clean)
        except: return None
    return x

def normalize_and_clean(df):
    # 1. Orientation Fix
    first_col_vals = df.iloc[:, 0].astype(str).str.lower().tolist()
    metric_hits = sum(1 for v in first_col_vals if 'revenue' in v or 'net income' in v or 'assets' in v)
    
    if metric_hits > 0:
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
    df = df.loc[:, ~df.columns.duplicated()]
    
    # 3. Year Detection
    if 'Year' not in df.columns:
        for col in df.columns:
            try:
                sample = float(df[col].iloc[0])
                if 1900 < sample < 2100:
                    df.rename(columns={col: 'Year'}, inplace=True)
                    break
            except: pass
            
    # 4. Numeric Force
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
    if 'Year' in results.columns: results = results.sort_values('Year')
    def get(col): return results[col] if col in results else pd.Series([None]*len(results))

    # Profitability
    results['Gross Margin'] = results.apply(lambda x: safe_pct(x.get('Revenue',0) - x.get('COGS',0), x.get('Revenue')), axis=1)
    results['Net Margin'] = results.apply(lambda x: safe_pct(x.get('Net Income'), x.get('Revenue')), axis=1)
    results['ROA'] = results.apply(lambda x: safe_pct(x.get('Net Income'), x.get('Total Assets')), axis=1)
    results['ROE'] = results.apply(lambda x: safe_pct(x.get('Net Income'), x.get('Equity')), axis=1)
    
    # Liquidity
    results['Current Ratio'] = results.apply(lambda x: safe_div(x.get('Current Assets'), x.get('Current Liabilities')), axis=1)
    results['Quick Ratio'] = results.apply(lambda x: safe_div(
        (x.get('Cash', 0) or 0) + (x.get('Accounts Receivable', 0) or 0) + (x.get('Short Term Investments', 0) or 0),
        x.get('Current Liabilities')
    ), axis=1)
    
    # Leverage
    results['Debt-to-Assets'] = results.apply(lambda x: safe_pct(x.get('Total Liabilities'), x.get('Total Assets')), axis=1)
    results['Debt-to-Equity'] = results.apply(lambda x: safe_div(x.get('Total Liabilities'), x.get('Equity')), axis=1)

    # Growth
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
# 5. AI & VISUALS
# ==========================================

def get_traffic_light(val, metric, industry_data):
    if val is None or pd.isna(val): return "⚪", "No Data"
    ranges = industry_data.get(metric)
    if not ranges: return "⚪", "No Benchmark"
    low, high = ranges
    
    # Inverse metrics (lower is better)
    if metric in ['Debt-to-Equity', 'Debt-to-Assets']:
        if val <= low: return "🟢", "Strong"
        elif val <= high: return "🟡", "Average"
        else: return "🔴", "Risk"
    
    # Standard metrics (higher is better)
    if val >= high: return "🟢", "Strong"
    elif val >= low: return "🟡", "Average"
    else: return "🔴", "Weak"

def get_gemini_analysis(api_key, industry, data_str, context):
    if not api_key: return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        # PROMPT ENGINEERING
        prompt = f"""
        You are a seasoned CFO Consultant specializing in the {industry} sector.
        
        INDUSTRY CONTEXT:
        {context['focus']}
        SUCCESS INDICATORS: {context['good']}
        RISK INDICATORS: {context['bad']}
        INSTRUCTION: {context['prompt_instruction']}
        
        FINANCIAL DATA:
        {data_str}
        
        TASK:
        Write a strategic assessment.
        1. **Executive Scorecard:** (3 Bullet points on the most critical status).
        2. **Industry Specific Deep Dive:** Analyze the specific metrics that matter for {industry}.
        3. **Risk Analysis:** Identify specific red flags based on the data provided.
        4. **Strategic Roadmap:** 3 actionable, prioritized recommendations.
        
        TONE: Professional, Direct, Insightful.
        """
        return model.generate_content(prompt).text
    except: return None

def get_mock_analysis(industry, df):
    # Smart Mock logic based on last year's Net Income
    try:
        latest_ni = df.iloc[-1].get('Net Income', 0)
        status = "struggling" if latest_ni < 0 else "growing"
    except: status = "analyzing"
    
    return f"""
    ### 📝 Preliminary Analyst Notes (Mock Mode)
    * **Industry:** {industry}
    * **Observation:** The company appears to be {status} based on the latest Net Income figures.
    
    #### 🔍 Key Observations
    1. **Trend Analysis:** Review the Visual Trends tab to see if Revenue and Profit are correlated.
    2. **Liquidity Check:** Ensure the Current Ratio is above 1.5x (See Benchmarks).
    3. **Capital Structure:** Check Debt-to-Equity to ensure leverage is sustainable.
    
    *(To unlock the full AI Strategic Report, please enter a Gemini API Key in the sidebar)*
    """

def chat_with_data(user_query, df_context, api_key):
    if not api_key: return "I need an API Key to answer specific questions. (Mock Mode Active)"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"Context:\n{df_context.to_string()}\nQuestion: {user_query}\nAnswer as a CFO in 2 sentences."
        return model.generate_content(prompt).text
    except: return "Connection Error."

# ==========================================
# 6. MAIN APPLICATION
# ==========================================

def main():
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = st.text_input("Gemini API Key (Optional)", type="password")
        st.caption("Leave blank to use Mock Mode.")
        st.divider()
        
        def clear_cache():
            st.session_state.processed_data = None
            st.session_state.analysis_results = None
            st.session_state.chat_history = []
            
        uploaded_file = st.file_uploader("📂 Upload Financials", type=['csv', 'xlsx'], on_change=clear_cache)
        selected_industry = st.selectbox("🏭 Select Industry", list(INDUSTRY_BENCHMARKS.keys()))
        
        if st.session_state.processed_data is not None:
            if st.button("🚀 Analyze Data", type="primary", use_container_width=True):
                with st.spinner("Processing Financial Models..."):
                    df_clean = normalize_and_clean(st.session_state.processed_data)
                    st.session_state.analysis_results = calculate_metrics(df_clean)
                    st.session_state.common_size = calculate_common_size(df_clean)
                    st.rerun()

    # --- HERO SECTION ---
    if not uploaded_file and st.session_state.analysis_results is None:
        st.title("FinSight AI")
        st.markdown("### Institutional-Grade Financial Analysis")
        st.markdown("---")
        
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown('<div class="metric-card"><h4>📥 1. Universal Import</h4><p>Upload standard or messy Excel/CSV files. We fix the formatting automatically.</p></div>', unsafe_allow_html=True)
        with c2: st.markdown('<div class="metric-card"><h4>🧠 2. Contextual AI</h4><p>Our engine adjusts its analysis criteria based on the specific industry you select.</p></div>', unsafe_allow_html=True)
        with c3: st.markdown('<div class="metric-card"><h4>🚦 3. Smart Benchmarking</h4><p>Instant visual feedback on where you stand against industry leaders.</p></div>', unsafe_allow_html=True)
        return

    # --- DASHBOARD ---
    if st.session_state.analysis_results is not None:
        results = st.session_state.analysis_results
        common = st.session_state.common_size
        latest = results.iloc[-1]
        
        st.title(f"📊 Financial Analysis: {selected_industry}")
        
        # KPI ROW
        k1, k2, k3, k4 = st.columns(4)
        with k1: 
            val = latest.get('Revenue Growth')
            st.metric("Revenue Growth", f"{val:.1f}%" if pd.notna(val) else "N/A", delta_color="normal")
        with k2: 
            val = latest.get('Net Margin')
            st.metric("Net Margin", f"{val:.1f}%" if pd.notna(val) else "N/A")
        with k3: 
            val = latest.get('Current Ratio')
            st.metric("Current Ratio", f"{val:.2f}x" if pd.notna(val) else "N/A")
        with k4: 
            val = latest.get('ROE')
            st.metric("ROE", f"{val:.1f}%" if pd.notna(val) else "N/A")

        st.markdown("---")
        
        # TABS
        tabs = st.tabs(["🚦 Scorecard", "📈 Trends", "📋 Deep Data", "🤖 Strategic Report", "💬 AI Chat"])
        
        # TAB 1: SCORECARD
        with tabs[0]:
            st.markdown('<div class="section-header">Industry Benchmarking</div>', unsafe_allow_html=True)
            
            bench_data = []
            metrics = ["Gross Margin", "Net Margin", "Current Ratio", "Quick Ratio", "Debt-to-Equity", "ROE"]
            for m in metrics:
                val = latest.get(m)
                icon, status = get_traffic_light(val, m, INDUSTRY_BENCHMARKS[selected_industry])
                val_fmt = f"{val:.1f}%" if "Margin" in m or "ROE" in m else f"{val:.2f}"
                if pd.isna(val): val_fmt = "N/A"
                bench_data.append({"Metric": m, "Value": val_fmt, "Indicator": icon, "Status": status})
            
            b_df = pd.DataFrame(bench_data)
            st.dataframe(b_df, hide_index=True, use_container_width=True)

        # TAB 2: TRENDS
        with tabs[1]:
            st.markdown('<div class="section-header">Visual Trajectory</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            
            with c1:
                if 'Revenue' in results and 'Net Income' in results:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=results['Year'], y=results['Revenue'], name='Revenue', marker_color='#6366F1'))
                    fig.add_trace(go.Scatter(x=results['Year'], y=results['Net Income'], name='Net Income', line=dict(color='#10B981', width=3)))
                    fig.update_layout(title="Top Line vs Bottom Line", height=400, template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                    
            with c2:
                if 'Current Ratio' in results:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=results['Year'], y=results['Current Ratio'], name='Current Ratio', line=dict(color='#F59E0B', width=3)))
                    fig.add_trace(go.Scatter(x=results['Year'], y=[1.0]*len(results), name='Danger Zone', line=dict(color='red', dash='dot')))
                    fig.update_layout(title="Liquidity Health (Target > 1.0)", height=400, template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

        # TAB 3: DATA (STYLED)
        with tabs[2]:
            st.markdown('<div class="section-header">Financial Statements & Ratios</div>', unsafe_allow_html=True)
            
            st.write("### Ratio Analysis")
            # Gradient styling for better readability
            st.dataframe(
                results.set_index('Year').T.style.background_gradient(cmap="Blues", axis=1).format("{:.2f}"), 
                use_container_width=True
            )
            
            st.write("### Common Size Analysis")
            st.dataframe(
                common.set_index('Year').T.style.background_gradient(cmap="Greens", axis=1).format("{:.1f}%"), 
                use_container_width=True
            )

        # TAB 4: AI REPORT
        with tabs[3]:
            st.markdown('<div class="section-header">AI Strategic Assessment</div>', unsafe_allow_html=True)
            if st.button("Generate Pro Report", type="primary"):
                with st.spinner("Consulting AI Expert..."):
                    context = INDUSTRY_CONTEXT.get(selected_industry, INDUSTRY_CONTEXT["General"])
                    report = get_gemini_analysis(api_key, selected_industry, results.to_string(), context)
                    
                    if report:
                        st.markdown(report)
                    else:
                        st.info("API Key missing or invalid. Showing Mock Report.")
                        st.markdown(get_mock_analysis(selected_industry, results))

        # TAB 5: CHAT
        with tabs[4]:
            st.markdown('<div class="section-header">CFO Assistant</div>', unsafe_allow_html=True)
            
            for msg in st.session_state.chat_history:
                st.chat_message(msg['role']).write(msg['content'])
                
            if prompt := st.chat_input("Ask about the data..."):
                st.session_state.chat_history.append({'role':'user', 'content':prompt})
                st.chat_message('user').write(prompt)
                
                with st.spinner("Thinking..."):
                    resp = chat_with_data(prompt, results, api_key)
                    st.session_state.chat_history.append({'role':'assistant', 'content':resp})
                    st.chat_message('assistant').write(resp)

if __name__ == "__main__":
    main()

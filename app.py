import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai

# ==========================================
# 1. CONFIGURATION & PAGE SETUP
# ==========================================
st.set_page_config(
    page_title="AI Financial Analyst V3.0",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
if 'processed_data' not in st.session_state: st.session_state.processed_data = None
if 'common_size' not in st.session_state: st.session_state.common_size = None

# ==========================================
# 2. BENCHMARK DATABASE (12 INDUSTRIES)
# ==========================================
# Format: "Metric": (Min_Average, Max_Average)
# Note: For simplicity in traffic lights:
#   > Max_Average = Leader (Green)
#   Within Range = Average (Yellow)
#   < Min_Average = Risk (Red) (Logic inverted for debt ratios)

INDUSTRY_BENCHMARKS = {
    "Manufacturing": {
        "Gross Margin": (25, 35), "Net Margin": (5, 10), "Current Ratio": (1.2, 2.0), 
        "Quick Ratio": (0.8, 1.2), "Debt-to-Equity": (0.5, 1.5), "ROE": (10, 15)
    },
    "SaaS / Software": {
        "Gross Margin": (70, 85), "Net Margin": (10, 25), "Current Ratio": (1.5, 3.0), 
        "Quick Ratio": (1.5, 3.0), "Debt-to-Equity": (0.0, 0.5), "ROE": (15, 30)
    },
    "Retail / E-commerce": {
        "Gross Margin": (30, 50), "Net Margin": (3, 7), "Current Ratio": (1.2, 1.8), 
        "Quick Ratio": (0.2, 0.6), "Debt-to-Equity": (0.5, 1.2), "ROE": (15, 25)
    },
    "Healthcare / Biotech": {
        "Gross Margin": (55, 75), "Net Margin": (10, 20), "Current Ratio": (2.0, 4.0), 
        "Quick Ratio": (1.8, 3.5), "Debt-to-Equity": (0.2, 0.8), "ROE": (10, 20)
    },
    "Construction / Engineering": {
        "Gross Margin": (10, 20), "Net Margin": (2, 5), "Current Ratio": (1.1, 1.5), 
        "Quick Ratio": (0.9, 1.2), "Debt-to-Equity": (0.8, 2.0), "ROE": (12, 18)
    },
    "Energy / Oil & Gas": {
        "Gross Margin": (35, 50), "Net Margin": (8, 15), "Current Ratio": (1.0, 1.5), 
        "Quick Ratio": (0.8, 1.2), "Debt-to-Equity": (0.5, 1.0), "ROE": (8, 15)
    },
    "Real Estate / REITs": {
        "Gross Margin": (60, 70), "Net Margin": (20, 40), "Current Ratio": (0.5, 1.5), 
        "Quick Ratio": (0.5, 1.5), "Debt-to-Equity": (1.5, 3.0), "ROE": (5, 10)
    },
    "Financial Services / Banks": {
        "Gross Margin": (90, 100), "Net Margin": (20, 35), "Current Ratio": (1.0, 1.2), # Less relevant for banks
        "Quick Ratio": (1.0, 1.2), "Debt-to-Equity": (2.0, 5.0), "ROE": (8, 12)
    },
    "Consumer Goods": {
        "Gross Margin": (35, 50), "Net Margin": (5, 10), "Current Ratio": (1.2, 2.0), 
        "Quick Ratio": (0.6, 1.0), "Debt-to-Equity": (0.5, 1.5), "ROE": (15, 25)
    },
    "Telecommunications": {
        "Gross Margin": (45, 60), "Net Margin": (8, 15), "Current Ratio": (0.8, 1.2), 
        "Quick Ratio": (0.6, 1.0), "Debt-to-Equity": (1.0, 2.5), "ROE": (10, 20)
    },
    "Automotive": {
        "Gross Margin": (10, 20), "Net Margin": (3, 6), "Current Ratio": (1.1, 1.4), 
        "Quick Ratio": (0.8, 1.0), "Debt-to-Equity": (1.0, 2.0), "ROE": (10, 18)
    },
    "Hospitality / Restaurants": {
        "Gross Margin": (60, 75), "Net Margin": (4, 9), "Current Ratio": (0.8, 1.5), 
        "Quick Ratio": (0.4, 0.8), "Debt-to-Equity": (1.5, 3.0), "ROE": (20, 40)
    }
}

COLUMN_MAPPING = {
    "Year": ["year", "fiscal year", "fy"],
    "Revenue": ["revenue", "total revenue", "sales", "net sales"],
    "COGS": ["cogs", "cost of goods sold", "cost of sales"],
    "Operating Expenses": ["operating expenses", "opex", "sg&a"],
    "Operating Income": ["operating income", "operating profit", "ebit"],
    "Interest Expense": ["interest expense", "interest"],
    "Net Income": ["net income", "net profit", "net earnings"], # Priority 1
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
    "Equity": ["shareholders' equity", "equity", "total equity", "stockholders' equity"],
    "Operating Cash Flow": ["operating cash flow", "cash from operations", "net cash from operating activities"],
    "CapEx": ["capital expenditures", "capex", "purchases of property"]
}

# ==========================================
# 3. ROBUST DATA PROCESSING
# ==========================================

def clean_currency(x):
    """Converts '$1,000.00' or '1,000' to float 1000.0"""
    if isinstance(x, str):
        # Remove '$', ',', ' ' and parens for negative numbers
        clean_str = x.replace('$', '').replace(',', '').replace(' ', '')
        if '(' in clean_str and ')' in clean_str:
            clean_str = '-' + clean_str.replace('(', '').replace(')', '')
        try:
            return float(clean_str)
        except ValueError:
            return None # Failed to parse
    return x

def normalize_and_clean(df):
    """Deep cleaning of data types and column names."""
    # 1. Standardize Header Names
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    # 2. Map to Internal Standard Names
    renamed_cols = {}
    found_targets = set()
    
    # Priority: Exact match first, then fuzzy
    # We loop through standard keys to find matches in the user df
    for standard, variations in COLUMN_MAPPING.items():
        if standard in found_targets: continue # Already found this standard col
        
        match_found = False
        
        # A. Exact Match Check
        for col in df.columns:
            if col in variations:
                renamed_cols[col] = standard
                found_targets.add(standard)
                match_found = True
                break
        
        # B. Fuzzy Match Check (if no exact match)
        if not match_found:
            for col in df.columns:
                # Avoid overwriting already mapped columns
                if col in renamed_cols: continue
                
                for v in variations:
                    if v in col:
                        renamed_cols[col] = standard
                        found_targets.add(standard)
                        match_found = True
                        break
                if match_found: break

    # Rename columns
    df = df.rename(columns=renamed_cols)
    
    # 3. Remove Duplicate Columns (Crucial Fix for Ambiguity Error)
    # If "Net Income" appears twice, keep the first one
    df = df.loc[:, ~df.columns.duplicated()]

    # 4. Force Numeric Types
    # Apply cleaning to all columns except 'Year'
    for col in df.columns:
        if col != 'Year':
            df[col] = df[col].apply(clean_currency)
            df[col] = pd.to_numeric(df[col], errors='coerce') # Force NaN if not number
            
    return df

def safe_div(n, d):
    """Robust division handling zero, None, and NaN."""
    try:
        if d is None or n is None: return None
        if pd.isna(d) or pd.isna(n): return None
        if d == 0: return None
        return n / d
    except:
        return None

# ==========================================
# 4. CALCULATION ENGINE
# ==========================================

def calculate_metrics(df):
    results = df.copy()
    results = results.sort_values('Year')
    
    # Helper to safe get column
    def get_col(col_name):
        return results[col_name] if col_name in results else pd.Series([None]*len(results))

    # --- Profitability ---
    rev = get_col('Revenue')
    cogs = get_col('COGS')
    net_inc = get_col('Net Income')
    assets = get_col('Total Assets')
    equity = get_col('Equity')
    
    results['Gross Margin'] = results.apply(lambda x: safe_div(x.get('Revenue',0) - x.get('COGS',0), x.get('Revenue')) * 100 if 'Revenue' in x and 'COGS' in x else None, axis=1)
    results['Net Margin'] = results.apply(lambda x: safe_div(x.get('Net Income'), x.get('Revenue')) * 100, axis=1)
    results['ROA'] = results.apply(lambda x: safe_div(x.get('Net Income'), x.get('Total Assets')) * 100, axis=1)
    results['ROE'] = results.apply(lambda x: safe_div(x.get('Net Income'), x.get('Equity')) * 100, axis=1)

    # --- Liquidity ---
    cur_assets = get_col('Current Assets')
    cur_liab = get_col('Current Liabilities')
    cash = get_col('Cash')
    receivables = get_col('Accounts Receivable')
    short_invest = get_col('Short Term Investments')
    
    results['Current Ratio'] = results.apply(lambda x: safe_div(x.get('Current Assets'), x.get('Current Liabilities')), axis=1)
    
    # Quick Ratio Logic: (Cash + AR + ST Invest) / CL
    results['Quick Ratio'] = results.apply(
        lambda x: safe_div(
            (x.get('Cash', 0) or 0) + (x.get('Accounts Receivable', 0) or 0) + (x.get('Short Term Investments', 0) or 0),
            x.get('Current Liabilities')
        ), axis=1
    )

    # --- Leverage ---
    tot_liab = get_col('Total Liabilities')
    results['Debt-to-Assets'] = results.apply(lambda x: safe_div(x.get('Total Liabilities'), x.get('Total Assets')) * 100, axis=1)
    results['Debt-to-Equity'] = results.apply(lambda x: safe_div(x.get('Total Liabilities'), x.get('Equity')), axis=1)

    # --- Growth ---
    for m in ['Revenue', 'Net Income', 'Total Assets']:
        if m in results:
            results[f'{m} Growth'] = results[m].pct_change() * 100

    return results

def calculate_common_size(df):
    common = df.copy()
    # IS as % of Revenue
    if 'Revenue' in df:
        cols_to_map = ['COGS', 'Gross Profit', 'Operating Expenses', 'Net Income', 'Operating Income']
        for c in cols_to_map:
            if c in df:
                common[f'{c} (%)'] = common.apply(lambda x: safe_div(x[c], x['Revenue']) * 100, axis=1)
    
    # BS as % of Assets
    if 'Total Assets' in df:
        cols_to_map = ['Current Assets', 'Inventory', 'Accounts Receivable', 'Cash', 'PP&E', 'Total Liabilities', 'Equity']
        for c in cols_to_map:
            if c in df:
                common[f'{c} (%)'] = common.apply(lambda x: safe_div(x[c], x['Total Assets']) * 100, axis=1)
                
    return common

# ==========================================
# 5. VISUALIZATION & AI
# ==========================================

def get_traffic_light(val, metric, industry_data):
    """Returns Color, Status, Distance for Traffic Light Visuals."""
    if val is None or pd.isna(val): return "⚪", "No Data", ""
    
    ranges = industry_data.get(metric)
    if not ranges: return "⚪", "No Benchmark", ""
    
    low, high = ranges
    
    # Logic for metrics where HIGHER is BETTER (Margins, ROE)
    is_inverse = metric in ['Debt-to-Equity', 'Debt-to-Assets']
    
    if not is_inverse:
        if val >= high: return "🟢", "Leader", f"+{val-high:.1f} pts"
        elif val >= low: return "🟡", "Average", "In Range"
        else: return "🔴", "Weak/Risk", f"{val-low:.1f} pts"
    else:
        # Logic for metrics where LOWER is BETTER (Debt)
        if val <= low: return "🟢", "Leader", "Low Risk"
        elif val <= high: return "🟡", "Average", "In Range"
        else: return "🔴", "Weak/Risk", "High Risk"

def get_gemini_analysis(api_key, industry, data_str):
    if not api_key: return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        Act as a Senior Financial Analyst. 
        Industry: {industry}
        Data: 
        {data_str}
        
        Task: Provide a professional executive summary, 3 strengths, 3 weaknesses, and recommendations.
        Constraint: Ignore 'None' or 'NaN' values.
        """
        return model.generate_content(prompt).text
    except: return None

def get_mock_analysis(industry, df):
    return f"""
    ### 🤖 AI Analyst Report (Mock Mode)
    **Industry Context:** {industry}
    
    The company shows specific signals relevant to the {industry} sector. 
    Analysis of the latest fiscal year indicates a need to focus on **Operational Efficiency** and **Cash Flow Management**.
    
    #### ✅ Key Strengths
    * **Revenue Base:** Established market presence visible in top-line numbers.
    * **Asset Coverage:** Asset base appears sufficient to support current operations.
    
    #### ⚠️ Identified Risks
    * **Benchmark Deviation:** Certain profitability ratios are trailing the industry leaders.
    * **Liquidity Pressure:** Cash reserves relative to short-term obligations require monitoring.
    
    #### 🚀 Recommendations
    1. **Expense Audit:** Conduct a full review of Operating Expenses to improve Net Margins.
    2. **Inventory Optimization:** Reduce Days Sales in Inventory to free up working capital.
    3. **Debt Structuring:** Evaluate refinancing options if Debt-to-Equity remains elevated.
    """

# ==========================================
# 6. MAIN APPLICATION
# ==========================================

def main():
    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = st.text_input("Gemini API Key (Optional)", type="password")
        st.divider()
        
        def clear_cache():
            st.session_state.processed_data = None
            st.session_state.analysis_results = None
            
        uploaded_file = st.file_uploader("Upload Financials", type=['csv', 'xlsx'], on_change=clear_cache)
        selected_industry = st.selectbox("Select Industry", list(INDUSTRY_BENCHMARKS.keys()))
        
        if st.session_state.processed_data is not None:
            if st.button("🚀 Analyze Financials", type="primary"):
                with st.spinner("Processing Complex Data..."):
                    # 1. Clean & Normalize
                    df_clean = normalize_and_clean(st.session_state.processed_data)
                    # 2. Calculate
                    st.session_state.analysis_results = calculate_metrics(df_clean)
                    st.session_state.common_size = calculate_common_size(df_clean)
                    st.rerun()

    # --- Main Content ---
    st.title("💰 AI Financial Analyst V3.0 (Robust)")
    
    # File Loader
    if uploaded_file and st.session_state.processed_data is None:
        try:
            if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
            else: df = pd.read_excel(uploaded_file)
            st.session_state.processed_data = df
            st.success("✅ File Loaded Successfully")
        except Exception as e:
            st.error(f"Error Loading File: {e}")

    # Dashboard
    if st.session_state.analysis_results is not None:
        results = st.session_state.analysis_results
        common = st.session_state.common_size
        latest = results.iloc[-1]
        
        tab1, tab2, tab3, tab4 = st.tabs(["🚦 Benchmarks", "📊 Visual Trends", "🔬 Deep Data", "🤖 AI Insights"])
        
        # TAB 1: TRAFFIC LIGHT BENCHMARKS
        with tab1:
            st.subheader(f"Performance vs. {selected_industry} Standards")
            st.caption("🟢 Leader | 🟡 Average | 🔴 Weak/Risk")
            
            # Build Traffic Light Table
            bench_data = []
            metrics_to_show = ["Gross Margin", "Net Margin", "Current Ratio", "Quick Ratio", "Debt-to-Equity", "ROE"]
            
            for m in metrics_to_show:
                val = latest.get(m)
                icon, status, dist = get_traffic_light(val, m, INDUSTRY_BENCHMARKS[selected_industry])
                val_fmt = f"{val:.1f}%" if "Margin" in m or "ROE" in m else f"{val:.2f}"
                if val is None or pd.isna(val): val_fmt = "N/A"
                
                bench_data.append({
                    "Metric": m,
                    "Your Value": val_fmt,
                    "Status": icon,
                    "Assessment": status,
                    "Variance": dist
                })
            
            st.dataframe(pd.DataFrame(bench_data), use_container_width=True, hide_index=True)
            
            # Show Raw Benchmark Ranges for Reference
            with st.expander("See Industry Benchmark Ranges"):
                st.write(INDUSTRY_BENCHMARKS[selected_industry])

        # TAB 2: VISUAL TRENDS
        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                if 'Revenue' in results and 'Net Income' in results:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=results['Year'], y=results['Revenue'], name='Revenue', marker_color='#3B82F6'))
                    fig.add_trace(go.Scatter(x=results['Year'], y=results['Net Income'], name='Net Income', line=dict(color='#10B981', width=3)))
                    fig.update_layout(title="Top Line vs. Bottom Line", legend=dict(orientation="h"))
                    st.plotly_chart(fig, use_container_width=True)
            with c2:
                if 'Current Ratio' in results and 'Quick Ratio' in results:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=results['Year'], y=results['Current Ratio'], name='Current Ratio', line=dict(color='#6366F1', width=3)))
                    fig.add_trace(go.Scatter(x=results['Year'], y=results['Quick Ratio'], name='Quick Ratio', line=dict(color='#EF4444', dash='dot')))
                    fig.update_layout(title="Liquidity Health Check", legend=dict(orientation="h"))
                    st.plotly_chart(fig, use_container_width=True)
            
            if 'Total Assets' in results and 'Total Liabilities' in results:
                fig3 = go.Figure()
                fig3.add_trace(go.Bar(x=results['Year'], y=results['Total Assets'], name='Assets', marker_color='#0EA5E9'))
                fig3.add_trace(go.Bar(x=results['Year'], y=results['Total Liabilities'], name='Liabilities', marker_color='#F43F5E'))
                fig3.update_layout(title="Solvency Structure (Assets vs Liabilities)", barmode='group')
                st.plotly_chart(fig3, use_container_width=True)

        # TAB 3: DEEP DATA TABLES
        with tab3:
            st.subheader("Financial Ratios (All Years)")
            st.dataframe(results.set_index('Year').T.style.format(lambda x: "N/A" if pd.isna(x) else "{:.2f}".format(x)))
            
            st.subheader("Common Size Analysis")
            st.write("Income Statement (% of Revenue) | Balance Sheet (% of Assets)")
            st.dataframe(common.set_index('Year').T.style.format(lambda x: "N/A" if pd.isna(x) else "{:.1f}%".format(x)))

        # TAB 4: AI INSIGHTS
        with tab4:
            st.subheader("🤖 Strategic Analysis")
            if st.button("Generate AI Report"):
                with st.spinner("Analyzing..."):
                    if api_key:
                        data_str = results.to_string()
                        analysis = get_gemini_analysis(api_key, selected_industry, data_str)
                        if analysis: st.markdown(analysis)
                        else: st.error("API Error. Showing Mock Data.")
                    
                    if not api_key or not analysis:
                        st.markdown(get_mock_analysis(selected_industry, results))

if __name__ == "__main__":
    main()

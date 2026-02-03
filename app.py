{\rtf1\ansi\ansicpg1252\cocoartf2709
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import pandas as pd\
import numpy as np\
import plotly.graph_objects as go\
import google.generativeai as genai\
import json\
import time\
import re\
from datetime import datetime\
\
# ==========================================\
# 1. CONFIGURATION & PAGE SETUP\
# ==========================================\
st.set_page_config(\
    page_title="AI Financial Analyst Pro",\
    page_icon="\uc0\u55357 \u56522 ",\
    layout="wide",\
    initial_sidebar_state="expanded"\
)\
\
# Initialize Session State\
if 'chat_history' not in st.session_state:\
    st.session_state.chat_history = []\
if 'analysis_results' not in st.session_state:\
    st.session_state.analysis_results = None\
if 'processed_data' not in st.session_state:\
    st.session_state.processed_data = None\
\
# ==========================================\
# 2. CONSTANTS & DATASETS\
# ==========================================\
\
# 2.1 Benchmark Data (3 Tiers: Average, Leaders, Failure Patterns)\
# Ranges are tuples: (min, max). For failure, it often implies "worse than X" or "within danger zone"\
INDUSTRY_BENCHMARKS = \{\
    "SaaS / Technology": \{\
        "average": \{"Gross Margin": (70, 80), "Net Margin": (5, 15), "Revenue Growth": (20, 40), "Current Ratio": (1.5, 2.5), "Debt-to-Assets": (10, 30)\},\
        "leaders": \{"Gross Margin": (82, 95), "Net Margin": (25, 40), "Revenue Growth": (50, 100), "Current Ratio": (2.5, 4.0), "Debt-to-Assets": (0, 10)\},\
        "failure_patterns": \{"Gross Margin": (0, 50), "Net Margin": (-100, -20), "Revenue Growth": (-50, 5), "Current Ratio": (0, 0.8), "Debt-to-Assets": (60, 100)\}\
    \},\
    "Retail / E-commerce": \{\
        "average": \{"Gross Margin": (25, 45), "Net Margin": (2, 5), "Revenue Growth": (5, 15), "Current Ratio": (1.2, 2.0), "Debt-to-Assets": (30, 50)\},\
        "leaders": \{"Gross Margin": (50, 65), "Net Margin": (7, 12), "Revenue Growth": (20, 35), "Current Ratio": (2.0, 3.0), "Debt-to-Assets": (10, 25)\},\
        "failure_patterns": \{"Gross Margin": (0, 15), "Net Margin": (-20, 0), "Revenue Growth": (-20, 0), "Current Ratio": (0, 0.9), "Debt-to-Assets": (70, 100)\}\
    \},\
    "Manufacturing": \{\
        "average": \{"Gross Margin": (20, 35), "Net Margin": (3, 8), "Revenue Growth": (3, 8), "Current Ratio": (1.2, 2.0), "Debt-to-Assets": (30, 50)\},\
        "leaders": \{"Gross Margin": (35, 50), "Net Margin": (10, 18), "Revenue Growth": (10, 20), "Current Ratio": (2.0, 3.5), "Debt-to-Assets": (10, 25)\},\
        "failure_patterns": \{"Gross Margin": (0, 15), "Net Margin": (-15, 0), "Revenue Growth": (-15, -2), "Current Ratio": (0, 0.8), "Debt-to-Assets": (65, 100)\}\
    \},\
    "Healthcare / Biotech": \{\
        "average": \{"Gross Margin": (50, 70), "Net Margin": (5, 15), "Revenue Growth": (10, 25), "Current Ratio": (1.5, 3.0), "Debt-to-Assets": (20, 40)\},\
        "leaders": \{"Gross Margin": (75, 90), "Net Margin": (20, 35), "Revenue Growth": (30, 60), "Current Ratio": (3.5, 6.0), "Debt-to-Assets": (0, 15)\},\
        "failure_patterns": \{"Gross Margin": (0, 40), "Net Margin": (-50, -10), "Revenue Growth": (-20, 5), "Current Ratio": (0, 1.0), "Debt-to-Assets": (60, 100)\}\
    \},\
    "Restaurants / Food Service": \{\
        "average": \{"Gross Margin": (60, 70), "Net Margin": (3, 6), "Revenue Growth": (2, 8), "Current Ratio": (0.8, 1.5), "Debt-to-Assets": (40, 60)\},\
        "leaders": \{"Gross Margin": (70, 80), "Net Margin": (8, 15), "Revenue Growth": (10, 20), "Current Ratio": (1.5, 2.5), "Debt-to-Assets": (20, 35)\},\
        "failure_patterns": \{"Gross Margin": (0, 55), "Net Margin": (-10, 0), "Revenue Growth": (-10, 0), "Current Ratio": (0, 0.6), "Debt-to-Assets": (70, 100)\}\
    \},\
    "Construction / Real Estate": \{\
        "average": \{"Gross Margin": (15, 25), "Net Margin": (2, 6), "Revenue Growth": (5, 12), "Current Ratio": (1.1, 1.8), "Debt-to-Assets": (40, 60)\},\
        "leaders": \{"Gross Margin": (25, 35), "Net Margin": (8, 15), "Revenue Growth": (15, 25), "Current Ratio": (2.0, 3.0), "Debt-to-Assets": (20, 35)\},\
        "failure_patterns": \{"Gross Margin": (0, 10), "Net Margin": (-15, 0), "Revenue Growth": (-20, 0), "Current Ratio": (0, 0.9), "Debt-to-Assets": (75, 100)\}\
    \},\
    "Energy / Utilities": \{\
        "average": \{"Gross Margin": (30, 45), "Net Margin": (8, 12), "Revenue Growth": (2, 6), "Current Ratio": (0.9, 1.3), "Debt-to-Assets": (50, 70)\},\
        "leaders": \{"Gross Margin": (45, 60), "Net Margin": (15, 22), "Revenue Growth": (8, 15), "Current Ratio": (1.4, 2.0), "Debt-to-Assets": (30, 45)\},\
        "failure_patterns": \{"Gross Margin": (0, 20), "Net Margin": (-10, 2), "Revenue Growth": (-15, 0), "Current Ratio": (0, 0.7), "Debt-to-Assets": (80, 100)\}\
    \},\
     "Financial Services": \{\
        "average": \{"Gross Margin": (90, 100), "Net Margin": (15, 25), "Revenue Growth": (5, 10), "Current Ratio": (1.0, 1.5), "Debt-to-Assets": (60, 80)\}, # Note: Margins vary by sub-sector\
        "leaders": \{"Gross Margin": (90, 100), "Net Margin": (30, 50), "Revenue Growth": (15, 30), "Current Ratio": (1.5, 3.0), "Debt-to-Assets": (40, 60)\},\
        "failure_patterns": \{"Gross Margin": (0, 70), "Net Margin": (-20, 5), "Revenue Growth": (-10, 2), "Current Ratio": (0, 0.9), "Debt-to-Assets": (90, 100)\}\
    \},\
     "Professional Services": \{\
        "average": \{"Gross Margin": (30, 50), "Net Margin": (10, 15), "Revenue Growth": (5, 12), "Current Ratio": (1.5, 2.5), "Debt-to-Assets": (20, 40)\},\
        "leaders": \{"Gross Margin": (50, 70), "Net Margin": (20, 30), "Revenue Growth": (15, 25), "Current Ratio": (2.5, 4.0), "Debt-to-Assets": (0, 15)\},\
        "failure_patterns": \{"Gross Margin": (0, 25), "Net Margin": (-10, 0), "Revenue Growth": (-15, 0), "Current Ratio": (0, 1.0), "Debt-to-Assets": (50, 100)\}\
    \}\
\}\
\
# 2.2 Column Mapping (Normalization)\
# Keys are the internal standard names, Values are lists of likely variations\
COLUMN_MAPPING = \{\
    "Year": ["year", "fiscal year", "period", "fy", "date"],\
    "Revenue": ["revenue", "total revenue", "sales", "turnover", "net sales"],\
    "COGS": ["cogs", "cost of goods sold", "cost of sales", "cost of revenue"],\
    "Operating Expenses": ["operating expenses", "opex", "sg&a", "selling, general and administrative"],\
    "Operating Income": ["operating income", "operating profit", "ebit"],\
    "Net Income": ["net income", "net profit", "earnings", "net earnings"],\
    "Total Assets": ["total assets", "assets"],\
    "Current Assets": ["current assets"],\
    "Total Liabilities": ["total liabilities", "liabilities"],\
    "Current Liabilities": ["current liabilities"],\
    "Equity": ["shareholders' equity", "equity", "total equity", "stockholders' equity"],\
    "Cash": ["cash", "cash and cash equivalents", "cash & equivalents"]\
\}\
\
# 2.3 Sample Data (SaaS Success vs. Retail Struggle)\
SAMPLE_DATA_SAAS = [\
    \{"Year": 2021, "Revenue": 1000000, "COGS": 200000, "Operating Expenses": 750000, "Net Income": 50000, "Total Assets": 800000, "Current Assets": 500000, "Total Liabilities": 300000, "Current Liabilities": 200000, "Equity": 500000, "Cash": 300000\},\
    \{"Year": 2022, "Revenue": 1500000, "COGS": 280000, "Operating Expenses": 900000, "Net Income": 320000, "Total Assets": 1200000, "Current Assets": 800000, "Total Liabilities": 400000, "Current Liabilities": 250000, "Equity": 800000, "Cash": 550000\},\
    \{"Year": 2023, "Revenue": 2400000, "COGS": 400000, "Operating Expenses": 1200000, "Net Income": 800000, "Total Assets": 2000000, "Current Assets": 1500000, "Total Liabilities": 500000, "Current Liabilities": 300000, "Equity": 1500000, "Cash": 1000000\}\
]\
\
SAMPLE_DATA_RETAIL = [\
    \{"Year": 2020, "Revenue": 5000000, "COGS": 3000000, "Operating Expenses": 1800000, "Net Income": 200000, "Total Assets": 4000000, "Current Assets": 2000000, "Total Liabilities": 3000000, "Current Liabilities": 1500000, "Equity": 1000000, "Cash": 400000\},\
    \{"Year": 2021, "Revenue": 4800000, "COGS": 3100000, "Operating Expenses": 1850000, "Net Income": -150000, "Total Assets": 3900000, "Current Assets": 1800000, "Total Liabilities": 3200000, "Current Liabilities": 1900000, "Equity": 700000, "Cash": 200000\},\
    \{"Year": 2022, "Revenue": 4200000, "COGS": 2900000, "Operating Expenses": 1900000, "Net Income": -600000, "Total Assets": 3500000, "Current Assets": 1200000, "Total Liabilities": 3400000, "Current Liabilities": 2000000, "Equity": 100000, "Cash": 50000\}\
]\
\
# ==========================================\
# 3. UTILITY FUNCTIONS\
# ==========================================\
\
def normalize_columns(df):\
    """Normalize user-uploaded columns to standard internal names."""\
    df.columns = [c.strip().lower() for c in df.columns]\
    renamed = \{\}\
    \
    for standard, variations in COLUMN_MAPPING.items():\
        for col in df.columns:\
            if col in variations:\
                renamed[col] = standard\
                break\
    \
    # Check if a column matches loosely (contains substring) if exact match fails\
    for standard, variations in COLUMN_MAPPING.items():\
        if standard not in renamed.values():\
            for col in df.columns:\
                if col not in renamed: # not yet mapped\
                    if any(v in col for v in variations):\
                        renamed[col] = standard\
                        break\
                        \
    return df.rename(columns=renamed)\
\
def safe_div(n, d):\
    """Safe division handling zero denominator."""\
    if d == 0 or pd.isna(d) or pd.isna(n):\
        return None\
    return n / d\
\
def format_metric(val, is_percent=True):\
    """Format numbers for display."""\
    if val is None:\
        return "N/A"\
    if is_percent:\
        return f"\{val:.1f\}%"\
    return f"\{val:.2f\}"\
\
def get_status_color(metric_name, value, benchmarks):\
    """Determine color status based on benchmark tiers."""\
    if value is None:\
        return "gray"\
    \
    # Get benchmark ranges\
    leaders = benchmarks['leaders'].get(metric_name)\
    failures = benchmarks['failure_patterns'].get(metric_name)\
    \
    # 1. Check Failure (Red)\
    if failures:\
        # Failure logic: usually if below min or matches range depending on metric nature\
        # For simplicity in this demo: if it falls INSIDE the failure range\
        if failures[0] <= value <= failures[1]:\
            return "red"\
            \
    # 2. Check Leaders (Green)\
    if leaders:\
        # Leader logic: usually if above min\
        if value >= leaders[0]:\
            return "green"\
            \
    # 3. Default to Average (Yellow)\
    return "orange"\
\
# ==========================================\
# 4. FINANCIAL CALCULATION ENGINE\
# ==========================================\
\
def calculate_metrics(df):\
    """\
    Calculate financial ratios and growth metrics.\
    Returns a dictionary of DataFrames for display.\
    """\
    df = df.sort_values('Year')\
    results = df.copy()\
    \
    # 1. Profitability\
    if 'Revenue' in df and 'COGS' in df:\
        results['Gross Margin'] = results.apply(lambda x: safe_div(x['Revenue'] - x['COGS'], x['Revenue']) * 100, axis=1)\
    \
    if 'Revenue' in df and 'Net Income' in df:\
        results['Net Margin'] = results.apply(lambda x: safe_div(x['Net Income'], x['Revenue']) * 100, axis=1)\
        \
    if 'Total Assets' in df and 'Net Income' in df:\
        results['ROA'] = results.apply(lambda x: safe_div(x['Net Income'], x['Total Assets']) * 100, axis=1)\
\
    # 2. Growth (YoY)\
    for col in ['Revenue', 'Net Income', 'Total Assets']:\
        if col in df:\
            results[f'\{col\} Growth'] = results[col].pct_change() * 100\
\
    # 3. Liquidity\
    if 'Current Assets' in df and 'Current Liabilities' in df:\
        results['Current Ratio'] = results.apply(lambda x: safe_div(x['Current Assets'], x['Current Liabilities']), axis=1)\
    \
    # 4. Leverage\
    if 'Total Liabilities' in df and 'Total Assets' in df:\
        results['Debt-to-Assets'] = results.apply(lambda x: safe_div(x['Total Liabilities'], x['Total Assets']) * 100, axis=1)\
        \
    return results\
\
# ==========================================\
# 5. AI INTEGRATION (GEMINI PRO + MOCK)\
# ==========================================\
\
def get_gemini_analysis(api_key, industry, data_summary, comparison_text):\
    """\
    Calls Gemini Pro to generate financial analysis.\
    """\
    try:\
        genai.configure(api_key=api_key)\
        model = genai.GenerativeModel('gemini-pro')\
        \
        prompt = f"""\
        You are a senior financial analyst. Analyze the following company in the \{industry\} industry.\
        \
        FINANCIAL DATA SUMMARY (Last 3 Years):\
        \{data_summary\}\
        \
        BENCHMARK COMPARISON:\
        \{comparison_text\}\
        \
        TASK:\
        1. Executive Summary (2-3 sentences on overall health).\
        2. Top 3 Strengths (with evidence).\
        3. Top 3 Weaknesses/Risks (with evidence).\
        4. Failure Pattern Check: Specifically highlight if any metrics match the "Failed Company" patterns provided.\
        5. 3-5 Actionable Strategic Recommendations.\
        \
        TONE: Professional, insightful, direct. No investment advice.\
        FORMAT: Use Markdown with bold headers.\
        """\
        \
        with st.spinner('\uc0\u55358 \u56598  Gemini is analyzing the financials...'):\
            response = model.generate_content(prompt)\
            return response.text\
    except Exception as e:\
        return f"API_ERROR: \{str(e)\}"\
\
def get_mock_analysis(industry, df_latest):\
    """\
    Sophisticated fallback analysis if API fails.\
    """\
    time.sleep(2) # Simulate thinking\
    \
    # Logic for mock insights\
    gm = df_latest.get('Gross Margin', 0)\
    nm = df_latest.get('Net Margin', 0)\
    cr = df_latest.get('Current Ratio', 0)\
    \
    strength = []\
    weakness = []\
    \
    if gm > 50: strength.append(f"Strong Gross Margin of \{gm:.1f\}% indicates pricing power.")\
    else: weakness.append(f"Low Gross Margin of \{gm:.1f\}% suggests high production costs.")\
    \
    if nm > 10: strength.append(f"Healthy Net Margin of \{nm:.1f\}% shows operational efficiency.")\
    elif nm < 0: weakness.append(f"Negative Net Margin of \{nm:.1f\}% indicates lack of profitability.")\
    \
    if cr < 1: weakness.append(f"Critical Liquidity: Current Ratio of \{cr:.2f\} indicates inability to cover short-term debts.")\
    \
    return f"""\
    **[MOCK ANALYSIS - API UNAVAILABLE]**\
    \
    ### \uc0\u55357 \u56522  Executive Summary\
    The company shows a mixed performance profile within the \{industry\} sector. While there are indicators of \{('strength' if nm > 0 else 'stress')\}, specific metrics require immediate management attention.\
    \
    ### \uc0\u9989  Strengths\
    1. **Margin Performance**: \{strength[0] if strength else "Stable revenue base."\}\
    2. **Asset Base**: Company maintains a significant asset base relative to liabilities.\
    3. **Growth Potential**: Recent trends suggest opportunity for scale if costs are managed.\
    \
    ### \uc0\u9888 \u65039  Weaknesses & Risks\
    1. **Profitability Concerns**: \{weakness[0] if weakness else "Margins are tightening."\}\
    2. **Liquidity**: \{weakness[1] if len(weakness) > 1 else "Cash flow monitoring is essential."\}\
    3. **Benchmark Deviation**: Performance lags behind industry leaders in key efficiency ratios.\
    \
    ### \uc0\u55357 \u57000  Risk Pattern Assessment\
    Analysis detects potential alignment with failure patterns in **\{'Liquidity' if cr < 1 else 'Profitability'\}**. This warrants a stress test of the balance sheet.\
    \
    ### \uc0\u55357 \u56960  Recommendations\
    1. **Cost Optimization**: Conduct a line-item audit of COGS to improve gross margins.\
    2. **Working Capital**: renegotiate payment terms with suppliers to improve the Current Ratio.\
    3. **Revenue Diversification**: Explore adjacent markets to reduce dependency on core streams.\
    """\
\
def chat_agent(user_query, context_data):\
    """\
    Simple mock chat agent that answers based on context.\
    In a real full production, this would also call Gemini with history.\
    """\
    user_query = user_query.lower()\
    if "gross margin" in user_query:\
        return "Gross Margin measures how much revenue you retain after direct costs. A declining margin suggests rising material costs or pricing pressure."\
    elif "recommend" in user_query or "improve" in user_query:\
        return "Focus on cutting COGS and extending accounts payable to boost your liquidity immediately."\
    elif "risk" in user_query:\
        return "The primary risk identified is the potential liquidity crunch if the Current Ratio drops below 1.0."\
    else:\
        return "That's a great question. Based on the analysis, the focus should be on stabilizing cash flow and improving operational efficiency to match industry leaders."\
\
# ==========================================\
# 6. MAIN APPLICATION UI\
# ==========================================\
\
def main():\
    # --- Sidebar ---\
    with st.sidebar:\
        st.header("\uc0\u9881 \u65039  Configuration")\
        \
        # API Key Input\
        api_key = st.text_input("Gemini API Key (Optional)", type="password", help="Leave empty to use Mock Engine")\
        \
        st.divider()\
        st.subheader("1. Upload Data")\
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])\
        \
        st.write("OR")\
        \
        # Sample Data Buttons\
        col1, col2 = st.columns(2)\
        with col1:\
            if st.button("Load SaaS (Good)"):\
                st.session_state.processed_data = pd.DataFrame(SAMPLE_DATA_SAAS)\
                st.rerun()\
        with col2:\
            if st.button("Load Retail (Bad)"):\
                st.session_state.processed_data = pd.DataFrame(SAMPLE_DATA_RETAIL)\
                st.rerun()\
                \
        st.divider()\
        st.subheader("2. Select Industry")\
        selected_industry = st.selectbox("Industry Benchmark", list(INDUSTRY_BENCHMARKS.keys()))\
        \
        if st.session_state.processed_data is not None:\
            if st.button("\uc0\u55357 \u56589  Analyze Financials", type="primary"):\
                # Trigger Analysis\
                with st.spinner("Crunching numbers..."):\
                    df = normalize_columns(st.session_state.processed_data)\
                    metrics_df = calculate_metrics(df)\
                    st.session_state.analysis_results = metrics_df\
                    # Clear chat history on new analysis\
                    st.session_state.chat_history = []\
                    st.rerun()\
\
    # --- Main Content ---\
    st.title("\uc0\u55357 \u56522  AI Financial Analyst & Benchmarker")\
    st.markdown("Upload your financial statements to receive institutional-grade analysis, failure pattern detection, and strategic recommendations.")\
    \
    # Process Uploaded File\
    if uploaded_file and st.session_state.processed_data is None:\
        try:\
            if uploaded_file.name.endswith('.csv'):\
                df = pd.read_csv(uploaded_file)\
            else:\
                df = pd.read_excel(uploaded_file)\
            st.session_state.processed_data = df\
            st.success("File uploaded successfully! Click 'Analyze Financials' in the sidebar.")\
        except Exception as e:\
            st.error(f"Error reading file: \{e\}")\
\
    # Display Results if Analysis exists\
    if st.session_state.analysis_results is not None:\
        results = st.session_state.analysis_results\
        latest_year = results.iloc[-1]\
        benchmarks = INDUSTRY_BENCHMARKS[selected_industry]\
        \
        # --- TAB STRUCTURE ---\
        tab1, tab2, tab3, tab4 = st.tabs(["\uc0\u55357 \u56520  Financial Trends", "\u55356 \u57263  Benchmark Comparison", "\u55358 \u56598  AI Insights", "\u55357 \u56492  AI Assistant"])\
        \
        # TAB 1: FINANCIAL TRENDS\
        with tab1:\
            st.subheader("Key Financial Metrics (Year-over-Year)")\
            \
            # Formatting DataFrame for display\
            display_df = results.set_index('Year').T\
            st.dataframe(display_df.style.format("\{:.2f\}"))\
            \
            # Optional: Simple Charts\
            st.subheader("Visual Trends")\
            chart_col1, chart_col2 = st.columns(2)\
            \
            with chart_col1:\
                if 'Revenue' in results.columns:\
                    fig = go.Figure()\
                    fig.add_trace(go.Bar(x=results['Year'], y=results['Revenue'], name='Revenue', marker_color='#4F46E5'))\
                    fig.update_layout(title="Revenue Growth", height=300)\
                    st.plotly_chart(fig, use_container_width=True)\
            \
            with chart_col2:\
                if 'Net Income' in results.columns:\
                    fig = go.Figure()\
                    fig.add_trace(go.Scatter(x=results['Year'], y=results['Net Income'], name='Net Income', line=dict(color='#10B981', width=3)))\
                    fig.update_layout(title="Net Income Trend", height=300)\
                    st.plotly_chart(fig, use_container_width=True)\
\
        # TAB 2: BENCHMARK COMPARISON\
        with tab2:\
            st.subheader(f"Performance vs. \{selected_industry\} Benchmarks")\
            st.markdown("Comparison against Industry Average, Leaders, and Known Failure Patterns.")\
            \
            # Benchmark Table Construction\
            benchmark_data = []\
            \
            metrics_to_check = ['Gross Margin', 'Net Margin', 'Revenue Growth', 'Current Ratio', 'Debt-to-Assets']\
            \
            for m in metrics_to_check:\
                if m in latest_year:\
                    val = latest_year[m]\
                    if val is None: continue\
                    \
                    status = get_status_color(m, val, benchmarks)\
                    status_icon = "\uc0\u55357 \u57314 " if status == "green" else "\u55357 \u56628 " if status == "red" else "\u55357 \u57313 "\
                    \
                    # Formatting ranges for display\
                    avg_range = f"\{benchmarks['average'][m][0]\} - \{benchmarks['average'][m][1]\}"\
                    leader_range = f"> \{benchmarks['leaders'][m][0]\}"\
                    \
                    # Calculate Distance\
                    # Distance logic: How far from average midpoint\
                    avg_mid = sum(benchmarks['average'][m]) / 2\
                    distance = val - avg_mid\
                    dist_str = f"\{distance:+.1f\} pts vs Avg"\
                    \
                    benchmark_data.append(\{\
                        "Metric": m,\
                        "Your Value": format_metric(val, is_percent=(m != 'Current Ratio')),\
                        "Status": status_icon,\
                        "Industry Avg": avg_range,\
                        "Industry Leaders": leader_range,\
                        "Variance": dist_str\
                    \})\
            \
            b_df = pd.DataFrame(benchmark_data)\
            st.table(b_df)\
            \
            st.info("\uc0\u55357 \u57314  = Leader Tier | \u55357 \u57313  = Average Range | \u55357 \u56628  = Risk/Failure Pattern")\
\
        # TAB 3: AI INSIGHTS\
        with tab3:\
            st.subheader("Generative AI Strategic Analysis")\
            \
            # Prepare context for AI\
            data_str = results.to_string()\
            comp_str = b_df.to_string()\
            \
            if st.button("Generate New Analysis"):\
                if api_key:\
                    ai_response = get_gemini_analysis(api_key, selected_industry, data_str, comp_str)\
                    if "API_ERROR" in ai_response:\
                        st.error("Gemini API Error. Falling back to Mock Engine.")\
                        st.markdown(get_mock_analysis(selected_industry, latest_year))\
                    else:\
                        st.markdown(ai_response)\
                else:\
                    st.warning("No API Key detected. Using Mock Analysis Engine.")\
                    st.markdown(get_mock_analysis(selected_industry, latest_year))\
            else:\
                st.write("Click the button above to generate insights.")\
\
        # TAB 4: CHAT INTERFACE\
        with tab4:\
            st.subheader("\uc0\u55357 \u56492  Financial Assistant")\
            st.markdown("Ask questions about the analysis (e.g., *'Why is my current ratio red?'*)")\
            \
            # Display chat history\
            for message in st.session_state.chat_history:\
                with st.chat_message(message["role"]):\
                    st.markdown(message["content"])\
            \
            # User Input\
            if prompt := st.chat_input("Ask a follow-up question..."):\
                st.session_state.chat_history.append(\{"role": "user", "content": prompt\})\
                with st.chat_message("user"):\
                    st.markdown(prompt)\
                \
                # Get Response (Mock or AI)\
                # Note: In full version, pass chat history to Gemini\
                response = chat_agent(prompt, results)\
                \
                with st.chat_message("assistant"):\
                    st.markdown(response)\
                st.session_state.chat_history.append(\{"role": "assistant", "content": response\})\
\
if __name__ == "__main__":\
    main()}

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
import re
from typing import Dict, Optional, Tuple, List

# ==========================================
# 1. CONFIGURATION & BENCHMARK DATA
# ==========================================

st.set_page_config(
    page_title="Financial Analyst AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the dots and metrics
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .status-dot {
        height: 12px;
        width: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 5px;
    }
    .dot-green { background-color: #28a745; }
    .dot-yellow { background-color: #ffc107; }
    .dot-red { background-color: #dc3545; }
    .small-font { font-size: 0.85rem; color: #666; }
    </style>
""", unsafe_allow_html=True)

# 12 Industries with specific benchmark ranges (Avg, Leader, Failure)
# Structure: (Min_Avg, Max_Avg), (Min_Leader, Max_Leader), (Min_Fail, Max_Fail)
# Note: These are generalized realistic ranges for demonstration.
INDUSTRY_BENCHMARKS = {
    "SaaS / Technology": {
        "gross_margin": ((70, 80), (82, 95), (30, 60)),
        "net_margin": ((10, 20), (25, 40), (-50, 0)),
        "current_ratio": ((1.5, 2.5), (3.0, 5.0), (0, 0.9)),
        "debt_to_equity": ((0.5, 1.0), (0, 0.4), (2.0, 10.0)),
        "revenue_growth": ((15, 30), (40, 100), (-20, 5))
    },
    "Retail / E-commerce": {
        "gross_margin": ((45, 55), (58, 65), (20, 35)),
        "net_margin": ((3, 6), (7, 12), (-10, 1)),
        "current_ratio": ((1.2, 2.0), (2.0, 3.5), (0, 0.9)),
        "debt_to_equity": ((1.0, 2.0), (0.5, 1.0), (3.0, 10.0)),
        "revenue_growth": ((5, 10), (15, 25), (-15, 0))
    },
    "Manufacturing": {
        "gross_margin": ((25, 35), (38, 45), (10, 20)),
        "net_margin": ((5, 10), (12, 18), (-15, 2)),
        "current_ratio": ((1.2, 2.0), (2.5, 4.0), (0, 1.0)),
        "debt_to_equity": ((0.8, 1.5), (0.2, 0.6), (2.5, 8.0)),
        "revenue_growth": ((3, 8), (10, 20), (-10, 0))
    },
    "Healthcare / Biotech": {
        "gross_margin": ((50, 70), (75, 90), (20, 40)),
        "net_margin": ((10, 18), (20, 35), (-100, -5)), # Bio often burns cash
        "current_ratio": ((2.0, 3.5), (4.0, 8.0), (0, 1.2)),
        "debt_to_equity": ((0.4, 0.8), (0, 0.3), (1.5, 5.0)),
        "revenue_growth": ((5, 15), (20, 50), (-10, 0))
    },
    "Restaurants / Food Service": {
        "gross_margin": ((60, 70), (72, 80), (40, 55)),
        "net_margin": ((3, 6), (8, 15), (-10, 1)),
        "current_ratio": ((0.8, 1.5), (1.8, 3.0), (0, 0.6)), # Restaurants run lean
        "debt_to_equity": ((1.5, 3.0), (0.5, 1.2), (4.0, 15.0)),
        "revenue_growth": ((4, 8), (10, 25), (-20, 0))
    },
    "Construction / Real Estate": {
        "gross_margin": ((15, 25), (28, 35), (5, 12)),
        "net_margin": ((4, 9), (10, 15), (-15, 2)),
        "current_ratio": ((1.3, 2.2), (2.5, 4.0), (0, 1.0)),
        "debt_to_equity": ((1.2, 2.5), (0.5, 1.0), (3.5, 12.0)),
        "revenue_growth": ((5, 12), (15, 30), (-25, 0))
    },
    "Energy / Utilities": {
        "gross_margin": ((30, 45), (50, 65), (10, 25)),
        "net_margin": ((8, 12), (15, 20), (-10, 5)),
        "current_ratio": ((1.0, 1.5), (2.0, 3.0), (0, 0.8)),
        "debt_to_equity": ((1.0, 2.0), (0.5, 0.9), (3.0, 10.0)),
        "revenue_growth": ((2, 6), (8, 15), (-10, -2))
    },
    "Professional Services": {
        "gross_margin": ((40, 50), (55, 65), (20, 30)),
        "net_margin": ((10, 15), (20, 30), (-5, 5)),
        "current_ratio": ((1.5, 2.5), (3.0, 5.0), (0, 1.1)),
        "debt_to_equity": ((0.3, 0.8), (0, 0.2), (1.5, 5.0)),
        "revenue_growth": ((8, 15), (20, 40), (-15, 0))
    },
    "Transportation / Logistics": {
        "gross_margin": ((15, 25), (28, 35), (5, 12)),
        "net_margin": ((4, 8), (10, 14), (-12, 1)),
        "current_ratio": ((1.1, 1.6), (1.8, 2.5), (0, 0.9)),
        "debt_to_equity": ((1.0, 2.0), (0.4, 0.8), (3.0, 9.0)),
        "revenue_growth": ((4, 10), (12, 20), (-15, -2))
    },
    "Media / Entertainment": {
        "gross_margin": ((35, 45), (50, 65), (15, 25)),
        "net_margin": ((8, 14), (18, 25), (-20, 0)),
        "current_ratio": ((1.4, 2.0), (2.5, 4.0), (0, 1.0)),
        "debt_to_equity": ((0.8, 1.5), (0.2, 0.6), (2.5, 8.0)),
        "revenue_growth": ((5, 15), (20, 40), (-20, 0))
    },
    "Agriculture": {
        "gross_margin": ((10, 20), (22, 30), (0, 8)),
        "net_margin": ((3, 7), (8, 12), (-10, 0)),
        "current_ratio": ((1.2, 2.0), (2.5, 3.5), (0, 0.9)),
        "debt_to_equity": ((0.5, 1.2), (0.1, 0.4), (2.0, 6.0)),
        "revenue_growth": ((2, 6), (8, 15), (-10, -2))
    },
    "Financial Services": {
        "gross_margin": ((90, 98), (99, 100), (50, 80)), # Different for banks, treating Rev as Net Interest Inc roughly
        "net_margin": ((15, 25), (30, 45), (-10, 5)),
        "current_ratio": ((1.0, 1.0), (1.0, 1.0), (0, 0.0)), # Not relevant for banks usually
        "debt_to_equity": ((5.0, 8.0), (2.0, 4.0), (10.0, 20.0)),
        "revenue_growth": ((4, 8), (10, 20), (-10, 0))
    }
}

# Extensive Column Mapping (The "Brain" for parsing)
COLUMN_MAPPING = {
    # Income Statement
    'revenue': ['revenue', 'sales', 'total revenue', 'net sales', 'total sales', 'gross revenue'],
    'cogs': ['cost of goods sold', 'cogs', 'cost of revenue', 'cost of sales', 'direct costs'],
    'gross_profit': ['gross profit', 'gross margin', 'gross income'],
    'operating_expenses': ['operating expenses', 'opex', 'total operating expenses', 'sg&a', 'selling, general and administrative'],
    'operating_income': ['operating income', 'operating profit', 'ebit', 'income from operations'],
    'net_income': ['net income', 'net profit', 'net earnings', 'profit after tax', 'net loss'],
    'interest_expense': ['interest expense', 'interest costs', 'finance costs'],
    'shares_outstanding': ['shares outstanding', 'weighted average shares', 'common shares outstanding'],
    
    # Balance Sheet
    'total_assets': ['total assets', 'assets'],
    'current_assets': ['current assets', 'total current assets'],
    'cash': ['cash', 'cash and cash equivalents', 'cash & equivalents', 'cash and equivalents'],
    'accounts_receivable': ['accounts receivable', 'receivables', 'net receivables', 'trade receivables'],
    'inventory': ['inventory', 'inventories', 'merchandise inventory', 'stock'],
    'total_liabilities': ['total liabilities', 'liabilities'],
    'current_liabilities': ['current liabilities', 'total current liabilities'],
    'accounts_payable': ['accounts payable', 'payables', 'trade payables'],
    'equity': ['shareholders equity', 'total equity', 'stockholders equity', 'total shareholders equity', 'equity'],
    'debt': ['total debt', 'long term debt', 'short term debt', 'notes payable', 'loans'], # Often requires sum of short+long
    
    # Cash Flow
    'operating_cash_flow': ['cash flow from operations', 'operating cash flow', 'net cash from operating activities', 'cash provided by operations'],
    'investing_cash_flow': ['cash flow from investing', 'investing cash flow', 'net cash from investing activities'],
    'financing_cash_flow': ['cash flow from financing', 'financing cash flow', 'net cash from financing activities'],
}

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================

def clean_column_name(col_name):
    """Normalizes string for comparison."""
    if not isinstance(col_name, str):
        return str(col_name)
    return col_name.lower().strip().replace('_', ' ').replace('-', ' ')

def map_columns(df):
    """Maps user columns to standard internal names."""
    mapped_data = pd.DataFrame(index=df.index)
    
    # Create a reverse lookup for debugging/display
    found_cols = {}

    for standard_name, potential_names in COLUMN_MAPPING.items():
        found = False
        for user_col in df.columns:
            cleaned_user = clean_column_name(user_col)
            if any(cleaned_user == p for p in potential_names):
                # Try to convert to numeric, coerce errors to NaN
                series = pd.to_numeric(df[user_col].replace('[\$,]', '', regex=True), errors='coerce')
                mapped_data[standard_name] = series
                found_cols[standard_name] = user_col
                found = True
                break
        
        if not found:
            mapped_data[standard_name] = np.nan
            
    return mapped_data, found_cols

def detect_format_and_parse(file_obj):
    """
    Auto-detects Vertical (Metrics in rows, Years in cols) vs Horizontal.
    Returns a standardized DataFrame where Index = Years (sorted asc), Columns = Metrics.
    """
    try:
        # Read file
        if file_obj.name.endswith('.csv'):
            df = pd.read_csv(file_obj)
        else:
            df = pd.read_excel(file_obj)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

    # Clean whitespace in headers
    df.columns = df.columns.astype(str).str.strip()
    
    # Strategy: Look for "Year" or Date-like patterns to determine orientation
    # Regex for year (e.g., 2021, 2022, FY23)
    year_pattern = r'20\d{2}|19\d{2}|FY\d{2}'
    
    # Check headers for years (Vertical layout usually has years in columns)
    header_years = [col for col in df.columns if re.search(year_pattern, col, re.IGNORECASE)]
    
    is_vertical = False
    
    if len(header_years) >= 2:
        # Likely Vertical (Metrics in first col, Years in other cols)
        is_vertical = True
    else:
        # Check first column for years (Horizontal layout)
        first_col = df.iloc[:, 0].astype(str)
        col_years = [val for val in first_col if re.search(year_pattern, val, re.IGNORECASE)]
        if len(col_years) >= 2:
            is_vertical = False
        else:
            # Fallback: Assume horizontal if columns look like metrics
            is_vertical = False

    if is_vertical:
        # Transpose
        # Assume first column is metric names
        df = df.set_index(df.columns[0])
        df = df.transpose()
        # Now Index is Years, Columns are Metrics
    else:
        # Assume one of the columns is "Year" or "Date"
        # Find the year column
        year_col = None
        for col in df.columns:
            if clean_column_name(col) in ['year', 'fiscal year', 'period', 'date']:
                year_col = col
                break
        
        if year_col:
            df = df.set_index(year_col)
        else:
            # Try to find which column looks like a year
            for col in df.columns:
                if df[col].astype(str).str.match(year_pattern).all():
                    df = df.set_index(col)
                    break
    
    # Clean Index (Years)
    # Extract just the year number if possible
    try:
        df.index = df.index.astype(str).str.extract(r'(\d{4})')[0].astype(float).astype(int)
    except:
        pass # Keep as is if regex fails (e.g. unique identifiers)
    
    df = df.sort_index() # Ensure chronological order (oldest to newest)
    return df

# ==========================================
# 3. FINANCIAL ANALYSIS ENGINE
# ==========================================

class FinancialAnalyzer:
    def __init__(self, df):
        self.raw_df, self.mapping_log = map_columns(df)
        self.metrics = pd.DataFrame(index=self.raw_df.index)
    
    def safe_div(self, a, b):
        """Safe division returning NaN if b is 0 or NaN."""
        return np.where((b == 0) | (pd.isna(b)) | (pd.isna(a)), np.nan, a / b)

    def calculate_metrics(self):
        d = self.raw_df # Shorthand
        m = self.metrics # Shorthand
        
        # --- Profitability ---
        m['Gross Margin (%)'] = self.safe_div(d['gross_profit'], d['revenue']) * 100
        m['Net Margin (%)'] = self.safe_div(d['net_income'], d['revenue']) * 100
        m['Operating Margin (%)'] = self.safe_div(d['operating_income'], d['revenue']) * 100
        m['ROA (%)'] = self.safe_div(d['net_income'], d['total_assets']) * 100
        m['ROE (%)'] = self.safe_div(d['net_income'], d['equity']) * 100
        
        # --- Liquidity ---
        m['Current Ratio'] = self.safe_div(d['current_assets'], d['current_liabilities'])
        # Quick Ratio: (Current Assets - Inventory) / Current Liab
        m['Quick Ratio'] = self.safe_div(d['current_assets'] - d['inventory'].fillna(0), d['current_liabilities'])
        m['Operating Cash Flow Ratio'] = self.safe_div(d['operating_cash_flow'], d['current_liabilities'])

        # --- Leverage ---
        m['Debt to Assets (%)'] = self.safe_div(d['total_liabilities'], d['total_assets']) * 100
        m['Debt to Equity'] = self.safe_div(d['total_liabilities'], d['equity'])
        m['Interest Coverage'] = self.safe_div(d['operating_income'], d['interest_expense'])

        # --- Efficiency ---
        m['Asset Turnover'] = self.safe_div(d['revenue'], d['total_assets'])
        
        # Inventory Turnover = COGS / Average Inventory (using current for simplicity or avg if prev year exists)
        # Using current inventory for point-in-time simplicity
        m['Inventory Turnover'] = self.safe_div(d['cogs'], d['inventory'])
        m['Days Sales Inventory'] = self.safe_div(365, m['Inventory Turnover'])
        
        # Receivables Turnover = Revenue / AR
        m['AR Turnover'] = self.safe_div(d['revenue'], d['accounts_receivable'])
        days_sales_outstanding = self.safe_div(365, m['AR Turnover'])
        
        # Accounts Payable Turnover (Est) = COGS / AP
        ap_turnover = self.safe_div(d['cogs'], d['accounts_payable'])
        days_payable_outstanding = self.safe_div(365, ap_turnover)

        # Cash Gap (Cash Conversion Cycle) = DSI + DSO - DPO
        # Only calc if all 3 exist
        m['Cash Conversion Cycle (Days)'] = days_sales_outstanding + m['Days Sales Inventory'] - days_payable_outstanding
        
        # --- Growth (YoY) ---
        m['Revenue Growth (%)'] = d['revenue'].pct_change() * 100
        m['Net Income Growth (%)'] = d['net_income'].pct_change() * 100
        m['Asset Growth (%)'] = d['total_assets'].pct_change() * 100
        
        # --- Valuation / Other ---
        m['EPS'] = self.safe_div(d['net_income'], d['shares_outstanding'])

        return m

# ==========================================
# 4. AI INSIGHT ENGINE (HYBRID)
# ==========================================

def get_status_color(value, metric_name, industry_data):
    """Determines Red/Yellow/Green based on industry benchmarks."""
    if pd.isna(value) or metric_name not in industry_data:
        return "gray"
    
    # Mapping metric keys to benchmark keys
    key_map = {
        'Gross Margin (%)': 'gross_margin',
        'Net Margin (%)': 'net_margin',
        'Current Ratio': 'current_ratio',
        'Revenue Growth (%)': 'revenue_growth',
        'Debt to Equity': 'debt_to_equity'
    }
    
    if metric_name not in key_map:
        return "gray"
        
    bench_key = key_map[metric_name]
    avg_rng, lead_rng, fail_rng = industry_data[bench_key]
    
    # Logic: 
    # Green if in Leader range or better
    # Red if in Failure range
    # Yellow otherwise
    
    # Handle "Lower is Better" metrics (e.g., Debt)
    lower_is_better = bench_key in ['debt_to_equity']
    
    if lower_is_better:
        if value <= lead_rng[1]: return "green"
        if value >= fail_rng[0]: return "red"
        return "yellow"
    else:
        if value >= lead_rng[0]: return "green"
        if value <= fail_rng[1]: return "red"
        return "yellow"

def generate_logic_based_insights(metrics_df, industry, industry_data):
    """
    Generates detailed textual analysis WITHOUT an API key.
    Uses extensive if/else logic to construct paragraphs.
    """
    latest = metrics_df.iloc[-1]
    prev = metrics_df.iloc[-2] if len(metrics_df) > 1 else latest
    
    report = []
    report.append(f"### 🤖 AI Financial Analysis (Synthetic Mode)\n")
    report.append(f"**Industry Context:** Analyzing against {industry} standards.\n")
    
    # 1. Profitability Analysis
    report.append("#### 💰 Profitability & Growth")
    gm = latest.get('Gross Margin (%)', np.nan)
    nm = latest.get('Net Margin (%)', np.nan)
    rev_g = latest.get('Revenue Growth (%)', np.nan)
    
    prof_text = ""
    if pd.isna(gm):
        prof_text += "Insufficient data to calculate Gross Margin. "
    else:
        bench = industry_data.get('gross_margin', ((0,0),(0,0),(0,0)))
        avg_low, avg_high = bench[0]
        if gm < avg_low:
            prof_text += f"Gross Margin of {gm:.1f}% is **below the industry average** ({avg_low}-{avg_high}%), suggesting pricing pressure or high direct costs. "
        elif gm > avg_high:
            prof_text += f"Gross Margin of {gm:.1f}% is **strong**, exceeding the industry average. "
        else:
            prof_text += f"Gross Margin of {gm:.1f}% is within healthy industry norms. "
            
    if not pd.isna(rev_g):
        if rev_g > 0:
            prof_text += f"Revenue is growing at {rev_g:.1f}% YoY. "
        else:
            prof_text += f"WARNING: Revenue contracted by {abs(rev_g):.1f}% this year. "
            
    report.append(prof_text)
    
    # 2. Liquidity Analysis
    report.append("#### 💧 Liquidity & Solvency")
    cr = latest.get('Current Ratio', np.nan)
    qr = latest.get('Quick Ratio', np.nan)
    ccc = latest.get('Cash Conversion Cycle (Days)', np.nan)
    
    liq_text = ""
    if not pd.isna(cr):
        if cr < 1.0:
            liq_text += f"**CRITICAL:** Current Ratio is {cr:.2f}, indicating the company may struggle to pay short-term obligations. "
        elif cr < 1.5:
            liq_text += f"Current Ratio of {cr:.2f} is tight but manageable. "
        else:
            liq_text += f"Liquidity is robust with a Current Ratio of {cr:.2f}. "
            
    if not pd.isna(ccc):
        liq_text += f"The Cash Conversion Cycle is {ccc:.0f} days. "
        if ccc > 100:
            liq_text += "This is a long cycle, potentially tying up significant cash in operations. "
    
    report.append(liq_text)

    # 3. Efficiency & Risk
    report.append("#### ⚙️ Efficiency & Risk")
    debt_eq = latest.get('Debt to Equity', np.nan)
    roa = latest.get('ROA (%)', np.nan)
    
    eff_text = ""
    if not pd.isna(debt_eq):
        if debt_eq > 2.0:
            eff_text += f"Leverage is high (Debt/Equity: {debt_eq:.2f}), increasing financial risk. "
        else:
            eff_text += "Leverage appears conservatively managed. "
            
    if not pd.isna(roa):
        if roa < 0:
            eff_text += "The company is generating negative returns on its assets. "
        elif roa > 10:
            eff_text += "Asset efficiency is excellent, generating over 10% return on assets. "
            
    report.append(eff_text)
    
    # 4. Recommendations
    report.append("#### 🚀 Strategic Recommendations")
    recs = []
    if not pd.isna(gm) and gm < industry_data['gross_margin'][0][0]:
        recs.append("- **Cost Review:** Audit COGS immediately. Negotiate with suppliers or review pricing strategy to lift Gross Margin.")
    if not pd.isna(cr) and cr < 1.2:
        recs.append("- **Cash Preservation:** Immediate focus on cash flow is needed. Delay CapEx and accelerate receivables collection.")
    if not pd.isna(rev_g) and rev_g < 5:
        recs.append("- **Growth Strategy:** Top-line growth is stagnant. Investigate new marketing channels or product line expansions.")
    
    if not recs:
        recs.append("- Continue monitoring key ratios quarterly.")
        recs.append("- Benchmark against top competitors to find marginal gains.")
        
    report.extend(recs)
    
    return "\n".join(report)

def generate_gemini_insights(api_key, metrics_df, industry):
    """Calls Gemini API for insights."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        # Construct Prompt
        data_str = metrics_df.to_string()
        prompt = f"""
        You are a senior financial analyst. Analyze the following financial metrics for a company in the {industry} industry.
        
        DATA:
        {data_str}
        
        REQUIREMENTS:
        1. Executive Summary of financial health.
        2. Strengths & Weaknesses (cite specific numbers).
        3. Risk Assessment (identify failure patterns).
        4. 3-5 Actionable Recommendations.
        5. Use professional tone.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ API Error: {str(e)}. Switching to Logic-Based Analysis..."

# ==========================================
# 5. MAIN UI
# ==========================================

def main():
    # Sidebar
    st.sidebar.title("Configuration")
    
    # API Key Handling
    api_key = st.sidebar.text_input("Gemini API Key (Optional)", type="password")
    st.sidebar.info("Leave blank to use the built-in Logic-Based Analyst.")
    
    # Inputs
    industry = st.sidebar.selectbox("Select Industry", list(INDUSTRY_BENCHMARKS.keys()))
    
    uploaded_file = st.sidebar.file_uploader("Upload Financials", type=['xlsx', 'csv'])
    
    # Clear Data Button (Session State)
    if 'data_cleared' not in st.session_state:
        st.session_state.data_cleared = False
        
    if st.sidebar.button("Clear Data"):
        st.session_state.data_cleared = True
        st.rerun()
        
    if st.session_state.data_cleared and uploaded_file:
        st.warning("Data cleared. Please re-upload or uncheck clear.")
        st.session_state.data_cleared = False
        return

    # Main Content
    st.title("📊 AI-Powered Financial Analyzer")
    st.markdown("Upload your Balance Sheet, Income Statement, and Cash Flow data (Excel/CSV) to get deep insights.")

    if uploaded_file:
        # 1. Parse Data
        raw_df = detect_format_and_parse(uploaded_file)
        
        if raw_df is not None:
            # 2. Run Calculations
            analyzer = FinancialAnalyzer(raw_df)
            metrics_df = analyzer.calculate_metrics()
            
            # 3. KPI Dashboard
            st.divider()
            st.header("📈 Financial Performance Scorecard")
            
            # Filter only calculated metrics (columns not all NaN)
            valid_metrics = metrics_df.dropna(axis=1, how='all')
            
            # Display KPIs with colored dots
            # We'll display the LATEST year's data
            latest_year = valid_metrics.index[-1]
            st.subheader(f"Results for FY {latest_year}")
            
            # Create columns for grid layout
            cols = st.columns(4)
            metric_names = valid_metrics.columns.tolist()
            
            for i, metric in enumerate(metric_names):
                val = valid_metrics.loc[latest_year, metric]
                if pd.isna(val): continue
                
                # Get Industry Data for color coding
                color = get_status_color(val, metric, INDUSTRY_BENCHMARKS[industry])
                dot_class = f"dot-{color}"
                
                with cols[i % 4]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="small-font">{metric}</div>
                        <div style="font-size: 1.5rem; font-weight: bold;">
                            <span class="status-dot {dot_class}"></span>
                            {val:,.2f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # 4. Interactive Charts
            st.divider()
            st.header("📊 Interactive Visualizations")
            
            chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs([
                "Revenue Growth", "Net Income Trend", "ROA", "Liquidity Health"
            ])
            
            with chart_tab1:
                if 'Revenue Growth (%)' in metrics_df.columns:
                    fig = px.bar(metrics_df, x=metrics_df.index, y='Revenue Growth (%)', 
                                 title="Year-over-Year Revenue Growth", color='Revenue Growth (%)',
                                 color_continuous_scale='RdYlGn')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("Insufficient data for Revenue Growth.")
            
            with chart_tab2:
                if 'net_income' in analyzer.raw_df.columns:
                    fig = px.line(analyzer.raw_df, x=analyzer.raw_df.index, y='net_income', 
                                  title="Net Income Trend", markers=True)
                    st.plotly_chart(fig, use_container_width=True)
            
            with chart_tab3:
                if 'ROA (%)' in metrics_df.columns:
                    fig = px.area(metrics_df, x=metrics_df.index, y='ROA (%)', title="Return on Assets")
                    st.plotly_chart(fig, use_container_width=True)
            
            with chart_tab4:
                if 'Current Ratio' in metrics_df.columns and 'Quick Ratio' in metrics_df.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=metrics_df.index, y=metrics_df['Current Ratio'], name='Current Ratio', mode='lines+markers'))
                    fig.add_trace(go.Scatter(x=metrics_df.index, y=metrics_df['Quick Ratio'], name='Quick Ratio', mode='lines+markers'))
                    fig.update_layout(title="Liquidity Health Check")
                    st.plotly_chart(fig, use_container_width=True)
            
            # 5. AI Insights Section
            st.divider()
            st.header("🤖 AI Analysis & Recommendations")
            
            if st.button("Generate Detailed Analysis"):
                with st.spinner("Analyzing data patterns..."):
                    if api_key:
                        analysis = generate_gemini_insights(api_key, metrics_df, industry)
                        st.markdown(analysis)
                    else:
                        analysis = generate_logic_based_insights(metrics_df, industry, INDUSTRY_BENCHMARKS[industry])
                        st.markdown(analysis)
                        st.caption("Generated using Logic-Based Analyst (No API Key Provided)")

            # 6. Raw Data Expander
            with st.expander("View Raw Data & Mappings"):
                st.write("### Normalized Data Used for Calc")
                st.dataframe(analyzer.raw_df)
                st.write("### Calculated Metrics")
                st.dataframe(metrics_df)

    else:
        # Sample Data Button for Demo
        if st.button("Load Sample Data"):
            # Create a sample DataFrame
            data = {
                'Year': [2021, 2022, 2023, 2024],
                'Revenue': [1000000, 1100000, 1050000, 1200000],
                'COGS': [600000, 650000, 700000, 720000],
                'Net Income': [100000, 120000, 50000, 130000],
                'Total Assets': [500000, 550000, 600000, 650000],
                'Current Assets': [200000, 220000, 180000, 250000],
                'Current Liabilities': [150000, 140000, 160000, 150000],
                'Inventory': [50000, 60000, 80000, 55000],
                'Receivables': [80000, 85000, 90000, 95000],
                'Payables': [40000, 45000, 50000, 42000]
            }
            df_sample = pd.DataFrame(data)
            df_sample.to_csv("sample_financials.csv", index=False)
            st.success("Sample data 'sample_financials.csv' created! Upload it to test.")

if __name__ == "__main__":
    main()

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
    .dot-gray { background-color: #adb5bd; }   
    .dot-red { background-color: #dc3545; }    
    .small-font { font-size: 0.85rem; color: #666; }
    </style>
""", unsafe_allow_html=True)

# Master Benchmark Data
INDUSTRY_BENCHMARKS = {
    "SaaS / Technology": {
        "gross_margin": ((70, 80), (82, 95), (30, 60)),
        "net_margin": ((10, 20), (25, 40), (-50, 0)),
        "current_ratio": ((1.5, 2.5), (3.0, 5.0), (0, 0.9)),
        "debt_to_equity": ((0.5, 1.0), (0, 0.4), (2.0, 10.0)),
        "revenue_growth": ((15, 30), (40, 100), (-20, 5)),
        "roa": ((5, 10), (15, 25), (-20, 0)),
        "roe": ((10, 20), (25, 40), (-30, 0)),
        "operating_margin": ((10, 20), (25, 35), (-20, 0)),
        "quick_ratio": ((1.5, 2.5), (3.0, 5.0), (0, 1.0)),
        "debt_to_assets": ((20, 40), (0, 15), (50, 90)),
        "asset_turnover": ((0.5, 0.8), (0.9, 1.5), (0, 0.4))
    },
    "Manufacturing": {
        "gross_margin": ((25, 35), (36, 45), (0, 20)),
        "net_margin": ((5, 9), (10, 15), (-10, 2)),
        "operating_margin": ((8, 12), (13, 20), (-5, 5)),
        "roa": ((5, 8), (9, 15), (-10, 2)),
        "roe": ((10, 15), (16, 25), (-15, 5)),
        "current_ratio": ((1.2, 1.8), (2.0, 3.0), (0, 1.0)),
        "quick_ratio": ((0.8, 1.2), (1.3, 2.0), (0, 0.6)),
        "operating_cash_flow_ratio": ((0.3, 0.5), (0.6, 1.0), (0, 0.1)),
        "debt_to_assets": ((30, 50), (10, 29), (60, 90)),
        "debt_to_equity": ((0.8, 1.5), (0, 0.7), (2.0, 5.0)),
        "interest_coverage": ((4, 7), (8, 20), (0, 2)),
        "inventory_turnover": ((5, 7), (8, 12), (0, 3)),
        "days_sales_inventory": ((50, 70), (30, 49), (90, 200)),
        "ar_turnover": ((7, 9), (10, 15), (0, 5)),
        "asset_turnover": ((0.8, 1.1), (1.2, 2.0), (0, 0.6)),
        "cash_conversion_cycle": ((50, 80), (20, 49), (100, 200)),
        "revenue_growth": ((3, 8), (9, 20), (-10, 0)),
        "net_income_growth": ((5, 10), (11, 25), (-15, 0)),
        "asset_growth": ((3, 6), (7, 15), (-5, 0))
    },
    "Healthcare / Biotech": {
        "gross_margin": ((50, 60), (61, 80), (0, 40)),
        "net_margin": ((10, 15), (16, 25), (-20, 0)),
        "operating_margin": ((15, 22), (23, 35), (-15, 5)),
        "roa": ((6, 10), (11, 20), (-15, 2)),
        "roe": ((12, 18), (19, 30), (-20, 5)),
        "current_ratio": ((2.0, 3.0), (3.1, 5.0), (0, 1.5)),
        "quick_ratio": ((1.5, 2.5), (2.6, 4.0), (0, 1.0)),
        "debt_to_assets": ((20, 40), (0, 19), (50, 80)),
        "debt_to_equity": ((0.3, 0.8), (0, 0.2), (1.0, 3.0)),
        "inventory_turnover": ((3, 5), (6, 10), (0, 2)),
        "revenue_growth": ((8, 15), (16, 40), (-5, 2))
    },
    "Restaurants / Food Service": {
        "gross_margin": ((60, 70), (71, 80), (30, 55)),
        "net_margin": ((3, 6), (7, 12), (-5, 1)),
        "current_ratio": ((0.8, 1.2), (1.3, 2.0), (0, 0.6)),
        "quick_ratio": ((0.4, 0.8), (0.9, 1.5), (0, 0.3)),
        "inventory_turnover": ((20, 30), (31, 50), (0, 15)),
        "cash_conversion_cycle": ((-20, 0), (-50, -21), (10, 50)),
        "debt_to_equity": ((1.5, 3.0), (0.5, 1.4), (4.0, 10.0))
    },
    "Construction / Real Estate": {
        "gross_margin": ((18, 25), (26, 35), (5, 15)),
        "net_margin": ((5, 9), (10, 15), (-5, 2)),
        "current_ratio": ((1.1, 1.5), (1.6, 2.5), (0, 1.0)),
        "debt_to_equity": ((1.2, 2.0), (0.5, 1.1), (2.5, 6.0)),
        "inventory_turnover": ((4, 6), (7, 12), (0, 3)),
        "days_sales_inventory": ((60, 90), (30, 59), (100, 300))
    },
    "Energy / Utilities": {
        "gross_margin": ((35, 45), (46, 60), (10, 25)),
        "net_margin": ((8, 12), (13, 20), (-5, 5)),
        "current_ratio": ((1.0, 1.5), (1.6, 2.5), (0, 0.9)),
        "debt_to_assets": ((40, 55), (20, 39), (60, 90)),
        "asset_turnover": ((0.4, 0.6), (0.7, 1.0), (0, 0.3))
    },
    "Professional Services": {
        "gross_margin": ((40, 50), (51, 65), (20, 35)),
        "net_margin": ((10, 15), (16, 25), (-5, 5)),
        "current_ratio": ((1.5, 2.2), (2.3, 4.0), (0, 1.2)),
        "roe": ((15, 25), (26, 40), (-10, 5)),
        "inventory_turnover": ((0, 0), (0, 0), (0, 0)) 
    },
    "Transportation / Logistics": {
        "gross_margin": ((18, 25), (26, 35), (5, 15)),
        "net_margin": ((4, 7), (8, 12), (-5, 1)),
        "current_ratio": ((1.1, 1.5), (1.6, 2.5), (0, 1.0)),
        "debt_to_equity": ((1.0, 1.8), (0.4, 0.9), (2.0, 5.0)),
        "inventory_turnover": ((12, 18), (19, 30), (0, 10))
    },
    "Media / Entertainment": {
        "gross_margin": ((40, 50), (51, 65), (20, 35)),
        "net_margin": ((10, 15), (16, 25), (-10, 5)),
        "current_ratio": ((1.4, 2.0), (2.1, 3.5), (0, 1.1)),
        "debt_to_equity": ((0.8, 1.5), (0.2, 0.7), (2.0, 5.0))
    },
    "Agriculture": {
        "gross_margin": ((15, 25), (26, 35), (0, 10)),
        "net_margin": ((4, 8), (9, 14), (-5, 2)),
        "current_ratio": ((1.3, 1.8), (1.9, 3.0), (0, 1.1)),
        "debt_to_assets": ((30, 50), (10, 29), (55, 85)),
        "inventory_turnover": ((2, 4), (5, 8), (0, 1.5))
    },
    "Financial Services": {
        "gross_margin": ((90, 99), (99, 100), (50, 80)),
        "net_margin": ((15, 22), (23, 35), (0, 10)),
        "roa": ((1.0, 1.5), (1.6, 3.0), (0, 0.8)), 
        "roe": ((8, 12), (13, 20), (-5, 5)),
        "debt_to_equity": ((5.0, 9.0), (2.0, 4.9), (10.0, 20.0))
    },
     "Retail / E-commerce": {
        "gross_margin": ((45, 55), (58, 65), (20, 35)),
        "net_margin": ((3, 6), (7, 12), (-10, 1)),
        "current_ratio": ((1.2, 2.0), (2.0, 3.5), (0, 0.9)),
        "debt_to_equity": ((1.0, 2.0), (0.5, 1.0), (3.0, 10.0)),
        "revenue_growth": ((5, 10), (15, 25), (-15, 0)),
        "inventory_turnover": ((4, 8), (9, 15), (0, 3))
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
    'debt': ['total debt', 'long term debt', 'short term debt', 'notes payable', 'loans'], 
    
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
    year_pattern = r'20\d{2}|19\d{2}|FY\d{2}'
    
    # Check headers for years (Vertical layout usually has years in columns)
    header_years = [col for col in df.columns if re.search(year_pattern, col, re.IGNORECASE)]
    
    is_vertical = False
    
    if len(header_years) >= 2:
        is_vertical = True
    else:
        # Check first column for years (Horizontal layout)
        first_col = df.iloc[:, 0].astype(str)
        col_years = [val for val in first_col if re.search(year_pattern, val, re.IGNORECASE)]
        if len(col_years) >= 2:
            is_vertical = False
        else:
            is_vertical = False

    if is_vertical:
        # Transpose
        # Assume first column is metric names
        df = df.set_index(df.columns[0])
        df = df.transpose()
        # Now Index is Years, Columns are Metrics
    else:
        # Assume one of the columns is "Year" or "Date"
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
    try:
        df.index = df.index.astype(str).str.extract(r'(\d{4})')[0].astype(float).astype(int)
    except:
        pass 
    
    df = df.sort_index() 
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
        d = self.raw_df 
        m = self.metrics 
        
        # --- Profitability ---
        m['Gross Margin (%)'] = self.safe_div(d['gross_profit'], d['revenue']) * 100
        m['Net Margin (%)'] = self.safe_div(d['net_income'], d['revenue']) * 100
        m['Operating Margin (%)'] = self.safe_div(d['operating_income'], d['revenue']) * 100
        m['ROA (%)'] = self.safe_div(d['net_income'], d['total_assets']) * 100
        m['ROE (%)'] = self.safe_div(d['net_income'], d['equity']) * 100
        
        # --- Liquidity ---
        m['Current Ratio'] = self.safe_div(d['current_assets'], d['current_liabilities'])
        m['Quick Ratio'] = self.safe_div(d['current_assets'] - d['inventory'].fillna(0), d['current_liabilities'])
        m['Operating Cash Flow Ratio'] = self.safe_div(d['operating_cash_flow'], d['current_liabilities'])

        # --- Leverage ---
        m['Debt to Assets (%)'] = self.safe_div(d['total_liabilities'], d['total_assets']) * 100
        m['Debt to Equity'] = self.safe_div(d['total_liabilities'], d['equity'])
        m['Interest Coverage'] = self.safe_div(d['operating_income'], d['interest_expense'])

        # --- Efficiency ---
        m['Asset Turnover'] = self.safe_div(d['revenue'], d['total_assets'])
        m['Inventory Turnover'] = self.safe_div(d['cogs'], d['inventory'])
        m['Days Sales Inventory'] = self.safe_div(365, m['Inventory Turnover'])
        m['AR Turnover'] = self.safe_div(d['revenue'], d['accounts_receivable'])
        days_sales_outstanding = self.safe_div(365, m['AR Turnover'])
        ap_turnover = self.safe_div(d['cogs'], d['accounts_payable'])
        days_payable_outstanding = self.safe_div(365, ap_turnover)
        m['Cash Conversion Cycle (Days)'] = days_sales_outstanding + m['Days Sales Inventory'] - days_payable_outstanding
        
        # --- Growth (YoY) ---
        m['Revenue Growth (%)'] = d['revenue'].pct_change() * 100
        m['Net Income Growth (%)'] = d['net_income'].pct_change() * 100
        m['Asset Growth (%)'] = d['total_assets'].pct_change() * 100
        
        # --- Valuation ---
        m['EPS'] = self.safe_div(d['net_income'], d['shares_outstanding'])

        return m

# ==========================================
# 4. AI INSIGHT ENGINE (HYBRID)
# ==========================================

def get_status_color(value, metric_name, industry_data):
    """
    Returns 'green' (better than avg), 'gray' (avg), or 'red' (worse than avg).
    """
    # 1. Check for missing value
    if pd.isna(value):
        return "gray"
    
    # 2. Map Metric Name to Dictionary Key
    key_map = {
        'Gross Margin (%)': 'gross_margin',
        'Net Margin (%)': 'net_margin',
        'Operating Margin (%)': 'operating_margin',
        'ROA (%)': 'roa',
        'ROE (%)': 'roe',
        'Current Ratio': 'current_ratio',
        'Quick Ratio': 'quick_ratio',
        'Operating Cash Flow Ratio': 'operating_cash_flow_ratio',
        'Debt to Assets (%)': 'debt_to_assets',
        'Debt to Equity': 'debt_to_equity',
        'Interest Coverage': 'interest_coverage',
        'Inventory Turnover': 'inventory_turnover',
        'Days Sales Inventory': 'days_sales_inventory',
        'AR Turnover': 'ar_turnover',
        'Asset Turnover': 'asset_turnover',
        'Cash Conversion Cycle (Days)': 'cash_conversion_cycle',
        'Revenue Growth (%)': 'revenue_growth',
        'Net Income Growth (%)': 'net_income_growth',
        'Asset Growth (%)': 'asset_growth'
    }
    
    # If we don't know this metric, return gray
    if metric_name not in key_map:
        return "gray"
        
    bench_key = key_map[metric_name]
    
    # 3. Check if we have benchmarks for this specific industry
    if bench_key not in industry_data:
        return "gray"
        
    # 4. Get the "Average" Range
    avg_range = industry_data[bench_key][0] 
    min_avg, max_avg = avg_range
    
    # 5. Define "Lower is Better" metrics
    lower_is_better = [
        'debt_to_assets', 
        'debt_to_equity', 
        'days_sales_inventory', 
        'cash_conversion_cycle'
    ]
    
    is_lower_good = bench_key in lower_is_better
    
    # 6. Determine Color
    if is_lower_good:
        if value < min_avg:
            return "green" # Lower than avg range = Better
        elif value > max_avg:
            return "red"   # Higher than avg range = Worse
        else:
            return "gray"  # Within avg range = Neutral
    else:
        # Standard "Higher is Better"
        if value > max_avg:
            return "green" # Higher than avg range = Better
        elif value < min_avg:
            return "red"   # Lower than avg range = Worse
        else:
            return "gray"  # Within avg range = Neutral

def generate_logic_based_insights(metrics_df, industry, industry_data):
    """
    Generates detailed textual analysis WITHOUT an API key.
    Uses extensive if/else logic to construct paragraphs.
    """
    latest = metrics_df.iloc[-1]
    
    report = []
    report.append(f"### 🤖 AI Financial Analysis (Synthetic Mode)\n")
    report.append(f"**Industry Context:** Analyzing against {industry} standards.\n")
    
    # 1. Profitability Analysis
    report.append("#### 💰 Profitability & Growth")
    gm = latest.get('Gross Margin (%)', np.nan)
    rev_g = latest.get('Revenue Growth (%)', np.nan)
    
    prof_text = ""
    if pd.isna(gm):
        prof_text += "Insufficient data to calculate Gross Margin. "
    else:
        # Check against average benchmark (index 0)
        bench_gm = industry_data.get('gross_margin', ((0,0),(0,0),(0,0)))
        avg_low, avg_high = bench_gm[0]
        
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
    # Generic logic checks
    if not pd.isna(gm) and gm < industry_data.get('gross_margin', [(0,0)])[0][0]:
        recs.append("- **Cost Review:** Audit COGS immediately. Negotiate with suppliers or review pricing strategy.")
    if not pd.isna(cr) and cr < 1.2:
        recs.append("- **Cash Preservation:** Immediate focus on cash flow is needed. Delay CapEx.")
    if not pd.isna(rev_g) and rev_g < 5:
        recs.append("- **Growth Strategy:** Top-line growth is stagnant. Investigate new marketing channels.")
    
    if not recs:
        recs.append("- Continue monitoring key ratios quarterly.")
        recs.append("- Benchmark against top competitors to find marginal gains.")
        
    report.extend(recs)
    
    return "\n".join(report)

def generate_gemini_insights(api_key, metrics_df, industry):
    """Calls Gemini API for insights."""
    try:
        genai.configure(api_key=api_key)
        # UPDATED: Using a specific, stable model version
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
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
# 5. CHATBOT FUNCTIONALITY
# ==========================================

def chat_with_data(api_key, user_query, metrics_df, industry):
    """Handles chat interaction."""
    if not api_key:
        # Fallback for No API Key (Simple Logic)
        user_query = user_query.lower()
        if "gross margin" in user_query:
            val = metrics_df['Gross Margin (%)'].iloc[-1]
            return f"The current Gross Margin is {val:.1f}%."
        elif "net income" in user_query:
            val = metrics_df['Net Margin (%)'].iloc[-1]
            return f"The Net Margin is {val:.1f}%."
        elif "current ratio" in user_query:
            val = metrics_df['Current Ratio'].iloc[-1]
            return f"The Current Ratio is {val:.2f}."
        elif "debt" in user_query:
            val = metrics_df['Debt to Equity'].iloc[-1]
            return f"Debt to Equity ratio is {val:.2f}."
        else:
            return "I am in 'Synthetic Mode' (No API Key). I can only answer basic questions about margins, ratios, and debt. Please provide an API key for full AI chat."

    try:
        genai.configure(api_key=api_key)
        # UPDATED: Using a specific, stable model version
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        # Context includes the full data
        data_context = metrics_df.to_string()
        prompt = f"""
        Context: You are a financial assistant analyzing a {industry} company.
        Financial Data:
        {data_context}
        
        User Question: {user_query}
        
        Answer concisely using the data provided.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"
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
# -------------------------------------------
            # NEW: Interactive Chatbot Section
            # -------------------------------------------
            st.divider()
            st.header("💬 Chat with Financial AI")
            st.markdown("Ask specific questions about the data (e.g., 'Why is the current ratio low?', 'How is debt trending?').")

            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat messages from history on app rerun
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # React to user input
            if prompt := st.chat_input("Ask a question about your financial data..."):
                # Display user message in chat message container
                st.chat_message("user").markdown(prompt)
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})

                with st.spinner("Thinking..."):
                    response = chat_with_data(api_key, prompt, metrics_df, industry)
                
                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

            # -------------------------------------------
            # End of Chatbot Section
            # -------------------------------------------
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

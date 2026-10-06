import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="Retail Executive Dashboard", layout="wide")
st.title("Commercial Performance & Executive Audit")

# 2. Ingest and Clean
@st.cache_data
def load_data():
    sales = pd.read_csv('Sales.csv', encoding='latin1')
    products = pd.read_csv('Products.csv', encoding='latin1')
    
    sales.columns = sales.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    products.columns = products.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    
    df = pd.merge(sales, products, on='productkey', how='left')
    df['unit_price_clean'] = df['unit_price_usd'].str.replace(r'[\$,]', '', regex=True).astype(float)
    df['revenue'] = df['quantity'] * df['unit_price_clean']
    df['order_date'] = pd.to_datetime(df['order_date'])
    return df

df = load_data()

# 3. Helper Function for Retention (Defined here so Python knows it!)
def order_frequency_summary(data_df, order_col='order_number', customer_col='customerkey'):
    customers = data_df.groupby(customer_col).agg(total_orders=(order_col, 'nunique')).reset_index()
    
    conditions = [
        customers['total_orders'] == 1,
        customers['total_orders'] == 2,
        customers['total_orders'] == 3
    ]
    choices = ['1 Order', '2 Orders', '3 Orders']
    customers['order_count'] = np.select(conditions, choices, default='4+ Orders')
    
    result = customers.groupby('order_count').agg(total_customers=(customer_col, 'count')).reset_index()
    
    # Enforce correct sorting
    sort_order = {'1 Order': 1, '2 Orders': 2, '3 Orders': 3, '4+ Orders': 4}
    result['sort_key'] = result['order_count'].map(sort_order)
    result = result.sort_values('sort_key').drop(columns=['sort_key']).reset_index(drop=True)
    
    result['% of total customers'] = (result['total_customers'] / result['total_customers'].sum() * 100).round(2)
    return result

# 4. Interactive Sidebar Filter
category_list = ['All'] + sorted(df['category'].dropna().unique().tolist())
selected_category = st.sidebar.selectbox("Filter by Category", category_list)

if selected_category != 'All':
    filtered_df = df[df['category'] == selected_category]
else:
    filtered_df = df

# 5. Top-Level Executive KPI Scorecards
col1, col2, col3 = st.columns(3)
total_rev = filtered_df['revenue'].sum()
total_units = filtered_df['quantity'].sum()
avg_order_val = total_rev / filtered_df['order_number'].nunique() if filtered_df['order_number'].nunique() > 0 else 0

col1.metric("Total Revenue", f"${total_rev:,.2f}")
col2.metric("Total Units Sold", f"{total_units:,.0f}")
col3.metric("Avg Order Value", f"${avg_order_val:,.2f}")

st.markdown("---")

# 6. Tabbed Analytical Navigation
tab1, tab2 = st.tabs(["Subcategory Pareto & Trends", "Customer Retention"])

# ==========================================
# TAB 1: PARETO & TRENDS
# ==========================================
with tab1:
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("Pareto Distribution")
        pareto_df = (
            filtered_df.groupby('subcategory')
            .agg(total_revenue=('revenue', 'sum'))
            .sort_values(by='total_revenue', ascending=False)
            .reset_index()
        )
        pareto_df['cum_revenue'] = pareto_df['total_revenue'].cumsum()
        pareto_df['cum_pct'] = (pareto_df['cum_revenue'] / total_rev * 100).round(2) if total_rev > 0 else 0
        
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.bar(pareto_df['subcategory'], pareto_df['total_revenue'] / 1e6, color='steelblue')
        ax1.set_xticklabels(pareto_df['subcategory'], rotation=90, fontsize=8)
        ax1.set_ylabel("Revenue ($M)", color='steelblue')
        
        ax2 = ax1.twinx()
        ax2.plot(pareto_df['subcategory'], pareto_df['cum_pct'], color='darkorange', marker='o')
        ax2.axhline(80, color='red', linestyle='--', linewidth=1.5)
        ax2.set_ylim(0, 105)
        ax2.set_ylabel("Cumulative %", color='darkorange')
        
        st.pyplot(fig1)

    with chart_col2:
        st.subheader("Annual Revenue Trend")
        trend_df = filtered_df.groupby(filtered_df['order_date'].dt.year)['revenue'].sum().reset_index()
        trend_df.columns = ['Year', 'Revenue']
        
        fig2, ax_trend = plt.subplots(figsize=(8, 5))
        ax_trend.plot(trend_df['Year'], trend_df['Revenue'] / 1e6, marker='s', color='#2ca02c', linewidth=2.5)
        ax_trend.set_ylabel("Revenue ($M)")
        ax_trend.set_xlabel("Year")
        ax_trend.grid(True, linestyle=':', alpha=0.6)
        
        st.pyplot(fig2)
        
    st.dataframe(pareto_df, use_container_width=True)

# ==========================================
# TAB 2: CUSTOMER RETENTION
# ==========================================
with tab2:
    st.subheader("Customer Repeat Purchase Distribution")
    
    freq_df = order_frequency_summary(filtered_df)
    
    total_cust = freq_df['total_customers'].sum()
    repeat_cust = freq_df[freq_df['order_count'] != '1 Order']['total_customers'].sum()
    rpr_pct = (repeat_cust / total_cust * 100) if total_cust > 0 else 0
    
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Unique Customers", f"{total_cust:,}")
    kpi2.metric("Repeat Buyers (>1 Order)", f"{repeat_cust:,}")
    kpi3.metric("Repeat Purchase Rate (RPR)", f"{rpr_pct:.1f}%")
    
    fig_ret, ax_ret = plt.subplots(figsize=(8, 4))
    bars = ax_ret.bar(freq_df['order_count'], freq_df['total_customers'], color='#1f77b4', width=0.5)
    
    for bar, pct in zip(bars, freq_df['% of total customers']):
        yval = bar.get_height()
        ax_ret.text(
            bar.get_x() + bar.get_width() / 2, 
            yval + (max(freq_df['total_customers']) * 0.02), 
            f"{int(yval):,}\n({pct:.1f}%)", 
            ha='center', 
            va='bottom', 
            fontsize=9,
            fontweight='bold'
        )
    
    ax_ret.set_ylim(0, max(freq_df['total_customers']) * 1.25)
    ax_ret.spines[['top', 'right', 'left']].set_visible(False)
    ax_ret.yaxis.grid(True, linestyle='--', alpha=0.4)
    
    st.pyplot(fig_ret)
    st.dataframe(freq_df, use_container_width=True)
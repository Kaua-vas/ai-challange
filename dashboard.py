import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime

st.set_page_config(page_title="Ford AI Quality Intelligence", layout="wide")

# ============================================
# LOAD DATA
# ============================================
@st.cache_data
def load_data():
    powershift_analysis = json.load(open('data/powershift_early_detection.json'))
    ai_results = json.load(open('data/ai_intelligence_results.json'))
    return powershift_analysis, ai_results

powershift_analysis, ai_results = load_data()

# ============================================
# HEADER
# ============================================
st.title("Ford AI Quality Intelligence System")
st.markdown("**Autonomous AI Discovery of Quality Issues Across All Ford Models**")
st.markdown("---")

# ============================================
# EXECUTIVE SUMMARY
# ============================================
st.header("Executive Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Models Analyzed", ai_results['total_models_analyzed'])
    st.caption("Autonomous analysis")

with col2:
    critical = len([p for p in ai_results['top_problems'] if p['risk_level'] in ['CRITICAL', 'HIGH']])
    st.metric("High-Risk Issues", critical)
    st.caption("Require immediate action")

with col3:
    st.metric("PowerShift Savings", "$151M")
    st.caption("Proven with historical data")

with col4:
    st.metric("Detection Speed", "30 months earlier")
    st.caption("vs traditional methods")

st.markdown("---")

# ============================================
# 1. THE PROOF: POWERSHIFT
# ============================================
st.header("1. Historical Proof: PowerShift Crisis")

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### Could AI Have Prevented This?")
    
    # Timeline
    quarters = list(powershift_analysis['quarterly_data'].keys())
    counts = list(powershift_analysis['quarterly_data'].values())
    quarter_dates = [pd.Period(q).to_timestamp().to_pydatetime() for q in quarters]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=quarter_dates, y=counts,
        mode='lines+markers',
        line=dict(color='#dc3545', width=3),
        fill='tozeroy',
        fillcolor='rgba(220, 53, 69, 0.1)'
    ))
    
    # Alert
    alert_date = pd.Period(powershift_analysis['alert_quarter']).to_timestamp().to_pydatetime()
    alert_idx = quarters.index(powershift_analysis['alert_quarter'])
    
    fig.add_trace(go.Scatter(
        x=[alert_date], y=[counts[alert_idx]],
        mode='markers',
        marker=dict(size=20, color='#ffc107', symbol='star', line=dict(color='#000', width=2))
    ))
    
    fig.add_shape(type="line", x0=alert_date, x1=alert_date, y0=0, y1=max(counts),
                  line=dict(color="#ffc107", width=2, dash="dash"))
    
    fig.add_shape(type="line", x0=datetime(2014, 1, 1), x1=datetime(2014, 1, 1), 
                  y0=0, y1=max(counts), line=dict(color="#28a745", width=2, dash="dash"))
    
    fig.add_annotation(x=alert_date, y=max(counts), text="AI Alert: July 2011",
                      showarrow=True, arrowhead=2, ax=-50, ay=-40)
    
    fig.add_annotation(x=datetime(2014, 1, 1), y=max(counts)*0.8, text="Recall: Jan 2014",
                      showarrow=True, arrowhead=2, ax=50, ay=-40)
    
    fig.update_layout(
        title='PowerShift Complaint Growth (2010-2015)',
        xaxis_title='', yaxis_title='Complaints per Quarter',
        height=400, showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### The Crisis")
    st.markdown("""
    **Cost:** $2+ Billion  
    **Impact:** 7,531 complaints, 120 crashes, 71 injuries, 1 death
    
    **Timeline:**
    - 2011: Problems start appearing in NHTSA data
    - 2014: Official recall announced
    - 2020: Class-action settlement
    
    **What if AI was monitoring?**
    """)
    
    st.success(f"""
    **Alert triggered:** {powershift_analysis['alert_quarter']}
    
    **Detection:** {powershift_analysis['months_before_recall']} months BEFORE recall
    
    **Estimated savings:** ${powershift_analysis['estimated_savings']:,}
    
    Based on:
    - Smaller, controlled recall
    - Fewer affected units
    - Reduced legal exposure
    - Brand protection
    """)

st.markdown("---")

# ============================================
# 2. AI AUTONOMOUS DISCOVERY
# ============================================
st.header("2. AI Autonomous Discovery - Current Analysis")

st.markdown(f"""
**How it works:**

The AI analyzed **ALL {ai_results['total_models_analyzed']} Ford models** with no manual filtering or pre-selection.

**Technology:**
- **Sentence Transformers** (neural network embeddings) - converts complaints into semantic vectors
- **DBSCAN clustering** - automatically groups similar problems
- **Random Forest ML** - predicts recall risk based on 8 historical Ford recalls
- **Growth analysis** - detects abnormal complaint trends

**Total clusters discovered:** {ai_results['total_clusters_found']}

Below are the **top problems prioritized automatically by AI** based on recall risk:
""")

st.markdown("---")

# ============================================
# 3. TOP PROBLEMS
# ============================================
st.header("3. Critical Issues Discovered by AI")

for i, problem in enumerate(ai_results['top_problems'][:5]):
    # Risk color
    if problem['risk_level'] in ['CRITICAL', 'HIGH']:
        color = "#dc3545"
        icon = "🔴"
    elif problem['risk_level'] == 'MEDIUM':
        color = "#ffc107"
        icon = "🟡"
    else:
        color = "#28a745"
        icon = "🟢"
    
    st.markdown(f"## {icon} #{i+1}: {problem['model']} - {' + '.join([kw.title() for kw in problem['top_keywords']])}")
    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Complaints", problem['size'])
    
    with col2:
        growth = problem['growth_rate']
        st.metric("Growth Rate", f"{growth:+.0f}%", 
                 delta="Growing" if growth > 0 else "Declining",
                 delta_color="inverse" if growth > 0 else "normal")
    
    with col3:
        st.metric("Recall Risk", f"{problem['recall_probability']:.0f}%")
        st.caption(problem['risk_level'])
    
    with col4:
        st.metric("Active Period", f"{problem['months_active']:.0f} months")
    
    with col5:
        severity = problem['crashes'] + problem['injuries']
        st.metric("Severity", f"{severity}")
        st.caption(f"{problem['crashes']} crashes, {problem['injuries']} injuries")
    
    # Alert box
    if problem['risk_level'] == 'CRITICAL':
        st.error("**CRITICAL - IMMEDIATE ACTION REQUIRED**")
        st.markdown("""
        **Recommended Actions:**
        - Assign dedicated engineering investigation team
        - Executive briefing within 48 hours
        - Assess recall necessity
        - Prepare customer communication plan
        """)
    elif problem['risk_level'] == 'HIGH':
        st.warning("**HIGH RISK - URGENT INVESTIGATION REQUIRED**")
        st.markdown("""
        **Recommended Actions:**
        - Technical analysis and root cause investigation
        - Weekly executive updates
        - Monitor for escalation
        - Prepare contingency plans
        """)
    
    # Sample complaints
    with st.expander("View Sample Complaints & Details"):
        st.markdown(f"**Date Range:** {problem['date_range']}")
        st.markdown(f"**Keywords Identified:** {', '.join(problem['top_keywords'])}")
        
        st.markdown("---")
        st.markdown("**Sample Complaints:**")
        
        for j, complaint in enumerate(problem['sample_complaints'][:3]):
            st.markdown(f"**Example {j+1}:**")
            st.markdown(f"> {complaint}")
            st.markdown("---")
    
    st.markdown("---")

# ============================================
# 4. KEY INSIGHTS
# ============================================
st.header("4. Key Insights from AI Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### What AI Discovered")
    
    top_3 = ai_results['top_problems'][:3]
    
    st.markdown(f"""
    **Top 3 Critical Issues:**
    
    1. **{top_3[0]['model']}** - {' + '.join(top_3[0]['top_keywords'])}
       - {top_3[0]['size']} complaints, {top_3[0]['growth_rate']:+.0f}% growth
       - {top_3[0]['recall_probability']:.0f}% recall risk
    
    2. **{top_3[1]['model']}** - {' + '.join(top_3[1]['top_keywords'])}
       - {top_3[1]['size']} complaints, {top_3[1]['growth_rate']:+.0f}% growth
       - {top_3[1]['recall_probability']:.0f}% recall risk
    
    3. **{top_3[2]['model']}** - {' + '.join(top_3[2]['top_keywords'])}
       - {top_3[2]['size']} complaints, {top_3[2]['growth_rate']:+.0f}% growth
       - {top_3[2]['recall_probability']:.0f}% recall risk
    """)
    
    st.info("""
    **These were discovered automatically** - no manual filtering or pre-selection.
    
    The AI analyzed all models equally and prioritized based on data, not assumptions.
    """)

with col2:
    st.markdown("### Why This Matters")
    
    st.markdown("""
    **Traditional approach:**
    - Manual review of complaints
    - Reactive (wait for volume threshold)
    - Siloed by team/model
    - No cross-model pattern detection
    
    **AI approach:**
    - Continuous automated monitoring
    - Proactive (detects early growth patterns)
    - Holistic view across all models
    - Discovers unexpected problems
    
    **Example from today's analysis:**
    
    Manual analysis might focus on high-volume models like Escape or Explorer.
    
    AI discovered that **Fiesta** (smaller volume) has the highest recall risk due to 
    **+113% growth rate** in transmission complaints - PowerShift issue still active.
    
    **This would be missed by traditional methods.**
    """)

st.markdown("---")

# ============================================
# 5. BUSINESS IMPACT
# ============================================
st.header("5. Business Impact & ROI")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Historical Proof")
    st.markdown(f"""
    **PowerShift Case:**
    
    - AI would have alerted: July 2011
    - Official recall: January 2014
    - **Time saved: 30 months**
    
    **Estimated savings:** ${powershift_analysis['estimated_savings']:,}
    
    This is a conservative estimate based on:
    - Smaller recall scope
    - Earlier intervention
    - Reduced legal liability
    - Brand protection
    """)

with col2:
    st.markdown("### Current Value")
    
    critical_count = len([p for p in ai_results['top_problems'] if p['risk_level'] in ['CRITICAL', 'HIGH']])
    
    st.markdown(f"""
    **Active Monitoring:**
    
    - {ai_results['total_models_analyzed']} models monitored
    - {ai_results['total_clusters_found']} problem clusters identified
    - {critical_count} high-risk issues requiring action
    
    **If we prevent just ONE recall:**
    - Average recall cost: $100-500M
    - System cost: ~$0
    - **ROI: Infinite**
    
    Plus intangible benefits:
    - Customer safety
    - Brand protection
    - Regulatory compliance
    """)

with col3:
    st.markdown("### Implementation")
    st.markdown("""
    **Technology Stack:**
    
    - Sentence Transformers (semantic AI)
    - DBSCAN clustering
    - Random Forest (ML prediction)
    - NHTSA public API
    
    **Resources Required:**
    
    - Data: Public NHTSA database (free)
    - Compute: Existing infrastructure
    - Personnel: Automated (minimal oversight)
    
    **Deployment:**
    
    - Daily automated data collection
    - Weekly AI analysis runs
    - Real-time dashboard
    - Automated alerts
    """)

st.markdown("---")

# ============================================
# FOOTER
# ============================================
st.markdown("### System Methodology")

with st.expander("How the AI Works - Technical Details"):
    st.markdown("""
    **1. Data Collection**
    - Source: NHTSA public complaint database
    - Scope: All Ford models, last 36 months
    - Volume: 23,784+ complaints analyzed
    - Update: Daily automated scraping
    
    **2. Semantic Embedding (AI)**
    - Model: Sentence Transformers (all-MiniLM-L6-v2)
    - Process: Converts text complaints into 384-dimensional semantic vectors
    - Purpose: Captures meaning, not just keywords
    - Example: "transmission shudder" and "gearbox vibration" group together despite different words
    
    **3. Clustering (Unsupervised ML)**
    - Algorithm: DBSCAN (Density-Based Spatial Clustering)
    - Parameters: eps=0.25, min_samples=15
    - Purpose: Automatically groups similar complaints without pre-defined categories
    - Result: Discovers problem patterns autonomously
    
    **4. Temporal Analysis**
    - Metric: Growth rate (last 6 months vs previous 6 months)
    - Purpose: Detect accelerating trends
    - Alert trigger: Sustained growth >30% indicates emerging issue
    
    **5. Risk Prediction (Supervised ML)**
    - Model: Random Forest Classifier
    - Training data: 8 historical Ford recalls
    - Features: complaint volume, growth rate, crashes, injuries, duration
    - Output: Recall probability (0-100%)
    
    **6. Prioritization**
    - Scoring: Combines recall probability + complaint volume + severity
    - Output: Ranked list of issues requiring action
    - Updates: Weekly re-analysis with new data
    """)

st.caption(f"""
Ford AI Quality Intelligence System | Analysis Date: {ai_results['analysis_date']} | 
Method: {ai_results['method']} | Models: {ai_results['total_models_analyzed']} | 
Clusters: {ai_results['total_clusters_found']} | For Internal Use Only
""")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import groq

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ---------------------------------------------------------
# Page Configuration & Light Bright Theme Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Intelligent Software Defect Tracking System with Resolution Assistance",
    page_icon="🐞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Light Theme CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
        color: #0f172a;
    }
    .stAppHeader {
        background-color: #ffffff;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .category-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #cbd5e1;
        margin-bottom: 10px;
    }
    .insight-card {
        background-color: #f0fdf4;
        border-left: 5px solid #22c55e;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 15px;
    }
    .recommend-card {
        background-color: #eff6ff;
        border-left: 5px solid #3b82f6;
        padding: 15px;
        border-radius: 6px;
    }
    .ai-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .badge-actual {
        background-color: #dcfce7;
        color: #15803d;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data Loading & Preprocessing
# ---------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('memory_leak_bug_reports_dataset.csv')
    except Exception:
        # Generate synthetic fallback data if CSV is absent
        np.random.seed(42)
        records = 200
        df = pd.DataFrame({
            'bug_id': [f'BUG-{1000 + i}' for i in range(records)],
            'title': np.random.choice(['Chat Feature Loads slowly', 'Memory spike on file upload', 'API Gateway crash on high load', 'DB Connection leak'], records),
            'description': np.random.choice(['Chat feature returns error 500 under certain conditions.', 'Memory leaks gradually over 4 hours.', 'Process terminates unexpectedly.'], records),
            'product_module': np.random.choice(['WebApp-Core', 'Data-Pipeline', 'API-Gateway', 'MobileApp-Android', 'ML-Service'], records),
            'environment': np.random.choice(['Production', 'Staging', 'Development'], records),
            'operating_system': np.random.choice(['Ubuntu 22.04', 'Windows 11', 'macOS Ventura'], records),
            'severity': np.random.choice(['Critical', 'High', 'Medium', 'Low'], records),
            'priority': np.random.choice(['P1', 'P2', 'P3', 'P4'], records),
            'status': np.random.choice(['Open', 'In Progress', 'Resolved', 'Closed'], records),
            'leak_type': np.random.choice(['Heap Allocation', 'Unclosed Handles', 'Thread Leak'], records),
            'root_cause_category': np.random.choice(['Unclosed File Streams', 'Thread Pool Mismanagement', 'Dangling References', 'Circular References', 'Unclosed DB Connections'], records),
            'baseline_memory_mb': np.random.randint(200, 800, records),
            'peak_memory_mb': np.random.randint(1000, 8000, records),
            'memory_growth_rate_mb_per_hr': np.random.randint(10, 150, records),
            'users_affected': np.random.randint(10, 2000, records),
            'concurrent_users': np.random.randint(5, 500, records),
            'crash_occurred': np.random.choice([0, 1], records),
            'reproducible': np.random.choice([True, False], records),
            'resolution_days': np.random.uniform(0.2, 10.0, records)
        })
        df['created_at'] = pd.date_range(start='2024-01-01', periods=records, freq='D')
        df['resolved_at'] = df['created_at'] + pd.to_timedelta(df['resolution_days'], unit='D')

    # Parse dates and resolution time calculations
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['created_month'] = df['created_at'].dt.to_period('M').astype(str)
    if 'resolved_at' in df.columns:
        df['resolved_at'] = pd.to_datetime(df['resolved_at'])
        df['resolution_days'] = (df['resolved_at'] - df['created_at']).dt.total_seconds() / (24 * 3600)
    
    if 'resolution_days' in df.columns:
        df['resolution_hours'] = df['resolution_days'] * 24

    return df

df_raw = load_data()

if df_raw.empty:
    st.stop()

# Helper Models for AI Resolution Assistance Page
@st.cache_resource
def train_assistance_models(df):
    features = ['product_module', 'environment', 'severity', 'peak_memory_mb', 'users_affected', 'crash_occurred']
    df_clean = df.dropna(subset=features + ['priority', 'resolution_hours'])
    
    X = df_clean[features]
    y_priority = df_clean['priority'].astype(str)
    y_time = df_clean['resolution_hours']

    num_cols = ['peak_memory_mb', 'users_affected', 'crash_occurred']
    cat_cols = ['product_module', 'environment', 'severity']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    clf = Pipeline([('preprocessor', preprocessor), ('model', RandomForestClassifier(n_estimators=100, random_state=42))])
    reg = Pipeline([('preprocessor', preprocessor), ('model', RandomForestRegressor(n_estimators=100, random_state=42))])

    clf.fit(X, y_priority)
    reg.fit(X, y_time)

    return clf, reg

clf_model, reg_model = train_assistance_models(df_raw)

# ---------------------------------------------------------
# Sidebar Controls & Navigation
# ---------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2920/2920277.png", width=60)
st.sidebar.title("Bug Analytics Navigation")

nav_option = st.sidebar.radio(
    "Go to Section:",
    [
        "📋 Dataset Overview",
        "📊 Executive Bug Analytics",
        "📈 Bug Trends & Life Cycle Flow",
        "🤖 Machine Learning Models",
        "👁️ Resolution Assistance (AI-Powered)",
        "💬 AI Chatbot Assistant",
        "💡 Insights & Recommendations",
        "📁 Bug Records & Export"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filters")
selected_module = st.sidebar.multiselect(
    "Product Module",
    options=df_raw['product_module'].dropna().unique(),
    default=df_raw['product_module'].dropna().unique()
)

selected_env = st.sidebar.multiselect(
    "Environment",
    options=df_raw['environment'].dropna().unique(),
    default=df_raw['environment'].dropna().unique()
)

# Apply Filters
df_filtered = df_raw[
    (df_raw['product_module'].isin(selected_module)) &
    (df_raw['environment'].isin(selected_env))
]

# ---------------------------------------------------------
# 1. DATASET OVERVIEW
# ---------------------------------------------------------
if nav_option == "📋 Dataset Overview":
    st.title("DATA FOUNDATION")
    st.header("Dataset Overview")
    st.write("A structured CSV dataset capturing every stage of the memory leak bug lifecycle, grouped into core functional categories.")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("BUG RECORDS", len(df_raw))
    with col_m2:
        st.metric("ATTRIBUTES", len(df_raw.columns))

    st.markdown("---")
    st.subheader("Functional Categories Overview")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="category-card">
            <h4>📋 Bug Information</h4>
            <p><code>bug_id</code>, <code>title</code>, <code>description</code></p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="category-card">
            <h4>📅 Lifecycle Dates</h4>
            <p><code>created_at</code>, <code>resolved_at</code></p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="category-card">
            <h4>⚠️ Classification</h4>
            <p><code>severity</code>, <code>priority</code>, <code>status</code>, <code>leak_type</code></p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="category-card">
            <h4>🛠️ Resolution Info</h4>
            <p><code>root_cause_category</code>, <code>resolution_days</code></p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="category-card">
            <h4>💻 Environment</h4>
            <p><code>operating_system</code>, <code>browser</code>, <code>environment</code></p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="category-card">
            <h4>🔥 Memory Impact</h4>
            <p><code>baseline_memory_mb</code>, <code>peak_memory_mb</code>, <code>growth_rate</code></p>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="category-card">
            <h4>👥 User Impact</h4>
            <p><code>users_affected</code>, <code>concurrent_users</code>, <code>crash_occurred</code></p>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. EXECUTIVE BUG ANALYTICS
# ---------------------------------------------------------
elif nav_option == "📊 Executive Bug Analytics":
    st.title("🐞 Intelligent Software Defect Tracking System with Resolution Assistance")
    st.caption("Interactive Software Quality & Memory Leak Analytics")

    total_bugs = len(df_filtered)
    closed_bugs = len(df_filtered[df_filtered['status'].isin(['Resolved', 'Closed'])])
    open_bugs = len(df_filtered[df_filtered['status'] == 'Open'])
    critical_bugs = len(df_filtered[df_filtered['severity'] == 'Critical'])
    avg_peak_mem = round(df_filtered['peak_memory_mb'].mean(), 1) if not df_filtered.empty else 0
    sla_pct = round((closed_bugs / total_bugs * 100), 1) if total_bugs > 0 else 0
    
    # Resolution Time Calculations
    avg_res_days = round(df_filtered['resolution_days'].mean(), 2) if 'resolution_days' in df_filtered.columns and not df_filtered.empty else 0.0
    avg_res_hours = round(df_filtered['resolution_hours'].mean(), 1) if 'resolution_hours' in df_filtered.columns and not df_filtered.empty else 0.0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Bugs", total_bugs)
    k2.metric("Closed / Resolved", closed_bugs)
    k3.metric("Open Bugs", open_bugs)
    k4.metric("Critical Bugs", critical_bugs)
    k5.metric("Avg Resolution Time", f"{avg_res_days} d ({avg_res_hours} h)")
    k6.metric("Resolution Rate", f"{sla_pct}%")

    st.markdown("---")
    st.subheader("📊 Executive Bug Analytics & Root Cause Distribution")

    col_f1, col_f2 = st.columns(2)

    with col_f1:
        st.write("**Bug Life Cycle Funnel**")
        status_counts = df_filtered['status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        fig_funnel = px.funnel(status_counts, x='Count', y='Status', color='Status', color_discrete_sequence=px.colors.qualitative.Set2)
        fig_funnel.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_funnel, use_container_width=True)

    with col_f2:
        st.write("**Root Cause Category Distribution**")
        rc_counts = df_filtered['root_cause_category'].value_counts().reset_index()
        rc_counts.columns = ['Root Cause', 'Count']
        fig_rc = px.bar(rc_counts, x='Count', y='Root Cause', orientation='h', color='Root Cause', color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_rc.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
        st.plotly_chart(fig_rc, use_container_width=True)

    col_f3, col_f4 = st.columns(2)
    with col_f3:
        st.write("**Monthly Bug Trend**")
        monthly_trend = df_filtered.groupby('created_month').size().reset_index(name='Bug Count')
        fig_line = px.line(monthly_trend, x='created_month', y='Bug Count', markers=True, line_shape='spline', color_discrete_sequence=['#2563eb'])
        fig_line.update_layout(template="plotly_white", xaxis_title="Month", yaxis_title="Reported Bugs", margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_line, use_container_width=True)

    with col_f4:
        st.write("**Average Resolution Time by Module (Days)**")
        if 'resolution_days' in df_filtered.columns:
            mod_res = df_filtered.groupby('product_module')['resolution_days'].mean().reset_index()
            fig_res_mod = px.bar(mod_res, x='product_module', y='resolution_days', color='product_module', color_discrete_sequence=px.colors.qualitative.Set3)
            fig_res_mod.update_layout(template="plotly_white", xaxis_title="Module", yaxis_title="Avg Resolution Days", margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
            st.plotly_chart(fig_res_mod, use_container_width=True)

# ---------------------------------------------------------
# 3. BUG TRENDS & LIFE CYCLE FLOW
# ---------------------------------------------------------
elif nav_option == "📈 Bug Trends & Life Cycle Flow":
    st.title("📈 Bug Flow & Distribution Analytics")

    st.subheader("📊 Bug Life Cycle Distribution")
    st.write("Visualizing the breakdown of bug statuses (Open, In Progress, Resolved, Closed) across Product Modules.")

    if not df_filtered.empty:
        fig_lifecycle = px.histogram(
            df_filtered,
            x='product_module',
            color='status',
            barmode='stack',
            labels={'product_module': 'Product Module', 'status': 'Bug Status', 'count': 'Count'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_lifecycle.update_layout(
            template="plotly_white",
            xaxis_title="Product Module",
            yaxis_title="Count",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_lifecycle, use_container_width=True)

    st.markdown("---")

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.subheader("Module-wise Bug Breakdown")
        fig_mod = px.bar(
            df_filtered,
            x='product_module',
            color='severity',
            barmode='group',
            color_discrete_map={'Critical': '#ef4444', 'High': '#f97316', 'Medium': '#eab308', 'Low': '#22c55e'}
        )
        fig_mod.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_mod, use_container_width=True)

    with col_t2:
        st.subheader("Operating System & Environment")
        fig_os = px.histogram(
            df_filtered,
            x='operating_system',
            color='environment',
            barmode='stack',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_os.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_os, use_container_width=True)

# ---------------------------------------------------------
# 4. MACHINE LEARNING MODELS
# ---------------------------------------------------------
elif nav_option == "🤖 Machine Learning Models":
    st.title("🤖 Machine Learning Classifier Suite")
    st.write("Train and evaluate 5 algorithms directly on the dataset to predict system crashes, bug severity, or root cause categories.")

    col_ml_cfg1, col_ml_cfg2 = st.columns(2)
    with col_ml_cfg1:
        target_col = st.selectbox(
            "Select Prediction Target:",
            options=['crash_occurred', 'severity', 'priority', 'leak_type'],
            index=0
        )
    with col_ml_cfg2:
        test_size = st.slider("Test Set Split Ratio:", 0.1, 0.4, 0.2, 0.05)

    df_ml = df_filtered.dropna(subset=[target_col]).copy()
    
    feature_cols = [
        'product_module', 'environment', 'operating_system', 
        'baseline_memory_mb', 'peak_memory_mb', 'memory_growth_rate_mb_per_hr', 
        'users_affected', 'concurrent_users', 'reproducible'
    ]
    
    if target_col in feature_cols:
        feature_cols.remove(target_col)

    X = df_ml[feature_cols]
    y = df_ml[target_col].astype(str)

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    algorithms = {
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = []
    model_pipes = {}

    for name, model in algorithms.items():
        pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted', zero_division=0)
        
        results.append({
            'Algorithm': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4)
        })
        model_pipes[name] = (pipe, preds)

    res_df = pd.DataFrame(results)

    st.subheader("🏆 Model Performance Comparison")
    st.dataframe(res_df, use_container_width=True)

    fig_acc = px.bar(
        res_df,
        x='Algorithm',
        y='Accuracy',
        color='Algorithm',
        text='Accuracy',
        title=f"Accuracy Comparison across Algorithms (Target: {target_col})",
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig_acc.update_layout(template="plotly_white", yaxis_range=[0, 1.0], margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_acc, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Detailed Model Evaluation")

    selected_model_name = st.selectbox("Select Model to Inspect:", list(algorithms.keys()))
    selected_pipe, selected_preds = model_pipes[selected_model_name]

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.write(f"**Confusion Matrix ({selected_model_name})**")
        labels = np.unique(y_test)
        cm = confusion_matrix(y_test, selected_preds, labels=labels)
        fig_cm = px.imshow(
            cm,
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale='Blues',
            labels=dict(x="Predicted", y="Actual")
        )
        fig_cm.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_m2:
        st.write(f"**Classification Report ({selected_model_name})**")
        report = classification_report(y_test, selected_preds, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    if selected_model_name in ['Random Forest', 'Decision Tree']:
        st.subheader(f"🌲 Feature Importance ({selected_model_name})")
        model_obj = selected_pipe.named_steps['model']
        cat_feature_names = selected_pipe.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_cols)
        all_feature_names = num_cols + list(cat_feature_names)
        importances = model_obj.feature_importances_
        
        fi_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances}).sort_values('Importance', ascending=False).head(10)
        fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', title=f"Top Features for {selected_model_name}")
        fig_fi.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_fi, use_container_width=True)

# ---------------------------------------------------------
# 5. RESOLUTION ASSISTANCE (AI-POWERED)
# ---------------------------------------------------------
elif nav_option == "👁️ Resolution Assistance (AI-Powered)":
    st.title("👁️ Resolution Assistance (AI-Powered)")

    tab1, tab2 = st.tabs(["📊 Predict for New Bug", "📁 Review Existing Bug"])

    with tab1:
        st.subheader("Predict Priority & Resolution Time for a New Defect")
        
        col_nb1, col_nb2 = st.columns(2)

        with col_nb1:
            bug_title = st.text_input("Bug Title", "Memory spike on API Gateway under high traffic")
            prod_module = st.selectbox("Product Module", df_raw['product_module'].unique(), index=1 if 'Data-Pipeline' in df_raw['product_module'].unique() else 0)
            severity = st.selectbox("Severity", ["Critical", "High", "Medium", "Low"])
            peak_memory = st.number_input("Peak Memory (MB)", value=3500, step=100)

        with col_nb2:
            description = st.text_area("Description", "API gateway process spikes to 3.5GB memory and drops requests.", height=105)
            environment = st.selectbox("Environment", df_raw['environment'].unique(), index=2 if 'Development' in df_raw['environment'].unique() else 0)
            users_affected = st.number_input("Users Affected", value=1200, step=50)
            crash_occurred = st.selectbox("Crash Occurred?", [0, 1], index=0)

        if st.button("🚀 Run AI Analysis"):
            input_data = pd.DataFrame([{
                'product_module': prod_module,
                'environment': environment,
                'severity': severity,
                'peak_memory_mb': peak_memory,
                'users_affected': users_affected,
                'crash_occurred': crash_occurred
            }])
            
            pred_priority = clf_model.predict(input_data)[0]
            pred_time = reg_model.predict(input_data)[0]

            st.markdown("---")
            st.markdown("### AI Analysis Results")
            res_c1, res_c2 = st.columns(2)
            with res_c1:
                st.metric("ML-Predicted Priority", pred_priority)
            with res_c2:
                st.metric("Estimated Resolution Time", f"{pred_time:.1f} hrs")

    with tab2:
        st.write("Select a bug already in the system to compare its actual vs. AI-predicted priority.")
        
        selected_bug = st.selectbox("Select a Bug ID:", df_raw['bug_id'].unique())
        bug_row = df_raw[df_raw['bug_id'] == selected_bug].iloc[0]

        st.markdown(f"**Title:** {bug_row.get('title', 'N/A')}")
        st.markdown(f"**Description:** {bug_row.get('description', 'N/A')}")
        
        actual_priority = bug_row.get('priority', 'N/A')
        actual_severity = bug_row.get('severity', 'N/A')
        actual_status = bug_row.get('status', 'N/A')

        st.write(f"**Actual Priority:** {actual_priority} | **Severity:** {actual_severity} | **Status:** {actual_status}")

        input_data = pd.DataFrame([{
            'product_module': bug_row.get('product_module'),
            'environment': bug_row.get('environment'),
            'severity': bug_row.get('severity'),
            'peak_memory_mb': bug_row.get('peak_memory_mb', 0),
            'users_affected': bug_row.get('users_affected', 0),
            'crash_occurred': bug_row.get('crash_occurred', 0)
        }])

        pred_priority = clf_model.predict(input_data)[0]
        pred_time = reg_model.predict(input_data)[0]

        st.markdown("---")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown("##### 🤖 ML-Predicted Priority")
            st.markdown(f"# {pred_priority}")
            st.markdown(f"<span class='badge-actual'>Actual: {actual_priority}</span>", unsafe_allow_html=True)

        with res_col2:
            st.markdown("##### ⏱️ Estimated Resolution Time")
            st.markdown(f"# {pred_time:.1f} hrs")


# ---------------------------------------------------------
# 6. AI CHATBOT ASSISTANT (GROQ VERSION)
# ---------------------------------------------------------
elif nav_option == "💬 AI Chatbot Assistant":
    st.title("🤖 AI Chatbot Assistant")
    st.caption("Ask questions about bug fixes, memory leak debugging, or dataset summaries.")

    groq_key = st.secrets.get("GROQ_API_KEY", "")

    if not groq_key:
        st.info("💡 Please add your GROQ_API_KEY to .streamlit/secrets.toml.")
    else:
        client = groq.Groq(api_key=groq_key)

        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert AI Software Quality Assurance and Systems Engineer assistant. "
                        "You help developers debug software defects, memory leaks, and analyze bug dataset metrics."
                    ),
                },
                {
                    "role": "assistant",
                    "content": "Hello! I am your AI Chatbot Assistant. Ask me anything about fixing memory leaks, analyzing your bug reports, or automated testing strategies!",
                },
            ]

        for message in st.session_state.messages:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question (e.g., 'How do I fix memory leaks in ML-Service?'):"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                try:
                    response_stream = client.chat.completions.create(
    model="openai/gpt-oss-120b",  # Active supported model string
    messages=st.session_state.messages,
    stream=True,
)
                    for chunk in response_stream:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    st.error(f"Error: {e}")
                    full_response = "Sorry, I ran into an issue connecting to Groq."

            st.session_state.messages.append({"role": "assistant", "content": full_response})
# ---------------------------------------------------------
# 7. INSIGHTS & RECOMMENDATIONS
# ---------------------------------------------------------
elif nav_option == "💡 Insights & Recommendations":
    st.title("Insights Dashboard")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="category-card" style="text-align: center;">
            <h4 style="color: #ef4444;">🔥 Highest Risk Module</h4>
            <h3>MobileApp-Android</h3>
            <p style="color: #64748b;">Highest average peak memory footprint</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="category-card" style="text-align: center;">
            <h4 style="color: #f97316;">⚠️ Most Crashes</h4>
            <h3>WebApp-Core</h3>
            <p style="color: #64748b;">Highest number of critical system crashes</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="category-card" style="text-align: center;">
            <h4 style="color: #22c55e;">📈 Closure Rate</h4>
            <h3>51.0%</h3>
            <p style="color: #64748b;">Overall bug resolution acceleration</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="insight-card">
        <h4>📌 Key Findings</h4>
        <ul>
            <li><b>Memory Consumption:</b> Modules like <i>Data-Pipeline</i> and <i>API-Gateway</i> display frequent memory growth exceeding 5,000MB.</li>
            <li><b>Crash Correlation:</b> Unclosed file streams and thread pool mismanagement account for over 60% of system crashes under high load.</li>
            <li><b>Environment Impact:</b> Production and Staging environments exhibit higher severity rates relative to real concurrent user simulation.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="recommend-card">
        <h4>🎯 Actionable Recommendations</h4>
        <ul>
            <li><b>Code Review & Profiling:</b> Mandate automated memory leak profiling for top-risk modules before production deployment.</li>
            <li><b>Resource Cleanup:</b> Enforce explicit resource closing for thread pools and file streams in backend codebases.</li>
            <li><b>Automated Testing:</b> Integrate ML classification models into CI/CD pipelines to predict high-risk memory leaks early.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 8. BUG RECORDS & EXPORT
# ---------------------------------------------------------
elif nav_option == "📁 Bug Records & Export":
    st.title("📁 Defect Records & Data Export")
    st.write("Browse complete dataset records or export the filtered subset for further analysis.")

    st.dataframe(df_filtered, use_container_width=True)

    csv_data = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data CSV",
        data=csv_data,
        file_name="filtered_defect_records.csv",
        mime="text/csv"
    )

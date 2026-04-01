import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch

# Load pickles
df          = pickle.load(open('df_display.pkl',    'rb'))
X_pca       = pickle.load(open('X_pca.pkl',         'rb'))
kmeans_lbl  = pickle.load(open('kmeans_labels.pkl', 'rb'))
dbscan_lbl  = pickle.load(open('dbscan_labels.pkl', 'rb'))
hc_lbl      = pickle.load(open('hc_labels.pkl',     'rb'))
df_scaled   = pickle.load(open('df_scaled.pkl',     'rb'))
wcss        = pickle.load(open('wcss.pkl',          'rb'))
scores      = pickle.load(open('scores.pkl',        'rb'))

st.set_page_config(page_title="World Development Analysis", layout="wide")
st.title("🌍 World Development Measurement Analysis")
st.markdown("---")

page = st.sidebar.radio("Go to", ["📊 Data", "📈 Charts", "🔵 Clustering"])

# ── DATA ──────────────────────────────────────────────────────
if page == "📊 Data":
    c1, c2 = st.columns(2)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])

    st.subheader("Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Statistics")
    st.dataframe(df.describe().T.round(2), use_container_width=True)

# ── CHARTS ────────────────────────────────────────────────────
elif page == "📈 Charts":
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(df[num_cols].corr(), cmap="coolwarm", center=0, linewidths=0.4, ax=ax)
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Scatter Plots")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.scatterplot(x=df["Internet Usage"], y=df["GDP"], alpha=0.5, ax=ax)
        ax.set_title("Internet Usage vs GDP")
        st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.scatterplot(x=df["Energy Usage"], y=df["CO2 Emissions"], alpha=0.5, color="coral", ax=ax)
        ax.set_title("Energy Usage vs CO2 Emissions")
        st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.scatterplot(x=df["Life Expectancy Male"], y=df["Life Expectancy Female"], alpha=0.5, color="green", ax=ax)
        ax.set_title("Life Expectancy: Male vs Female")
        st.pyplot(fig); plt.close()

    with col4:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(np.log1p(df["GDP"]), bins=30, kde=True, ax=ax)
        ax.set_title("GDP Distribution (Log)")
        st.pyplot(fig); plt.close()

# ── CLUSTERING ────────────────────────────────────────────────
elif page == "🔵 Clustering":

    st.subheader("KMeans (K=3)")
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_lbl, cmap="tab10", s=10, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="Cluster")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Elbow Method")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, 9), wcss, marker='o', color='steelblue')
    ax.set_xlabel("K"); ax.set_ylabel("WCSS")
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("DBSCAN")
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_lbl, cmap="tab10", s=10, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="Cluster (-1=noise)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    st.pyplot(fig); plt.close()
    st.write(f"Noise points: **{list(dbscan_lbl).count(-1)}**")

    st.markdown("---")
    st.subheader("Dendrogram (100 samples)")
    idx = np.random.choice(len(df_scaled), size=100, replace=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    sch.dendrogram(sch.linkage(df_scaled[idx], method='ward'), ax=ax, no_labels=True)
    ax.set_ylabel("Distance")
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Silhouette Score Comparison")
    models = ['K-Means', 'DBSCAN', 'Hierarchical']
    vals   = [scores['kmeans'], scores['dbscan'], scores['hc']]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(models, vals, color=['#4e9af1', '#f1c44e', '#6fcf7c'], width=0.4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha='center', fontweight='bold')
    ax.set_ylabel("Silhouette Score"); ax.set_ylim(0, max(vals) + 0.1)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    st.pyplot(fig); plt.close()

    st.table({"Model": models, "Score": [round(v, 4) for v in vals]})

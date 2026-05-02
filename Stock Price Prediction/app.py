import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

:root {
    --bg: #0a0f1e;
    --panel: #111827;
    --border: #1e2d45;
    --accent: #00d4ff;
    --accent2: #7c3aed;
    --green: #00e676;
    --red: #ff1744;
    --text: #e2e8f0;
    --muted: #64748b;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--panel);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Header */
.hero {
    text-align: center;
    padding: 2rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hero h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -1px;
}
.hero p {
    color: var(--muted);
    font-size: 0.9rem;
    margin: 0.4rem 0 0;
    font-family: 'Space Mono', monospace;
}

/* Metric cards */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    flex: 1;
    min-width: 140px;
}
.metric-card .label {
    font-size: 0.72rem;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.4rem;
}
.metric-card .value {
    font-size: 1.6rem;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
}
.metric-card .value.green { color: var(--green); }
.metric-card .value.red   { color: var(--red);   }
.metric-card .value.blue  { color: var(--accent); }
.metric-card .value.purple{ color: var(--accent2);}

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    color: var(--muted) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff22, #7c3aed22);
    border: 1px solid var(--accent);
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background: var(--accent);
    color: var(--bg);
}

/* Selectbox / slider */
.stSelectbox label, .stSlider label, .stRadio label {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Info box */
.info-box {
    background: #00d4ff11;
    border: 1px solid #00d4ff33;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    color: var(--text);
    margin: 1rem 0;
}
.warn-box {
    background: #ff174411;
    border: 1px solid #ff174433;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    color: var(--text);
    margin: 1rem 0;
}

/* Progress text */
.stProgress > div > div { background: var(--accent) !important; }

/* Matplotlib figures */
.stPlotlyChart, .stPyplot { border-radius: 12px; overflow: hidden; }

/* Divider */
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>STOCK PRICE PREDICTOR</h1>
    <p>LSTM Neural Network · S&P 500 · 2013–2018</p>
</div>
""", unsafe_allow_html=True)

# ─── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("all_stocks_5yr.csv", on_bad_lines='skip')
    df['date'] = pd.to_datetime(df['date'])
    return df

data = load_data()
all_companies = sorted(data['Name'].unique())

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ SETTINGS")
    st.markdown("---")

    company = st.selectbox("🏢 Select Company", all_companies,
                           index=all_companies.index('AAPL'))

    st.markdown("---")
    st.markdown("### 🧠 MODEL CONFIG")

    epochs = st.slider("Training Epochs", 5, 30, 10, step=5)
    lookback = st.slider("Lookback Window (days)", 30, 90, 60, step=10)
    train_split = st.slider("Train / Test Split", 0.80, 0.95, 0.95, step=0.05)

    st.markdown("---")
    st.markdown("### 📊 CHART OPTIONS")
    show_volume = st.checkbox("Show Volume", value=True)
    show_ma = st.checkbox("Show Moving Averages", value=True)

    st.markdown("---")
    run_model = st.button("🚀 TRAIN & PREDICT")

    st.markdown("---")
    st.markdown("""
<div style='font-family:Space Mono,monospace; font-size:0.65rem; color:#64748b; line-height:1.6'>
DISCLAIMER<br>
This tool is for educational purposes only. Not financial advice.
</div>
""", unsafe_allow_html=True)

# ─── Filter Company Data ────────────────────────────────────────────────────────
company_data = data[data['Name'] == company].sort_values('date').reset_index(drop=True)

# ─── Compute Quick Stats ───────────────────────────────────────────────────────
latest_close  = company_data['close'].iloc[-1]
prev_close    = company_data['close'].iloc[-2]
price_change  = latest_close - prev_close
pct_change    = (price_change / prev_close) * 100
avg_volume    = company_data['volume'].mean()
price_range   = company_data['high'].max() - company_data['low'].min()

change_class  = "green" if price_change >= 0 else "red"
change_symbol = "▲" if price_change >= 0 else "▼"

# ─── Metrics Row ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="metric-row">
  <div class="metric-card">
    <div class="label">Latest Close</div>
    <div class="value blue">${latest_close:.2f}</div>
  </div>
  <div class="metric-card">
    <div class="label">1-Day Change</div>
    <div class="value {change_class}">{change_symbol} {abs(pct_change):.2f}%</div>
  </div>
  <div class="metric-card">
    <div class="label">Avg Daily Volume</div>
    <div class="value purple">{avg_volume/1e6:.1f}M</div>
  </div>
  <div class="metric-card">
    <div class="label">5yr Price Range</div>
    <div class="value blue">${price_range:.2f}</div>
  </div>
  <div class="metric-card">
    <div class="label">Total Records</div>
    <div class="value">{len(company_data):,}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Price History", "🤖 LSTM Prediction", "📊 Data Explorer"])

# ════════════════════════════════════════════════════════════════════
# TAB 1 – Price History
# ════════════════════════════════════════════════════════════════════
with tab1:
    fig, axes = plt.subplots(2 if show_volume else 1, 1,
                             figsize=(14, 8 if show_volume else 5),
                             facecolor='#0a0f1e')
    ax = axes[0] if show_volume else axes

    ax.set_facecolor('#111827')
    ax.plot(company_data['date'], company_data['close'],
            color='#00d4ff', linewidth=1.5, label='Close', zorder=3)
    ax.plot(company_data['date'], company_data['open'],
            color='#7c3aed', linewidth=1, alpha=0.6, label='Open', zorder=2)

    if show_ma:
        ma20  = company_data['close'].rolling(20).mean()
        ma50  = company_data['close'].rolling(50).mean()
        ma200 = company_data['close'].rolling(200).mean()
        ax.plot(company_data['date'], ma20,  color='#f59e0b', linewidth=1, linestyle='--', alpha=0.7, label='MA 20')
        ax.plot(company_data['date'], ma50,  color='#10b981', linewidth=1, linestyle='--', alpha=0.7, label='MA 50')
        ax.plot(company_data['date'], ma200, color='#ef4444', linewidth=1, linestyle='--', alpha=0.7, label='MA 200')

    ax.fill_between(company_data['date'], company_data['low'], company_data['high'],
                    alpha=0.07, color='#00d4ff')
    ax.set_title(f'{company} — Price History', color='#e2e8f0', fontsize=13,
                 fontfamily='monospace', pad=12)
    ax.set_ylabel('Price (USD)', color='#64748b', fontsize=9)
    ax.tick_params(colors='#64748b', labelsize=8)
    ax.spines[:].set_color('#1e2d45')
    ax.legend(framealpha=0, labelcolor='#e2e8f0', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    ax.grid(color='#1e2d45', linewidth=0.5, linestyle='--')

    if show_volume:
        ax2 = axes[1]
        ax2.set_facecolor('#111827')
        colors = ['#00e676' if c >= o else '#ff1744'
                  for c, o in zip(company_data['close'], company_data['open'])]
        ax2.bar(company_data['date'], company_data['volume'],
                color=colors, alpha=0.7, width=1.5)
        ax2.set_ylabel('Volume', color='#64748b', fontsize=9)
        ax2.set_title('Volume', color='#e2e8f0', fontsize=10, fontfamily='monospace')
        ax2.tick_params(colors='#64748b', labelsize=8)
        ax2.spines[:].set_color('#1e2d45')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')
        ax2.grid(color='#1e2d45', linewidth=0.5, linestyle='--')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════════════════════════════
# TAB 2 – LSTM Prediction
# ════════════════════════════════════════════════════════════════════
with tab2:
    if not run_model:
        st.markdown("""
<div class="info-box">
👈 Configure your model in the sidebar, then click <strong>TRAIN & PREDICT</strong> to start.
<br><br>
The LSTM model will train on historical closing prices and predict future values using a sliding window approach.
</div>
""", unsafe_allow_html=True)

        # Show architecture diagram
        st.markdown("#### 🧠 Model Architecture")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
<div class="metric-card">
<div class="label">Layer 1</div>
<div class="value blue" style="font-size:1rem">LSTM (64 units)</div>
<div style="color:#64748b;font-size:0.75rem;margin-top:0.3rem">return_sequences=True</div>
</div>
""", unsafe_allow_html=True)
            st.markdown(f"""
<div class="metric-card">
<div class="label">Layer 3</div>
<div class="value blue" style="font-size:1rem">Dense (32 units)</div>
<div style="color:#64748b;font-size:0.75rem;margin-top:0.3rem">ReLU Activation</div>
</div>
""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
<div class="metric-card">
<div class="label">Layer 2</div>
<div class="value purple" style="font-size:1rem">LSTM (64 units)</div>
<div style="color:#64748b;font-size:0.75rem;margin-top:0.3rem">return_sequences=False</div>
</div>
""", unsafe_allow_html=True)
            st.markdown(f"""
<div class="metric-card">
<div class="label">Layer 4</div>
<div class="value purple" style="font-size:1rem">Dense (1 unit)</div>
<div style="color:#64748b;font-size:0.75rem;margin-top:0.3rem">Output — Close Price</div>
</div>
""", unsafe_allow_html=True)

    else:
        # ─── Train Model ───────────────────────────────────────────────────────
        import tensorflow as tf
        from tensorflow import keras
        from sklearn.preprocessing import MinMaxScaler

        st.markdown(f"#### 🚀 Training LSTM on **{company}** — {epochs} epochs, {lookback}-day window")

        close_data = company_data[['close']].values
        training_len = int(np.ceil(len(close_data) * train_split))

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(close_data)

        train = scaled[:training_len]
        x_train, y_train = [], []
        for i in range(lookback, len(train)):
            x_train.append(train[i-lookback:i, 0])
            y_train.append(train[i, 0])
        x_train = np.array(x_train).reshape(-1, lookback, 1)
        y_train = np.array(y_train)

        # Build model
        model = keras.models.Sequential([
            keras.layers.LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
            keras.layers.LSTM(64),
            keras.layers.Dense(32),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train with progress
        progress_bar = st.progress(0)
        status      = st.empty()
        loss_vals   = []

        class StreamlitCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                loss_vals.append(logs['loss'])
                pct = int((epoch + 1) / epochs * 100)
                progress_bar.progress(pct)
                status.markdown(f"<span style='font-family:Space Mono,monospace;color:#00d4ff;font-size:0.8rem'>"
                                f"Epoch {epoch+1}/{epochs} · Loss: {logs['loss']:.6f}</span>",
                                unsafe_allow_html=True)

        model.fit(x_train, y_train, epochs=epochs,
                  batch_size=32, verbose=0,
                  callbacks=[StreamlitCallback()])

        status.markdown("<span style='color:#00e676;font-family:Space Mono,monospace;font-size:0.8rem'>✓ Training complete!</span>",
                        unsafe_allow_html=True)

        # ─── Predict ───────────────────────────────────────────────────────────
        test_input = scaled[training_len - lookback:, :]
        x_test = []
        for i in range(lookback, len(test_input)):
            x_test.append(test_input[i-lookback:i, 0])
        x_test = np.array(x_test).reshape(-1, lookback, 1)

        predictions = scaler.inverse_transform(model.predict(x_test))
        y_test      = close_data[training_len:]

        mse  = np.mean((predictions - y_test) ** 2)
        rmse = np.sqrt(mse)
        mae  = np.mean(np.abs(predictions - y_test))
        r2   = 1 - np.sum((y_test - predictions)**2) / np.sum((y_test - np.mean(y_test))**2)

        # ─── Error Metrics ─────────────────────────────────────────────────────
        st.markdown(f"""
<div class="metric-row">
  <div class="metric-card">
    <div class="label">RMSE</div>
    <div class="value {'green' if rmse<5 else 'red'}">${rmse:.2f}</div>
  </div>
  <div class="metric-card">
    <div class="label">MAE</div>
    <div class="value blue">${mae:.2f}</div>
  </div>
  <div class="metric-card">
    <div class="label">R² Score</div>
    <div class="value {'green' if r2>0.9 else 'blue'}">{r2:.4f}</div>
  </div>
  <div class="metric-card">
    <div class="label">MSE</div>
    <div class="value purple">{mse:.2f}</div>
  </div>
</div>
""", unsafe_allow_html=True)

        # ─── Prediction Chart ──────────────────────────────────────────────────
        train_df = company_data.iloc[:training_len]
        test_df  = company_data.iloc[training_len:].copy()
        test_df['Predictions'] = predictions

        fig2, axes2 = plt.subplots(2, 1, figsize=(14, 10), facecolor='#0a0f1e')

        # Top: full prediction chart
        ax = axes2[0]
        ax.set_facecolor('#111827')
        ax.plot(train_df['date'], train_df['close'],
                color='#64748b', linewidth=1.2, label='Training Data', alpha=0.8)
        ax.plot(test_df['date'], test_df['close'],
                color='#00d4ff', linewidth=1.5, label='Actual Price')
        ax.plot(test_df['date'], test_df['Predictions'],
                color='#00e676', linewidth=1.5, linestyle='--', label='Predicted Price')
        ax.axvline(x=train_df['date'].iloc[-1], color='#7c3aed',
                   linestyle=':', linewidth=1.5, alpha=0.8, label='Train/Test Split')
        ax.fill_between(test_df['date'], test_df['close'], test_df['Predictions'],
                        alpha=0.1, color='#00d4ff')
        ax.set_title(f'{company} — LSTM Price Prediction (Train/Test)',
                     color='#e2e8f0', fontsize=13, fontfamily='monospace', pad=12)
        ax.set_ylabel('Price (USD)', color='#64748b', fontsize=9)
        ax.tick_params(colors='#64748b', labelsize=8)
        ax.spines[:].set_color('#1e2d45')
        ax.legend(framealpha=0, labelcolor='#e2e8f0', fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        ax.grid(color='#1e2d45', linewidth=0.5, linestyle='--')

        # Bottom: training loss
        ax3 = axes2[1]
        ax3.set_facecolor('#111827')
        ax3.plot(range(1, len(loss_vals)+1), loss_vals,
                 color='#7c3aed', linewidth=2, marker='o', markersize=4)
        ax3.fill_between(range(1, len(loss_vals)+1), loss_vals,
                         alpha=0.2, color='#7c3aed')
        ax3.set_title('Training Loss (MSE)', color='#e2e8f0',
                      fontsize=11, fontfamily='monospace', pad=10)
        ax3.set_xlabel('Epoch', color='#64748b', fontsize=9)
        ax3.set_ylabel('Loss', color='#64748b', fontsize=9)
        ax3.tick_params(colors='#64748b', labelsize=8)
        ax3.spines[:].set_color('#1e2d45')
        ax3.grid(color='#1e2d45', linewidth=0.5, linestyle='--')
        ax3.set_xticks(range(1, len(loss_vals)+1))

        plt.tight_layout(pad=2)
        st.pyplot(fig2)
        plt.close()

        # ─── Test Period Detail ────────────────────────────────────────────────
        st.markdown("#### 🔍 Test Period — Actual vs Predicted")
        fig3, ax4 = plt.subplots(figsize=(14, 4), facecolor='#0a0f1e')
        ax4.set_facecolor('#111827')
        ax4.plot(test_df['date'], test_df['close'],
                 color='#00d4ff', linewidth=1.8, label='Actual')
        ax4.plot(test_df['date'], test_df['Predictions'],
                 color='#00e676', linewidth=1.8, linestyle='--', label='Predicted')
        error = test_df['close'].values - predictions.flatten()
        ax4.fill_between(test_df['date'],
                         test_df['close'], test_df['Predictions'],
                         where=(error > 0), alpha=0.15, color='#00d4ff', label='Overestimate')
        ax4.fill_between(test_df['date'],
                         test_df['close'], test_df['Predictions'],
                         where=(error < 0), alpha=0.15, color='#ff1744', label='Underestimate')
        ax4.set_ylabel('Price (USD)', color='#64748b', fontsize=9)
        ax4.tick_params(colors='#64748b', labelsize=8)
        ax4.spines[:].set_color('#1e2d45')
        ax4.legend(framealpha=0, labelcolor='#e2e8f0', fontsize=8)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax4.grid(color='#1e2d45', linewidth=0.5, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

# ════════════════════════════════════════════════════════════════════
# TAB 3 – Data Explorer
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### 📊 Multi-Stock Comparison")

    selected = st.multiselect("Choose companies to compare",
                              all_companies,
                              default=['AAPL', 'GOOGL', 'AMZN', 'NVDA', 'MSFT'])

    if selected:
        fig4, ax5 = plt.subplots(figsize=(14, 6), facecolor='#0a0f1e')
        ax5.set_facecolor('#111827')
        palette = ['#00d4ff', '#7c3aed', '#00e676', '#f59e0b', '#ef4444',
                   '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#a855f7']

        for i, comp in enumerate(selected):
            cdf = data[data['Name'] == comp].sort_values('date')
            # Normalize to 100 at start
            norm = cdf['close'] / cdf['close'].iloc[0] * 100
            ax5.plot(cdf['date'], norm, label=comp,
                     color=palette[i % len(palette)], linewidth=1.6)

        ax5.set_title('Normalized Price Performance (Base = 100)',
                      color='#e2e8f0', fontsize=13, fontfamily='monospace', pad=12)
        ax5.set_ylabel('Normalized Price', color='#64748b', fontsize=9)
        ax5.tick_params(colors='#64748b', labelsize=8)
        ax5.spines[:].set_color('#1e2d45')
        ax5.legend(framealpha=0, labelcolor='#e2e8f0', fontsize=9, ncol=3)
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax5.get_xticklabels(), rotation=30, ha='right')
        ax5.grid(color='#1e2d45', linewidth=0.5, linestyle='--')
        ax5.axhline(y=100, color='#1e2d45', linewidth=1)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    st.markdown("---")
    st.markdown(f"#### 🗂️ Raw Data — {company}")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        n_rows = st.slider("Rows to show", 10, 100, 20, step=10)
    with col_b:
        sort_order = st.radio("Sort", ["Latest first", "Oldest first"])

    disp = company_data.sort_values('date', ascending=(sort_order == "Oldest first")).head(n_rows)
    disp['date'] = disp['date'].dt.strftime('%Y-%m-%d')
    st.dataframe(disp.style
                 .format({'open': '${:.2f}', 'high': '${:.2f}',
                          'low': '${:.2f}', 'close': '${:.2f}',
                          'volume': '{:,.0f}'}),
                 use_container_width=True)

    # Summary stats
    st.markdown(f"#### 📋 Summary Statistics — {company}")
    stats = company_data[['open', 'high', 'low', 'close', 'volume']].describe().round(2)
    st.dataframe(stats, use_container_width=True)

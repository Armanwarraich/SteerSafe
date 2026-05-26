"""
SteerSafe Dashboard — reads from session_log.csv
No webcam here — run detection.py separately for live detection
Run: streamlit run app/streamlit_app.py
"""
import streamlit as st
import pandas as pd
import os
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(BASE_DIR, 'logs', 'session_log.csv')

st.set_page_config(
    page_title="SteerSafe Dashboard",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 SteerSafe — Session Dashboard")
st.markdown("Run `python src/detection.py` in a separate terminal for live detection.")
st.markdown("---")

# Sidebar
refresh = st.sidebar.slider("Refresh every N seconds", 1, 10, 3)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** ResNet-152 (ONNX)")
st.sidebar.markdown("**Accuracy:** 99.10%")
st.sidebar.markdown("**F1 Score:** 99.09%")
st.sidebar.markdown("**Dataset:** MRL Eye (84,898 images)")

placeholder = st.empty()

while True:
    with placeholder.container():

        if not os.path.exists(LOG_PATH):
            st.warning("No session log found. Start detection.py first.")

        else:
            try:
                df = pd.read_csv(LOG_PATH)
            except Exception:
                st.info("Reading log file...")
                time.sleep(refresh)
                continue

            if df.empty:
                st.info("Detection running — no events logged yet.")

            else:
                # Top metrics
                col1, col2, col3, col4 = st.columns(4)

                total_alerts = len(df[df['event'] == 'DROWSY_ALERT'])
                total_yawns  = len(df[df['event'] == 'YAWN'])
                avg_ear      = round(df['ear'].mean(), 4)
                min_ear      = round(df['ear'].min(), 4)

                col1.metric("🚨 Total Alerts", total_alerts)
                col2.metric("😮 Total Yawns",  total_yawns)
                col3.metric("👁 Avg EAR",      avg_ear)
                col4.metric("📉 Min EAR",      min_ear)

                st.markdown("---")

                # EAR chart
                st.subheader("EAR Score Over Session")
                st.line_chart(df[['ear']].rename(columns={'ear': 'EAR Score'}))
                st.markdown("---")

                # Event log + summary
                log_col, summary_col = st.columns(2)

                with log_col:
                    st.subheader("Event Log")
                    alert_df = df[df['event'].isin(['DROWSY_ALERT', 'YAWN'])]
                    if not alert_df.empty:
                        st.dataframe(
                            alert_df.tail(15),
                            width='stretch',
                            hide_index=True
                        )
                    else:
                        st.success("No drowsy events this session ✅")

                with summary_col:
                    st.subheader("Session Summary")
                    drowsy_pct = round((total_alerts / max(len(df), 1)) * 100, 2)
                    st.markdown(f"**Total frames logged:** {len(df)}")
                    st.markdown(f"**Drowsy alerts fired:** {total_alerts}")
                    st.markdown(f"**Yawns detected:** {total_yawns}")
                    st.markdown(f"**Average EAR:** {avg_ear}")
                    st.markdown(f"**Lowest EAR recorded:** {min_ear}")
                    st.markdown(f"**Alert rate:** {drowsy_pct}%")

                    if total_alerts == 0:
                        st.success("Driver was alert throughout session ✅")
                    elif total_alerts < 3:
                        st.warning("Minor drowsiness detected ⚠️")
                    else:
                        st.error("High drowsiness — unsafe driving session 🚨")

    time.sleep(refresh)
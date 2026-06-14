import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Predicting Steel Column Response at Elevated Temperatures", layout="wide")

page_bg_color = """
<style>
[data-testid="stAppViewContainer"] { background-color: #F4F8FA !important; }
[data-testid="stHeader"] { background-color: rgba(0,0,0,0) !important; }
h1, h2, h3, h4, h5, h6, p, span, div, label { color: #2C3E50 !important; }

<style>
/* Nút PLOTTING RESULTS */
div.stButton > button:first-child {
    background-color: #2E86C1 !important;
    color: white !important; /* Chỉnh màu cho nút cha */
    border: none;
    border-radius: 6px;
    font-weight: 800 !important;
    font-size: 18px !important;
}
/* Ép màu cho chữ bên trong nút (phần tử con) */
div.stButton > button:first-child p {
    color: white !important;
}

/* Nút DOWNLOAD */
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #3498db, #2980b9) !important;
    color: white !important; /* Chỉnh màu cho nút cha */
    border: none !important;
    border-radius: 6px;
    font-weight: 700 !important;
    font-size: 16px !important;
}
/* Ép màu cho chữ bên trong nút (phần tử con) */
[data-testid="stDownloadButton"] button p {
    color: white !important;
}
</style>
"""
st.markdown(page_bg_color, unsafe_allow_html=True)

@st.cache_resource
def load_models():
    loaded_model_data = joblib.load("model.pkl")
    scalerX = joblib.load("scaler_X.pkl")
    scalerY = joblib.load("scaler_y.pkl")
    
    if isinstance(loaded_model_data, dict):
        if "model" in loaded_model_data:
            model = loaded_model_data["model"]
        elif "final_model" in loaded_model_data:
            model = loaded_model_data["model"]
        else:
            model = list(loaded_model_data.values())[0]
    else:
        model = loaded_model_data
        
    return model, scalerX, scalerY

model, scalerX, scalerY = load_models()

def predict_xgboost_output(input_array):
    input_scaled = scalerX.transform(input_array)
    output_scaled = model.predict(input_scaled)
    output_scaled = output_scaled.reshape(-1, 1) 
    output_log = scalerY.inverse_transform(output_scaled)
    output_original = np.expm1(output_log)
    return output_original.flatten()

st.title("Load-Displacement Curve Prediction")
st.markdown("Using Hybrid XGBoost Model to predict the structural capacity of steel columns.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Geometric Parameters (mm)")
    h_in = st.number_input("Section Height ($h$)", value=600.0, step=10.0)
    bf_in = st.number_input("Flange Width ($b_f$)", value=300.0, step=10.0)
    tw_in = st.number_input("Web Thickness ($t_w$)", value=15.5, step=0.5)
    tf_in = st.number_input("Flange Thickness ($t_f$)", value=30.0, step=1.0)
    L_in = st.number_input("Column Length ($L$)", value=5000.0, step=100.0)

with col2:
    st.subheader("2. Material & Temperature")
    fy_in = st.number_input("Steel Yield Strength ($f_y$) [MPa]", value=235.0, step=5.0)
    T_in = st.number_input("Temperature ($T$) [°C]", value=20.0, step=10.0)

with col3:
    st.subheader("3. Chart Settings")
    step_size = st.number_input("Displacement Step (mm)", value=0.2, step=0.1)

st.markdown("---")

if st.button("PLOTTING RESULTS", type="primary", use_container_width=True):
    with st.spinner('Calculating...'):
        max_di = 15.0
        num_steps = int(max_di / step_size) + 1
        di_range = np.linspace(0, max_di, num_steps)
        di_range = np.round(di_range, 2)
        
        X_curve = np.zeros((num_steps, 8))
        X_curve[:, 0] = h_in
        X_curve[:, 1] = bf_in
        X_curve[:, 2] = tw_in
        X_curve[:, 3] = tf_in
        X_curve[:, 4] = fy_in
        X_curve[:, 5] = T_in
        X_curve[:, 6] = L_in
        X_curve[:, 7] = di_range
        
        pi_pred = predict_xgboost_output(X_curve)
        
        res_col1, res_col2 = st.columns([2, 1])
        
        with res_col1:
            st.markdown("### Load-Displacement Curve")
            
            import matplotlib.font_manager as fm
            from matplotlib.ticker import MultipleLocator
            
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['Times New Roman']
            plt.rcParams['mathtext.fontset'] = 'stix' 
            
            fig, ax = plt.subplots(figsize=(7, 5))
            
            ax.plot(di_range, pi_pred, color='Red', linestyle='-', linewidth=1.5)
            
            max_force = np.max(pi_pred)
            ax.set_xlim(0, max_di)
            ax.set_ylim(0, max_force * 1.05)
            ax.set_xlabel('Axial Displacement, $d_i$ (mm)', fontsize=12)
            ax.set_ylabel('Axial Force, $P_i$ (kN)', fontsize=12)
            ax.xaxis.set_major_locator(MultipleLocator(3))
            ax.tick_params(axis='both', which='major', labelsize=11, direction='in', length=5, width=1, top=False, right=False)
            ax.grid(True, which='major', linestyle='--', color='gray', alpha=0.3)
            ax.legend(loc='best', frameon=False, fontsize=11)

            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
                spine.set_color('black')                
            fig.tight_layout()
            st.pyplot(fig)
                        
        with res_col2:
            st.markdown("### Detailed Data")
            df_curve = pd.DataFrame({
                'Displacement (mm)': di_range,
                'Force (kN)': np.round(pi_pred, 2)
            })
            st.dataframe(df_curve, height=400)
            
            csv = df_curve.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Data (.csv)",
                data=csv,
                file_name=f'Load_Displacement_Curve.csv',
                mime='text/csv',
                use_container_width=True
            )
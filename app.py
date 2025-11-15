# app.py - PHIÊN BẢN CHUYÊN NGHIỆP NHẤT - 3 MÔ HÌNH CỐ ĐỊNH, CHỌN LÀ CHẠY NGAY!
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ====================== CẤU HÌNH TRANG ======================
st.set_page_config(page_title="Phát hiện gian lận giao dịch", page_icon="shield", layout="centered")

# ====================== CSS SIÊU ĐẸP ======================
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); padding: 2rem; border-radius: 20px;}
    .header-title {font-size: 3.3rem; font-weight: 900; text-align: center; color: #1b5e20; text-shadow: 2px 2px 8px rgba(0,0,0,0.2);}
    .school {font-size: 2.4rem; font-weight: 800; text-align: center; color: #1b5e20;}
    .model-box {background-color: #e8f5e8; padding: 1rem; border-radius: 15px; border: 3px solid #81c784; text-align: center;}
    .result-box {padding: 2.5rem; border-radius: 20px; text-align: center; font-size: 2.4rem; font-weight: bold; margin: 2rem 0; box-shadow: 0 10px 25px rgba(0,0,0,0.2);}
    .fraud {background-color: #ffebee; color: #c62828; border: 6px solid #e57373;}
    .safe {background-color: #e8f5e8; color: #2e7d32; border: 6px solid #81c784;}
    .footer {text-align: center; padding: 2rem; color: #1b5e20; margin-top: 4rem; border-top: 3px solid #81c784;}
</style>
""", unsafe_allow_html=True)

# ====================== HEADER ======================
st.markdown("<h1 class='school'>TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN VÀ TRUYỀN THÔNG</h1>", unsafe_allow_html=True)
st.markdown("<h1 class='header-title'>HỆ THỐNG PHÁT HIỆN GIAN LẬN GIAO DỊCH NGÂN HÀNG</h1>", unsafe_allow_html=True)
st.markdown("---")

# ====================== 3 MÔ HÌNH CỐ ĐỊNH ======================
@st.cache_resource
def load_models():
    with st.spinner("Đang tải 3 mô hình AI... (chỉ tải 1 lần)"):
        model_xgb = joblib.load("XGBoost.pkl")        # Đặt đúng tên file bạn up lên
        model_rf = joblib.load("RandomForest.pkl")    # Đặt đúng tên file
        model_lr = joblib.load("Logistic.pkl")  # hoặc tên file LogReg của bạn
    return {"XGBoost (Mạnh nhất)": model_xgb, "Random Forest": model_rf, "Logistic Regression": model_lr}

models_dict = load_models()
st.success("ĐÃ TẢI THÀNH CÔNG 3 MÔ HÌNH AI!")

# Người dùng chọn mô hình
selected_model_name = st.selectbox(
    "Chọn mô hình dự đoán",
    options=list(models_dict.keys()),
    format_func=lambda x: f"{x} - Độ chính xác cao"
)

model = models_dict[selected_model_name]
st.markdown(f"<div class='model-box'>Đang sử dụng: <strong>{selected_model_name}</strong></div>", unsafe_allow_html=True)

st.markdown("---")

# ====================== NHẬP DỮ LIỆU ======================
st.subheader("Nhập thông tin giao dịch cần kiểm tra")
col1, col2 = st.columns(2)

with col1:
    step = st.number_input("Step (giờ)", min_value=1, value=1, step=1)
    type_trans = st.selectbox("Loại giao dịch", ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"])
    amount = st.number_input("Số tiền giao dịch", min_value=0.0, value=100000.0, format="%.2f")
    oldbalanceOrg = st.number_input("Số dư NGƯỜI GỬI trước GD", min_value=0.0, value=500000.0, format="%.2f")
    newbalanceOrig = st.number_input("Số dư NGƯỜI GỬI sau GD", min_value=0.0, value=400000.0, format="%.2f")

with col2:
    oldbalanceDest = st.number_input("Số dư NGƯỜI NHẬN trước GD", min_value=0.0, value=0.0, format="%.2f")
    newbalanceDest = st.number_input("Số dư NGƯỜI NHẬN sau GD", min_value=0.0, value=100000.0, format="%.2f")

# ====================== DỰ ĐOÁN ======================
if st.button("PHÁT HIỆN GIAN LẬN NGAY", type="primary", use_container_width=True):
    with st.spinner("Đang phân tích giao dịch bằng AI..."):
        balanceDiffOrig = oldbalanceOrg - newbalanceOrig
        balanceDiffDest = oldbalanceDest - newbalanceDest

        # Mapping Label Encoding (nếu cần)
        type_mapping = {"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4}

        input_raw = {
            'step': step, 'type': type_trans, 'amount': amount,
            'oldbalanceOrg': oldbalanceOrg, 'newbalanceOrig': newbalanceOrig,
            'oldbalanceDest': oldbalanceDest, 'newbalanceDest': newbalanceDest,
            'balanceDiffOrig': balanceDiffOrig, 'balanceDiffDest': balanceDiffDest
        }
        input_df = pd.DataFrame([input_raw])

        # Thử dự đoán trực tiếp
        try:
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]
        except:
            # Nếu lỗi → thử Label Encoding
            input_df['type'] = type_mapping[type_trans]
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]

        prediction = int(pred)

    # ====================== KẾT QUẢ SIÊU ĐẸP ======================
    st.markdown("## KẾT QUẢ DỰ ĐOÁN")

    fig, ax = plt.subplots(figsize=(8,5))
    ax.pie([1-prob, prob], labels=['Hợp lệ', 'Gian lận'], autopct='%1.2f%%',
           colors=['#81C784', '#FF5252'], startangle=90, textprops={'fontsize': 16, 'fontweight': 'bold'})
    ax.set_title(f"Kết quả từ mô hình: {selected_model_name}", fontsize=16, fontweight='bold', color='#1b5e20')
    st.pyplot(fig)

    if prediction == 1:
        st.markdown(f"<div class='result-box fraud'>CẢNH BÁO: GIAO DỊCH CÓ DẤU HIỆU GIAN LẬN!<br><h2>Xác suất: {prob:.2%}</h2></div>", unsafe_allow_html=True)
        st.error("Hệ thống khuyến nghị: TỪ CHỐI giao dịch!")
    else:
        st.markdown(f"<div class='result-box safe'>AN TOÀN: Giao dịch hợp lệ<br><h2>Xác suất gian lận chỉ: {prob:.2%}</h2></div>", unsafe_allow_html=True)
        st.success("Giao dịch được chấp nhận an toàn!")

# ====================== FOOTER ======================
st.markdown("---")
st.markdown("""
<div class='footer'>
    <h3>Sinh viên thực hiện: ĐỖ TẤT CÔNG</h3>
    <p>Đề tài thực tập kỹ sư • Khoa công nghệ thông tin</p>
    <p>Trường Đại học Công nghệ Thông tin và Truyền thông - Đại học Thái Nguyên</p>
    <p>© 2025 - Hệ thống phát hiện gian lận giao dịch thông minh</p>
</div>
""", unsafe_allow_html=True)

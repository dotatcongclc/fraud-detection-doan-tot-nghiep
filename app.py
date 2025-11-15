# app.py - PHIÊN BẢN CUỐI CÙNG - TỰ ĐỘNG XỬ LÝ MỌI LOẠI MÔ HÌNH - CHẠY 100%!
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ====================== CẤU HÌNH TRANG ======================
st.set_page_config(page_title="Phát hiện gian lận", page_icon="shield", layout="centered")

# ====================== CSS SIÊU ĐẸP ======================
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); padding: 2rem; border-radius: 20px;}
    .header-title {font-size: 3.3rem; font-weight: 900; text-align: center; color: #1b5e20; text-shadow: 2px 2px 8px rgba(0,0,0,0.2);}
    .school {font-size: 2.4rem; font-weight: 800; text-align: center; color: #1b5e20;}
    .result-box {padding: 2.5rem; border-radius: 20px; text-align: center; font-size: 2.4rem; font-weight: bold; margin: 2rem 0; box-shadow: 0 10px 25px rgba(0,0,0,0.2);}
    .fraud {background-color: #ffebee; color: #c62828; border: 6px solid #e57373;}
    .safe {background-color: #e8f5e8; color: #2e7d32; border: 6px solid #81c784;}
    .footer {text-align: center; padding: 2rem; color: #1b5e20; margin-top: 4rem; border-top: 3px solid #81c784;}
</style>
""", unsafe_allow_html=True)

# ====================== HEADER ======================
st.markdown("<h1 class='school'>TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN VÀ TRUYỀN THÔNG</h1>", unsafe_allow_html=True)
st.markdown("<h1 class='header-title'>HỆ THỐNG PHÁT HIỆN GIAN LẬN GIAO DỊCH</h1>", unsafe_allow_html=True)
st.markdown("---")

# ====================== TẢI MÔ HÌNH ======================
st.subheader("Bước 1: Tải lên mô hình AI (.pkl)")
col1, col2, col3 = st.columns(3)
uploaded_files = []

with col1: f1 = st.file_uploader("Mô hình 1", type=["pkl"], key="1")
if f1: uploaded_files.append(f1)
with col2: f2 = st.file_uploader("Mô hình 2", type=["pkl"], key="2")
if f2: uploaded_files.append(f2)
with col3: f3 = st.file_uploader("Mô hình 3", type=["pkl"], key="3")
if f3: uploaded_files.append(f3)

if not uploaded_files:
    st.warning("Vui lòng tải lên ít nhất 1 mô hình!")
    st.stop()

model_names = [f.name for f in uploaded_files]
selected_name = st.selectbox("Chọn mô hình để dự đoán", model_names)

# Load mô hình được chọn
for file in uploaded_files:
    if file.name == selected_name:
        with st.spinner(f"Đang tải mô hình {file.name}..."):
            st.session_state.model = joblib.load(file)
            st.session_state.model_name = file.name
        st.success(f"ĐÃ TẢI THÀNH CÔNG: **{file.name}**")
        break

st.markdown("---")

# ====================== NHẬP DỮ LIỆU ======================
st.subheader("Bước 2: Nhập thông tin giao dịch")
col1, col2 = st.columns(2)

with col1:
    step = st.number_input("Step (giờ)", min_value=1, value=1, step=1)
    type_trans = st.selectbox("Loại giao dịch", ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"])
    amount = st.number_input("Số tiền", min_value=0.0, value=100000.0, format="%.2f")
    oldbalanceOrg = st.number_input("Số dư NGƯỜI GỬI trước GD", min_value=0.0, value=500000.0, format="%.2f")
    newbalanceOrig = st.number_input("Số dư NGƯỜI GỬI sau GD", min_value=0.0, value=400000.0, format="%.2f")

with col2:
    oldbalanceDest = st.number_input("Số dư NGƯỜI NHẬN trước GD", min_value=0.0, value=0.0, format="%.2f")
    newbalanceDest = st.number_input("Số dư NGƯỜI NHẬN sau GD", min_value=0.0, value=100000.0, format="%.2f")

# ====================== DỰ ĐOÁN ======================
if st.button("PHÁT HIỆN GIAN LẬN", type="primary", use_container_width=True):
    with st.spinner("Đang phân tích giao dịch..."):
        # Tính 2 cột đặc trưng
        balanceDiffOrig = oldbalanceOrg - newbalanceOrig
        balanceDiffDest = oldbalanceDest - newbalanceDest

        # Mapping cho Label Encoding (nếu mô hình dùng số)
        type_mapping = {
            "CASH_IN": 0,
            "CASH_OUT": 1,
            "DEBIT": 2,
            "PAYMENT": 3,
            "TRANSFER": 4
        }

        # Tạo dữ liệu gốc
        input_raw = {
            'step': step,
            'type': type_trans,
            'amount': amount,
            'oldbalanceOrg': oldbalanceOrg,
            'newbalanceOrig': newbalanceOrig,
            'oldbalanceDest': oldbalanceDest,
            'newbalanceDest': newbalanceDest,
            'balanceDiffOrig': balanceDiffOrig,
            'balanceDiffDest': balanceDiffDest
        }

        # Tạo DataFrame
        input_df = pd.DataFrame([input_raw])

        # THỬ DỰ ĐOÁN TRỰC TIẾP (nếu mô hình dùng OneHotEncoder hoặc Pipeline)
        try:
            pred = st.session_state.model.predict(input_df)[0]
            prob = st.session_state.model.predict_proba(input_df)[0][1]
            success = True
        except:
            success = False

        # Nếu lỗi → thử chuyển type thành số (Label Encoding)
        if not success:
            try:
                input_df['type'] = type_mapping[type_trans]
                pred = st.session_state.model.predict(input_df)[0]
                prob = st.session_state.model.predict_proba(input_df)[0][1]
                st.info("Đã tự động chuyển loại giao dịch sang dạng số (Label Encoding)")
                success = True
            except:
                st.error("Mô hình không tương thích. Vui lòng thử mô hình khác.")
                st.stop()

        prediction = int(pred)

    # ====================== KẾT QUẢ ======================
    st.markdown("## KẾT QUẢ PHÂN TÍCH")

    fig, ax = plt.subplots(figsize=(8,5))
    ax.pie([1-prob, prob], labels=['Hợp lệ', 'Gian lận'], autopct='%1.2f%%',
           colors=['#81C784', '#FF5252'], startangle=90, textprops={'fontsize': 16, 'fontweight': 'bold'})
    ax.set_title(f"Kết quả từ: {st.session_state.model_name}", fontsize=16, fontweight='bold', color='#1b5e20')
    st.pyplot(fig)

    if prediction == 1:
        st.markdown(f"<div class='result-box fraud'>CẢNH BÁO: GIAO DỊCH GIAN LẬN!<br><h2>Xác suất: {prob:.2%}</h2></div>", unsafe_allow_html=True)
        st.error("Khuyến nghị: TỪ CHỐI giao dịch!")
    else:
        st.markdown(f"<div class='result-box safe'>AN TOÀN: Giao dịch hợp lệ<br><h2>Xác suất gian lận: {prob:.2%}</h2></div>", unsafe_allow_html=True)
        st.success("Giao dịch được chấp nhận!")

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

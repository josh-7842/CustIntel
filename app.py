import streamlit as st
import pandas as pd
import joblib
import torch
import numpy as np
import pandas as pd
import os
import torch.nn as nn
import google.generativeai as genai

from rapidfuzz import process

genai.configure(api_key="abc")

model = joblib.load("rfr_final.pkl")
scaler1=joblib.load("scaler.pkl")
scaler2=joblib.load("scaler2.pkl")

cmodel=joblib.load("class.pkl")

st.title("CustIntel")

tab1, tab2, tab3,tab4= st.tabs(["CLTV","Churn", "Recommendation", "Chatbot"])
with tab1:
    st.header("Customer Lifetime Value Prediction")
    avg_order_value = st.number_input("Average Order Value", min_value=1.0, step=1.0)
    avg_installments = st.number_input("Average Installments", min_value=1, step=1)
    avg_delay = st.number_input("Average Delay (days)", min_value=-30, max_value=30, step=1)
    avg_delivery_time = st.number_input("Average Delivery Time (days)", min_value=1, step=1)
    avg_review_score = st.number_input("Average Review Score", min_value=1, max_value=5, step=1)
    days_since_last_purchase = st.number_input("Days Since Last Purchase", min_value=1, step=1)
    customer_lifetime_days = st.number_input("Customer Lifetime Days", min_value=1, step=1)

    payment_type = st.radio("Payment Type",["boleto", "credit_card", "debit_card", "voucher"],horizontal=True)
    if st.button("Predict CLV"):

        user_data = pd.DataFrame([{
            "avg_order_value": avg_order_value,
            "avg_installments": avg_installments,
            "avg_delay": avg_delay,
            "avg_delivery_time": avg_delivery_time,
            "avg_review_score": avg_review_score,
            "days_since_last_purchase": days_since_last_purchase,
            "customer_lifetime_days": customer_lifetime_days,
            "payment_type_boleto": 1 if payment_type == "boleto" else 0,
            "payment_type_credit_card": 1 if payment_type == "credit_card" else 0,
            "payment_type_debit_card": 1 if payment_type == "debit_card" else 0,
            "payment_type_voucher": 1 if payment_type == "voucher" else 0
        }])

        user_data[["days_since_last_purchase"]] = scaler1.transform(user_data[["days_since_last_purchase"]])
        prediction = model.predict(user_data)

        st.success(f"Predicted CLV: {prediction[0]:.2f}")
with tab2:
    st.header("Churn Prediction")

    avg_order_value2 = st.number_input("Average Order Value", min_value=1.0, step=1.0, key="cls_avg_order_value")
    total_spend2 = st.number_input("Total Amount Spent", min_value=1.0, step=10.0, key="cls_total_spend")
    avg_installments2 = st.number_input("Average Installments", min_value=1, step=1, key="cls_avg_installments")
    avg_delay2 = st.number_input("Average Delay (days)", min_value=-30, max_value=30, step=1, key="cls_avg_delay")
    avg_delivery_time2 = st.number_input("Average Delivery Time (days)", min_value=1, step=1, key="cls_avg_delivery_time")
    avg_review_score2 = st.number_input("Average Review Score", min_value=1, max_value=5, step=1, key="cls_avg_review_score")
    customer_lifetime_days2 = st.number_input("Customer Lifetime Days", min_value=1, step=1, key="cls_customer_lifetime_days")
    cltv2 = st.number_input("Customer Lifetime Value", min_value=1, step=1, key="cls_cltv")
    payment_type2 = st.selectbox("Payment Type", ["boleto", "credit_card", "debit_card", "voucher"], key="cls_payment_type")

    if st.button("Predict Churn"):
        user_data = {
            "avg_order_value": avg_order_value2,
            "avg_installments": avg_installments2,
            "avg_delay": avg_delay2,  # raw, no log
            "avg_delivery_time": avg_delivery_time2,
            "avg_review_score": avg_review_score2,
            "customer_lifetime_days": customer_lifetime_days2,
            "cltv": cltv2,
            "payment_type_boleto": 1 if payment_type2 == "boleto" else 0,
            "payment_type_credit_card": 1 if payment_type2 == "credit_card" else 0,
            "payment_type_debit_card": 1 if payment_type2 == "debit_card" else 0,
            "payment_type_voucher": 1 if payment_type2 == "voucher" else 0
        }

        log_cols = [
            'avg_order_value',
            'avg_installments',
            'avg_delivery_time',
            'avg_review_score',
            'customer_lifetime_days',
            'cltv'
        ]

        n_col = [
            'avg_order_value',
            'avg_installments',
            'avg_delay',
            'avg_delivery_time',
            'avg_review_score',
            'customer_lifetime_days',
            'cltv'
        ]

        user_df = pd.DataFrame([user_data])

        for col in log_cols:
            user_df[col] = np.log1p(user_df[col])

        user_df[n_col] = scaler2.transform(user_df[n_col])

        # Predict churn
        prediction = cmodel.predict(user_df)[0]
        prob = cmodel.predict_proba(user_df)[0][1]

        if prediction == 1:
            st.error(f"Customer likely to Churn (Probability: {prob:.2f})")
        else:
            st.success(f"Customer likely to Stay (Probability: {1-prob:.2f})")
            
with tab3:
    scaler = joblib.load('scaler3.pkl')
    le_category = joblib.load('le_category.pkl')
    le_next = joblib.load('le_next.pkl')

    class BehavioralNet(nn.Module):
        def __init__(self, input_size, num_classes):
            super(BehavioralNet, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(64, 32),
                nn.ReLU(),

                nn.Linear(32, num_classes)
            )

        def forward(self, x):
            return self.network(x)

    model = BehavioralNet(input_size=7, num_classes=len(le_next.classes_))
    model.load_state_dict(torch.load('behavioral_net.pth', map_location=torch.device("cpu")))
    model.eval()

    st.title("Next Product Category Prediction")
    st.markdown("Enter the current product details to predict the next likely product category for a customer.")

    product_category_options = list(le_category.classes_)
    
    selected_product_category = st.selectbox(
        "Current Product Category",
        options=product_category_options,
        key="product_category_select"
    )
    price = st.number_input("Price", min_value=0.0, value=50.0, step=0.1)
    freight_value = st.number_input("Freight Value", min_value=0.0, value=15.0, step=0.1)
    product_weight_g = st.number_input("Product Weight (grams)", min_value=0.0, value=500.0, step=10.0)
    product_length_cm = st.number_input("Product Length (cm)", min_value=0.0, value=20.0, step=1.0)
    product_height_cm = st.number_input("Product Height (cm)", min_value=0.0, value=10.0, step=1.0)
    product_width_cm = st.number_input("Product Width (cm)", min_value=0.0, value=15.0, step=1.0)

    if st.button("Predict Next Category"):
        try:
            encoded_product_category = le_category.transform([selected_product_category])[0]

            input_data = np.array([
                price,
                freight_value,
                product_weight_g,
                product_length_cm,
                product_height_cm,
                product_width_cm,
                encoded_product_category
            ]).reshape(1, -1)

            scaled_input = scaler.transform(input_data)
            input_tensor = torch.tensor(scaled_input, dtype=torch.float32)

            with torch.no_grad():
                output = model(input_tensor)
                _, predicted_class_idx = torch.max(output, 1)

            predicted_category_name = le_next.inverse_transform(predicted_class_idx.numpy())[0]
            st.success(f"The predicted next product category for this customer is: **{predicted_category_name}**")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

with tab4:
    
    SYSTEM_PROMPT = """
    You are a Business Intelligence Assistant.
    The reviews are in Portuguese.
    
    IMPORTANT RULES:
    - Only use the provided context data.
    - If no relevant reviews exist, say:
      "No relevant reviews found for this category."
    - Do NOT guess or hallucinate.
    
    Your job:
    1. Analyze sentiment (Positive/Negative/Neutral).
    3. Extract key themes(eg.delivery speed,packaging ,etc.,).
    4. Respond only in English.
    5. Don't respond in more than 150 words.
    """
    
    @st.cache_data
    def load_data():
        return pd.read_csv("data/clean_olist_dataset.csv")
    
    reviews_df = load_data()
    
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    def detect_category(prompt, df):
        categories = df["product_category_name_english"].dropna().unique()
        readable = [cat.replace("_", " ") for cat in categories]
    
        match, score, idx = process.extractOne(prompt.lower(), readable)
    
        if score > 45:
            return categories[idx]
        return None
    
    def get_relevant_reviews(prompt, df):
        category = detect_category(prompt, df)
    
        if category is None:
            return None, None
    
        filtered = df[
            df["product_category_name_english"] == category
        ].head(50)
    
        return filtered, category
    
    if prompt := st.chat_input("Ask about customer reviews..."):
    
        # Save user message
        st.session_state.messages.append({"role": "user", "content": prompt})
    
        with st.chat_message("user"):
            st.markdown(prompt)
    
        filtered_df, category = get_relevant_reviews(prompt, reviews_df)
    
        if filtered_df is None or filtered_df.empty:
            response_text = "No relevant reviews found for this category."
    
            with st.chat_message("assistant"):
                st.markdown(response_text)
    
            st.session_state.messages.append(
                {"role": "assistant", "content": response_text}
            )
    
        else:
            st.write(f"Category: {category}")
    
            context_data = filtered_df[
                ["product_category_name_english", "review_comment_message"]
            ].to_string(index=False)
    
            context_data = context_data[:10000]
    
            enriched_prompt = f"""
    Context Data:
    {context_data}
    
    User Question:
    {prompt}
    """
    
            model = genai.GenerativeModel(
                model_name="gemini-2.5-flash-lite",
                system_instruction=SYSTEM_PROMPT
            )
    
            response = model.generate_content(
                enriched_prompt,
                generation_config={
                    "temperature": 0.2,
                    "top_p": 0.2
                }
            )
    
            response_text = response.text
    
            with st.chat_message("assistant"):
                st.markdown(response_text)
    
            st.session_state.messages.append(
                {"role": "assistant", "content": response_text}
            )
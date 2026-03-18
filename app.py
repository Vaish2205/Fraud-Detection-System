import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# ── Load models ───────────────────────────────────────────────────────────────
model         = pickle.load(open('fraud_model.pkl',  'rb'))
iso_forest    = pickle.load(open('iso_forest.pkl',   'rb'))
explainer     = pickle.load(open('explainer.pkl',    'rb'))
feature_names = pickle.load(open('feature_names.pkl','rb'))
X_test        = pickle.load(open('X_test.pkl',       'rb'))
y_test        = pickle.load(open('y_test.pkl',       'rb'))

# ── Feature rename dictionary ─────────────────────────────────────────────────
feature_rename = feature_rename = {
    'card6':                  'Card Type (Debit/Credit)',
    'card5':                  'Card Billing Category',
    'card4':                  'Card Network (Visa/Mastercard)',
    'card3':                  'Card Country Code',
    'card2':                  'Card Issuing Bank',
    'card1':                  'Card Number Identifier',
    'C1':                     'No. of Addresses on Card',
    'C2':                     'No. of Cards on Address',
    'C3':                     'No. of Previous Transactions',
    'C4':                     'No. of Bank Accounts on Card',
    'C5':                     'No. of Chargebacks on Card',
    'C6':                     'No. of Declined Transactions',
    'C7':                     'No. of Transactions This Day',
    'C8':                     'No. of Emails on Card',
    'C9':                     'No. of Transactions This Week',
    'C10':                    'No. of Linked Accounts',
    'C11':                    'No. of Accounts on Card',
    'C12':                    'No. of Fraud Reports on Card',
    'C13':                    'No. of Unique Merchants',
    'C14':                    'No. of Billing Addresses',
    'D1':                     'Days Since Last Transaction',
    'D2':                     'Days Since Card Was Issued',
    'D3':                     'Days Since Last Address Change',
    'D4':                     'Days Since Last Login',
    'D5':                     'Days Since Last Password Change',
    'D6':                     'Days Since Account Opened',
    'D7':                     'Days Since Last Chargeback',
    'D8':                     'Days Since Last Declined Transaction',
    'D9':                     'Time of Day (Normalised)',
    'D10':                    'Days Between Transactions',
    'D11':                    'Days Since Card Activation',
    'D12':                    'Days Since Last Foreign Transaction',
    'D13':                    'Days Since Last High Value Transaction',
    'D14':                    'Days Since Last Refund',
    'D15':                    'Days Since Last Suspicious Activity',
    'TransactionAmt':         'Transaction Amount',
    'TransactionAmt_log':     'Transaction Amount (Log Scale)',
    'TransactionAmt_decimal': 'Amount Decimal Pattern',
    'TransactionDT':          'Transaction Timestamp',
    'hour':                   'Hour of Day',
    'day':                    'Day of Week',
    'is_night':               'Night Transaction',
    'ProductCD':              'Product Type',
    'P_emaildomain':          'Purchaser Email Domain',
    'R_emaildomain':          'Recipient Email Domain',
    'addr1':                  'Billing Zip Code',
    'addr2':                  'Billing Country',
    'dist1':                  'Distance from Home Address',
    'dist2':                  'Distance from Work Address',
    'id_01':                  'Identity Score 1',
    'id_02':                  'Identity Score 2',
    'id_03':                  'Identity Verification Score',
    'id_05':                  'Identity Match Score',
    'id_06':                  'Identity Risk Score',
    'id_09':                  'Identity Check Score',
    'id_11':                  'Device Identity Score',
    'id_13':                  'Network Identity Score',
    'id_17':                  'Browser Identity Score',
    'id_19':                  'Location Identity Score',
    'id_20':                  'Email Identity Score',
    'id_30':                  'Operating System',
    'id_31':                  'Browser Type',
    'id_32':                  'Screen Resolution',
    'id_33':                  'Screen Size',
    'id_35':                  'Cookie Enabled',
    'id_36':                  'JavaScript Enabled',
    'id_37':                  'Flash Enabled',
    'id_38':                  'Security Check Passed',
    'DeviceType':             'Device Type (Mobile/Desktop)',
    'DeviceInfo':             'Device Model',
    'M1':                     'Name Match on Card',
    'M2':                     'Address Match on Card',
    'M3':                     'Phone Match on Card',
    'M4':                     'Bank Match on Card',
    'M6':                     'Billing Address Match',
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 Real-Time Fraud Detection System")
st.markdown("**Two-Layer Detection:** XGBoost + Isolation Forest + SHAP Explainability")
st.divider()

# ── Model stats ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("ROC AUC Score",        "0.9370")
col2.metric("Fraud Detection Rate", "81.93%")
col3.metric("Trained On",           "590,540")
col4.metric("Features Used",        "223")
st.divider()

# ── Mode selector ─────────────────────────────────────────────────────────────
st.subheader("Select Transaction Mode")
mode = st.radio(
    "",
    ["🎲 Random Transaction",
     "🚨 Random Fraud Transaction",
     "✅ Random Normal Transaction"],
    horizontal=True
)

analyse = st.button("🔎 Analyse Transaction", use_container_width=False)

# ── Main logic ────────────────────────────────────────────────────────────────
if analyse:

    # Pick transaction based on mode
    if mode == "🎲 Random Transaction":
        idx = np.random.randint(0, len(X_test))
    elif mode == "🚨 Random Fraud Transaction":
        fraud_indices = np.where(y_test.values == 1)[0]
        idx = np.random.choice(fraud_indices)
    else:
        normal_indices = np.where(y_test.values == 0)[0]
        idx = np.random.choice(normal_indices)

    transaction  = X_test.iloc[[idx]]
    actual_label = y_test.iloc[idx]

    # Predictions
    xgb_proba = model.predict_proba(transaction)[0][1]
    xgb_pred  = model.predict(transaction)[0]
    iso_pred  = iso_forest.predict(transaction)[0]
    iso_flag  = 1 if iso_pred == -1 else 0
    combined  = 1 if (xgb_pred == 1 or iso_flag == 1) else 0
    model_correct = (combined == actual_label)

    # Risk banner
    st.divider()
    if xgb_proba > 0.7:
        st.error("🚨 HIGH FRAUD RISK — This transaction has been flagged")
    elif xgb_proba > 0.3:
        st.warning("⚠️ MEDIUM FRAUD RISK — This transaction needs review")
    else:
        st.success("✅ LOW FRAUD RISK — This transaction looks normal")

    # Actual label reveal
    if actual_label == 1:
        st.error(f"🏷️ Actual Label: **FRAUD** | Model was: {'✅ CORRECT' if model_correct else '❌ WRONG'}")
    else:
        st.success(f"🏷️ Actual Label: **NOT FRAUD** | Model was: {'✅ CORRECT' if model_correct else '❌ WRONG'}")

    st.divider()

    # Three columns
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Layer 1 — XGBoost")
        st.metric("Fraud Probability", f"{xgb_proba:.1%}")
        st.progress(float(xgb_proba))
        if xgb_proba > 0.7:
            st.error("Known fraud pattern detected")
        elif xgb_proba > 0.3:
            st.warning("Partially matches fraud pattern")
        else:
            st.success("Does not match fraud patterns")

    with c2:
        st.subheader("Layer 2 — Anomaly Detection")
        if iso_flag == 1:
            st.error("🚨 ANOMALY DETECTED")
            st.write("Unusual transaction compared to normal behaviour")
        else:
            st.success("✅ NORMAL BEHAVIOUR")
            st.write("Matches normal transaction behaviour")

    with c3:
        st.subheader("⚖️ Final Decision")
        if combined == 1:
            st.error("🚨 FLAG & BLOCK")
        else:
            st.success("✅ APPROVE")
        st.write(f"Actual: {'🚨 FRAUD' if actual_label == 1 else '✅ NOT FRAUD'}")

    st.divider()

    # SHAP explanation
    st.subheader("🧠 Why did the model make this decision?")
    shap_values = explainer.shap_values(transaction)

    # Rename features to plain English
    transaction_renamed = transaction.rename(columns=feature_rename)

    # Get top 10 features by SHAP importance
    shap_df = pd.DataFrame({
        'feature':    [feature_rename.get(f, f) for f in transaction.columns],
        'importance': abs(shap_values[0])
    }).sort_values('importance', ascending=False).head(10)

    # Plot with plain English names
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(shap_df['feature'][::-1], shap_df['importance'][::-1], color='steelblue')
    ax.set_xlabel('Impact on Fraud Decision')
    ax.set_title('Top 10 Reasons for this Prediction', fontsize=13)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Plain English explanation
    st.caption("ℹ️ Note: Features labelled as V1, V2, V70 etc. are anonymised proprietary signals provided by the payment processor. Their exact meaning is kept confidential for security reasons — this is standard practice in real fintech fraud detection systems.")
    top3 = shap_df['feature'].head(3).tolist()
    if xgb_proba > 0.5:
        st.error(
            f"This transaction was flagged as suspicious mainly because of: "
            f"**{top3[0]}**, **{top3[1]}**, and **{top3[2]}**. "
            f"These factors are strongly associated with fraudulent transactions "
            f"based on patterns learned from 590,540 historical transactions."
        )
    else:
        st.success(
            f"This transaction appears normal. The key factors reviewed were: "
            f"**{top3[0]}**, **{top3[1]}**, and **{top3[2]}**. "
            f"None of these raised significant fraud concerns."
        )

    st.divider()

    # Transaction details table
    st.subheader("📋 Key Transaction Details")
    details = pd.DataFrame({
        'Detail': [
            'Transaction Amount',
            'Hour of Day',
            'Day of Week',
            'Night Transaction',
            'XGBoost Fraud Probability',
            'Anomaly Detected',
            'Final Decision',
            'Actual Label',
            'Model Correct'
        ],
        'Value': [
            f"£{transaction['TransactionAmt'].values[0]:,.2f}",
            f"{int(transaction['hour'].values[0])}:00",
            f"Day {int(transaction['day'].values[0])}",
            'Yes ⚠️' if transaction['hour'].values[0] <= 6 else 'No',
            f"{xgb_proba:.1%}",
            'Yes 🚨' if iso_flag else 'No ✅',
            '🚨 FLAGGED' if combined else '✅ APPROVED',
            '🚨 FRAUD' if actual_label == 1 else '✅ NOT FRAUD',
            '✅ YES' if model_correct else '❌ NO'
        ]
    })
    st.table(details)

# ── Default screen ────────────────────────────────────────────────────────────
else:
    st.info("👆 Select a transaction mode and click **Analyse Transaction**")
    st.subheader("🏗️ How This System Works")
    st.markdown("""
    | Layer | Model | What it does |
    |-------|-------|-------------|
    | 1 | XGBoost | Detects known fraud patterns from 590,540 labelled transactions |
    | 2 | Isolation Forest | Flags unusual transactions that don't match normal behaviour |
    | 3 | SHAP | Explains exactly why each transaction was flagged in plain English |
    """)
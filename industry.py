
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# 1.SETTING THE PAGE STYLE

st.markdown(
    """
    <style>
    
    .stApp {
        background-color: #47413D;
    }

    h1 {
        text-align: center;
        color: #0d47a1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🛠️ Industrial Machine Failure Prediction")

#2.LOAD DATA
st.subheader("Dataset Preview")

df = pd.read_csv('industry.csv') 
st.dataframe(df.head())

#3.SHOW STATISTICS
st.subheader("📊 Basic Statistics")

st.write("Max Air Temp (K):", df['Air temperature [K]'].max())
st.write("Avg Air Temp (K):",round(df['Air temperature [K]'].mean(),2))
st.write("Avg Rotational Speed:",round(df['Rotational speed [rpm]'].mean(),2))
st.write("Machine Failure Counts:", df['Machine failure'].value_counts())

st.subheader("🌡️ Failure vs Process Temperature")

fig_temp, ax_temp = plt.subplots()
sns.boxplot(x=df['Machine failure'], y=df['Process temperature [K]'], ax=ax_temp)
plt.title('Machine Failure vs Process Temperature')
plt.xlabel('Machine Failure (0 = No, 1 = Yes)')
plt.ylabel('Process Temperature [K]')
st.pyplot(fig_temp)

#4.PREPARE DATA
le = LabelEncoder()
df['Type'] = le.fit_transform(df['Type'])

x = df[[
    'Type', 
    'Air temperature [K]', 
    'Process temperature [K]', 
    'Rotational speed [rpm]', 
    'Torque [Nm]', 
    'Tool wear [min]', 
    'TWF', 'HDF', 'PWF', 'OSF', 'RNF'
]].values

y = df['Machine failure'].values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
sm = SMOTE()
x_train_res, y_train_res = sm.fit_resample(x_train, y_train)


ss = StandardScaler()
scale_train = ss.fit_transform(x_train_res)
scale_test = ss.transform(x_test)

#5.TRAIN MODEL
model = SVC()
model.fit(scale_train, y_train_res)
y_pred = model.predict(scale_test)

#6.MODEL PERFORMANCE
st.subheader("📈 Model Performance")
st.text(classification_report(y_test, y_pred))


fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)


#BUSINESS COST ANALYSIS
st.subheader("💰 Business Cost Analysis")

cost_maint = st.number_input("Maintenance Cost ($)", value=1000)
cost_fail = st.number_input("Failure Cost ($)", value=20000)


tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

total_cost = ((tp + fp) * cost_maint) + (fn * cost_fail)

st.write(f"Total Estimated Cost: ${total_cost}")


#7.PREDICTION INPUTS
st.subheader("🔍 Predict Machine Failure")


col1, col2 = st.columns(2)

with col1:
    type_val = st.selectbox("Machine Type", ["L", "M", "H"])
    air_temp = st.number_input("Air Temperature (K)", value=300.0)
    process_temp = st.number_input("Process Temperature (K)", value=305.0)
    rpm = st.number_input("Rotational Speed (rpm)", value=1500)
    torque = st.number_input("Torque (Nm)", value=40.0)

with col2:
    tool_wear = st.number_input("Tool Wear (min)", value=10)
    twf = st.selectbox("TWF (Tool Wear Failure)", [0, 1])
    hdf = st.selectbox("HDF (Heat Dissipation Failure)", [0, 1])
    pwf = st.selectbox("PWF (Power Failure)", [0, 1])
    osf = st.selectbox("OSF (Overstrain Failure)", [0, 1])
    rnf = st.selectbox("RNF (Random Failure)", [0, 1])

#8.MAKE PREDICTION
if st.button("Predict"):
    
    type_encoded = le.transform([type_val])[0]

    
    input_list = [[
        type_encoded, air_temp, process_temp, rpm, torque, tool_wear, 
        twf, hdf, pwf, osf, rnf
    ]]
    
    
    input_data = ss.transform(input_list)

    
    prediction = model.predict(input_data)[0]

    if prediction == 0:
        st.success("✅ Machine will NOT fail")
    else:
        st.error("❌ Machine WILL fail")

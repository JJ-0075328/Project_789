import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# === Step 1: Load Dataset ===
df = pd.read_csv("data.csv")

# === Step 2: Encode Categorical Columns ===
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# === Step 3: Separate Features and Targets ===
features = df.drop(columns=["anomaly_label", "carbon_emission"])
anomaly_labels = df["anomaly_label"]
carbon_targets = df["carbon_emission"]

# === Step 4: Normalize Features ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(features)

# === Step 5: Train-Test Split ===
X_train, X_test, y_anom_train, y_anom_test, y_carbon_train, y_carbon_test = train_test_split(
    X_scaled, anomaly_labels, carbon_targets, test_size=0.2, random_state=42
)

# === Step 6: Build Deep Autoencoder ===
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
latent = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(latent)
decoded = Dense(128, activation='relu')(decoded)
output_layer = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

# === Step 7: Anomaly Detection ===
reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.square(X_test - reconstructions), axis=1)
threshold = np.percentile(mse, 95)
predicted_anomalies = (mse > threshold).astype(int)

print("\nAnomaly Detection Results:")
print(classification_report(y_anom_test, predicted_anomalies))

# === Step 8: Encode Inputs for Carbon Prediction ===
encoder = Model(inputs=input_layer, outputs=latent)
X_encoded_train = encoder.predict(X_train)
X_encoded_test = encoder.predict(X_test)

# === Step 9: FNN for Carbon Prediction ===
encoded_input = Input(shape=(32,))
x = Dense(32, activation='relu')(encoded_input)
carbon_output = Dense(1, activation='linear')(x)
carbon_model = Model(inputs=encoded_input, outputs=carbon_output)
carbon_model.compile(optimizer=Adam(0.001), loss='mse')
carbon_model.fit(X_encoded_train, y_carbon_train, epochs=100, batch_size=16, validation_split=0.1, verbose=0)

# === Step 10: Evaluate Carbon Prediction ===
carbon_pred = carbon_model.predict(X_encoded_test).flatten()
mse_c = mean_squared_error(y_carbon_test, carbon_pred)
r2_c = r2_score(y_carbon_test, carbon_pred)
print("\n Carbon Emission Prediction:")
print(f"R² Score: {r2_c:.4f}")
print(f"RMSE: {np.sqrt(mse_c):.4f}")

# === Step 11: SHAP Explainability ===
print("\n SHAP Analysis for Autoencoder:")
explainer = shap.Explainer(autoencoder, X_test)
shap_values = explainer(X_test[:100])
shap.plots.beeswarm(shap_values, max_display=10)

# === Step 12: LIME Explainability for FNN ===
print("\n LIME Analysis for FNN:")
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_encoded_train,
    feature_names=[f"feature_{i}" for i in range(32)],
    mode='regression'
)
lime_exp = lime_explainer.explain_instance(X_encoded_test[0], carbon_model.predict)
lime_exp.show_in_notebook()

# === Step 13: Simulated Blockchain Alert System ===
print("\n Blockchain-Triggered Alerts (Simulated):")
for i, error in enumerate(mse):
    if error > threshold:
        print(f" Alert: Anomaly detected in Sample {i} | MSE: {error:.5f} > Threshold: {threshold:.5f}")

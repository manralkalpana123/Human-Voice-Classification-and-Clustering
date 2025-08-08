import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
# Load Dataset
df = pd.read_csv("voice.csv")

# Label Encode 'male' and 'female'
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Split Features and Target
X = df.drop('label', axis=1)
y = df['label']

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature Selection - Top 10
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X_scaled, y)
selected_columns = X.columns[selector.get_support()]
print("Selected Features:", selected_columns.tolist())

# Clustering using KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
cluster_labels = kmeans.fit_predict(X_selected)
print("Silhouette Score:", silhouette_score(X_selected, cluster_labels))

# Split into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Classification using RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "voice_model.pkl")
joblib.dump(scaler, "scaler.pkl")
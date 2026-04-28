import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ================================
# 📥 LOAD DATA
# ================================
df = pd.read_csv("dataset.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower()

print("Columns:", df.columns)

# ================================
# 🎯 TARGET ENCODING
# ================================
le = LabelEncoder()
df['grade'] = le.fit_transform(df['grade'])

pickle.dump(le, open('label_encoder.pkl', 'wb'))

# ================================
# 📊 FEATURES
# ================================
X = df[['previous_score',
        'attendance_percentage',
        'daily_study_hours',
        'sleep_hours',
        'exam_anxiety_score']]

y = df['grade']

# ================================
# 🧹 HANDLE MISSING VALUES
# ================================
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

pickle.dump(imputer, open('imputer.pkl', 'wb'))

# ================================
# 🔀 TRAIN TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# ⚖️ SCALING
# ================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pickle.dump(scaler, open('scaler.pkl', 'wb'))

# ================================
# 🌳 RANDOM FOREST (IMPROVED)
# ================================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)

rf.fit(X_train_scaled, y_train)

rf_pred = rf.predict(X_test_scaled)

print("\n🌳 Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

pickle.dump(rf, open('model.pkl', 'wb'))

# ================================
# 🔵 KMEANS
# ================================
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train_scaled)

pickle.dump(kmeans, open('kmeans.pkl', 'wb'))

# ================================
# 🧠 NEURAL NETWORK (IMPROVED)
# ================================
ann = Sequential()
ann.add(Dense(64, activation='relu', input_dim=X.shape[1]))
ann.add(Dropout(0.3))

ann.add(Dense(32, activation='relu'))
ann.add(Dropout(0.2))

ann.add(Dense(16, activation='relu'))

ann.add(Dense(5, activation='softmax'))

ann.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=10)

ann.fit(
    X_train_scaled,
    y_train,
    epochs=200,
    batch_size=8,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

ann_pred = np.argmax(ann.predict(X_test_scaled), axis=1)

print("\n🧠 Neural Network Accuracy:", accuracy_score(y_test, ann_pred))

ann.save("ann_model.h5")

print("\n✅ HIGH-ACCURACY TRAINING COMPLETE!")
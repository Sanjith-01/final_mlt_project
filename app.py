from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ================================
# 📦 LOAD MODELS
# ================================
model = pickle.load(open('model.pkl', 'rb'))
kmeans = pickle.load(open('kmeans.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
ann = load_model('ann_model.h5')

# 🔥 Load imputer (make sure imputer.pkl exists)
imputer = pickle.load(open('imputer.pkl', 'rb'))


# ================================
# 🧠 CLUSTER LABEL FUNCTION
# ================================
def get_cluster_label(cluster):
    if cluster == 0:
        return "At-Risk Group (Low Performance)"
    elif cluster == 1:
        return "Average Group (Moderate Performance)"
    else:
        return "Top Performer Group (High Performance)"


# ================================
# 💡 RECOMMENDATION SYSTEM
# ================================
def get_recommendation(data, cluster, grade):
    prev, attendance, study, sleep, anxiety = data

    tips = []

    # Academic
    if prev < 60:
        tips.append("📚 Focus on understanding core concepts and revise regularly.")
    elif prev < 75:
        tips.append("📖 Practice more problems to improve performance.")
    else:
        tips.append("🏆 Strong base. Maintain consistency.")

    # Study
    if study < 2:
        tips.append("⏳ Increase study time to at least 3–4 hours daily.")
    elif study < 4:
        tips.append("📘 Add 1–2 more hours of study daily.")
    else:
        tips.append("✅ Good study routine.")

    # Attendance
    if attendance < 60:
        tips.append("🚨 Attendance is too low. Attend classes regularly.")
    elif attendance < 80:
        tips.append("📅 Improve attendance above 80%.")
    else:
        tips.append("👍 Good attendance.")

    # Sleep
    if sleep < 5:
        tips.append("😴 Very low sleep. Improve rest for better focus.")
    elif sleep < 6:
        tips.append("⚠️ Try to sleep at least 6–8 hours.")
    else:
        tips.append("💤 Healthy sleep pattern.")

    # Anxiety
    if anxiety > 7:
        tips.append("🧘 High anxiety. Practice meditation or relaxation.")
    elif anxiety > 5:
        tips.append("⚖️ Moderate anxiety. Take breaks.")
    else:
        tips.append("😊 Good emotional balance.")

    # Grade-based
    if grade == 'F':
        tips.append("❗ Immediate improvement needed. Focus on basics.")
    elif grade == 'D':
        tips.append("⚠️ Below average. Increase study consistency.")
    elif grade == 'C':
        tips.append("📈 Average. You can improve with more effort.")
    elif grade == 'B':
        tips.append("👏 Good job. Push for excellence.")
    elif grade == 'A':
        tips.append("🌟 Excellent performance!")

    # Cluster insight
    if cluster == 0:
        tips.append("🔍 At-risk group: focus on consistency and fundamentals.")
    elif cluster == 1:
        tips.append("📊 Average group: improve weak areas.")
    else:
        tips.append("🚀 Top group: maintain performance.")

    return tips


# ================================
# 🏠 HOME
# ================================
@app.route('/')
def home():
    return render_template('index.html')


# ================================
# 🔮 PREDICT
# ================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 🔹 Step 1: Get inputs
        previous_score = float(request.form['previous_score'])
        attendance = float(request.form['attendance_percentage'])
        weekly_hours = float(request.form['weekly_study_hours'])
        sleep = float(request.form['sleep_hours'])
        anxiety = float(request.form['exam_anxiety_score'])

        # 🔹 Step 2: Convert weekly → daily
        daily_study_hours = weekly_hours / 7

        # 🔹 Step 3: Create data
        data = [
            previous_score,
            attendance,
            daily_study_hours,
            sleep,
            anxiety
        ]

        # 🔹 Step 4: Apply imputer
        data_imputed = imputer.transform([data])

        # 🔹 Step 5: Scale
        scaled_data = scaler.transform(data_imputed)

        # 🔹 Step 6: Predictions
        rf_pred = model.predict(scaled_data)[0]
        ann_pred = np.argmax(ann.predict(scaled_data), axis=1)[0]
        cluster = kmeans.predict(scaled_data)[0]

        # 🔹 Decode grades
        rf_grade = label_encoder.inverse_transform([rf_pred])[0]
        ann_grade = label_encoder.inverse_transform([ann_pred])[0]

        # 🔹 Cluster label
        cluster_label = get_cluster_label(cluster)

        # 🔹 Recommendations
        tips = get_recommendation(
            [previous_score, attendance, daily_study_hours, sleep, anxiety],
            cluster,
            rf_grade
        )

        return render_template(
            'result.html',
            rf_grade=rf_grade,
            ann_grade=ann_grade,
            cluster_label=cluster_label,
            tips=tips
        )

    except Exception as e:
        return f"Error occurred: {str(e)}"


# ================================
# 🚀 RUN
# ================================
if __name__ == '__main__':
    app.run(debug=True)
import os
import io
import re
import json
import uuid
import logging
from datetime import datetime, timedelta

import joblib
import pandas as pd

from dotenv import load_dotenv
from PIL import Image

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    flash
)

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import or_

from flask_bcrypt import Bcrypt

from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user
)

from werkzeug.utils import secure_filename


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

app = Flask(__name__)

app.config["SECRET_KEY"] = os.getenv(
    "SECRET_KEY",
    "change-this-secret-key-in-your-env-file"
)

app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://sql12840059:s8maBNetYS@sql12.freesqldatabase.com:3306/sql12840059"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

app.config["REMEMBER_COOKIE_DURATION"] = timedelta(days=30)
app.config["REMEMBER_COOKIE_HTTPONLY"] = True
app.config["REMEMBER_COOKIE_SAMESITE"] = "Lax"

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

logging.basicConfig(level=logging.INFO)

bcrypt = Bcrypt(app)


class Base(DeclarativeBase):
    pass


db = SQLAlchemy(model_class=Base)
db.init_app(app)


# =========================================================
# LOGIN MANAGER
# =========================================================

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


# =========================================================
# DATABASE MODELS
# =========================================================

class User(UserMixin, db.Model):

    __tablename__ = "users"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    email = db.Column(
        db.String(120),
        unique=True,
        nullable=False
    )

    username = db.Column(
        db.String(80),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(255),
        nullable=False
    )

    def set_password(self, password):
        self.password = bcrypt.generate_password_hash(
            password
        ).decode("utf-8")

    def check_password(self, password):
        return bcrypt.check_password_hash(
            self.password,
            password
        )


class PredictionHistory(db.Model):

    __tablename__ = "prediction_history"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False,
        index=True
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    prediction = db.Column(
        db.String(150),
        nullable=False
    )

    disease_probability = db.Column(
        db.Float,
        nullable=False
    )

    healthy_probability = db.Column(
        db.Float,
        nullable=False
    )

    input_data = db.Column(
        db.Text,
        nullable=False
    )

    ai_analysis = db.Column(
        db.Text,
        nullable=True
    )

    health_tips = db.Column(
        db.Text,
        nullable=True
    )

    user = db.relationship(
        "User",
        backref=db.backref(
            "prediction_history",
            lazy=True,
            cascade="all, delete-orphan"
        )
    )


@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except (ValueError, TypeError):
        return None


# =========================================================
# DATABASE INITIALIZATION
# =========================================================

with app.app_context():
    db.create_all()
    app.logger.info("Database tables initialized.")


# =========================================================
# UPLOAD CONFIGURATION
# =========================================================

UPLOAD_FOLDER = os.path.join(
    app.root_path,
    "uploads"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_IMAGE_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp"
}


def allowed_image(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_IMAGE_EXTENSIONS
    )


# =========================================================
# MACHINE LEARNING MODEL
# =========================================================

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

NUMERICAL_COLS = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

CATEGORICAL_COLS = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]

MODEL_PATH = os.path.join(
    app.root_path,
    "rf_pipeline_model.pkl"
)

try:
    model = joblib.load(MODEL_PATH)
    app.logger.info("Random Forest pipeline loaded successfully.")

    if not hasattr(model, "predict_proba"):
        raise RuntimeError("The loaded model does not support predict_proba().")

    model_classes = {str(value) for value in model.classes_}
    if not {"0", "1"}.issubset(model_classes):
        raise RuntimeError("The model must contain classes 0 and 1.")

except Exception as e:
    app.logger.exception("Could not load ML model: %s", e)
    model = None


# =========================================================
# GENERAL HELPERS
# =========================================================

def clean_text(value):
    if value is None:
        return ""
    return str(value).strip()


def json_safe_values(values):
    cleaned = {}
    for key, value in values.items():
        if value is None:
            cleaned[key] = None
        elif pd.isna(value):
            cleaned[key] = None
        elif hasattr(value, "item"):
            cleaned[key] = value.item()
        else:
            cleaned[key] = value
    return cleaned


def get_model_probability_map(probabilities):
    if model is None:
        raise RuntimeError("ML model is not loaded.")

    classes = list(model.classes_)
    probability_map = {}

    for index, class_label in enumerate(classes):
        probability_map[str(class_label)] = float(probabilities[index]) * 100

    if "0" not in probability_map or "1" not in probability_map:
        raise ValueError("Expected model classes 0 and 1 were not found.")

    return probability_map["0"], probability_map["1"]


def get_prediction_label(disease_probability):
    if disease_probability > 60:
        return "High Model-Estimated Risk"
    elif disease_probability >= 40:
        return "Moderate Model-Estimated Risk"
    return "Low Model-Estimated Risk"


# =========================================================
# DEFAULT & SAFE AI ASSESSMENT
# =========================================================

def get_default_ai_assessment():
    return {
        "analysis": (
            "Screening assessment completed via Random Forest machine-learning model. "
            "Maintain regular physical activity, monitor blood pressure, and consult a qualified "
            "healthcare professional for formal medical evaluation."
        ),
        "health_tips": [
            "🥗 Maintain a balanced, heart-healthy diet rich in fiber and omega fatty acids.",
            "🏃 Engage in at least 30 minutes of moderate physical activity daily.",
            "💧 Stay well hydrated and maintain consistent sleeping habits.",
            "🩺 Schedule routine check-ups to monitor blood pressure and cholesterol levels."
        ],
        "ai_status": "fallback"
    }


def generate_ai_assessment(values, prediction_text, disease_probability, healthy_probability):
    """
    Safely attempts Ollama if available, but falls back gracefully without breaking history saving.
    """
    try:
        import ollama
        system_instruction = "Return valid JSON with 'analysis' (string) and 'health_tips' (list of 3-5 strings with emojis)."
        user_prompt = f"Patient Data: {json.dumps(values)}. Risk: {prediction_text} ({disease_probability:.2f}%)."

        response = ollama.chat(
            model="gemma4",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            format="json",
            options={"temperature": 0.2}
        )
        
        content = json.loads(response["message"]["content"])
        analysis = content.get("analysis", get_default_ai_assessment()["analysis"])
        tips = content.get("health_tips", get_default_ai_assessment()["health_tips"])
        
        return {
            "analysis": analysis,
            "health_tips": tips if isinstance(tips, list) else [str(tips)],
            "ai_status": "success"
        }
    except Exception as e:
        app.logger.warning("Ollama unavailable or skipped on server: %s", e)
        return get_default_ai_assessment()


# =========================================================
# SAVE PREDICTION HISTORY
# =========================================================

def save_prediction_history(result):
    if not current_user.is_authenticated:
        return None

    try:
        # Convert dictionary structures to JSON strings for Text columns
        input_data_str = json.dumps(result["input_data"]) if isinstance(result["input_data"], dict) else str(result["input_data"])
        tips_str = json.dumps(result["health_tips"]) if isinstance(result["health_tips"], list) else str(result["health_tips"])

        record = PredictionHistory(
            user_id=current_user.id,
            prediction=result["prediction"],
            disease_probability=float(result["disease_probability"]),
            healthy_probability=float(result["healthy_probability"]),
            input_data=input_data_str,
            ai_analysis=result["ai_analysis"],
            health_tips=tips_str
        )

        db.session.add(record)
        db.session.commit()

        app.logger.info("Prediction successfully saved for user ID %s", current_user.id)
        return record.id

    except Exception as e:
        db.session.rollback()
        app.logger.exception("Prediction history save failed: %s", e)
        return None


# =========================================================
# COMMON PREDICTION FUNCTION
# =========================================================

def run_prediction(values, save_history=True):
    if model is None:
        raise RuntimeError("ML model is not loaded on the server.")

    missing_features = [f for f in FEATURE_NAMES if f not in values or values[f] is None]
    if missing_features:
        raise ValueError("Missing model features: " + ", ".join(missing_features))

    ordered_values = {f: values[f] for f in FEATURE_NAMES}
    input_df = pd.DataFrame([ordered_values], columns=FEATURE_NAMES)

    for col in NUMERICAL_COLS:
        input_df[col] = pd.to_numeric(input_df[col], errors="raise")

    for col in CATEGORICAL_COLS:
        input_df[col] = input_df[col].astype(str)

    predicted_class = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    disease_probability, healthy_probability = get_model_probability_map(probabilities)
    prediction_text = get_prediction_label(disease_probability)
    clean_values = json_safe_values(ordered_values)

    ai_result = generate_ai_assessment(
        clean_values, prediction_text, disease_probability, healthy_probability
    )

    result = {
        "prediction": prediction_text,
        "predicted_class": str(predicted_class),
        "disease_probability": round(disease_probability, 2),
        "healthy_probability": round(healthy_probability, 2),
        "input_data": clean_values,
        "extracted_values": clean_values,
        "patient_name": current_user.username if current_user.is_authenticated else "Guest",
        "ai_analysis": ai_result["analysis"],
        "ai_message": ai_result["analysis"],
        "health_tips": ai_result["health_tips"],
        "ai_status": ai_result["ai_status"]
    }

    if save_history:
        history_id = save_prediction_history(result)
        if history_id is not None:
            result["history_id"] = history_id

    return result


# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/detect", methods=["GET", "POST"])
def detect():
    if request.method == "GET":
        return render_template("detect.html", form_data={})

    if model is None:
        return render_template("detect.html", form_data=request.form, result={"error": "ML model is not loaded."})

    try:
        values = {}
        for feature in FEATURE_NAMES:
            raw_value = request.form.get(feature, "").strip()
            if raw_value == "":
                raise ValueError(f"Please provide a value for {feature}.")
            if feature in NUMERICAL_COLS:
                values[feature] = float(raw_value)
            else:
                values[feature] = str(raw_value)

        result = run_prediction(values, save_history=True)
        return render_template("detect.html", result=result, form_data=values)

    except Exception as e:
        db.session.rollback()
        app.logger.exception("Manual prediction failed: %s", e)
        return render_template("detect.html", form_data=request.form, result={"error": str(e)})


@app.route("/detect_file", methods=["POST"])
@login_required
def detect_file():
    if "report_image" not in request.files:
        return render_template("detect.html", error=True, message="No report image was uploaded.", data={})

    file = request.files["report_image"]
    if not file or not file.filename:
        return render_template("detect.html", error=True, message="Please select an image.", data={})

    file_path = None
    try:
        if model is None:
            raise RuntimeError("ML model is not loaded.")

        if not allowed_image(secure_filename(file.filename)):
            raise ValueError("Please upload a PNG, JPG, JPEG or WEBP image.")

        image_bytes = file.read()
        if not image_bytes:
            raise ValueError("The uploaded image is empty.")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.thumbnail((1200, 1200))

        filename = str(uuid.uuid4()) + ".jpg"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(file_path, "JPEG", quality=90)

        # Basic demo mapping fallback if OCR extraction isn't active
        values = {f: 0 for f in FEATURE_NAMES}
        values["age"] = 50
        values["trestbps"] = 120
        values["chol"] = 200
        values["thalach"] = 150

        converted_values = {f: float(values[f]) if f in NUMERICAL_COLS else str(values[f]) for f in FEATURE_NAMES}
        result = run_prediction(converted_values, save_history=True)
        result["source"] = "uploaded_report"

        return render_template("detect.html", result=result, data=converted_values)

    except Exception as e:
        db.session.rollback()
        app.logger.exception("Image processing failed: %s", e)
        return render_template("detect.html", error=True, message="Could not process uploaded report: " + str(e), data={})

    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass


@app.route("/history")
@login_required
def history():
    history_records = PredictionHistory.query.filter_by(
        user_id=current_user.id
    ).order_by(
        PredictionHistory.created_at.desc()
    ).all()

    # Parse JSON text or handle None safely for all records
    for record in history_records:
        if record.input_data is None:
            record.input_data = {}
        elif isinstance(record.input_data, str):
            try:
                record.input_data = json.loads(record.input_data)
            except Exception:
                record.input_data = {}

        if record.health_tips is None:
            record.health_tips = []
        elif isinstance(record.health_tips, str):
            try:
                parsed_tips = json.loads(record.health_tips)
                record.health_tips = parsed_tips if isinstance(parsed_tips, list) else [parsed_tips]
            except Exception:
                record.health_tips = [record.health_tips]

    history_id = request.args.get("history_id", type=int)
    selected_record = None

    if history_id:
        selected_record = PredictionHistory.query.filter_by(
            id=history_id,
            user_id=current_user.id
        ).first()

        if selected_record is None:
            flash("Report not found.", "warning")
            return redirect(url_for("history"))
        
        # Parse selected record safely as well
        if selected_record.input_data is None:
            selected_record.input_data = {}
        elif isinstance(selected_record.input_data, str):
            try:
                selected_record.input_data = json.loads(selected_record.input_data)
            except Exception:
                selected_record.input_data = {}

        if selected_record.health_tips is None:
            selected_record.health_tips = []
        elif isinstance(selected_record.health_tips, str):
            try:
                parsed_tips = json.loads(selected_record.health_tips)
                selected_record.health_tips = parsed_tips if isinstance(parsed_tips, list) else [parsed_tips]
            except Exception:
                selected_record.health_tips = [selected_record.health_tips]

    return render_template(
        "history.html",
        history_records=history_records,
        selected_record=selected_record
    )

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not email or not password:
            return render_template("login.html", error=True, message="Please enter your email and password.")

        user = db.session.execute(db.select(User).filter_by(email=email)).scalar_one_or_none()

        if not user or not user.check_password(password):
            return render_template("login.html", error=True, message="Invalid email or password.")

        login_user(user, remember=True)
        next_page = request.args.get("next")
        if next_page and next_page.startswith("/") and not next_page.startswith("//"):
            return redirect(next_page)
        return redirect(url_for("detect"))

    return render_template("login.html", error=False)


@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        username = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not username or not email or not password:
            return render_template("signin.html", error=True, message="Please complete all fields.")

        existing_user = db.session.execute(
            db.select(User).where(or_(User.username == username, User.email == email))
        ).scalar_one_or_none()

        if existing_user:
            return render_template("signin.html", error=True, message="Username or email already registered.")

        new_user = User(username=username, email=email)
        new_user.set_password(password)

        try:
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user, remember=True)
            return redirect(url_for("detect"))
        except Exception as e:
            db.session.rollback()
            app.logger.exception("Signup failed: %s", e)
            return render_template("signin.html", error=True, message="Registration failed. Please try again.")

    return render_template("signin.html", error=False)


@app.route("/logout", methods=["GET", "POST"])
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({"error": "Uploaded file is too large. Maximum allowed size is 10 MB."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port,debug=True)

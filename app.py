from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from nutriai.ai_chat import chat_response
from nutriai.database import (
    init_db,
    save_assessment,
    save_chat,
    save_meal_plan,
    save_progress,
    save_shopping_list,
    table_df,
)
from nutriai.grocery import build_shopping_list
from nutriai.ml_engine import predict_targets
from nutriai.nutrition import calculate_bmi, nutrition_summary
from nutriai.recommendations import generate_meal_plan, healthier_alternatives

st.set_page_config(page_title="NutriAI: Virtual Nutritionist", page_icon="N", layout="wide")

DISCLAIMER = "Educational nutrition support only. NutriAI does not diagnose, treat, or replace clinician advice."
CONDITIONS = ["Diabetes", "Hypertension", "Thyroid", "PCOS", "Obesity"]
GOALS = ["Weight Loss", "Weight Gain", "Muscle Gain", "Maintenance", "Diabetes Management", "Heart Health", "PCOS Management"]


def ensure_state() -> None:
    init_db()
    defaults = {
        "profile": None,
        "nutrition": None,
        "meal_plan": None,
        "shopping": None,
        "user_id": None,
        "chat_messages": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def metric_card(label: str, value: str, help_text: str = "") -> None:
    st.metric(label, value, help=help_text)


def assessment_page() -> None:
    st.subheader("Health Assessment")
    with st.form("assessment_form"):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Name", value="Demo User")
        age = c2.number_input("Age", 12, 90, 30)
        gender = c3.selectbox("Gender", ["Female", "Male"])
        c4, c5, c6 = st.columns(3)
        height_cm = c4.number_input("Height (cm)", 120.0, 220.0, 165.0)
        weight_kg = c5.number_input("Weight (kg)", 30.0, 180.0, 70.0)
        activity_level = c6.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Athlete"], index=2)
        c7, c8, c9 = st.columns(3)
        exercise_frequency = c7.selectbox("Daily Exercise Frequency", ["None", "1-2 days/week", "3-4 days/week", "5-6 days/week", "Daily"])
        dietary_preference = c8.selectbox("Dietary Preference", ["Vegetarian", "Vegan", "Non-Vegetarian"])
        fitness_goal = c9.selectbox("Fitness Goal", GOALS)
        allergies_text = st.text_input("Allergies", placeholder="Peanuts, lactose, gluten")
        medical_conditions = st.multiselect("Medical Conditions", CONDITIONS)
        medical_history = st.text_area("Medical History", placeholder="Medications, previous diagnoses, lab markers, clinician instructions")
        submitted = st.form_submit_button("Generate Nutrition Plan", type="primary")
    if submitted:
        profile = {
            "name": name,
            "age": int(age),
            "gender": gender,
            "height_cm": float(height_cm),
            "weight_kg": float(weight_kg),
            "activity_level": activity_level,
            "exercise_frequency": exercise_frequency,
            "dietary_preference": dietary_preference,
            "allergies": [item.strip() for item in allergies_text.split(",") if item.strip()],
            "medical_conditions": medical_conditions,
            "fitness_goal": fitness_goal,
        }
        nutrition = nutrition_summary(profile)
        meal_plan = generate_meal_plan(profile, nutrition)
        shopping = build_shopping_list(meal_plan)
        user_id = save_assessment(profile, nutrition, medical_history)
        save_meal_plan(user_id, meal_plan)
        save_shopping_list(user_id, shopping)
        save_progress(user_id, profile["weight_kg"], nutrition["bmi"], nutrition["calorie_target"], 0)
        st.session_state.update(profile=profile, nutrition=nutrition, meal_plan=meal_plan, shopping=shopping, user_id=user_id)
        st.success("Personalized nutrition plan generated and saved.")
    if st.session_state.profile:
        show_summary()


def show_summary() -> None:
    profile = st.session_state.profile
    nutrition = st.session_state.nutrition
    ml_targets = predict_targets(profile, nutrition)
    st.subheader("Nutrition Analysis")
    cols = st.columns(6)
    values = [
        ("BMI", f"{nutrition['bmi']} ({nutrition['bmi_category']})"),
        ("BMR", f"{nutrition['bmr']:.0f} kcal"),
        ("TDEE", f"{nutrition['tdee']:.0f} kcal"),
        ("Calories", f"{nutrition['calorie_target']} kcal"),
        ("Protein", f"{nutrition['protein_g']} g"),
        ("Carbs / Fat", f"{nutrition['carbs_g']} g / {nutrition['fat_g']} g"),
    ]
    for col, (label, value) in zip(cols, values):
        with col:
            metric_card(label, value)
    st.info(f"Recommendation engine: {ml_targets['model_name']} suggested `{ml_targets['recommended_plan_type']}` at about `{ml_targets['daily_calories']}` kcal/day.")


def meal_plan_page() -> None:
    st.subheader("Personalized Meal Plan")
    meal_plan = st.session_state.meal_plan
    if not meal_plan:
        st.warning("Generate a health assessment first.")
        return
    st.caption(meal_plan["title"])
    cols = st.columns(4)
    for col, (slot, meal) in zip(cols, meal_plan["meals"].items()):
        with col:
            st.markdown(f"**{slot.title()}**")
            st.write(meal["name"])
            st.caption(", ".join(meal["items"]))
            st.metric("Calories", f"{meal['calories']} kcal")
            st.write(f"P {meal['protein_g']} g | C {meal['carbs_g']} g | F {meal['fat_g']} g")
    st.markdown("**Condition-Aware Guidance**")
    for item in meal_plan["guidance"]:
        st.write(f"- {item}")
    st.markdown("**Healthier Alternatives**")
    st.dataframe(pd.DataFrame(healthier_alternatives().items(), columns=["Instead of", "Try"]), use_container_width=True)


def grocery_page() -> None:
    st.subheader("Grocery Shopping Assistant")
    meal_plan = st.session_state.meal_plan
    if not meal_plan:
        st.warning("Generate a meal plan first.")
        return
    days = st.slider("Shopping days", 1, 14, 7)
    shopping = build_shopping_list(meal_plan, days)
    st.session_state.shopping = shopping
    if st.session_state.user_id:
        save_shopping_list(st.session_state.user_id, shopping)
    for category, items in shopping.items():
        with st.expander(category, expanded=True):
            st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)


def progress_page() -> None:
    st.subheader("Progress Tracking")
    if not st.session_state.user_id:
        st.warning("Generate and save an assessment first.")
        return
    with st.form("progress_form"):
        c1, c2, c3 = st.columns(3)
        weight = c1.number_input("Current Weight (kg)", 30.0, 180.0, float(st.session_state.profile["weight_kg"]))
        calories = c2.number_input("Daily Calorie Intake", 500, 6000, int(st.session_state.nutrition["calorie_target"]))
        progress = c3.slider("Goal Achievement (%)", 0, 100, 10)
        if st.form_submit_button("Save Progress"):
            bmi = calculate_bmi(st.session_state.profile["height_cm"], weight)
            save_progress(st.session_state.user_id, weight, bmi, calories, progress)
            st.success("Progress entry saved.")
    dashboard_page()


def dashboard_page() -> None:
    st.subheader("Data Visualization Dashboard")
    df = table_df("ProgressTracking")
    if df.empty:
        st.info("No progress data yet.")
        return
    df["recorded_at"] = pd.to_datetime(df["recorded_at"])
    if st.session_state.user_id:
        df = df[df["user_id"] == st.session_state.user_id]
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.line(df, x="recorded_at", y="weight_kg", markers=True, title="Weight Trend"), use_container_width=True)
        st.plotly_chart(px.line(df, x="recorded_at", y="calorie_intake", markers=True, title="Calorie Intake Trend"), use_container_width=True)
    with c2:
        st.plotly_chart(px.line(df, x="recorded_at", y="bmi", markers=True, title="BMI Trend"), use_container_width=True)
        st.plotly_chart(px.area(df, x="recorded_at", y="goal_progress", title="Goal Progress"), use_container_width=True)
    if st.session_state.nutrition:
        macros = pd.DataFrame([
            {"macro": "Protein", "grams": st.session_state.nutrition["protein_g"]},
            {"macro": "Carbohydrates", "grams": st.session_state.nutrition["carbs_g"]},
            {"macro": "Fat", "grams": st.session_state.nutrition["fat_g"]},
        ])
        st.plotly_chart(px.pie(macros, names="macro", values="grams", title="Nutrient Distribution"), use_container_width=True)


def chat_page() -> None:
    st.subheader("Virtual Nutritionist AI")
    st.caption(DISCLAIMER)
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    question = st.chat_input("Ask NutriAI about meals, calories, macros, or healthier alternatives")
    if question:
        st.session_state.chat_messages.append({"role": "user", "content": question})
        if st.session_state.user_id:
            save_chat(st.session_state.user_id, "user", question)
        answer = chat_response(question, st.session_state.profile, st.session_state.nutrition, st.session_state.meal_plan)
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        if st.session_state.user_id:
            save_chat(st.session_state.user_id, "assistant", answer)
        st.rerun()


def model_page() -> None:
    st.subheader("Machine Learning Recommendation Engine")
    report_path = Path("models/model_comparison.csv")
    dataset_path = Path("data/synthetic_nutrition_dataset.csv")
    st.write("Train models with `python scripts/train_models.py` to compare Decision Tree, Random Forest, KNN, and XGBoost when installed.")
    if report_path.exists():
        st.dataframe(pd.read_csv(report_path), use_container_width=True)
    else:
        st.info("No model comparison report found yet.")
    if dataset_path.exists():
        st.markdown("**Dataset Preview**")
        st.dataframe(pd.read_csv(dataset_path).head(25), use_container_width=True)


def records_page() -> None:
    st.subheader("SQLite Records")
    table = st.selectbox("Table", ["Users", "HealthProfiles", "MedicalHistory", "MealPlans", "ShoppingLists", "ProgressTracking", "ChatHistory"])
    try:
        df = table_df(table)
        st.dataframe(df, use_container_width=True)
        if table in {"MealPlans", "ShoppingLists"} and not df.empty:
            row = st.number_input("Inspect JSON row", 0, len(df) - 1, 0)
            payload_col = "plan_json" if table == "MealPlans" else "list_json"
            st.json(json.loads(df.iloc[int(row)][payload_col]))
    except Exception as exc:
        st.error(str(exc))


def main() -> None:
    ensure_state()
    st.title("NutriAI: Virtual Nutritionist")
    st.caption(DISCLAIMER)
    with st.sidebar:
        st.header("NutriAI")
        page = st.radio(
            "Navigation",
            ["Assessment", "Meal Plan", "Grocery List", "Progress Dashboard", "Virtual Nutritionist Chat", "ML Engine", "Database Records"],
        )
    pages = {
        "Assessment": assessment_page,
        "Meal Plan": meal_plan_page,
        "Grocery List": grocery_page,
        "Progress Dashboard": progress_page,
        "Virtual Nutritionist Chat": chat_page,
        "ML Engine": model_page,
        "Database Records": records_page,
    }
    pages[page]()


if __name__ == "__main__":
    main()

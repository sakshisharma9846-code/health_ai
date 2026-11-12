import json
import os
import re
from functools import wraps

from flask import (Flask, flash, jsonify, redirect, render_template, request,
                   session, url_for)

from ml_model import SymptomModel

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-me")

with open(os.path.join(BASE, "data", "problems.json"), encoding="utf-8") as f:
    PROBLEMS = json.load(f)
BY_ID = {p["id"]: p for p in PROBLEMS}
model = SymptomModel(PROBLEMS)  # trains in about a second at start-up
print(f"ML model ready. Held-out accuracy (synthetic data): {model.accuracy:.2%}")

EMERGENCY = re.compile(
    r"chest pain|can.?t breathe|difficulty breathing|unconscious|severe bleeding|"
    r"stroke|seizure|poison|overdose|suicid|kill myself|self.?harm", re.I)
EMERGENCY_MSG = ("This could be an emergency. Please call 112 (India) or go to the nearest hospital now. "
                 "If you are thinking about harming yourself, call Tele-MANAS 14416 or talk to someone you trust right away.")


def age_group(age):
    if age < 12:
        return dict(kid=True, label="Child (under 12)",
                    med="Children should not take medicines on their own. Doses depend on weight and the exact product, so always ask a paediatrician or pharmacist. Never give aspirin to a child.",
                    life="Sleep 9–11 hours, play outdoors for at least an hour, limit screen time, eat balanced home-cooked meals, drink enough water and wash hands often.")
    if age < 18:
        return dict(kid=False, label="Teenager (12–17)",
                    med="Teenagers should use any medicine only with a parent’s or doctor’s approval and follow the label. Avoid aspirin. Do not use pain gels or other medicines regularly without advice.",
                    life="Sleep 8–10 hours, avoid energy drinks and junk food, stay active for 60 minutes daily, keep screens off before bed and talk to a trusted adult about stress.")
    if age < 60:
        return dict(kid=False, label="Adult (18–59)",
                    med="Adults can generally use the medicine above as per the label. Do not exceed the dose, and ask a doctor first if you are pregnant or breastfeeding, have liver, kidney, stomach ulcer or asthma problems, or are allergic to it.",
                    life="Sleep 7–8 hours, exercise about 150 minutes a week, limit caffeine, alcohol and smoking, eat regular balanced meals, take posture breaks and get a yearly health check-up.")
    return dict(kid=False, label="Senior (60+)",
                med="Older adults are more sensitive to medicines. Paracetamol-type options are usually safer. Avoid diclofenac and similar painkillers without a doctor’s advice (risk to kidneys, stomach and blood pressure). Cetirizine can cause drowsiness and falls. Check for interactions with your BP, sugar or heart medicines.",
                life="Walk or do gentle yoga daily, get calcium, vitamin D and sunlight, check BP and sugar regularly, stay hydrated, make your home fall-safe, never stop prescribed medicines on your own and stay socially connected.")


def bmi_text(user):
    try:
        b = user["weight"] / (user["height"] / 100) ** 2
    except (KeyError, ZeroDivisionError):
        return ""
    cat = "underweight" if b < 18.5 else "normal" if b < 25 else "overweight" if b < 30 else "in the obesity range"
    return f"Your BMI is {b:.1f} ({cat})."


def login_required(fn):
    @wraps(fn)
    def wrapper(*a, **k):
        if "user" not in session:
            return redirect(url_for("login"))
        return fn(*a, **k)
    return wrapper


def info_required(fn):
    @wraps(fn)
    @login_required
    def wrapper(*a, **k):
        if "age" not in session["user"]:
            return redirect(url_for("info"))
        return fn(*a, **k)
    return wrapper


@app.route("/")
def home():
    return render_template("index.html", count=len(PROBLEMS))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        if not name:
            flash("Please enter your name.")
        elif not re.match(r"^\S+@\S+\.\S+$", email):
            flash("Please enter a valid email.")
        elif not re.match(r"^[6-9]\d{9}$", phone):
            flash("Please enter a valid 10-digit phone number.")
        else:
            session["user"] = dict(session.get("user", {}), name=name, email=email, phone=phone)
            return redirect(url_for("info"))
    return render_template("login.html", u=session.get("user", {}))


@app.route("/info", methods=["GET", "POST"])
@login_required
def info():
    if request.method == "POST":
        try:
            age = int(request.form["age"]); weight = float(request.form["weight"])
            height = float(request.form["height"]); sleep = float(request.form["sleep"])
            assert 1 <= age <= 120 and 1 <= weight <= 300 and 50 <= height <= 250 and 0 <= sleep <= 24
        except (ValueError, KeyError, AssertionError):
            flash("Please enter valid values in all fields.")
        else:
            session["user"] = dict(session["user"], age=age, weight=weight, height=height, sleep=sleep)
            return redirect(url_for("problems"))
    return render_template("info.html", u=session["user"])


@app.route("/problems")
@info_required
def problems():
    u = session["user"]
    tips = [bmi_text(u)]
    tips.append(("You sleep %g hours. Getting 7–8 hours helps reduce fatigue, headaches and stress." % u["sleep"])
                if u["sleep"] < 7 else "Your sleep (%g hours) is in a healthy range." % u["sleep"])
    return render_template("problems.html", problems=PROBLEMS, tips=tips, g=age_group(u["age"]), u=u)


@app.route("/problem/<int:pid>")
@info_required
def problem(pid):
    p = BY_ID.get(pid)
    if not p:
        return redirect(url_for("problems"))
    u = session["user"]
    return render_template("problem.html", p=p, g=age_group(u["age"]), u=u)


@app.route("/checker", methods=["GET", "POST"])
@info_required
def checker():
    results, text = [], ""
    if request.method == "POST":
        text = request.form.get("symptoms", "").strip()
        if text:
            results = [(BY_ID[i], round(pr * 100)) for i, pr in model.predict(text)]
    return render_template("checker.html", results=results, text=text,
                           emergency=bool(EMERGENCY.search(text)), msg=EMERGENCY_MSG)


@app.route("/chat")
@info_required
def chat():
    return render_template("chat.html", ai_on=bool(os.getenv("ANTHROPIC_API_KEY")))


def offline_reply(text):
    pid, prob = model.predict(text, 1)[0]
    p = BY_ID[pid]
    return (f"(Offline mode) Your description looks closest to **{p['name']}**.\n"
            f"Possible causes: {p['causes']}\nWhat can help: {p['treat']}\n"
            "This is general information, not a diagnosis. Please see a doctor if it is serious or does not improve.")


@app.route("/api/chat", methods=["POST"])
@info_required
def api_chat():
    data = request.get_json(force=True)
    text = str(data.get("message", "")).strip()[:1000]
    if not text:
        return jsonify(reply="Please type a message.")
    if EMERGENCY.search(text):
        return jsonify(reply=EMERGENCY_MSG)
    history = [m for m in data.get("history", [])[-10:] if m.get("role") in ("user", "assistant")]
    u = session["user"]
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            system = ("You are a friendly health-information assistant inside a student project website. "
                      f"The user is {u['age']} years old, {u['weight']} kg, {u['height']} cm, sleeps {u['sleep']} h/day. "
                      "Give general information, home care, lifestyle tips and clear signs for when to see a doctor. "
                      "Never diagnose. Do not give medicine doses for children; tell them to ask a paediatrician. "
                      "Remind the user that you are not a doctor. Keep answers under 150 words, in simple English.")
            msgs = [{"role": m["role"], "content": str(m["content"])[:1000]} for m in history]
            msgs.append({"role": "user", "content": text})
            r = anthropic.Anthropic().messages.create(
                model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"),
                max_tokens=500, system=system, messages=msgs)
            return jsonify(reply=r.content[0].text)
        except Exception as e:  # fall back so the demo never breaks
            print("AI error:", e)
    return jsonify(reply=offline_reply(text))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)

"""Goluuu Bot Core API - savage Hinglish gym & fitness coach backend."""
import base64
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Optional

import httpx
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import (Column, DateTime, Float, ForeignKey, Integer, String,
                        Text, create_engine)
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:////tmp/goluuu.db")  # /tmp = serverless-safe
API_KEY = os.getenv("GOLUUU_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Ordered failover chain; override with GEMINI_MODELS="a,b,c"
GEMINI_MODELS = [m.strip() for m in os.getenv(
    "GEMINI_MODELS", "gemini-3.8-flash,gemini-2.5-flash").split(",") if m.strip()]

# --------------------------------------------------------------------------
# Fancy font engine  ("Claim" -> "Cʟαιм")
# First letter of each word is kept, the rest are lowercased and mapped.
# --------------------------------------------------------------------------
_FANCY = dict(zip(
    "abcdefghijklmnopqrstuvwxyz",
    "αʙᴄᴅєꜰɢнιᴊᴋʟмησρǫʀѕтυᴠωxʏᴢ"))


def fancy(text: str) -> str:
    out, new_word = [], True
    for ch in text:
        if ch.isalpha() and ch.isascii():
            out.append(ch if new_word else _FANCY[ch.lower()])
            new_word = False
        else:
            out.append(ch)
            new_word = not (ch.isalnum() or ch in "'’")
    return "".join(out)


# --------------------------------------------------------------------------
# AES-256-GCM encryption
# --------------------------------------------------------------------------
def _load_key() -> bytes:
    raw = os.getenv("GOLUUU_AES_KEY")  # base64 of 32 random bytes
    if raw:
        key = base64.b64decode(raw)
        if len(key) != 32:
            raise RuntimeError("GOLUUU_AES_KEY must decode to exactly 32 bytes")
        return key
    secret = os.getenv("GOLUUU_SECRET")
    if not secret:
        raise RuntimeError("Set GOLUUU_AES_KEY (preferred) or GOLUUU_SECRET")
    salt = os.getenv("GOLUUU_SALT", "goluuu-v1").encode()
    return hashlib.scrypt(secret.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)


_AES = None


def _aes() -> AESGCM:
    global _AES
    if _AES is None:
        _AES = AESGCM(_load_key())
    return _AES


def encrypt(plain: str) -> str:
    nonce = os.urandom(12)
    return base64.b64encode(nonce + _aes().encrypt(nonce, plain.encode(), None)).decode()


def decrypt(token: str) -> str:
    blob = base64.b64decode(token)
    return _aes().decrypt(blob[:12], blob[12:], None).decode()


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False)
Base = declarative_base()


def _now():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    telegram_id = Column(String(32), unique=True, index=True, nullable=False)
    name = Column(String(80), nullable=False)
    age = Column(Integer, nullable=False)
    sex = Column(String(1), nullable=False)          # "m" / "f"
    height_cm = Column(Float, nullable=False)
    weight_kg = Column(Float, nullable=False)
    activity = Column(String(16), default="moderate")
    goal = Column(String(8), default="maintain")     # cut / maintain / bulk
    reason_enc = Column(Text)                        # AES-256-GCM encrypted
    created_at = Column(DateTime(timezone=True), default=_now)
    workouts = relationship("WorkoutLog", back_populates="user", cascade="all, delete-orphan")
    diets = relationship("DietPlan", back_populates="user", cascade="all, delete-orphan")


class WorkoutLog(Base):
    __tablename__ = "workout_logs"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    exercise = Column(String(80), nullable=False)
    sets = Column(Integer, nullable=False)
    reps = Column(Integer, nullable=False)
    weight_kg = Column(Float, default=0)
    notes_enc = Column(Text)
    logged_at = Column(DateTime(timezone=True), default=_now)
    user = relationship("User", back_populates="workouts")


class DietPlan(Base):
    __tablename__ = "diet_plans"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    calories = Column(Integer)
    protein_g = Column(Integer)
    carbs_g = Column(Integer)
    fats_g = Column(Integer)
    plan_text = Column(Text)
    created_at = Column(DateTime(timezone=True), default=_now)
    user = relationship("User", back_populates="diets")


Base.metadata.create_all(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------------------------------
# Metabolic computations
# --------------------------------------------------------------------------
ACTIVITY = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725, "athlete": 1.9}
GOAL_ADJ = {"cut": 0.80, "maintain": 1.0, "bulk": 1.10}


def compute_metrics(sex: str, age: int, height_cm: float, weight_kg: float,
                    activity: str, goal: str) -> dict:
    """Mifflin-St Jeor BMR, TDEE, macros, BMI."""
    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + (5 if sex == "m" else -161)
    tdee = bmr * ACTIVITY[activity]
    calories = tdee * GOAL_ADJ[goal]
    protein = weight_kg * (2.2 if goal == "cut" else 2.0)
    fats = calories * 0.25 / 9
    carbs = max((calories - protein * 4 - fats * 9) / 4, 0)
    bmi = weight_kg / ((height_cm / 100) ** 2)
    cat = ("underweight" if bmi < 18.5 else "normal" if bmi < 25
           else "overweight" if bmi < 30 else "obese")
    return {"bmr": round(bmr), "tdee": round(tdee), "calories": round(calories),
            "protein_g": round(protein), "carbs_g": round(carbs), "fats_g": round(fats),
            "bmi": round(bmi, 1), "bmi_category": cat}


# --------------------------------------------------------------------------
# Gemini with model failover
# --------------------------------------------------------------------------
PERSONA = (
    "You are Goluuu, a savage Hinglish (Hindi+English, Roman script) gym & fitness coach. "
    "Roast the user's excuses, laziness and bad form with aggressive gym-bro humour, then give "
    "one concrete fix. Never insult body, caste, religion, gender or family. Keep it short and "
    "punchy. If the user mentions an eating disorder, self-harm, injury or pain, drop the roast, "
    "be kind and suggest seeing a doctor or professional.")


async def gemini(prompt: str, image: Optional[tuple] = None, system: str = PERSONA) -> str:
    if not GEMINI_API_KEY:
        raise HTTPException(503, "GEMINI_API_KEY not configured")
    parts = [{"text": prompt}]
    if image:
        mime, data = image
        parts.append({"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode()}})
    body = {"system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}]}
    errors = []
    async with httpx.AsyncClient(timeout=30) as client:
        for model in GEMINI_MODELS:
            try:
                r = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    headers={"x-goog-api-key": GEMINI_API_KEY}, json=body)
                r.raise_for_status()
                text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                return text.strip()
            except Exception as e:  # try next model on any failure
                errors.append(f"{model}: {type(e).__name__}")
    raise HTTPException(502, f"All Gemini models failed ({'; '.join(errors)})")


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
def auth(x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Invalid API key")


app = FastAPI(title="Goluuu Bot Core API", dependencies=[Depends(auth)])


class UserIn(BaseModel):
    telegram_id: str
    name: str
    age: int = Field(ge=13, le=100)
    sex: str = Field(pattern="^[mf]$")
    height_cm: float = Field(ge=100, le=250)
    weight_kg: float = Field(ge=30, le=300)
    activity: str = "moderate"
    goal: str = "maintain"
    reason: Optional[str] = None  # transformation reason, stored encrypted


class WorkoutIn(BaseModel):
    exercise: str
    sets: int = Field(ge=1, le=50)
    reps: int = Field(ge=1, le=500)
    weight_kg: float = Field(default=0, ge=0)
    notes: Optional[str] = None


def _user(db: Session, tg: str) -> User:
    u = db.query(User).filter_by(telegram_id=tg).first()
    if not u:
        raise HTTPException(404, "User not found")
    return u


def _metrics(u: User) -> dict:
    return compute_metrics(u.sex, u.age, u.height_cm, u.weight_kg, u.activity, u.goal)


@app.get("/health")
def health():
    return {"status": "ok", "models": GEMINI_MODELS}


@app.post("/users")
def upsert_user(p: UserIn, db: Session = Depends(get_db)):
    if p.activity not in ACTIVITY or p.goal not in GOAL_ADJ:
        raise HTTPException(422, f"activity in {list(ACTIVITY)}, goal in {list(GOAL_ADJ)}")
    u = db.query(User).filter_by(telegram_id=p.telegram_id).first() or User(telegram_id=p.telegram_id)
    for f in ("name", "age", "sex", "height_cm", "weight_kg", "activity", "goal"):
        setattr(u, f, getattr(p, f))
    if p.reason:
        u.reason_enc = encrypt(p.reason)
    db.add(u)
    db.commit()
    return {"telegram_id": u.telegram_id, "metrics": _metrics(u),
            "message": fancy(f"Profile saved, {u.name}. Ab bahane band, kaam shuru.")}


@app.get("/users/{tg}")
def get_user(tg: str, include_reason: bool = False, db: Session = Depends(get_db)):
    u = _user(db, tg)
    out = {c.name: getattr(u, c.name) for c in u.__table__.columns if c.name != "reason_enc"}
    out["created_at"] = str(u.created_at)
    out["metrics"] = _metrics(u)
    if include_reason and u.reason_enc:
        out["reason"] = decrypt(u.reason_enc)
    return out


@app.post("/users/{tg}/workouts")
def log_workout(tg: str, p: WorkoutIn, db: Session = Depends(get_db)):
    u = _user(db, tg)
    w = WorkoutLog(user_id=u.id, exercise=p.exercise, sets=p.sets, reps=p.reps,
                   weight_kg=p.weight_kg, notes_enc=encrypt(p.notes) if p.notes else None)
    db.add(w)
    db.commit()
    return {"id": w.id, "volume_kg": p.sets * p.reps * p.weight_kg,
            "message": fancy("Logged. Jo likha nahi, woh hua nahi.")}


@app.get("/users/{tg}/workouts")
def list_workouts(tg: str, limit: int = 30, db: Session = Depends(get_db)):
    u = _user(db, tg)
    rows = (db.query(WorkoutLog).filter_by(user_id=u.id)
            .order_by(WorkoutLog.logged_at.desc()).limit(min(limit, 200)).all())
    return [{"id": w.id, "exercise": w.exercise, "sets": w.sets, "reps": w.reps,
             "weight_kg": w.weight_kg, "notes": decrypt(w.notes_enc) if w.notes_enc else None,
             "logged_at": str(w.logged_at)} for w in rows]


@app.post("/users/{tg}/diet")
async def generate_diet(tg: str, preference: str = "veg", db: Session = Depends(get_db)):
    u = _user(db, tg)
    m = _metrics(u)
    prompt = (f"Make a 1-day Indian {preference} meal plan hitting {m['calories']} kcal, "
              f"{m['protein_g']}g protein, {m['carbs_g']}g carbs, {m['fats_g']}g fat. "
              "List meals with rough quantities in a compact list, end with one savage line.")
    text = fancy(await gemini(prompt))
    d = DietPlan(user_id=u.id, calories=m["calories"], protein_g=m["protein_g"],
                 carbs_g=m["carbs_g"], fats_g=m["fats_g"], plan_text=text)
    db.add(d)
    db.commit()
    return {"id": d.id, "macros": m, "plan": text}


@app.post("/users/{tg}/roast")
async def roast(tg: str, situation: str, db: Session = Depends(get_db)):
    u = _user(db, tg)
    reply = await gemini(f"User {u.name} (goal: {u.goal}) says: {situation}\nRoast and motivate.")
    return {"reply": fancy(reply)}


@app.post("/users/{tg}/form-check")
async def form_check(tg: str, exercise: str = Form(...), description: str = Form(""),
                     media: Optional[UploadFile] = File(None), db: Session = Depends(get_db)):
    u = _user(db, tg)
    image = None
    if media:
        data = await media.read()
        if len(data) > 10 * 1024 * 1024:
            raise HTTPException(413, "Max 10MB")
        image = (media.content_type or "image/jpeg", data)
    reply = await gemini(
        f"Exercise: {exercise}. User note: {description or 'none'}. "
        "Judge the form, roast mistakes, give max 3 fixes.", image=image)
    return {"reply": fancy(reply)}

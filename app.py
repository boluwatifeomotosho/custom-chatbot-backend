from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow import keras
import random
import logging
from datetime import datetime
import os
import re
import requests
import traceback
from config import get_config

from database import init_db, save_user_context, load_user_context, load_all_memory

def merge_learned_intents_into_json():
    """Merge newly learned patterns into intents.json dynamically."""
    from database import get_all_learned_intents
    learned = get_all_learned_intents()

    if not learned:
        logger.info("No learned intents found to merge.")
        return False

    with open("intents.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Add new intent for each learned item
    for item in learned:
        found = False
        for intent in data["intents"]:
            if intent["tag"] == "learned":
                intent["patterns"].append(item["pattern"])
                intent["responses"].append(item["response"])
                found = True
                break

        if not found:
            data["intents"].append({
                "tag": "learned",
                "patterns": [item["pattern"]],
                "responses": [item["response"]]
            })

    # Write updated intents back to file
    with open("intents.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Merged {len(learned)} learned intents into intents.json")
    return True



# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# ------------------------------
# Setup
# ------------------------------
Config = get_config()

app = Flask(__name__)
app.config.from_object(Config)
CORS(app, resources={r"/api/*": {"origins": Config.CORS_ORIGINS}})
logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)

lemmatizer = WordNetLemmatizer()

# Globals
model = None
words = None
embedding_model = None
classes = None
intents = None
user_context = {}
special_names = {}
MEMORY_FILE = "user_memory.json"


# ------------------------------
# Persistent Memory Functions
# ------------------------------
def save_memory():
    try:
        for user_id, context in user_context.items():
            save_user_context(user_id, context)
    except Exception as e:
        logger.error(f"Error saving memory: {e}")



def load_memory():
    global user_context
    try:
        user_context = load_all_memory()
        logger.info(f"✅ Loaded {len(user_context)} users from SQLite memory")
    except Exception as e:
        logger.error(f"Error loading memory: {e}")



# ------------------------------
# Model + Data
# ------------------------------
def load_model_and_data():
    global model, words, classes, intents, special_names
    try:
        # Reset globals first
        model = None
        words = None
        classes = None
        intents = None


        # Load intents
        if os.path.exists("intents.json"): # This can stay as it's in the Backend folder
            with open("intents.json", "r", encoding='utf-8') as file:
                intents = json.load(file)
                logger.info("✅ Intents loaded successfully")
        else:
            logger.error("❌ intents.json not found!")
            return False

        # Load words
        if os.path.exists("Models/words.pkl"):
            with open("Models/words.pkl", "rb") as f:
                words = pickle.load(f)
                logger.info(f"✅ Words loaded: {len(words) if words else 0} words")
        else:
            logger.error("❌ Models/words.pkl not found! Please train the model first.")
            return False

        # Load classes
        if os.path.exists("Models/classes.pkl"):
            with open("Models/classes.pkl", "rb") as f:
                classes = pickle.load(f)
                logger.info(f"✅ Classes loaded: {len(classes) if classes else 0} classes")
        else:
            logger.error("❌ Models/classes.pkl not found! Please train the model first.")
            return False

        # Load model
        if os.path.exists("Models/chatbot_model.h5"):
            model = keras.models.load_model("Models/chatbot_model.h5")
            # prepare_intent_embeddings() # Defer this to lazy loading
            logger.info("✅ Model loaded successfully")
        else:
            logger.error("❌ chatbot_model.h5 not found! Please train the model first.")
            return False

        # Load special names
        if os.path.exists("special_names.json"):
            with open("special_names.json", "r", encoding='utf-8') as file:
                special_names = json.load(file)
                logger.info(f"✅ Loaded {len(special_names)} special name responses")
        else:
            logger.warning("⚠️ special_names.json not found, special name replies will be disabled.")
            special_names = {}

        load_memory()
        logger.info("🎉 All components loaded successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Error loading model/data: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

from sentence_transformers import SentenceTransformer, util

intent_embeddings = []

def prepare_intent_embeddings():
    global intent_embeddings, embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("✅ Embedding model loaded successfully")
    intent_embeddings = []

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            emb = embedding_model.encode(pattern, convert_to_tensor=True)
            intent_embeddings.append({
                "intent": intent["tag"],
                "pattern": pattern,
                "embedding": emb
            })
    logger.info(f"✅ Precomputed {len(intent_embeddings)} intent embeddings for fallback matching")

# ------------------------------
# NLP Utilities
# ------------------------------
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]


def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    logger.info(f"Found in bag: {w}")
    return np.array(bag)


def predict_class(sentence, model):
    if model is None or words is None or classes is None:
        logger.warning("Model, words, or classes not loaded properly")
        return [{"intent": "fallback", "probability": "1.0"}]

    try:
        p = bow(sentence, words)
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)

        predicted_intents = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

        # Debug logging
        logger.info(f"Input: '{sentence}' -> Predicted intents: {predicted_intents}")

        return predicted_intents or [{"intent": "fallback", "probability": "1.0"}]
    except Exception as e:
        logger.error(f"Error in predict_class: {str(e)}")
        return [{"intent": "fallback", "probability": "1.0"}]

def semantic_fallback(sentence, threshold=0.55):
    """
    When model confidence is too low, use semantic similarity to find the best-matching intent.
    """
    global embedding_model
    # Lazy load the embedding model on first use
    if embedding_model is None:
        logger.info("Lazy loading embedding model for semantic fallback...")
        prepare_intent_embeddings()

    if not intent_embeddings or embedding_model is None:
        return [{"intent": "fallback", "probability": "1.0"}]

    query_emb = embedding_model.encode(sentence, convert_to_tensor=True) # type: ignore
    best_match = None
    best_score = 0

    for item in intent_embeddings:
        score = util.cos_sim(query_emb, item["embedding"]).item()
        if score > best_score:
            best_score = score
            best_match = item

    if best_score >= threshold and best_match:
        return [{"intent": best_match["intent"], "probability": str(best_score)}]

    return [{"intent": "fallback", "probability": "1.0"}]



# ------------------------------
# Dynamic Response Helpers
# ------------------------------
def get_weather(city="Lagos"):
    try:
        api_key = app.config.get("OPENWEATHER_API_KEY", "")
        if not api_key:
            return "Weather info's off for now 😅"
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url).json()
        if "main" in response:
            temp = response["main"]["temp"]
            desc = response["weather"][0]["description"]
            return f"Right now in {city}, it's {temp}°C with {desc}."
        return "Couldn't fetch weather data."
    except Exception:
        return "Weather API kinda tweaking rn 💀"


def get_time():
    now = datetime.now()
    return f"It's {now.strftime('%I:%M %p')} already."


def get_joke():
    jokes = [
        "Why do programmers hate nature? Too many bugs 😂",
        "My code works… I have no idea why 😭",
        "You're like a missing semicolon — crucial but chaotic 😏"
    ]
    return random.choice(jokes)


def get_compliment():
    compliments = [
        "You're honestly built different 🔥",
        "You've got that main character vibe 😎",
        "You're smarter than you think, no cap 💯"
    ]
    return random.choice(compliments)


# ------------------------------
# Bolu Personality Tone
# ------------------------------
def bolu_tone(response):
    endings = ["😂", "😭", "😏", "💀", "😮‍💨"]
    slang_map = {
        "hello": "yo",
        "hi": "wazza",
        "alright": "aiit",
        "goodbye": "later",
        "thanks": "appreciate it dawg",
        "yes": "yeah",
        "no": "nahhh",
        "okay": "aiit bet"
    }

    for k, v in slang_map.items():
        response = re.sub(rf"\b{k}\b", v, response, flags=re.IGNORECASE)

    if random.random() < 0.4:
        response += " " + random.choice(endings)
    if random.random() < 0.3:
        response = "" + response
    return response


# ------------------------------
# Context + Response Logic
# ------------------------------

def get_response(ints, intents_json, user_id, msg):
    # Validate ints parameter
    if not ints or len(ints) == 0:
        # Check learned responses before giving fallback
        from database import get_all_learned_intents
        learned_data = get_all_learned_intents()

        for item in learned_data:
            if item["pattern"].lower() in msg.lower():
                return bolu_tone(item["response"])

        return bolu_tone("I'm not too sure what you mean.")

    tag = ints[0]["intent"]
    context = user_context.get(user_id, {})
    mood = context.get("mood")
    name = context.get("name")
    last_intent = context.get("last_intent")

    logger.info(f"Processing intent: {tag} for user: {user_id}")

    # Handle continuation (context carryover)
    if "what about" in msg.lower() and last_intent == "weather":
        return bolu_tone(get_weather())

    result = None # Initialize result

    # Dynamic function mapping
    dynamic_responses = {
        "weather": get_weather,
        "time": get_time,
        "joke": get_joke,
        "compliment": get_compliment,
    }

    if tag in dynamic_responses:
        result = dynamic_responses[tag]()
    elif tag == "greeting":
            if name:
                if mood == "sad":
                    result = f"yo {name}, feeling better today? 😅"
                elif mood == "happy":
                    result = f"yo {name}! still vibing high today 🔥"
                else:
                    result = f"yo {name}! wassup 😎"
            else:
                result = random.choice(["yo!", "hey there 👋", "wassup!", "how's it going?"])
    else:
        # Default response from intents.json
        if intents_json and "intents" in intents_json: # Check if intents_json is not None
            for i in intents_json.get("intents", []):
                if i.get("tag") == tag:
                    # Check for mood-specific responses first
                    if mood and mood in i:
                        result = random.choice(i[mood])
                    elif "responses" in i: # Fallback to general responses
                        result = random.choice(i["responses"])
                    break # Exit loop once tag is found

    res = bolu_tone(result)

    # Friendship flavor
    if len(context.get("history", [])) > 15:
        res = "Ayy we're basically besties now 😂 " + res

    user_context[user_id]["last_intent"] = tag
    save_memory()
    return res


# ------------------------------
# Core Chat Logic
# ------------------------------
def chatbot_response(msg, user_id="default"):
    try:
        if user_id not in user_context:
            user_context[user_id] = {"history": [], "mood": None, "pending_learn": None}

        # 🧠 If user is teaching the bot something new
        from database import store_learned_intent, get_learned_count

        # 1️⃣ If the user is confirming a pending learn
        pending = user_context[user_id].get("pending_learn")
        if pending:
            if any(word in msg.lower() for word in ["yes", "yeah", "sure", "yup", "ok", "of course"]):
                # Save it permanently
                store_learned_intent(user_id, pending["pattern"], msg)
                user_context[user_id]["pending_learn"] = None
                save_memory()

                count = get_learned_count()
                response = f"Got it! I’ll now reply with that whenever someone says '{pending['pattern']}' 😎"

                if count % 10 == 0:
                    response += "\nAlso, I’ve learned quite a bit lately 👀 — retraining myself now..."
                    try:
                        merge_learned_intents_into_json()
                        from train_model import train_chatbot_model
                        train_chatbot_model()
                        load_model_and_data()
                        response += "\n✅ Done! I’m now smarter than before 🔥"
                    except Exception as e:
                        logger.error(f"Auto-retrain failed: {e}")
                        response += "\n⚠️ Tried to retrain, but something went wrong 😅"

                user_context[user_id]["history"].append({"user": msg, "bot": response})
                save_memory()
                return bolu_tone(response)

            elif any(word in msg.lower() for word in ["no", "nah", "nope", "cancel"]):
                user_context[user_id]["pending_learn"] = None
                save_memory()
                return bolu_tone("Aiit, won’t learn that then 😅")

            # This part seems to be a continuation of the teaching flow
            prev_input = user_context[user_id]["pending_learn"]["pattern"]
            user_context[user_id]["pending_learn"]["response"] = msg
            save_memory()
            return bolu_tone(f"So you want me to say '{msg}' when someone says '{prev_input}', right? 😏 (yes/no)")

        # 2️⃣ Detect if the bot didn’t understand and user is teaching something new
        if "don’t understand" in msg.lower() or "teach" in msg.lower():
            user_context[user_id]["pending_learn"] = {"pattern": user_context[user_id].get("last_user_input", msg)}
            save_memory()
            return bolu_tone("Aiit bet. What should I reply when someone says that? 🤔")

        user_context[user_id]["last_user_input"] = msg

        # -----------------------------
        # 🔹 Step 1: Smart Name Detection (runs before intent)
        # -----------------------------
        # More specific pattern to avoid false positives like "I'm doing well"
        name_pattern = r"\b(?:my name is|call me|i'm|i am)\s+([A-Za-z]+)\b"
        stop_words = [ # Words that are likely not names
            "doing", "feeling", "going", "trying", "getting", "looking", "thinking",
            "tired", "sad", "angry", "happy", "depressed", "pissed", "fine", "good", "great",
            "sure", "curious", "okay"
        ]

        # Use the original message casing to better detect capitalized names
        name_match = re.search(name_pattern, msg, re.IGNORECASE)
        if name_match:
            possible_name = name_match.group(1)

            # Skip if the "name" looks like a mood or common word
            if possible_name.lower() not in stop_words:
                user_context[user_id]["name"] = possible_name

                # Optional: Personalized replies
                # Capitalize to match the keys in special_names.json
                if possible_name.capitalize() in special_names:
                    response = bolu_tone(special_names[possible_name.capitalize()])
                else:
                    response = bolu_tone(f"Aiitt {possible_name}, nice to meet you 😎")

                user_context[user_id]["history"].append({"user": msg, "bot": response})
                save_memory()
                return response

        # -----------------------------
        # 🔹 Step 2: Predict intent using model
        # -----------------------------
        ints = predict_class(msg, model)
        top_confidence = float(ints[0]["probability"])
        # Hybrid semantic fallback trigger
        if top_confidence < 0.80 or ints[0]["intent"] == "fallback":
            logger.info(f"[{user_id}] Using semantic fallback (confidence={top_confidence:.2f})")
            ints = semantic_fallback(msg)

        # -----------------------------
        # 🔹 Step 3: Mood Detection
        # -----------------------------
        text_lower = msg.lower()
        mood = None
        if any(word in text_lower for word in ["sad", "down", "tired", "not good", "depressed"]):
            mood = "sad"
        elif any(word in text_lower for word in ["angry", "mad", "pissed"]):
            mood = "angry"
        elif any(word in text_lower for word in ["happy", "good", "great", "fine", "excited"]):
            mood = "happy"

        if mood:
            user_context[user_id]["mood"] = mood

        # -----------------------------
        # 🔹 Step 4: Generate final response
        # -----------------------------
        res = get_response(ints, intents, user_id, msg)


        # Friendship flavor
        if len(user_context[user_id].get("history", [])) > 15:
            res = "Ayy we’re basically besties now 😂 " + res

        # Save message history
        user_context[user_id]["history"].append({"user": msg, "bot": res})
        if len(user_context[user_id]["history"]) > 30:
            user_context[user_id]["history"].pop(0)

        save_memory()
        return res

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "Something went off on my end — gimme a sec 💀"



# ------------------------------
# API Routes
# ------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    user_id = data.get("user_id", "default")

    if not user_message:
        return jsonify({"error": "Message cannot be empty"}), 400

    logger.info(f"[{user_id}] User: {user_message}")
    bot_response = chatbot_response(user_message, user_id)
    logger.info(f"[{user_id}] Bot: {bot_response}")

    return jsonify({
        "response": bot_response,
        "timestamp": datetime.now().isoformat()
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "words_loaded": words is not None,
        "classes_loaded": classes is not None,
        "intents_loaded": intents is not None,
        "timestamp": datetime.now().isoformat()
    })


@app.route("/memory/<user_id>", methods=["GET"])
def memory_view(user_id):
    """
    🔍 View what the bot remembers about a specific user.
    """
    if user_id not in user_context:
        return jsonify({
            "message": f"No memory found for user '{user_id}'.",
            "known_users": list(user_context.keys())
        }), 404

    memory = user_context[user_id]
    logger.info(f"🧠 Memory viewed for user: {user_id}")

    return jsonify({
        "user_id": user_id,
        "name": memory.get("name", None),
        "mood": memory.get("mood", None),
        "last_intent": memory.get("last_intent", None),
        "history_count": len(memory.get("history", [])),
        "recent_history": memory.get("history", [])[-20:],  # last 20 exchanges
        "timestamp": datetime.now().isoformat()
    })


@app.route("/train", methods=["POST"])
def trigger_training():
    """Trigger model training"""
    try:
        logger.info("🚀 Starting model training...")
        from train_model import train_chatbot_model

        # Train the model
        success = train_chatbot_model()

        if success:
            logger.info("✅ Training completed, reloading model...")
            # Force reload the model
            load_success = load_model_and_data()

            if load_success:
                return jsonify({
                    "message": "Model training completed and loaded successfully!",
                    "model_loaded": model is not None,
                    "words_loaded": words is not None,
                    "classes_loaded": classes is not None
                })
            else:
                return jsonify({"error": "Training completed but failed to load model"}), 500
        else:
            return jsonify({"error": "Training failed"}), 500

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": f"Training failed: {str(e)}"}), 500


@app.route("/reload", methods=["POST"])
def reload_model():
    """Force reload the model and data"""
    try:
        logger.info("🔄 Force reloading model...")
        success = load_model_and_data()
        prepare_intent_embeddings()

        if success:
            return jsonify({
                "message": "Model reloaded successfully!",
                "model_loaded": model is not None,
                "words_loaded": words is not None,
                "classes_loaded": classes is not None,
                "intents_loaded": intents is not None
            })
        else:
            return jsonify({"error": "Failed to reload model"}), 500

    except Exception as e:
        logger.error(f"Error reloading model: {str(e)}")
        return jsonify({"error": f"Reload failed: {str(e)}"}), 500


@app.route("/debug/<user_id>", methods=["GET"])
def debug_info(user_id):
    """Debug endpoint to check model status"""
    return jsonify({
        "model_loaded": model is not None,
        "words_loaded": words is not None,
        "classes_loaded": classes is not None,
        "intents_loaded": intents is not None,
        "user_context": user_context.get(user_id, {}),
        "total_users": len(user_context),
        "words_count": len(words) if words else 0,
        "classes_count": len(classes) if classes else 0,
        "intents_count": len(intents.get("intents", [])) if intents else 0
    })

@app.route("/retrain", methods=["POST"])
def retrain():
    """Manually trigger retraining using learned intents."""
    try:
        merge_learned_intents_into_json()
        from train_model import train_chatbot_model
        train_chatbot_model()
        load_model_and_data()
        return jsonify({"message": "Model retrained successfully!"})
    except Exception as e:
        logger.error(f"Retrain failed: {e}")
        return jsonify({"error": str(e)}), 500


# ------------------------------
# Startup
# ------------------------------
# Initialize the database before starting the app
init_db()
# Load the model at startup for production readiness
logger.info("🚀 Server starting... Loading models and data.")
load_model_and_data()

# ------------------------------
if __name__ == "__main__":
    from waitress import serve
    port = app.config.get('PORT', 5000)
    # Use waitress for a more stable server that avoids segmentation faults
    serve(app, host=app.config.get('HOST', '0.0.0.0'), port=port)

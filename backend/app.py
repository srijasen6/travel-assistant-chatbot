from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf # type: ignore
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import random
import json
import pickle
import os

app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')
CORS(app)

# Load NLP model and data
model_path = os.path.join(os.path.dirname(__file__), 'nlp_model/travel_chatbot_model.h5')
data_path = os.path.join(os.path.dirname(__file__), 'nlp_model/travel_chatbot_data.pkl')

# Initialize variables
words = []
classes = []
model = None
lemmatizer = WordNetLemmatizer()

def load_model():
    global model, words, classes
    
    # Load model and data
    model = tf.keras.models.load_model(model_path)
    
    data = pickle.load(open(data_path, "rb"))
    words = data['words']
    classes = data['classes']

# Load model when starting the app
load_model()

# Sample travel-related intents (expanded version)
intents = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "Hey", "Good day", "Good morning", "Good afternoon"],
            "responses": ["Hello! How can I help with your travel plans today?", "Hi there! Ready to plan your next adventure?", "Welcome! Where are you dreaming of traveling to?"]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "Goodbye", "See you later", "That's all", "I'm done"],
            "responses": ["Safe travels!", "Bon voyage!", "Have a great trip!", "Happy journeys!", "Come back if you need more travel tips!"]
        },
        {
            "tag": "thanks",
            "patterns": ["Thanks", "Thank you", "That's helpful", "I appreciate it", "Awesome thanks"],
            "responses": ["Happy to help!", "You're welcome!", "My pleasure!", "Enjoy your trip!", "Anytime! Safe travels!"]
        },
        {
            "tag": "travel_destinations",
            "patterns": ["Where should I travel?", "Recommend places to visit", "Best travel destinations", 
                        "Top vacation spots", "Where to go on holiday?", "Popular tourist destinations"],
            "responses": ["Popular destinations right now are Bali, Japan, and Italy.", 
                         "It depends on your interests! Beach, mountains, or cities?", 
                         "For beaches: Maldives or Thailand. For culture: Japan or Italy. For adventure: New Zealand or Costa Rica.",
                         "Consider Portugal for great food and beaches, or Vietnam for culture and value."]
        },
        {
            "tag": "visa_requirements",
            "patterns": ["Do I need a visa?", "Visa requirements", "Travel documents needed", 
                        "Passport requirements", "Entry requirements", "Do I need visa for"],
            "responses": ["Visa requirements depend on your nationality and destination. Which country are you asking about?",
                         "I can check visa requirements. Where are you planning to go?",
                         "Some countries offer visa-free entry. Tell me your destination and nationality."]
        },
        {
            "tag": "packing_list",
            "patterns": ["What should I pack?", "Packing list", "What to bring on trip", 
                        "Travel essentials", "What to pack for vacation"],
            "responses": ["Essentials: passport, tickets, medications, chargers. For beach: swimwear, sunscreen. For cold: layers, warm jacket.",
                         "Pack according to weather: check forecasts. Don't forget travel adapters and comfortable shoes!",
                         "I recommend packing cubes to stay organized. Bring copies of important documents too."]
        },
        {
            "tag": "flight_information",
            "patterns": ["Best time to book flights", "Cheap flight tips", "When to buy plane tickets",
                        "Flight deals", "How to find cheap flights", "Airfare trends"],
            "responses": ["Book international flights 2-3 months in advance for best prices. Tuesdays often have deals.",
                         "Use flight comparison tools. Be flexible with dates for better prices.",
                         "Consider nearby airports and red-eye flights for savings. Mid-week flights are often cheaper."]
        },
        {
            "tag": "accommodation",
            "patterns": ["Best places to stay", "Hotel recommendations", "Where to book accommodation",
                        "Airbnb vs hotel", "Best areas to stay in", "Hostel recommendations"],
            "responses": ["It depends on budget and style. Hotels offer service, Airbnbs more space, hostels for socializing.",
                         "Look for places with good public transport links. Read recent reviews carefully.",
                         "For cities, stay central if possible. For beaches, consider proximity to water."]
        },
        {
            "tag": "local_transport",
            "patterns": ["Getting around in", "Public transport options", "Best way to travel in",
                        "Should I rent a car in", "Taxi apps in", "Local transportation"],
            "responses": ["Many cities have great public transport. Research metro/bus passes for savings.",
                         "Ride-sharing apps often work internationally. Check if Uber/Lyft operates there.",
                         "In some places, bikes or scooters are great options. Always check safety first."]
        },
        {
            "tag": "travel_insurance",
            "patterns": ["Do I need travel insurance?", "Best travel insurance", "Insurance for trip",
                        "Medical coverage abroad", "Is travel insurance worth it"],
            "responses": ["Yes! It covers medical emergencies, trip cancellations, and lost luggage. Worth the peace of mind.",
                         "Compare policies for coverage limits. Check if your credit card offers any protection.",
                         "Especially important for international travel. Make sure it covers COVID-related issues too."]
        },
        {
            "tag": "currency_exchange",
            "patterns": ["Best way to exchange money", "Currency exchange tips", "Should I exchange before traveling",
                        "Using credit cards abroad", "ATM fees overseas"],
            "responses": ["Avoid airport exchanges - terrible rates. Use ATMs for better rates or get a fee-free card.",
                         "Notify your bank before travel. Some cards have no foreign transaction fees.",
                         "Carry some local currency for arrival, but don't exchange too much in advance."]
        },
        {
            "tag": "safety_tips",
            "patterns": ["Is it safe to travel to", "Travel safety tips", "How to stay safe abroad",
                        "Tourist scams to avoid", "Dangerous areas in"],
            "responses": ["Check government travel advisories. Generally, be aware of surroundings and don't flash valuables.",
                         "Research common scams at your destination. Use hotel safes for passports.",
                         "Make copies of important documents. Know emergency numbers for your destination."]
        },
        {
            "tag": "food_recommendations",
            "patterns": ["Must-try foods in", "Local dishes to try", "Best restaurants in",
                        "Food specialties of", "Where to eat in", "Street food safety"],
            "responses": ["Try the local specialties! Look for busy places - usually a good sign for quality.",
                         "Food tours are great for sampling safely. Check hygiene ratings where available.",
                         "Ask locals for recommendations away from tourist areas for authentic experiences."]
        },
        {
            "tag": "budget_travel",
            "patterns": ["Traveling on a budget", "How to save money traveling", "Cheap travel tips",
                        "Backpacking advice", "Affordable destinations", "Travel hacks"],
            "responses": ["Consider shoulder season travel. Eat where locals eat. Use public transport.",
                         "Southeast Asia, Eastern Europe and Central America offer great value.",
                         "House-sitting or work exchanges can reduce accommodation costs."]
        },
        {
            "tag": "solo_travel",
            "patterns": ["Tips for solo travelers", "Traveling alone", "Is it safe to travel solo",
                        "Best places for solo travel", "Meeting people while traveling alone"],
            "responses": ["Great for solo travel: Japan, Thailand, Portugal. Stay in hostels to meet people.",
                         "Always share your itinerary with someone. Trust your instincts in new situations.",
                         "Join free walking tours or hostel activities to connect with other travelers."]
        },
        {
            "tag": "family_travel",
            "patterns": ["Traveling with kids", "Family vacation tips", "Best destinations for families",
                        "Flying with children", "Kid-friendly activities in"],
            "responses": ["Look for destinations with good healthcare. Resorts often have kids clubs.",
                         "Pack snacks and entertainment for flights. Consider apartment rentals for space.",
                         "Japan and Scandinavia are very family-friendly with great infrastructure."]
        },
        {
            "tag": "weather_seasons",
            "patterns": ["Best time to visit", "Weather in during", "Rainy season in",
                        "When is peak season in", "Off-season travel to", "Climate in"],
            "responses": ["Shoulder seasons often have good weather with fewer crowds and lower prices.",
                         "Check seasonal weather patterns. Some places have monsoon seasons to avoid.",
                         "Peak season means higher prices but best weather. Off-season can offer great deals."]
        },
        {
            "tag": "travel_health",
            "patterns": ["Vaccinations needed for", "Travel health precautions", "Medications to bring",
                        "Altitude sickness prevention", "Traveler's diarrhea prevention", "Travel clinic"],
            "responses": ["Check CDC or WHO recommendations for your destination. Some places require yellow fever vaccine.",
                         "Pack a basic first aid kit. Bring enough prescription meds plus copies of prescriptions.",
                         "Drink bottled water in developing countries. Wash hands frequently to avoid illness."]
        },
        {
            "tag": "cultural_etiquette",
            "patterns": ["Cultural norms in", "Local customs in", "Dress code in",
                        "Things to avoid in", "Cultural do's and don'ts", "Tipping etiquette in"],
            "responses": ["Research local customs before you go. Dress modestly in conservative countries.",
                         "Learn basic greetings in local language. Be mindful of religious customs.",
                         "Tipping varies widely - in Japan it can be offensive, in US it's expected."]
        },
        {
            "tag": "adventure_travel",
            "patterns": ["Adventure activities in", "Best hiking trails", "Scuba diving locations",
                        "Extreme sports destinations", "Safari options", "Outdoor adventures"],
            "responses": ["New Zealand for adrenaline sports. Costa Rica for eco-adventures. Nepal for trekking.",
                         "Great Barrier Reef for diving. Patagonia for hiking. South Africa for safaris.",
                         "Always use reputable operators for adventure activities. Check safety records."]
        },
        {
            "tag": "romantic_getaways",
            "patterns": ["Honeymoon destinations", "Romantic vacations", "Best couples retreats",
                        "Anniversary trip ideas", "Luxury romantic hotels", "Secluded beaches"],
            "responses": ["Classic romantic spots: Maldives, Santorini, Bali. For cities: Paris, Venice.",
                         "Consider overwater bungalows in Bora Bora or private villas in Tuscany.",
                         "All-inclusive resorts take the stress out of planning for couples."]
        },
        {
            "tag": "travel_technology",
            "patterns": ["Best travel apps", "Useful websites for travel", "Tech gadgets for trips",
                        "Phone plans abroad", "Offline maps", "VPN for travel"],
            "responses": ["Essential apps: Google Maps (download offline), Google Translate, XE Currency.",
                         "Get a local SIM or international plan. VPN helps access content and secure WiFi.",
                         "Power bank is a must. Consider e-reader for books without weight."]
        },
        {
            "tag": "sustainable_travel",
            "patterns": ["Eco-friendly travel", "Sustainable tourism", "Green hotels",
                        "Reducing travel footprint", "Ethical animal tourism", "Responsible travel"],
            "responses": ["Choose direct flights when possible. Pack reusable water bottle/utensils.",
                         "Support local businesses. Avoid activities that exploit animals.",
                         "Look for eco-certified accommodations. Offset your carbon emissions."]
        },
        {
            "tag": "travel_planning",
            "patterns": ["How to plan a trip", "Trip itinerary help", "Travel checklist",
                        "Steps to plan vacation", "Organizing travel", "Pre-trip preparation"],
            "responses": ["Start with dates/budget, then flights, accommodation, activities. Leave some flexibility.",
                         "Make a folder with all confirmations. Check passport validity and visa needs.",
                         "Create a rough daily plan but don't over-schedule. Research opening hours/holidays."]
        },
        {
            "tag": "language_help",
            "patterns": ["Basic phrases in", "Language translation", "Do they speak English in",
                        "How to say hello in", "Language barrier tips", "Learning local language"],
            "responses": ["Learning hello, please, thank you goes far. Google Translate works offline.",
                         "Many tourist areas have English speakers, but rural areas may not.",
                         "Translation apps with camera function help with menus/signs."]
        },
        {
            "tag": "travel_failures",
            "patterns": ["Lost luggage", "Missed flight", "Passport lost",
                        "Travel emergencies", "Getting sick abroad", "Travel problems"],
            "responses": ["For lost passport, contact your embassy immediately. Travel insurance helps with many issues.",
                         "Keep essentials in carry-on in case luggage is delayed. Know airline policies.",
                         "For medical issues, contact insurance provider. Many hotels have doctor contacts."]
        }
    ]
}

# NLP functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list):
    tag = intents_list[0]['intent']
    list_of_intents = intents['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json['message']
    ints = predict_class(message)
    res = get_response(ints)
    return jsonify({'response': res})

if __name__ == '__main__':
    app.run(debug=True)
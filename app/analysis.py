
from functools import lru_cache

import numpy as np
from sklearn.linear_model import LogisticRegression
from transformers import pipeline


# 1. SENTIMENT ANALYSIS

@lru_cache(maxsize=1)
def get_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
    )


def analyze_sentiment(text):
    model = get_sentiment_model()
    result = model(text[:512])[0]

    label = result["label"].lower()

    if "positive" in label:
        sentiment = "Positive"
    elif "negative" in label:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    confidence = round(float(result["score"]) * 100, 2)
    return sentiment, confidence


# 2. EXTRACT CUSTOMER SIGNALS

import re

BUYING_PHRASES = [
    "ready to buy",
    "want to buy",
    "like to buy",
    "ready to purchase",
    "want to purchase",
    "place the order",
    "place an order",
    "place my order",
    "make a purchase",
    "buy it",
    "purchase it",
]

BUYING_WORDS = [
    "buy",
    "purchase",
    "order",
]

URGENCY_WORDS = [
    "today",
    "tomorrow",
    "this week",
    "immediately",
    "as soon as possible",
]

OBJECTION_WORDS = [
    "expensive",
    "too high",
    "budget",
    "discount",
    "problem",
    "disappointed",
    "not decided",
]

POSITIVE_WORDS = [
    "like",
    "love",
    "good",
    "acceptable",
    "ready",
    "interested",
    "works for me",
]


def is_negated(text, start, window=5):
    """Check whether a buying phrase is preceded by a negation."""
    preceding_words = text[:start].split()[-window:]
    preceding_text = " ".join(preceding_words)

    negations = [
        "not",
        "no",
        "never",
        "don't",
        "dont",
        "do not",
        "does not",
        "did not",
        "cannot",
        "can't",
        "cant",
        "won't",
        "will not",
        "would not",
        "wouldn't",
        "not want to",
        "not interested in",
    ]

    return any(
        re.search(r"\b" + re.escape(phrase) + r"\b", preceding_text)
        for phrase in negations
    )


def count_positive_buying_signals(text):
    """Count buying signals while ignoring directly negated phrases."""
    text = text.lower()
    count = 0

    phrases = sorted(BUYING_PHRASES, key=len, reverse=True)

    for phrase in phrases:
        for match in re.finditer(r"\b" + re.escape(phrase) + r"\b", text):
            if not is_negated(text, match.start()):
                count += 1

    # Count standalone buying words, avoiding duplicate phrase matches.
    for word in BUYING_WORDS:
        for match in re.finditer(r"\b" + re.escape(word) + r"\b", text):
            start, end = match.span()

            covered_by_phrase = any(
                phrase_match.start() <= start
                and phrase_match.end() >= end
                for phrase in phrases
                for phrase_match in re.finditer(
                    r"\b" + re.escape(phrase) + r"\b", text
                )
            )

            if not covered_by_phrase and not is_negated(text, start):
                count += 1

    return min(count, 5)


def count_matches(text, words):
    text = text.lower()
    return sum(text.count(word) for word in words)


def extract_features(text):
    text = text.lower()

    return {
        "buying_intent": count_positive_buying_signals(text),
        "urgency": min(count_matches(text, URGENCY_WORDS), 5),
        "objections": min(count_matches(text, OBJECTION_WORDS), 5),
        "positive_interest": min(count_matches(text, POSITIVE_WORDS), 5),
        "follow_up_needed": int(
            any(
                phrase in text
                for phrase in [
                    "think about it",
                    "discuss",
                    "send me",
                    "not decided",
                    "next week",
                    "need more time",
                    "not ready",
                ]
            )
        ),
    }



FEATURE_NAMES = [
    "buying_intent",
    "urgency",
    "objections",
    "positive_interest",
    "follow_up_needed",
]


# 3. SYNTHETIC TRAINING DATA FOR DEMONSTRATION
# Each row represents:
# buying intent, urgency, objections, positive interest, follow-up needed
# Target: 1 = likely to buy; 0 = unlikely to buy.
# This is not real customer data.

TRAINING_X = np.array([
    [5, 4, 0, 5, 0],
    [4, 3, 0, 4, 0],
    [4, 2, 1, 4, 0],
    [3, 3, 1, 4, 0],
    [3, 1, 0, 3, 0],
    [2, 1, 1, 3, 1],
    [2, 0, 2, 3, 1],
    [1, 0, 2, 2, 1],
    [0, 0, 3, 0, 1],
    [0, 0, 4, 0, 0],
    [0, 0, 5, 0, 0],
    [1, 0, 4, 1, 1],
    [2, 1, 3, 2, 1],
    [3, 1, 3, 3, 1],
    [4, 2, 2, 4, 1],
    [1, 0, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [3, 2, 0, 3, 0],
    [2, 2, 0, 2, 0],
    [4, 3, 3, 4, 1],
])

TRAINING_Y = np.array([
    1, 1, 1, 1, 1,
    1, 1, 0, 0, 0,
    0, 0, 0, 1, 1,
    0, 0, 1, 1, 1,
])


@lru_cache(maxsize=1)
def get_purchase_model():
    model = LogisticRegression(random_state=42)
    model.fit(TRAINING_X, TRAINING_Y)
    return model


def estimate_purchase_probability(features):
    values = np.array([
        features[name] for name in FEATURE_NAMES
    ]).reshape(1, -1)

    model = get_purchase_model()
    positive_class_index = list(model.classes_).index(1)

    probability = model.predict_proba(values)[0][positive_class_index]
    return round(float(probability) * 100, 2)


# 4. CUSTOMER INSIGHTS AND RECOMMENDATIONS

def create_insights(text, features, sentiment, probability):
    text = text.lower()
    insights = []
    recommendations = []

    if features["buying_intent"] > 0:
        insights.append(
            "The customer mentions buying, purchasing, or placing an order."
        )
    else:
        insights.append(
            "The customer has not expressed a clear buying action."
        )

    if features["objections"] > 0:
        insights.append(
            "The conversation contains possible concerns about price, "
            "service, or making a decision."
        )

    if features["follow_up_needed"]:
        insights.append(
            "The customer may need more information or time before deciding."
        )

    if any(word in text for word in ["price", "expensive", "budget"]):
        recommendations.append(
            "Explain the price and discuss suitable offers or alternatives, "
            "if available."
        )

    if any(word in text for word in ["disappointed", "problem"]):
        recommendations.append(
            "Address the customer's complaint before focusing on a sale."
        )

    if features["follow_up_needed"]:
        recommendations.append(
            "Send the requested information and arrange a suitable follow-up."
        )

    if features["buying_intent"] > 0 and probability >= 60:
        recommendations.append(
            "Confirm availability, answer final questions, and explain "
            "how to place the order."
        )
    elif probability < 40:
        recommendations.append(
            "Understand the customer's main concern without pressuring them."
        )

    if not recommendations:
        recommendations.append(
            "Ask what additional information the customer needs."
        )

    return insights, recommendations


# 5. COMPLETE ANALYSIS FOR ONE CALL

def analyze_call(call):
    transcript = call["transcript"]

    sentiment, confidence = analyze_sentiment(transcript)
    features = extract_features(transcript)
    probability = estimate_purchase_probability(features)

    insights, recommendations = create_insights(
        transcript, features, sentiment, probability
    )

    return {
        "call_id": call["id"],
        "title": call["title"],
        "customer": call["customer"],
        "sentiment": sentiment,
        "sentiment_confidence": confidence,
        "purchase_probability": probability,
        "features": features,
        "insights": insights,
        "recommendations": recommendations,
        "transcript": transcript,
        "model_note": (
            "The purchase estimate uses Logistic Regression trained on "
            "synthetic demonstration examples. It is not validated for "
            "real customer decisions."
        ),
    }

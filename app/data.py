
CALLS = [
    {
        "id": "call_001",
        "title": "Interested customer",
        "customer": "Customer 1",
        "scenario": "High purchase intent",
        "transcript": (
            "Hello, I really like this product. It has all the features "
            "I need. I want to place the order today. Can you deliver it "
            "tomorrow? The price is acceptable, and I am ready to buy."
        ),
    },
    {
        "id": "call_002",
        "title": "Customer comparing options",
        "customer": "Customer 2",
        "scenario": "Information seeking",
        "transcript": (
            "Hello, I am looking for information about this product. "
            "Can you explain its features and warranty? I am comparing "
            "it with two other products. I have not decided yet. "
            "Please send me the details so I can think about it."
        ),
    },
    {
        "id": "call_003",
        "title": "Dissatisfied customer",
        "customer": "Customer 3",
        "scenario": "Negative experience",
        "transcript": (
            "I am disappointed with the service. The product is too "
            "expensive, and nobody answered my previous questions. "
            "I do not want to place an order now. Please explain the "
            "problem and tell me whether you can offer a better price."
        ),
    },
    {
        "id": "call_004",
        "title": "Interested but concerned about price",
        "customer": "Customer 4",
        "scenario": "Needs follow-up",
        "transcript": (
            "I like the product and its features. However, the price is "
            "a little high for my budget. I need to discuss it with my "
            "family before deciding. Could you send me the available "
            "discounts? I may buy it next week if the price works for me."
        ),
    },
]


def get_all_calls():
    return CALLS


def get_call_by_id(call_id: str):
    for call in CALLS:
        if call["id"] == call_id:
            return call
    return None

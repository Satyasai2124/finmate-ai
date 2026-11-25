import random

CITY_TIERS = {
    "tier1": ["mumbai", "delhi", "bangalore", "bengaluru", "chennai", "hyderabad", "kolkata", "pune"],
    "tier2": ["ahmedabad", "jaipur", "lucknow", "kanpur", "nagpur", "indore", "bhopal", "visakhapatnam", "patna", "vadodara", "coimbatore", "kochi", "chandigarh", "guwahati", "mysore", "mysuru"],
    "tier3": []
}

COST_MULTIPLIERS = {"tier1": 1.3, "tier2": 1.0, "tier3": 0.75}

FOOD_ITEMS = {
    "biryani": {"cook": (60, 90), "tiffin": (80, 120), "online": (150, 250)},
    "rice": {"cook": (20, 35), "tiffin": (40, 60), "online": (80, 130)},
    "dal": {"cook": (15, 25), "tiffin": (30, 50), "online": (70, 110)},
    "roti": {"cook": (10, 20), "tiffin": (25, 40), "online": (50, 90)},
    "chapati": {"cook": (10, 20), "tiffin": (25, 40), "online": (50, 90)},
    "thali": {"cook": (40, 70), "tiffin": (60, 100), "online": (120, 200)},
    "dosa": {"cook": (15, 30), "tiffin": (35, 55), "online": (80, 140)},
    "idli": {"cook": (10, 20), "tiffin": (25, 40), "online": (60, 100)},
    "paneer": {"cook": (50, 80), "tiffin": (70, 110), "online": (130, 220)},
    "chicken": {"cook": (70, 110), "tiffin": (90, 140), "online": (160, 280)},
    "egg": {"cook": (15, 25), "tiffin": (30, 50), "online": (60, 100)},
    "pasta": {"cook": (30, 50), "tiffin": (50, 80), "online": (100, 180)},
    "noodles": {"cook": (20, 40), "tiffin": (40, 65), "online": (80, 150)},
    "pizza": {"cook": (60, 100), "tiffin": (80, 130), "online": (150, 350)},
    "burger": {"cook": (40, 70), "tiffin": (60, 100), "online": (100, 200)},
    "sandwich": {"cook": (20, 35), "tiffin": (35, 60), "online": (70, 130)},
    "paratha": {"cook": (15, 30), "tiffin": (30, 50), "online": (60, 110)},
    "poha": {"cook": (10, 20), "tiffin": (20, 35), "online": (50, 90)},
    "upma": {"cook": (10, 20), "tiffin": (20, 35), "online": (50, 90)},
    "samosa": {"cook": (15, 25), "tiffin": (20, 35), "online": (40, 80)},
    "pav bhaji": {"cook": (30, 50), "tiffin": (45, 70), "online": (90, 160)},
    "chole bhature": {"cook": (30, 50), "tiffin": (50, 80), "online": (100, 170)},
    "fish": {"cook": (60, 100), "tiffin": (80, 130), "online": (150, 260)},
    "mutton": {"cook": (100, 160), "tiffin": (130, 200), "online": (200, 400)},
    "fried rice": {"cook": (25, 45), "tiffin": (45, 70), "online": (90, 160)},
    "pulao": {"cook": (30, 50), "tiffin": (50, 75), "online": (90, 160)},
    "maggi": {"cook": (10, 20), "tiffin": (25, 40), "online": (50, 90)},
    "omelette": {"cook": (10, 20), "tiffin": (25, 40), "online": (50, 90)},
    "salad": {"cook": (20, 35), "tiffin": (35, 55), "online": (70, 120)},
    "soup": {"cook": (15, 30), "tiffin": (30, 50), "online": (70, 120)},
}

DEFAULT_COST = {"cook": (30, 60), "tiffin": (50, 80), "online": (100, 180)}


def get_city_tier(location: str) -> str:
    loc = location.lower().strip()
    for tier, cities in CITY_TIERS.items():
        for city in cities:
            if city in loc:
                return tier
    return "tier3"


def predict_spending(budget: float, food_item: str, location: str) -> dict:
    food_key = food_item.lower().strip()
    costs = FOOD_ITEMS.get(food_key, DEFAULT_COST)
    tier = get_city_tier(location)
    multiplier = COST_MULTIPLIERS[tier]

    options = []
    for mode, label, icon, desc in [
        ("cook", "Cook at Home", "🏠", "Buy ingredients and cook yourself"),
        ("tiffin", "Tiffin / Local Shop", "🍱", "Order from nearby tiffin center or local eatery"),
        ("online", "Online Order", "📱", "Order via Swiggy, Zomato, etc."),
    ]:
        low = round(costs[mode][0] * multiplier)
        high = round(costs[mode][1] * multiplier)
        estimated = round((low + high) / 2)
        within_budget = budget >= low
        savings = round(budget - estimated) if within_budget else 0

        options.append({
            "mode": label,
            "icon": icon,
            "description": desc,
            "cost_low": low,
            "cost_high": high,
            "estimated": estimated,
            "within_budget": within_budget,
            "savings": max(savings, 0),
        })

    affordable = [o for o in options if o["within_budget"]]
    if affordable:
        best = min(affordable, key=lambda x: x["estimated"])
    else:
        best = min(options, key=lambda x: x["estimated"])

    return {
        "budget": budget,
        "food_item": food_item.title(),
        "location": location,
        "city_tier": tier.replace("tier", "Tier "),
        "options": options,
        "best_option": best["mode"],
        "best_estimated": best["estimated"],
        "affordable_count": len(affordable),
        "available_items": sorted([k.title() for k in FOOD_ITEMS.keys()]),
    }

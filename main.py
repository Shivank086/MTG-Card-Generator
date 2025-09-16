import pandas as pd
import re
from collections import Counter, defaultdict
from problog.program import PrologString
from problog import get_evaluatable
from collections import Counter

CSV_FILE = "mtg_cards.csv"

#load dataset
df = pd.read_csv(CSV_FILE, low_memory=False)
df.columns = [c.strip() for c in df.columns]


#parse mana costs
COLOR_LETTERS = list("WUBRG")  
def parse_mana_cost(mana_str):
    if pd.isna(mana_str) or not isinstance(mana_str, str):
        return 0, []
    tokens = re.findall(r"\{([^}]+)\}", mana_str)
    cmc = 0
    colors = set()
    for t in tokens:
        if t.isdigit():
            cmc += int(t)
        else:
            cmc += 1
        for ch in t:
            if ch in COLOR_LETTERS:
                colors.add(ch)
    return cmc, sorted(list(colors))

df["CMC_parsed"], df["ColorList"] = zip(*df["Mana Cost"].map(parse_mana_cost))
df["ColorPrimary"] = df["ColorList"].apply(lambda c: "".join(c) if c else "Colorless")


#power/toughness
def parse_pt(val):
    if pd.isna(val):
        return None
    try:
        return int(val)
    except ValueError:
        return None
df["PowerNum"] = df["Power"].map(parse_pt)
df["ToughnessNum"] = df["Toughness"].map(parse_pt)


#detect creature
df["IsCreature"] = df["Type"].fillna("").str.contains("Creature", case=False)

#parse creature subtypes
def parse_subtypes(type_str):
    if not isinstance(type_str, str) or type_str.strip() == "":
        return []
    if "—" in type_str:
        parts = type_str.split("—", 1)[1]
    else:
        parts = type_str.replace("Creature", "").strip()
    return [t.strip() for t in parts.split() if t.strip()]

df["SubTypes"] = df["Type"].apply(parse_subtypes)


#Correlation analysis
creatures = df[df["IsCreature"] & df["PowerNum"].notna() & df["ToughnessNum"].notna()]
power_corr = creatures["CMC_parsed"].corr(creatures["PowerNum"])
tough_corr = creatures["CMC_parsed"].corr(creatures["ToughnessNum"])

print("\nCMC vs Power/Toughness Correlation (creatures only)")
print(f"Correlation (CMC ↔ Power): {power_corr:.3f}")
print(f"Correlation (CMC ↔ Toughness): {tough_corr:.3f}")

#creature type by color 
type_counts_by_color = defaultdict(Counter)
for _, row in creatures.iterrows():
    for sub in row["SubTypes"]:
        type_counts_by_color[row["ColorPrimary"]][sub] += 1

#key words in abilities
keywords = {
    "flying": "Flying",
    "trample": "Trample",
    "lifelink": "Lifelink",
    "deathtouch": "Deathtouch",
    "haste": "Haste",
    "vigilance": "Vigilance",
    "menace": "Menace",
    "mill": "Mill",
    "draw": "Card Draw",
    "counter target": "Counterspell"
}

ability_counts = defaultdict(Counter)
for _, row in df.iterrows():
    text = str(row.get("Text", "")).lower()
    for kw, label in keywords.items():
        if kw in text:
            ability_counts[row["ColorPrimary"]][label] += 1

print("\nCommon Abilities by Color")
for color, counter in ability_counts.items():
    print(f"{color}: {counter.most_common(5)}")


#problog


flying_counts = Counter()
total_flying = 0

for i, row in df.head(200).iterrows():
    text = str(row.get("Text", "")).lower()
    if "flying" in text:
        colors = row["ColorPrimary"]
        for ch in colors:
            flying_counts[ch] += 1
            total_flying += 1

problog_facts = []
for color, count in flying_counts.items():
    p = count / total_flying
    problog_facts.append(f"{p:.4f}::color('{color.lower()}').")

problog_model = f"""
% Probabilities of being a color given flying
{chr(10).join(problog_facts)}

query(color('b')).
"""

problog_program = PrologString(problog_model)
result = get_evaluatable().create_from(problog_program).evaluate()

print("\nProbability that a card with Flying is Black:")

key = list(result.keys())[0]
print("P(Black | Flying) = {:.3f}".format(result[key]))

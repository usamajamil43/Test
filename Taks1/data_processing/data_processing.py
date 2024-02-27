import pandas as pd

def process_item(item):
    details = item[2] if len(item) > 2 else {}
    nutritional_info = details.get("nutritionalInfo", {})
    kcal = nutritional_info.get("kcal", None)
    fat = nutritional_info.get("fat", None)
    protein = nutritional_info.get("protein", None)
    allergens = ", ".join(nutritional_info.get("allergens", []))
    available = details.get("available", True)
    return kcal, fat, protein, allergens, available

def json_to_dataframe(json_chunks):
    # Prepare lists to hold data
    categories, ids, names, prices, kcals, fats, proteins, allergens_list, availabilities, details_str = [], [], [], [], [], [], [], [], [], []

    # Iterate through each category and item
    for chunk in json_chunks:
        for category, items in chunk.items():
            if category not in ["Location", "Menus"]:  # Adjust as necessary
                for item_id, item in items.items():
                    categories.append(category)
                    ids.append(item_id)
                    names.append(item[0])
                    prices.append(item[1])

                    kcal, fat, protein, allergens, available = process_item(item)
                    kcals.append(kcal)
                    fats.append(fat)
                    proteins.append(protein)
                    allergens_list.append(allergens)
                    availabilities.append(available)

                    detail = f"{item[0]}, Price: ${item[1]}, Calories: {kcal}, Fat: {fat}g, Protein: {protein}g, Allergens: {allergens}, Available: {'Yes' if available else 'No'}"
                    details_str.append(detail)

    # Create DataFrame
    df = pd.DataFrame({
        'Category': categories,
        'ID': ids,
        'Name': names,
        'Price': prices,
        'Calories': kcals,
        'Fat': fats,
        'Protein': proteins,
        'Allergens': allergens_list,
        'Available': availabilities,
        'Details': details_str
    })

    return df




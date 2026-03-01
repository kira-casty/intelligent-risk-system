import joblib
import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance():
    model = joblib.load("model/rf_model_vehicle.pkl")
    feature_names = ["age","income","vehicle_age","past_claims","claim_amount","accident_severity"]

    importances = model.feature_importances_
    df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(8,5))
    plt.barh(df["Feature"], df["Importance"])
    plt.gca().invert_yaxis()
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()
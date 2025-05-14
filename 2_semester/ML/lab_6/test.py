import numpy as np
import json
from tensorflow.keras.models import load_model

driver_types = {
    0: 'Осторожный',
    1: 'Агрессивный',
    2: 'Непредсказуемый'
}
model = load_model("driver_behavior_model.h5")

with open("driver_behavior_test_data.json", "r") as f:
    test_data = json.load(f)

trips = test_data["trips"]

print("=== Результаты классификации поведения ===\n")
for label, trip in trips.items():
    trip_array = np.array(trip).reshape(1, len(trip), 3)
    prediction = model.predict(trip_array)
    class_index = np.argmax(prediction)
    class_name = driver_types[class_index]
    confidence = prediction[0][class_index]

    print(f"'{label}': предсказано — {class_name} ({confidence:.2f})")


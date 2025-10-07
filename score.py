import joblib
import json
import numpy as np

def init():
    global model
    model = joblib.load("model.joblib")  # load directly from local file

def run(data):
    try:
        data = np.array(json.loads(data)["data"])
        result = model.predict(data)
        return result.tolist()
    except Exception as e:
        return str(e)

# Local test
if __name__ == "__main__":
    init()
    test_data = json.dumps({"data": [[5.1, 3.5, 1.4, 0.2]]})
    print(run(test_data))




import joblib
class Model:
    def __init__(self,model_path):
        self.model = joblib.load(model_path)

    def predict(self,landmarks):
        return self.model.predict(landmarks)
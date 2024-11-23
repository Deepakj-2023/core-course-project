import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model



# img = Image.open(r'C:\Users\Deepak\Desktop\core_course_project\person1_virus_7.jpeg')
# img.show()
# img = img.convert('L')
# img = img.resize((36,36))
# img = np.asarray(img)
# img = img.reshape((1,36,36,1))
# img = img / 255.0
# print(img.shape)

model = load_model("models/pneumonia.h5")
# pred = np.argmax(model.predict(img)[0])
# print(pred)



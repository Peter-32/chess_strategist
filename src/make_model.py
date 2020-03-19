from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
data = plt.imread("../data/raw/board.png")
i = 0
j = 0
X = []
y = np.zeros([64,3])
record = 0
for i in range(8):
    for j in range(8):
        cord_x = 477 + 146*i
        cord_y = 193 + 146*j
        if i == 1:
            y[record, 0] = 1
        elif i == 6:
            y[record, 1] = 1
        else:
            y[record, 2] = 1
        X.append(data[cord_x:cord_x+146,cord_y:cord_y+146,:].flatten())
        record += 1

X = pd.DataFrame(X).values
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(1)
model.fit(X, y)

import pickle
def save_obj(obj, name ):
    with open('../models/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('../models/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

save_obj(model, "model")
save_obj(scaler, "scaler")
prediction = model.predict(X)
prediction

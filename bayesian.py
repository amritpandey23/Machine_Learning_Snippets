from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

outlook = ["sunny", "sunny", "overcast", "rain", "rain", "rain", "overcast", "sunny", "sunny", "rain", "sunny", "overcast", "overcast", "rain"]

temperature = ["hot", "hot", "hot", "mild", "cool", "cool", "cool", "mild", "cool", "mild", "mild", "mild", "hot", "mild"]

humidity = ["high", "high", "high", "high", "normal", "normal", "normal", "high", "normal", "normal", "normal", "high", "normal", "high"]

wind = ["weak", "strong", "weak", "weak", "weak", "strong", "strong", "weak", "weak", "weak", "strong", "strong", "weak", "strong"]

play_tenis = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no"]

le = preprocessing.LabelEncoder()

wea_encoded = le.fit_transform(outlook)
temp_encoded = le.fit_transform(temperature)
wind_encoded = le.fit_transform(wind)
hum_encoded = le.fit_transform(humidity)
label = le.fit_transform(play_tenis)

features = list(zip(wea_encoded, temp_encoded, wind_encoded, hum_encoded))

model = GaussianNB()
model.fit(features, label)

samp1 = [0, 1, 0, 1]
samp2 = [1, 2, 1, 0]
samp3 = [0,0,0,0]

print(model.predict([samp1]))
print(model.predict([samp2]))
print(model.predict([samp3]))
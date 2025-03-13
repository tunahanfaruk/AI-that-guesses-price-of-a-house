import pandas as pd
from sklearn.linear_model import LinearRegression

dataset = 'dataset.csv'
df = pd.read_csv(dataset)

X = df[['area', 'parking']] 
y = df['price']            

reg = LinearRegression()
reg.fit(X, y)

w = float(input("Area of a House: "))
q = float(input("Number of House Parkings: "))

guess_data = pd.DataFrame([[w, q]], columns=['area', 'parking'])

guess = reg.predict(guess_data)
print("Guessed price of the Houses:", guess[0])

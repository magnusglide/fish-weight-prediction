import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


df = pd.read_csv("Fish.csv")
cdf = df[["Weight", "Length1", "Height", "Width"]]


msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

train_x = np.asanyarray(train[["Length1"]])
train_y = np.asanyarray(train[["Weight"]])

test_x = np.asanyarray(test[["Length1"]])
test_y = np.asanyarray(test[["Weight"]])

poly = PolynomialFeatures(degree=10)
train_x_poly = poly.fit_transform(train_x)

clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly, train_y)


XX = np.arange(0.0, 60.0, 0.1)
yy = (
    clf.intercept_[0]
    + clf.coef_[0][1] * XX
    + clf.coef_[0][2] * np.power(XX, 2)
    + clf.coef_[0][3] * np.power(XX, 3)
    + clf.coef_[0][4] * np.power(XX, 4)
    + clf.coef_[0][5] * np.power(XX, 5)
    + clf.coef_[0][6] * np.power(XX, 6)
    + clf.coef_[0][7] * np.power(XX, 7)
    + clf.coef_[0][8] * np.power(XX, 8)
    + clf.coef_[0][9] * np.power(XX, 9)
    + clf.coef_[0][10] * np.power(XX, 10)
)


test_x_poly = poly.fit_transform(test_x)
test_y_ = clf.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y, test_y_))
print(
    "Predicted weight of a fish with length 26.9 is {}.\nThe actual weight is 300".format(
        clf.predict(poly.fit_transform([[26.9]]))
    )
)  # to check - weight should be 300

# plt.scatter(np.asanyarray(test[["Length1"]]), test_y, color="blue")
# plt.scatter(np.asanyarray(test[["Length1"]]), test_y_, color="red")
plt.scatter(cdf[["Length1"]], cdf[["Weight"]])
# plt.scatter(train_x, train_y_, color="green")
plt.scatter(test_x, test_y_, color="red")
plt.plot(XX, yy, "-r")
plt.xlabel("Length")
plt.ylabel("Weight")
plt.show()

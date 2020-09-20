from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score

# Irisの測定データの読み込み
# iris = datasets.load_iris()
# clf = svm.LinearSVC()
# clf.fit(iris.data, iris.target)
# print(clf.predict([[1.4, 3.5, 5.1, 0.2], [6.5, 2.6, 4.4, 1.4], [5.9, 3.0, 5.2, 1.5]]))

# データの読み込み
iris = datasets.load_iris()
x, y = iris.data, iris.target

# トレーニングデータとテストデータに分ける
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

# モデルの選択
model = svm.LinearSVC()
# model = RandomForestClassifier()
# 学習
model.fit(x_train, y_train)

# テストと評価
pred = model.predict(x_test)
print(accuracy_score(y_test, pred))

# 学習済みモデルを使う
print(model.predict([[1.4, 3.5, 5.1, 0.2], [6.5, 2.6, 4.4, 1.4], [5.9, 3.0, 5.2, 1.5]]))
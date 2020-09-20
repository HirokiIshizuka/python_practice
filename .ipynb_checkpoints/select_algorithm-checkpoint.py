{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "AdaBoostClassifier の正解率 =  0.9666666666666667\n",
      "BaggingClassifier の正解率 =  0.9666666666666667\n",
      "BernoulliNB の正解率 =  0.26666666666666666\n",
      "\u001b[33mWarning：\u001b[0m CalibratedClassifierCV : ('Liblinear failed to converge, increase the number of iterations.',)\n",
      "CategoricalNB の正解率 =  0.9666666666666667\n",
      "\u001b[31mError：\u001b[0m ClassifierChain : (\"__init__() missing 1 required positional argument: 'base_estimator'\",)\n",
      "ComplementNB の正解率 =  0.5333333333333333\n",
      "DecisionTreeClassifier の正解率 =  1.0\n",
      "\u001b[33mWarning：\u001b[0m DummyClassifier : ('The default value of strategy will change from stratified to prior in 0.24.',)\n",
      "ExtraTreeClassifier の正解率 =  0.9\n",
      "ExtraTreesClassifier の正解率 =  0.9666666666666667\n",
      "GaussianNB の正解率 =  0.9666666666666667\n",
      "GaussianProcessClassifier の正解率 =  1.0\n",
      "GradientBoostingClassifier の正解率 =  1.0\n",
      "HistGradientBoostingClassifier の正解率 =  0.9666666666666667\n",
      "KNeighborsClassifier の正解率 =  0.9666666666666667\n",
      "LabelPropagation の正解率 =  0.9333333333333333\n",
      "LabelSpreading の正解率 =  0.9333333333333333\n",
      "LinearDiscriminantAnalysis の正解率 =  1.0\n",
      "\u001b[33mWarning：\u001b[0m LinearSVC : ('Liblinear failed to converge, increase the number of iterations.',)\n",
      "\u001b[33mWarning：\u001b[0m LogisticRegression : ('lbfgs failed to converge (status=1):\\nSTOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\\n\\nIncrease the number of iterations (max_iter) or scale the data as shown in:\\n    https://scikit-learn.org/stable/modules/preprocessing.html\\nPlease also refer to the documentation for alternative solver options:\\n    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression',)\n",
      "\u001b[33mWarning：\u001b[0m LogisticRegressionCV : ('lbfgs failed to converge (status=1):\\nSTOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\\n\\nIncrease the number of iterations (max_iter) or scale the data as shown in:\\n    https://scikit-learn.org/stable/modules/preprocessing.html\\nPlease also refer to the documentation for alternative solver options:\\n    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression',)\n",
      "\u001b[33mWarning：\u001b[0m MLPClassifier : (\"Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.\",)\n",
      "\u001b[31mError：\u001b[0m MultiOutputClassifier : (\"__init__() missing 1 required positional argument: 'estimator'\",)\n",
      "MultinomialNB の正解率 =  0.5666666666666667\n",
      "NearestCentroid の正解率 =  1.0\n",
      "NuSVC の正解率 =  1.0\n",
      "\u001b[31mError：\u001b[0m OneVsOneClassifier : (\"__init__() missing 1 required positional argument: 'estimator'\",)\n",
      "\u001b[31mError：\u001b[0m OneVsRestClassifier : (\"__init__() missing 1 required positional argument: 'estimator'\",)\n",
      "\u001b[31mError：\u001b[0m OutputCodeClassifier : (\"__init__() missing 1 required positional argument: 'estimator'\",)\n",
      "PassiveAggressiveClassifier の正解率 =  0.8666666666666667\n",
      "Perceptron の正解率 =  0.9\n",
      "QuadraticDiscriminantAnalysis の正解率 =  1.0\n",
      "RadiusNeighborsClassifier の正解率 =  1.0\n",
      "RandomForestClassifier の正解率 =  0.9666666666666667\n",
      "RidgeClassifier の正解率 =  0.8666666666666667\n",
      "RidgeClassifierCV の正解率 =  0.8666666666666667\n",
      "SGDClassifier の正解率 =  0.5333333333333333\n",
      "SVC の正解率 =  1.0\n",
      "\u001b[31mError：\u001b[0m StackingClassifier : (\"__init__() missing 1 required positional argument: 'estimators'\",)\n",
      "\u001b[31mError：\u001b[0m VotingClassifier : (\"__init__() missing 1 required positional argument: 'estimators'\",)\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.utils import all_estimators\n",
    "import warnings\n",
    "\n",
    "# アヤメデータの読み込み\n",
    "iris_data = pd.read_csv(\"iris.csv\", encoding=\"utf-8\")\n",
    "\n",
    "# アヤメデータをラベルと入力データに分離する \n",
    "y = iris_data.loc[:,\"Name\"]\n",
    "x = iris_data.loc[:,[\"SepalLength\",\"SepalWidth\",\"PetalLength\",\"PetalWidth\"]]\n",
    "\n",
    "# 学習用とテスト用に分離する \n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, train_size = 0.8, shuffle = True)\n",
    "\n",
    "# classifierのアルゴリズム全てを取得する --- (※1)\n",
    "allAlgorithms = all_estimators(type_filter=\"classifier\")\n",
    "warnings.simplefilter(\"error\")\n",
    "\n",
    "for(name, algorithm) in allAlgorithms :\n",
    "  try :\n",
    "    # 各アリゴリズムのオブジェクトを作成 --- (※2)\n",
    "    clf = algorithm()\n",
    "\n",
    "    # 学習して、評価する --- (※3)\n",
    "    clf.fit(x_train, y_train)\n",
    "    y_pred = clf.predict(x_test)\n",
    "    print(name,\"の正解率 = \" , accuracy_score(y_test, y_pred))\n",
    "  \n",
    "  # WarningやExceptionの内容を表示する --- (※4)\n",
    "  except Warning as w :\n",
    "    print(\"\\033[33m\"+\"Warning：\"+\"\\033[0m\", name, \":\", w.args)\n",
    "  except Exception as e :\n",
    "    print(\"\\033[31m\"+\"Error：\"+\"\\033[0m\", name, \":\", e.args)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

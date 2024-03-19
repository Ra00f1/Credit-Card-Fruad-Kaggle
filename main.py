import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import sklearn.linear_model as lm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

"""
    Uncomment the following lines to display all the columns and rows
"""
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data

# Data Summary Function
def Data_Summary(data):
    print("Data Summary")
    print("=====================================")
    print("First 5 rows")
    print(data.describe())
    print("=====================================")
    print("Data types")
    print(data.info())
    print("=====================================")
    print("Data count")
    print(data.count())
    print("=====================================")
    print("Missing values")
    print(data.isnull().sum())
    print("=====================================")
    print("Data shape")
    print(data.shape)
    print("=====================================")
    print("Unique values in each column")
    print(data.nunique())
    print("=====================================")

    # describe the last column
    print("Last column description")
    print(data.iloc[:, -1].describe())
    # Visualize the last column
    temp_data = data.drop(data.index[0])
    plt.figure(figsize=(7, 6))
    plt.bar(temp_data.iloc[:, -1].unique(), temp_data.iloc[:, -1].value_counts())
    plt.show(block=True)

    print("---------------------------------------------------------------------------------")


def Visualize_Data(data):
    print("Visualizing Data")
    print("=====================================")
    # delete the first row as it only contains the column names
    new_data = data.drop(data.index[0])

    # print(new_data.iloc[:, 1:-1])
    viz_data = new_data.iloc[:, 1:-1]

    # Visualizing the data, x is Class(last column), y is the rest of the columns
    for column in viz_data.columns:
        plt.figure(figsize=(7, 6))
        plt.bar(new_data.iloc[:, -1], viz_data[column])
        plt.show(block=True)
        wait = input("Press Enter to continue")


def Preprocessing(data):
    print("Preprocessing")
    print("=====================================")
    # delete the first row as it only contains the column names
    # data = data.drop(data.index[0])
    # delete the first column as it only contains the Time value
    # data = data.drop(data.columns[0], axis=1)
    std_scaler = StandardScaler()

    scaled_time = std_scaler.fit_transform(data['Time'].values.reshape(-1,1))
    scaled_amount = std_scaler.fit_transform(data['Amount'].values.reshape(-1,1))

    # drop the original Time and Amount column
    data.drop(['Time'], axis=1, inplace=True)
    data.drop(['Amount'], axis=1, inplace=True)

    # insert the scaled Time and scaled Amount column
    data.insert(0, 'Scaled Time', scaled_time)
    data.insert(1, 'Scaled Amount', scaled_amount)

    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2, random_state=42)
    print("X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_train shape: ", y_train.shape)
    print("y_test shape: ", y_test.shape)

    print(X_train.head())
    print(X_test.head())
    print(y_train.head())
    print(y_test.head())
    print("---------------------------------------------------------------------------------")

    return X_train, X_test, y_train, y_test


def Machine_Learning(_train, X_test, y_train, y_test):
    print("Machine Learning")
    print("=====================================")

    log_reg = lm.LogisticRegression()
    log_reg.fit(X_train, y_train)
    log_reg_score = log_reg.score(X_test, y_test)
    print("Logistic Regression Testing Accuracy: ", log_reg_score)
    print("Logistic Regression Classification Report: ")
    print(classification_report(y_test, log_reg.predict(X_test)))
    print("=====================================")

    # Decision Tree
    Decision_tree = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=None, min_samples_leaf=5)
    Decision_tree.fit(X_train, y_train)
    Decision_tree_score = Decision_tree.score(X_test, y_test)
    print("Decision Tree Testing Accuracy: ", Decision_tree_score)
    print("Decision Tree Classification Report: ")
    print(classification_report(y_test, Decision_tree.predict(X_test)))
    print("=====================================")

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_score = knn.score(X_test, y_test)
    print("KNN Testing Accuracy: ", knn_score)
    print("=====================================")

    lin_reg = lm.LinearRegression()
    lin_reg.fit(X_train, y_train)
    lin_reg_score = lin_reg.score(X_test, y_test)
    print("Linear Regression Testing Accuracy: ", lin_reg_score)
    print("=====================================")

    Sgd_reg = lm.SGDRegressor()
    Sgd_reg.fit(X_train, y_train)
    Sgd_reg_score = Sgd_reg.score(X_test, y_test)
    print("SGD Testing Accuracy: ", Sgd_reg_score)
    print("=====================================")

    Ridge_reg = lm.Ridge()
    Ridge_reg.fit(X_train, y_train)
    Ridge_reg_score = Ridge_reg.score(X_test, y_test)
    print("Ridge Testing Accuracy: ", Ridge_reg_score)
    print("=====================================")

    Lasso_reg = lm.Lasso()
    Lasso_reg.fit(X_train, y_train)
    Lasso_reg_score = Lasso_reg.score(X_test, y_test)
    print("Lasso Testing Accuracy: ", Lasso_reg_score)
    print("=====================================")

    Elastic_reg = lm.ElasticNet()
    Elastic_reg.fit(X_train, y_train)
    Elastic_reg_score = Elastic_reg.score(X_test, y_test)
    print("Elastic Testing Accuracy: ", Elastic_reg_score)
    print("=====================================")

    Huber_reg = lm.HuberRegressor()
    Huber_reg.fit(X_train, y_train)
    Huber_reg_score = Huber_reg.score(X_test, y_test)
    print("Huber Testing Accuracy: ", Huber_reg_score)
    print("=====================================")

    Ransac_reg = lm.RANSACRegressor()
    Ransac_reg.fit(X_train, y_train)
    Ransac_reg_score = Ransac_reg.score(X_test, y_test)
    print("Ransac Testing Accuracy: ", Ransac_reg_score)
    print("=====================================")

    # Theil regression is not working (Low memory)
    # Theil_reg = lm.TheilSenRegressor()
    # Theil_reg.fit(X_train, y_train)
    # Theil_reg_score = Theil_reg.score(X_test, y_test)
    # print("Theil Testing Accuracy: ", Theil_reg_score)

    models = [log_reg, Decision_tree, lin_reg, Sgd_reg, Ridge_reg, Lasso_reg, Elastic_reg, Huber_reg, Ransac_reg]
    scores = [log_reg_score,
              Decision_tree_score,
              lin_reg_score,
              Sgd_reg_score,
              Ridge_reg_score,
              Lasso_reg_score,
              Elastic_reg_score,
              Huber_reg_score,
              Ransac_reg_score]

    print("---------------------------------------------------------------------------------")

    return models, scores


if __name__ == '__main__':
    data = read_csv('Data/creditcard.csv')
    # Data_Summary(data)
    # Visualize_Data(data)
    X_train, X_test, y_train, y_test = Preprocessing(data)

    models, scores = Machine_Learning(X_train, X_test, y_train, y_test)
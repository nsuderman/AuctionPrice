from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from eda import explore_data as eda


def train_and_testsplit(data, tcol, test_size=0.3):
    x = data.drop(tcol, axis=1)
    y = data[tcol]

    return train_test_split(x, y, test_size=test_size, random_state=50)


def build_model(model_name, model, data, tcol):
    x_train, x_test, y_train, y_test = train_and_testsplit(data, tcol)

    # Scale features
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    X_train_scaled = scaler.transform(x_train)
    X_test_scaled = scaler.transform(x_test)
    X_all = scaler.transform(data.drop(['price'], axis=1))

    model.fit(X_train_scaled, y_train)
    y_pred_train = model.predict(X_test_scaled)
    rmse_train = np.sqrt(mean_squared_error(model.predict(X_train_scaled), y_train))
    rmse_predict = np.sqrt(mean_squared_error(y_test, y_pred_train))
    r2score = r2_score(y_test, y_pred_train)

    y_pred_all = model.predict(X_all)

    temp = [model_name, rmse_train, rmse_predict, float(r2score), y_pred_train, model, scaler, y_pred_all]

    return temp


def run_models(df, tcol):
    col_names = ["Model_Name", "RMSE_Train", "RMSE_Predict", "R_Square", "Y_Pred_Train", "Model", "Scaler",
                 "Y_Pred_All"]
    result = pd.DataFrame(columns=col_names)

    models = {"Linear_Regression": LinearRegression(),
              "Lasso_Regression": Lasso(),
              "Ridge_Regression": Ridge(),
              "Elastic_Regression": ElasticNet(),
              "Decision_Tree_Regressor": DecisionTreeRegressor(),
              "Ada_Boost_Regressor": AdaBoostRegressor(),
              "Random_Forest_Regressor": RandomForestRegressor(),
              "Gradient_Boosting_Regressor": GradientBoostingRegressor(),
              "XG_Boost_Regressor": XGBRegressor(),
              "Support_Vector_Regressor": SVR(),
              "K_Nearest_Neighbor_Regressor": KNeighborsRegressor()}

    for m in models:
        result.loc[len(result)] = build_model(m, models[m], df, tcol)

    best_model_record = result[result.R_Square == result.R_Square.max()]
    print(f'\nBest Model is {best_model_record.iloc[0, 0]} with an R2 of {format(best_model_record.iloc[0, 3], ".3f")}')

    return result

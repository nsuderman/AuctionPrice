import pandas as pd
from scipy.stats import pearsonr
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from model import models


folder = os.path.abspath(os.getcwd()) + '\solution\\'


def question_1(df):
    # What are the most important characteristics and features that determine the selling price of a used car?
    # ~~~~~~~~~~~~~Find Significant/insignificant P Values from PearsonR
    col_names = ["feature", "corr_coeff", "p_value", "significant"]
    sig_features = pd.DataFrame(columns=col_names)
    for col in df.drop(['price'],axis=1):
        pearson_coeff, p_value = pearsonr(df[col], df['price'])
        significant=None
        if p_value < .05:
            significant = True
        else:
            significant = False
        sig_features.loc[len(sig_features)] = [col, pearson_coeff, p_value, significant]

    sig_features.sort_values(by=['p_value'], inplace=True)
    print('\nSignificant Features on Price:\n',sig_features[sig_features.p_value < .05])

    sig_features.to_csv(folder + '1. significant_features.csv',index=False)
    return sig_features


def question_2A(df):
    # Does a relative change in selling price over time differ significantly with respect to any of
    # the car characteristics, e.g., color, price range or features?

    #Output DF
    col_names = ["dependent", "independent_one", "independent_two", "p_value", "significant"]
    sig_cor_features = pd.DataFrame(columns=col_names)

    columns = ['paint_color', 'car_type', 'engine_power', 'feature_1', 'feature_2', 'feature_3', 'feature_4',
                    'feature_5', 'feature_6', 'feature_7', 'feature_8','registration_date']
    for col in columns:
        model = ols(f'price ~ C(sold_month) + C({col}) + C(sold_month):C({col})', data=df).fit()
        anova = sm.stats.anova_lm(model, typ=2)
        p_value = anova.loc[f'C(sold_month):C({col})']['PR(>F)']

        if p_value < .05:
            significant = True

            fig, ax1 = plt.subplots(figsize=(15, 15))
            ax2 = ax1.twinx()
            sns.lineplot(x='sold_month', y='price', hue=col, data=df, ci=None, ax=ax1)
            sns.countplot(x='sold_month', data=df, palette="Blues_d", alpha=.1, ax=ax2,
                          order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            plt.title(
                f'sold_month and {col} impact on price: \n 2-Way ANOVA shows statistical Significance at P-value '
                f'{format(p_value, ".5f")}', fontdict={'fontsize': 20})
            plt.xlabel('xlabel', fontsize=18)
            plt.ylabel('ylabel', fontsize=16)
            plt.savefig(folder + '\charts\\' + f'2a. price_month_{col}.png')
        else:
            significant = False
        # print(f'\n\n2-Way ANOVA {col} and Sold Month \n', anova)
        sig_cor_features.loc[len(sig_cor_features)] = ['price', 'sold_month', col,float(format(p_value, ".6f")),significant]

    print('\nSignificant Features Overtime on Price:\n',sig_cor_features[sig_cor_features.p_value < .05])
    sig_cor_features.to_csv(folder + '2a. significant_features_price_month.csv',index=False)
    return sig_cor_features


def question_2B(df): #!!!!!!!!!!!!!!!Needs work
    # ----------------------------
    # Are there any statistically significant seasonality patterns in pricing, e.g.certain car types being more
    # expensive in summer than winter?

    colors=[]
    text=""
    for val in df["paint_color"].unique():
        temp_df = df[df['paint_color'] == val]
        model = ols('price ~ C(sold_month)', data=temp_df).fit()
        anova = sm.stats.anova_lm(model, typ=2)
        p_value = anova.loc['C(sold_month)']['PR(>F)']

        if p_value < .05:
            colors.append(val)
            text = '\n' + text + f' {val} is Statistically with a P-Value of {format(p_value,".3f")}'

    colors.append('red')
    fig, ax1 = plt.subplots(figsize=(15, 15))
    ax2 = ax1.twinx()
    sns.lineplot(x='sold_month', y='price', hue="paint_color", data=df, ci=None, ax=ax1, hue_order=colors)
    sns.countplot(x='sold_month', data=df, palette="Blues_d", alpha=.1, ax=ax2, order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.title('Paint Color and Seasonality Impact on Price'+text, fontdict={'fontsize': 20})
    plt.savefig(folder + '\charts\\' + '2b. seasonality_pattern_paint.png')


    #, hue_order = ['coupe', 'convertible']
    type = []
    text = ""
    for val in df["car_type"].unique():
        temp_df = df[df['car_type'] == val]
        model = ols('price ~ C(sold_month)', data=temp_df).fit()
        anova = sm.stats.anova_lm(model, typ=2)
        p_value = anova.loc['C(sold_month)']['PR(>F)']

        if p_value < .05:
            type.append(val)
            text = text + f'\n{val} is Statistically with a P-Value of {format(p_value, ".3f")}'

    type.append('convertible')
    fig, ax1 = plt.subplots(figsize=(15, 15))
    ax1 = sns.lineplot(x='sold_month', y='price', hue="car_type", data=df, ci=None, hue_order=type)
    ax2 = ax1.twinx()
    ax2 = sns.countplot(x='sold_month', data=df, palette="Blues_d", alpha=.1, order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.title('Vehicle Type and Seasonality Impact on Price'+text, fontdict={'fontsize': 20})
    plt.savefig(folder + '\charts\\' + '2b. seasonality_pattern_type.png')

    fig, ax1 = plt.subplots(figsize=(15, 15))
    ax1 = sns.lineplot(x='sold_month', y='price',data=df, ci=None)
    ax2 = ax1.twinx()
    ax2 = sns.countplot(x='sold_month', data=df, palette="Blues_d", alpha=.1, order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.title('Sales Price By Month Plot', fontdict={'fontsize': 20})
    plt.savefig(folder + '\charts\\' + '2b. sales_price_by_month.png')


def question_3(df,scaled_df,model_df):
    # v
    df['y_predict_orig'] = model_df['Y_Pred_All'].iloc[0]

    # --- Create a copy of the scaled data for a future prediction
    scaled_df_future = scaled_df.copy()
    scaled_df_future["registration_date"] = scaled_df_future["registration_date"] - 10000
    scaled_df_future["mileage"] = scaled_df_future["mileage"] + 10000

    corr = scaled_df[['price','mileage','registration_date']].corr(method='pearson')
    print('\nCorreleation Matrix for Mileage and Registration Date\n',corr)


    # --- Run Prediction prediction
    scaler = model_df['Scaler'].iloc[0]
    X_scaled = scaler.transform(scaled_df_future.drop(['price'],axis=1))

    model = model_df['Model'].iloc[0]
    y_pred = model.predict(X_scaled)
    df['y_predict_future'] = y_pred
    df['loss_measure'] = df['y_predict_orig'] - df['y_predict_future']

    print(f'\nMinimum Loss is {df.loss_measure.min()}')
    print(f'Average Loss is {df.loss_measure.mean()}')
    print(f'Maximum Loss is {df.loss_measure.max()}')

    df_corr = scaled_df.corr()

    filtered_rows = df[(df.price > 20000) & (df.price < 30000)]
    filtered_rows = filtered_rows[filtered_rows.loss_measure < filtered_rows.loss_measure.quantile(0.25)]
    filtered_rows.sort_values(by=['loss_measure'],inplace=True)

    # similiar_cars_df = df[(df.model_key == best_buy['model_key'].iloc[0]) &
    #                        (df.mileage < best_buy['mileage'].iloc[0] + 10000) &
    #                        (df.mileage > best_buy['mileage'].iloc[0] - 10000) &
    #                        (df['registration_date'].str[-4:] == best_buy['registration_date'].iloc[0][-4:])]

    filtered_rows.head(10).to_csv(folder + '3. top_10_car_buys.csv')

    return df,filtered_rows.head(10)


def question_4(df,model_results):
    # Please share the out-of-sample accuracy metric for the model you used to answer the above questions.
    model_results.to_csv(folder + '4. best_model_results.csv')

    best_model_df = model_results[model_results.R_Square == model_results.R_Square.max()]

    results = pd.DataFrame({'Minimum_Loss':[df.loss_measure.min()],
                            'Average_Loss':[df.loss_measure.mean()],
                            'Maximum_Loss':[df.loss_measure.max()],
                            'Model': [best_model_df['Model_Name'].iloc[0]],
                            'Model_R2': [best_model_df['R_Square'].iloc[0]]})

    results.to_csv(folder + '4. model_metrics.csv')

    sns.distplot(df.loss_measure)
    plt.figure(figsize=(12, 10))
    sns.distplot(x=df.loss_measure)
    plt.title('Loss Measure Distribution Plot')
    plt.savefig(folder + '4. loss_measure_dist.png')




def question_5(df):  # !!!!!!!!!!!!!!!Needs work
    # Feel free to share any other interesting insights worth mentioning.

    pivot = pd.pivot_table(df, values='price', index=['sold_month'],aggfunc=('count'))
    pivot['month'] =pivot.index
    pivot['agg_func'] = 'Count'

    pivot1 = pd.pivot_table(df, values='price', index=['sold_month'],aggfunc=('mean'))
    pivot1['month'] = pivot1.index
    pivot1['agg_func'] = 'Average'
    results_pivot=pd.concat([pivot,pivot1])

    pivot2 = pd.pivot_table(df, values='price', index=['sold_month'],aggfunc=('sum'))
    pivot2['month'] = pivot1.index
    pivot2['agg_func'] = 'Total'
    results_pivot=pd.concat([results_pivot,pivot2])
    pivot.rename(columns={"price": "value"})

    pivot3 = pd.DataFrame({'month':[1,2,3,4,5,6,7,8,9],'price':pivot1['price'].pct_change()})
    pivot3['agg_func'] = 'Ave Sales % Change'
    results_pivot = pd.concat([results_pivot, pivot3])

    pivot4 = pd.DataFrame({'month': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'price': pivot2['price'].pct_change()})
    pivot4['agg_func'] = 'Total Sales % Change'
    results_pivot = pd.concat([results_pivot, pivot4])
    results_pivot.rename(columns={'price':'value'},inplace=True)
    results_pivot = results_pivot.fillna(0)

    count_sales = results_pivot[results_pivot.agg_func.isin(['Count'])]
    ave_sales = results_pivot[results_pivot.agg_func.isin(['Ave Sales % Change'])]
    total_sales = results_pivot[results_pivot.agg_func.isin(['Total Sales % Change'])]
    pearson_coeff1, p_value1 = pearsonr(ave_sales['value'], total_sales['value'])
    pearson_coeff2, p_value2 = pearsonr(count_sales['value'], total_sales['value'])

    fig, ax1 = plt.subplots(figsize=(15, 15))
    ax1 = sns.lineplot(x='month', y='value', hue='agg_func',ci=None,
                       data=results_pivot[results_pivot.agg_func.isin(['Ave Sales % Change','Total Sales % Change'])])
    ax1.set_ylabel("Sales Stats % Change")
    ax2 = ax1.twinx()
    ax2 = sns.barplot(x='month', y="value",palette="Blues_d",alpha=.1,
                      data=results_pivot[results_pivot.agg_func == 'Count'],order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ax2.set_ylabel("Cars Sold")
    plt.title(f'Sales Price - Percent Change\n\n'
              f'Ave Sales to Total Sales Significant P-Value of {format(p_value1,".3f")} with a correlation of {format(pearson_coeff1,".3f")} \n'
              f'Cars Sold to Total Sales Significant P-Value of {format(p_value2,".3f")} with a correlation of {format(pearson_coeff2,".3f")}\n\n'
              'It Appears Higher Sales has a Positive Leveraged Impact on Total Sales', fontdict={'fontsize': 20})
    plt.savefig(folder + '\charts\\' + '5. sales_perc_change.png')

import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder

plt.figure(figsize=(12,10))
sns.set_style("darkgrid")

chart_folder = os.path.abspath(os.getcwd()) + '\eda\charts\\'


def normalize_skew(array_series):
    return np.log(array_series)


def inverse_skew(array_series):
    return np.exp(array_series)


def VIF_colinearity(ind_var):
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif = pd.DataFrame()
    vif["vif_factor"] = [variance_inflation_factor(ind_var.values,i) for i in range(ind_var.shape[1])]
    vif["Feature"] = ind_var.columns

    return vif.sort_values("vif_factor", ascending=False)


def apply_PCA(x):
    col = []
    ncom = len(x.columns)

    from sklearn.preprocessing import StandardScaler
    x = StandardScaler().fit_transform(x)

    from sklearn.decomposition import PCA

    for i in range(1, ncom):
        pca = PCA(n_components=i)
        p_components = pca.fit_transform(x)
        exp_var_ratio = np.cumsum(pca.explained_variance_ratio_)
        if exp_var_ratio[i - 1] > 0.9:
            n_components = i
            break

    print("Explained Variance Ratio after PCA is: ", exp_var_ratio)

    # creating dataframe
    for j in range(1, n_components + 1):
        col.append("pc" + str(j))
    pcom = pd.DataFrame(data=p_components, columns=col)

    return pcom


def custom_summary(data):
    result = []
    for col in data.columns:
        stats = OrderedDict({"column_name":col,
                    "Count":round(data[col].count(),2),
                    "Minimum":round(data[col].min(),2),
                    "Quartile 1":round(data[col].quantile(0.25),2),
                    "Mean":round(data[col].mean(),2),
                    "Median":round(data[col].median(),2),
                    "Mode":round(data[col].mode(),2),
                    "Quartile 3":round(data[col].quantile(0.75),2),
                    "Maximum":round(data[col].max(),2),
                    "Variance":round(data[col].var(),2),
                    "Std. Dev.":round(data[col].std(),2),
                    "Kurtosis":round(data[col].kurt(),2),
                    "Skewness":round(data[col].skew(),2),
                    "IQR":data[col].quantile(0.75)-data[col].quantile(0.25)})
        result.append(stats)
        if data[col].skew()<-1:
            sk_label = "Highly Negatively Skewed"
        elif -1<=data[col].skew()<-0.5:
            sk_label = "Moderately Negatively Skewed"
        elif -0.5<= data[col].skew()<0:
            sk_label = "Fairly Symmetric (Negative)"
        elif 0<=data[col].skew()<0.5:
            sk_label= "Fairly Symmetric (Positive)"
        elif 0.5<=data[col].skew()<1:
            sk_label= "Moderately Positively Skewed"
        elif data[col].skew()>1:
            sk_label="Highly Positively Skewed"
        else:
            sk_label='error'
        stats['skewness comment']=sk_label

        #Outlier comment
        upperlim = stats['Quartile 3']+(1.5*stats['IQR'])
        lowerlim = stats['Quartile 1']-(1.5*stats['IQR'])
        if len([x for x in data[col] if x < lowerlim or x > upperlim]) > 0:
            outliercomments = 'Has Outlier'
        else:
            outliercomments = 'Has no outliers'
        stats['Outlier Comment'] = outliercomments
        result_df = pd.DataFrame(result)
    return result_df


def replace_outlier(data, col, method='Quartile', strategy='Median', print_results=True):
    col_data = data[col]

    # Using Quartile to set values
    if method == 'Quartile':
        q2 = data[col].median()
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        low_limit = q1 - 1.5 * iqr
        up_limit = q3 + 1.5 * iqr

    # Using STD to set values
    elif method == 'Standard Deviation':
        col_mean = data[col].mean()
        col_std = data[col].std()
        cutoff = col_std * 2
        low_limit = col_mean - cutoff
        up_limit = col_mean + cutoff

    else:
        if print_results:
            print("Error: Pass a correct method")

    # Printing Outliers
    outliers = data.loc[(col_data < low_limit) | (col_data > up_limit), col]
    outlier_density = round((len(outliers) / len(data)), 2)
    if len(outliers) == 0:
        if print_results:
            print(f'Feature\'{col}\'does not have any outlier')
    else:
        if print_results:
            print(f'Total no. of outliers are: {len(outliers)}\n')
            print(f'Outlier percentage: {outlier_density}\n')
            print(f'Outliers for \'{col}\'are : {np.sort(np.array(outliers))}\n')
            print(data[(col_data < low_limit) | (col_data > up_limit)])

    # Replacing Outliers

    if strategy == 'Median':
        data.loc[(col_data < low_limit) | (col_data > up_limit), col] = q2
    elif strategy == 'Mean':
        data.loc[(col_data < low_limit) | (col_data > up_limit), col] = col_mean
    else:
        if print_results:
            print("Error: Pass a correct Strategy")

    return data


def run_analysis_preprossing(df,print_results=False):
    if print_results:
        print('\nExploring Data')
        print(df.info())
        print(df.describe())
    print(df.info())

    scaled = pd.DataFrame()

    category_col = list(df.select_dtypes(['object']).columns)
    bool_col = list(df.select_dtypes(['bool']).columns)
    numeric_col = list(df.select_dtypes(['int64']).columns)
    numeric_col.remove('price')

    # All vehicle makes are BMW
    df.drop(['maker_key'],axis=1, inplace=True)

    # Find the missing data in the columns
    cols_missing_data = [col for col in df.columns if df[col].isnull().sum() > 0]
    if print_results:
        print('\nColumns missing data:', cols_missing_data)

    # Ordinal Data
    df['sold_at'] = pd.to_datetime(df["sold_at"]).dt.strftime("%Y%m%d").astype(int)
    df['registration_date'] = pd.to_datetime(df["registration_date"]).dt.strftime("%Y%m%d").astype(int)

    # Feature Engineering
    df['sold_month'] = df["sold_at"].astype(str).str[4:6].astype(int)
    df['sold_year'] = df["sold_at"].astype(str).str[:4].astype(int)
    df['summer'] = df['sold_month'].apply(lambda x: 1 if x in [3,4,5,6,7,8] else 0)
    scaled['summer'] = df['summer']
    df['diesel'] = df['fuel'].apply(lambda x: 1 if x == 'diesel' else 0)
    scaled['diesel'] = df['diesel']

    # Add Continuous data to scaled df
    scaled['registration_date'] = df['registration_date']
    scaled['sold_at'] = df['sold_at']
    scaled['mileage'] = df['mileage']
    scaled['engine_power'] = df['engine_power']
    scaled['sold_month'] = df['sold_year']
    scaled['sold_year'] = df['sold_year']

    if print_results:
        duplicate_rows_df = df[df.duplicated()]
        print('number of duplicate rows: ', duplicate_rows_df.shape)


    # Replace Outliers
    # for col in numeric_col:
    #   scaled = replace_outlier(scaled,col,method='Standard Deviation',strategy='Mean',print_results=print_results)

    for col in scaled.columns:
        plt.figure(figsize=(12,10))
        sns.scatterplot(x=scaled[col],y=df['price'])
        plt.title(f'{col} Scatter Plot')
        plt.savefig(chart_folder + f'{col}_scatter.png')

        plt.figure(figsize=(12, 10))
        sns.distplot(x=scaled[col])
        plt.title(f'{col} Distribution Plot')
        plt.savefig(chart_folder + f'{col}_Dist.png')
    #sns.boxplot(x=df['price'])
    #plt.show()


    # boolean to int
    for col in bool_col:
        scaled[col] = df[col].astype(int)

    # One Hot Encode ---Not using because I am getting better results with the Label Encoder
    # ohe = ['paint_color','car_type']
    # for col in ohe:
    #     dummies = pd.get_dummies(df[col], prefix=col,drop_first=True)
    #     scaled = pd.concat([scaled, dummies], axis=1)

    le = LabelEncoder()
    cols = ['model_key','paint_color','car_type']
    scaled[cols] = df[cols].apply(lambda x: le.fit_transform(x))

    # View Correlation
    # temp_df = pd.concat([scaled, df['price']], axis=1)
    # matrix = np.triu(temp_df.corr())
    # plt.figure(figsize=(24, 20))
    # sns.heatmap(temp_df.corr(), annot=True, mask=matrix, cmap='coolwarm')
    # plt.title(f'Correlation Matrix')
    # plt.savefig(chart_folder + 'correlation.png')

    scaled.drop(['sold_month','sold_year','paint_color','feature_7','sold_at','diesel','summer'],axis=1,inplace=True)

    # View Correlation
    temp_df = pd.concat([scaled, df['price']], axis=1)
    matrix = np.triu(temp_df.corr())
    plt.figure(figsize=(24, 20))
    sns.heatmap(temp_df.corr(), annot=True, mask=matrix, cmap='coolwarm')
    plt.title(f'Correlation Matrix')
    plt.savefig(chart_folder + 'correlation.png')


    # Find multicollinearity
    vif_col = VIF_colinearity(scaled)
    print(vif_col)

    # skewed_cols = ['registration_date','mileage','engine_power']
    # for c in skewed_cols:
    #     scaled[c] = scaled.loc[scaled[c] < 0, c] = scaled[c].mean()
    #     scaled[c] = normalize_skew(scaled[c])

    # Find Custom Summary
    summary = custom_summary(scaled)
    print(summary)

    scaled['price'] = df['price']

    return scaled, df
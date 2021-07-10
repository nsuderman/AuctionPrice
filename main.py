from data_collection import get_data
from eda import explore_data
from model import models
from solution import hp_questions as solution
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    # Data Collection~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    df = get_data.data_to_df()
    df_orig=df.copy()

    # Exploritory Data Analysis~~~~~~~~~~~~~~~~~~~~~~~~~~
    processed_data, df_not_scaled = explore_data.run_analysis_preprossing(df)

    # Model Data~~~~~~~~~~~~~~~~~~~~~~~~~~
    model_results = models.run_models(processed_data,'price')
    best_model_df = model_results[model_results.R_Square == model_results.R_Square.max()]


    # --------------------------
    # What are the most important characteristics and features that determine the selling price of a used car?
    sig_features = solution.question_1(processed_data)

    # ~~~~Analyze features against sold month to see changes against price
    sig_correlation_features = solution.question_2A(df_not_scaled)

    # ~~~~Analyze features against seasonality to see how prices change
    solution.question_2B(df_not_scaled)
    # sig_features = sig_features['feature'].tolist()
    #
    # plt.figure(figsize=(24, 20))
    # sns.pairplot(df[sig_features])
    # plt.savefig(chart_folder + 'pairplot.png')

    # ~~~~~~~~~~~Predict Something
    df_orig, top_10_best_buys = solution.question_3(df_orig,processed_data,best_model_df)
    best_buy = top_10_best_buys.head(1)
    print('The Best Car to buy is ', best_buy)


    # ~~~~~~~~~~~Measurement of Prediction
    solution.question_4(df_orig,model_results)

    # ~~~~~~~~~~~Other Observations
    solution.question_5(df_not_scaled)

    a=1




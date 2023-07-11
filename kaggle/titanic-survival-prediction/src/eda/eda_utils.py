# Exploratory Data Analysis.
from ydata_profiling import ProfileReport
import ppscore as pps
from autoviz.AutoViz_Class import AutoViz_Class

# Plotting.
import matplotlib.pyplot as plt
import seaborn as sns


def pandas_profile(df, title, output_file):
    """
    EDA function to apply pandas-profiling to a dataframe.
    Generates profile in HTML and renders as notebook frame.
    Stores HTML report to disk.

    Documentation: https://ydata-profiling.ydata.ai/docs
    """

    print("ProfileReport()...")
    profile = ProfileReport(df, title=title)

    print("profile.to_file()...")
    profile.to_file(output_file)

    print("profile.to_notebook_iframe()....")
    profile.to_notebook_iframe()


def generate_pps(df, target):
    """
    Generate a heatmap of predictive power scores.

    Documentation: https://github.com/8080labs/ppscore
    """

    pps.predictors(df, target)
    matrix_df = pps.matrix(df)[['x',
                                'y',
                                'ppscore']].pivot(columns='x',
                                                  index='y',
                                                  values='ppscore')
    sns.set(font_scale=.8)
    sns.heatmap(matrix_df,
                vmin=0,
                vmax=1,
                linewidths=1.5,
                annot=True,
                fmt=".2f")

    plt.show()


def autovisualize(df, target):
    """
    Generate a suite of visualizations for a dataframe.
    """
    AV = AutoViz_Class()

    dft = AV.AutoViz(filename='',
                     sep=',',
                     depVar=target,
                     dfte=df,
                     header=0,
                     verbose=1)



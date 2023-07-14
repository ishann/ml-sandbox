import pandas as pd

# Exploratory Data Analysis.
from ydata_profiling import ProfileReport
import ppscore as pps
from autoviz.AutoViz_Class import AutoViz_Class

# Plotting.
import matplotlib.pyplot as plt
import seaborn as sns

# Manifold visualization.
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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



def pca(df_trn):
    """
    Perform PCA. Pick first 2 PCs and visualize.
    """
    
    # Remove "object" dtypes and target variable.
    df_manifold = df_trn.drop(labels=["Name", "Ticket", "Survived"], axis=1)
    
    # Normalize before fitting and transforming.
    _, M = df_manifold.shape
    mean_, std_ = df_manifold.mean(), df_manifold.std()
    df_manifold=(df_manifold-mean_)/std_
    
    pca = PCA(n_components=M)
    df_manifold = pd.DataFrame(pca.fit_transform(df_manifold))
    df_manifold["Survived"] = df_trn["Survived"]
    
    plt.figure(figsize=(8,8))
    sns.scatterplot(
        x=0, y=1,
        palette=sns.color_palette("Dark2"),
        data=df_manifold,
        hue="Survived",
        legend="full",
    )

def tsne(df_trn):
    """
    Perform t-SNE. Pick first 2 PCs and visualize.
    """
    df_manifold = df_trn.drop(labels=["Name", "Ticket", "Survived"], axis=1)
    _, M = df_manifold.shape
    mean_, std_ = df_manifold.mean(), df_manifold.std()
    df_manifold=(df_manifold-mean_)/std_
    
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(df_manifold)
    
    df_manifold['tsne-2d-one'] = tsne_results[:,0]
    df_manifold['tsne-2d-two'] = tsne_results[:,1]
    df_manifold["Survived"] = df_trn["Survived"]
    
    plt.figure(figsize=(8,8))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        palette=sns.color_palette("Dark2"),
        data=df_manifold,
        hue="Survived",
        legend="full",
    )


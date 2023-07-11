import pandas as pd
import numpy as np

def str_to_numeric(df):
    """
    Convert string values to numeric values.
    TODO: Consider automatically detecting which
          string features to convert to numberics.
    """
    df["Sex"].replace({"female": 0,
                       "male":1},
                       inplace=True)

    df["Embarked"].replace({np.nan: 0,
                            "C": 1,
                            "S": 2,
                            "Q": 3},
                            inplace=True)

    return df

def impute_age(df_trn, df_tst):
    """
    Train: Use the median age of each (Pclass, SibSp) group, ie,
           correlated features, to fill missing values with the median within each group.
    Test: Impute age by using the median age of the corresponding
          (Pclass, SibSp) group from the training set.
    """
    print(df_trn.corr(numeric_only=True)["Age"].abs().sort_values(ascending=False)[1:])
    print("=> Age is significantly correlated with Pclass and SibSp.")

    df_trn["Age"] = df_trn.groupby(["Pclass", "SibSp"])["Age"].apply(lambda x: x.fillna(x.median()))
    
    # (Pclass, SibSp) = (3, 8) has only one sample, so we fill it with the median of
    # (Pclass, SibSp) = (3, 5), the closest available SibSp class.
    med_pclass3_sibsp5 = df_trn[df_trn["Pclass"]==3][df_trn["SibSp"]==3]["Age"].median()
    df_trn["Age"].fillna(med_pclass3_sibsp5, inplace=True)
    
    # Use medians from df_trn to fill missing values in df_tst.
    def fill_missing_age(df, med_age):
        
        pclass, sibsp = df["Pclass"].values[0], df["SibSp"].values[0]
        age = med_age[(pclass, sibsp)]
        
        df["Age"].fillna(age, inplace=True)
    
        return df
    
    med_age = df_trn.groupby(["Pclass", "SibSp"])["Age"].median() 
    
    df_tst["Age"] = df_tst.groupby(["Pclass", "SibSp"]).apply(
                        lambda x: fill_missing_age(x, med_age))["Age"]
    
    return df_trn, df_tst

def impute_cabin(df_trn, df_tst):
    """
    First, discretize cabins based on letter. Then, find correlation and
    figure out that Pclass is correlated with discretized cabin feature.
    
    Train: Use the median cabin class of each Pclass group to fill
           missing values with the median within each group.
    
    Test: Impute age by using the median cabin of the corresponding
          Pclass group from the training set.
 
    """
    
    # Figure out correlated features for discretized Cabin.
    mapper = {"Z": 0,
              "A":1, "B":2, "C":3, "D":4,
              "E":5, "F":6, "G":7, "T":8}

    cabin = df_trn.dropna(subset="Cabin")
    cabin["Cabin"] = cabin["Cabin"].apply(
                       lambda x: x[0]).astype("string").map(mapper)
    print(cabin.corr(numeric_only=True)["Cabin"].abs().sort_values(ascending=False)[1:])
    print("=> discretized Cabin feature is highly correlated with Pclass.",
          "\nSo, can use Pclass to impute Cabin.")
    
    def discretize_cabin(cabin_name):
    
        # if cabin_name.isna():
        if pd.isna(cabin_name):
            #print("Z for nan")
            return float("nan")
        else:
            #print("{} for {}".format(cabin_name[0], cabin_name))
            return cabin_name[0]
    
    df_trn["Cabin"] = df_trn["Cabin"].apply(
                        lambda x: discretize_cabin(x)).astype("string").map(mapper)
    
    df_trn["Cabin"] = df_trn.groupby(["Pclass"])["Cabin"].apply(lambda x: x.fillna(x.median()))
    
    # Use medians from df_trn to fill missing values in df_tst.
    def fill_missing_cabin(df, med_cabin):
        
        pclass = df["Pclass"].values[0]
        cabin = med_cabin[(pclass)]
        df["Cabin"].fillna(cabin, inplace=True)
        return df
    
    med_cabin = df_trn.groupby(["Pclass"])["Cabin"].median().astype(int)
    df_tst["Cabin"] = df_tst.groupby("Pclass").apply(
                        lambda x: fill_missing_cabin(x, med_cabin))["Cabin"]
    
    return df_trn, df_tst
    
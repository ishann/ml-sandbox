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
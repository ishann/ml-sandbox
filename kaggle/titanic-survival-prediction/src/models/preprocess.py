from sklearn.model_selection import train_test_split

def prep_feats_for_modeling(df_trn, df_tst, split_ratio=0.2):
    """
    Prepares features for modeling (both train and test sets).
    Split the training data into training and validation sets.
    """

    # Drop "Name" and "Ticket".
    df_trn = df_trn.drop(["Name", "Ticket"], axis=1)
    df_tst = df_tst.drop(["Name", "Ticket"], axis=1)
    
    # Split the training data into features and target variable.
    trn_X, trn_Y = df_trn.drop(["Survived"], axis=1), df_trn["Survived"]
    
    # Split the training data into training and validation sets.
    trn_X, val_X, trn_Y, val_Y = train_test_split(trn_X, trn_Y,
                                                  test_size=split_ratio,
                                                  random_state=42)
    
    trn_X, trn_Y = trn_X.to_numpy(), trn_Y.to_numpy()
    val_X, val_Y = val_X.to_numpy(), val_Y.to_numpy()

    return trn_X, trn_Y, val_X, val_Y, df_tst




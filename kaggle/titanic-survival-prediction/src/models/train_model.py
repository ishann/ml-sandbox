from lazypredict.Supervised import LazyClassifier

def baseline(trn_X, val_X, trn_Y, val_Y):
    """
    Get baselines using LazyPredict.
    """
    clf = LazyClassifier(verbose=0,
                         ignore_warnings=True,
                         custom_metric=None)
    models, _ = clf.fit(trn_X, val_X, trn_Y, val_Y)

    print(models)

    return models
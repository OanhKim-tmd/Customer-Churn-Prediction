import os
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)




def evaluate_model(model, X_test, y_test):
    """
    Calculate evaluation metrics
    """


    y_pred = model.predict(
        X_test
    )


    result = {

        "Accuracy": round(
            accuracy_score(
                y_test,
                y_pred
            ),
            4
        ),


        "Precision": round(
            precision_score(
                y_test,
                y_pred,
                zero_division=0
            ),
            4
        ),


        "Recall": round(
            recall_score(
                y_test,
                y_pred,
                zero_division=0
            ),
            4
        ),


        "F1-Score": round(
            f1_score(
                y_test,
                y_pred,
                zero_division=0
            ),
            4
        )
    }



    # Check probability output

    if hasattr(
        model,
        "predict_proba"
    ):

        y_prob = model.predict_proba(
            X_test
        )[:,1]


        result["ROC-AUC"] = round(
            roc_auc_score(
                y_test,
                y_prob
            ),
            4
        )


        result["PR-AUC"] = round(
            average_precision_score(
                y_test,
                y_prob
            ),
            4
        )


    else:

        result["ROC-AUC"] = None
        result["PR-AUC"] = None



    return result






def evaluate_all_models(models, X_test, y_test):
    """
    Evaluate all trained models
    """


    results = []


    for name, model in models.items():

        print(
            f"Evaluating {name}..."
        )


        score = evaluate_model(
            model,
            X_test,
            y_test
        )


        score["Model"] = name


        results.append(
            score
        )



    return pd.DataFrame(results)







def plot_confusion_matrix(
        model,
        X_test,
        y_test,
        model_name,
        save_dir
):


    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test
    )


    plt.title(
        f"{model_name} Confusion Matrix"
    )


    plt.savefig(
        os.path.join(
            save_dir,
            f"{model_name.lower()}_confusion_matrix.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()







def plot_roc_curve(
        model,
        X_test,
        y_test,
        model_name,
        save_dir
):


    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test
    )


    plt.title(
        f"{model_name} ROC Curve"
    )


    plt.savefig(
        os.path.join(
            save_dir,
            f"{model_name.lower()}_roc_curve.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()







def plot_precision_recall_curve(
        model,
        X_test,
        y_test,
        model_name,
        save_dir
):


    PrecisionRecallDisplay.from_estimator(
        model,
        X_test,
        y_test
    )


    plt.title(
        f"{model_name} Precision Recall Curve"
    )


    plt.savefig(
        os.path.join(
            save_dir,
            f"{model_name.lower()}_pr_curve.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()







def plot_feature_importance(
        model,
        feature_names,
        save_dir
):


    if not hasattr(
        model,
        "feature_importances_"
    ):
        return



    importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }
    )


    importance = importance.sort_values(
        by="Importance",
        ascending=False
    ).head(15)



    plt.figure(
        figsize=(8,6)
    )


    plt.barh(
        importance["Feature"],
        importance["Importance"]
    )


    plt.xlabel(
        "Importance"
    )


    plt.ylabel(
        "Feature"
    )


    plt.gca().invert_yaxis()


    plt.title(
        "Random Forest Feature Importance"
    )


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            save_dir,
            "random_forest_feature_importance.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()





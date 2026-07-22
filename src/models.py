import os
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score



def get_models():
    """
    Khai bao cac mo hinh su dung
    """

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced"
        ),


        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )
    }


    return models




def train_models(
        X_train,
        y_train,
        output_dir="../outputs/models"
):
    """
    Train models, evaluate using cross validation
    and save trained models
    """

    models = get_models()

    trained_models = {}

    cv_results = []


    os.makedirs(
        output_dir,
        exist_ok=True
    )


    print("--- Training Models ---")


    for name, model in models.items():


        print(
            f"Training {name} ..."
        )


        # Cross validation before final training

        cv_score = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="f1"
        )


        mean_cv = cv_score.mean()


        print(
            "CV F1 Score:",
            round(mean_cv,4)
        )


        cv_results.append(
            {
                "Model": name,
                "CV_F1": round(mean_cv,4)
            }
        )



        # Train final model

        model.fit(
            X_train,
            y_train
        )


        trained_models[name] = model



        # Save model

        filename = (
            name
            .replace(" ","_")
            .lower()
            + ".pkl"
        )


        joblib.dump(
            model,
            os.path.join(
                output_dir,
                filename
            )
        )


        print(
            f"Saved {filename}"
        )


        print("-"*30)



    # Save CV result

    cv_df = pd.DataFrame(
        cv_results
    )


    cv_df.to_csv(
        "../outputs/cv_results.csv",
        index=False
    )


    return trained_models




def save_feature_importance(
        model,
        feature_names,
        output_path="../outputs/feature_importance.csv"
):
    """
    Lay feature importance tu Random Forest
    """

    if hasattr(
        model,
        "feature_importances_"
    ):


        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": model.feature_importances_
            }
        )


        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        )


        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )


        importance_df.to_csv(
            output_path,
            index=False
        )


        print(
            "Saved feature importance"
        )




if __name__ == "__main__":

    from preprocessing import preprocess_data


    df = pd.read_csv(
        "../data/processed/cleaned_data.csv"
    )


    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)



    trained_models = train_models(
        X_train,
        y_train
    )


    rf_model = trained_models[
        "Random Forest"
    ]


    save_feature_importance(
        rf_model,
        X_train.columns
    )
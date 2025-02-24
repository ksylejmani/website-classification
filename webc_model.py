# Import necessary libraries
import pandas as pd
import xgboost as xgb
import re
import shap
import eli5
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import PartialDependenceDisplay
from eli5.sklearn import PermutationImportance
import seaborn as sns
import matplotlib.pyplot as plt

class WebsiteClassification:
    """
    This class handles the entire pipeline of website classification using machine learning.
    The pipeline includes data preprocessing, feature extraction, model training, and model interpretation.
    """

    def __init__(self):
        """
        Initializes the WebsiteClassification class by loading the dataset,
        performing feature extraction, training the model, and running various evaluations.
        """
        self.file_name = "website_classification"
        self.dataset = self.load_data(self.file_name)
        self.dataset = self.add_features(self.dataset)
        self.dataset = self.encode_category(self.dataset)
        self.train_set, self.validation_set = self.split_data(self.dataset)
        self.model = self.train_xgboost_model(self.train_set, self.validation_set)
        self.evaluate_model(self.model, self.validation_set)
        self.run_feature_analysis(self.model, self.validation_set)

    def load_data(self, file_name):
        """Loads dataset from the specified CSV file."""
        with open(f"{file_name}.csv", encoding="utf8") as file:
            return pd.read_csv(file)

    def split_data(self, dataset, test_size=0.4, random_state=42):
        """Splits the dataset into training and validation sets."""
        dataset = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)
        return train_test_split(dataset, test_size=test_size, random_state=random_state)

    def add_features(self, dataset):
        """
        Adds additional features to the dataset:
        - Word count
        - Unique word count
        - Average word length
        - Special character count
        - Sentiment score
        - URL length
        - Subdomain count
        """
        dataset['word_count'] = dataset['cleaned_website_text'].apply(lambda x: len(x.split()))
        dataset['unique_word_count'] = dataset['cleaned_website_text'].apply(lambda x: len(set(x.split())))
        dataset['avg_word_length'] = dataset['cleaned_website_text'].apply(lambda x: sum(len(word) for word in x.split()) / len(x.split()))
        dataset['special_char_count'] = dataset['cleaned_website_text'].apply(lambda x: len(re.findall(r'[^\w\s]', x)))
        dataset['sentiment'] = dataset['cleaned_website_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
        dataset['url_length'] = dataset['website_url'].apply(lambda x: len(x))
        dataset['subdomain_count'] = dataset['website_url'].apply(lambda x: len(re.findall(r'\.', x)) - 1)
        return dataset

    def encode_category(self, dataset):
        """Encodes the target category as numeric values."""
        label_encoder = LabelEncoder()
        dataset['Category'] = label_encoder.fit_transform(dataset['Category'])
        return dataset

    def train_xgboost_model(self, train_set, validation_set):
        """
        Trains an XGBoost classifier on the training set and evaluates on the validation set.
        Returns the trained model.
        """
        X_train = train_set.drop(columns=['Category', 'website_url', 'cleaned_website_text'])
        y_train = train_set['Category']
        X_val = validation_set.drop(columns=['Category', 'website_url', 'cleaned_website_text'])
        y_val = validation_set['Category']

        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=100)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

        return model

    def evaluate_model(self, model, validation_set):
        """
        Evaluates the trained model on the validation set and prints the performance metrics.
        """
        X_val = validation_set.drop(columns=['Category', 'website_url', 'cleaned_website_text'])
        y_val = validation_set['Category']
        y_pred = model.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred, average='macro')
        precision = precision_score(y_val, y_pred, average='macro')
        f1 = f1_score(y_val, y_pred, average='macro')
        confusion = confusion_matrix(y_val, y_pred)

        print(f"Validation Accuracy: {accuracy}")
        print(f"Validation Recall: {recall}")
        print(f"Validation Precision: {precision}")
        print(f"Validation F1 Score: {f1}")
        print(f"Confusion Matrix:\n{confusion}")

    def run_feature_analysis(self, model, validation_set):
        """
        Runs various feature analysis techniques, including:
        - Feature correlation visualization
        - Permutation importance
        - Partial dependence plots
        - SHAP values for model interpretability
        """
        self.visualize_feature_correlations(self.dataset)
        self.show_permutation_importance(model, validation_set)
        self.show_partial_dependence(model, validation_set, 'url_length')
        self.show_2d_partial_dependence(model, validation_set, ('subdomain_count', 'url_length'))
        self.explain_with_shap(model, validation_set)

    def visualize_feature_correlations(self, dataset):
        """Generates a heatmap of feature correlations."""
        dataset = dataset.drop(columns=['website_url', 'cleaned_website_text'])
        corr = dataset.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.show()

    def show_permutation_importance(self, model, validation_set):
        """Displays permutation importance for model features."""
        X_val = validation_set.drop(columns=['Category', 'website_url', 'cleaned_website_text'])
        y_val = validation_set['Category']

        perm = PermutationImportance(model, random_state=42).fit(X_val, y_val)
        print(eli5.format_as_text(eli5.explain_weights(perm, feature_names=X_val.columns.tolist())))

    def show_partial_dependence(self, model, validation_set, feature):
        """Plots partial dependence for a single feature."""
        X_val = validation_set.drop(columns=['Category', 'website_url', 'cleaned_website_text'])
        PartialDependenceDisplay.from_estimator(model, X_val, [feature], target=0)
        plt.title(f'Partial Dependence of {feature}')
        plt.show()

    def show_2d_partial_dependence(self, model, validation_set, features):
        """Plots 2D partial dependence for a pair of features."""
        X_val = validation_set.drop(columns=['Category', 'website_url', 'cleaned_website_text'])
        fig, ax = plt.subplots(figsize=(8, 6))
        feature_names = [tuple(features)]
        PartialDependenceDisplay.from_estimator(model, X_val, feature_names, target=0, ax=ax)
        plt.show()

    def explain_with_shap(self, model, validation_set):
        """Generates SHAP values and visualizes model interpretability."""
        X_val = validation_set.drop(columns=['Category', 'website_url', 'cleaned_website_text'])
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val)

        shap.initjs()
        force_plot = shap.force_plot(explainer.expected_value[1], shap_values[1], X_val)
        shap.save_html("shap_force_plot.html", force_plot)


if __name__ == "__main__":
    # Run the solution pipeline
    solution = WebsiteClassification()

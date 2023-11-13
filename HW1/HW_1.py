import pandas as pd
import numpy as np
import nltk
import ssl
nltk.download('all')
import re
from bs4 import BeautifulSoup
import sys
import subprocess
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.linear_model import Perceptron
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import warnings
warnings.filterwarnings('ignore')

def select_data(df):
    df = df[['review_body','star_rating']]
    df = df.dropna()
    data = df[df['star_rating']== 1].sample(n=20000)
    for i in range(2,6):
        data1 = df[df['star_rating']== i].sample(n=20000)
        data = data.append(data1, ignore_index=True)
    return data

# Data clearning
def word_lower(data):
    data['review_body'] = data['review_body'].str.lower()
    return data

def remove_HTML_URL(text):
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub('<.*?>+', '', text)
    return text

def remove_non_alphabetical(text):
    text = re.sub(r'[^A-Za-z0-9 ]+', '',text)
    return text

def remove_extra_spaces(text):
    res = " ".join(text.split())
    return res

def perform_contractions(text):
    list1 = []
    for word in text.split():
        list1.append(contractions.fix(word))
    return " ".join(list1)

def remove_stop_words(text):
    stop_words= stopwords.words('english')
    list1 = []
    for word in text.split():
        if word not in stop_words:
            list1.append(word)
    return ' '.join(list1)

def perform_lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    list1 = []
    for word in text.split():
        word = lemmatizer.lemmatize(word)
        list1.append(word)
    return ' '.join(list1)

def TF_IDF(list1):
    tr_idf_model  = TfidfVectorizer(max_features = 1300)
    tf_idf_vector = tr_idf_model.fit_transform(list1)
    tf_idf_array = tf_idf_vector.toarray()
    words_set = tr_idf_model.get_feature_names()
    df_tf_idf = pd.DataFrame(tf_idf_array, columns = words_set)
    return df_tf_idf, tf_idf_array

def print_summary(summary):
    list1 = [summary['class 0']['precision'],summary['class 1']['precision'],summary['class 2']['precision'],summary['class 3']['precision'],summary['class 4']['precision']]
    list2 = [summary['class 0']['recall'],summary['class 1']['recall'],summary['class 2']['recall'],summary['class 3']['recall'],summary['class 4']['recall']]
    list3 = [summary['class 0']['f1-score'],summary['class 1']['f1-score'],summary['class 2']['f1-score'],summary['class 3']['f1-score'],summary['class 4']['f1-score']]
    print(summary['class 0']['precision'],",",summary['class 0']['recall'],",",summary['class 0']['f1-score'])
    print(summary['class 1']['precision'],",",summary['class 1']['recall'],",",summary['class 1']['f1-score'])
    print(summary['class 2']['precision'],",",summary['class 2']['recall'],",",summary['class 2']['f1-score'])
    print(summary['class 3']['precision'],",",summary['class 3']['recall'],",",summary['class 3']['f1-score'])
    print(summary['class 4']['precision'],",",summary['class 4']['recall'],",",summary['class 4']['f1-score'])
    print(sum(list1)/len(list1),",",sum(list2)/len(list2),",",sum(list3)/len(list3))



if __name__ == '__main__':
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'contractions'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'bs4'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'nltk'])
    input_file = sys.argv[1]
    df = pd.read_table(input_file, index_col = False, sep="\\t")
    data = select_data(df)
    data['star_rating'] = data['star_rating'].apply(int)

    # data clearning
    data = word_lower(data)
    data['review_body'] = data['review_body'].apply(lambda x: remove_HTML_URL(x))
    data['review_body'] = data['review_body'].apply(lambda x: remove_non_alphabetical(x))
    data['review_body'] = data['review_body'].apply(lambda x: remove_extra_spaces(x))
    data['review_body'] = data['review_body'].apply(lambda x: perform_contractions(x))
    average_length_before = data['review_body'].apply(len).mean()
    after_average_length = data['review_body'].apply(len).mean()
    # print("step2:")
    print(average_length_before,",",after_average_length)

    #Preprocessing
    data['review_body'] = data['review_body'].apply(lambda x: remove_stop_words(x))
    data['review_body'] = data['review_body'].apply(lambda x: perform_lemmatization(x))
    after_pro_average_length = data['review_body'].apply(len).mean()
    # print("step3:")
    print(after_average_length,",",after_pro_average_length)

    #Feature Extraction
    review = data['review_body'].to_list()
    tf_idf_df,tf_idf_array = TF_IDF(review)
    X = tf_idf_array
    y = data['star_rating']
    x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(X, y,test_size=0.2, stratify = y, random_state=42)

    #Perceptron
    # print(f"Perceptron model:")
    Perceptron_model = Perceptron(eta0=0.01)
    Perceptron_model.fit(x_train_data,y_train_data)
    y_predict_data = Perceptron_model.predict(x_test_data)
    target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4']
    summary = classification_report(y_test_data, y_predict_data, target_names=target_names, output_dict = True)
    print_summary(summary)

    #SVM
    # print(f"SVM model:")
    SVM = SVC(kernel='linear')
    SVM.fit(x_train_data,y_train_data)
    y_predict_data = SVM.predict(x_test_data)
    summary = classification_report(y_test_data, y_predict_data, target_names=target_names, output_dict = True)
    print_summary(summary)

    #Logistic Regression
    # print(f"Logistic Regression model:")
    logistic_model = LogisticRegression(random_state=0)
    logistic_model.fit(x_train_data,y_train_data)
    y_predict_data = logistic_model.predict(x_test_data)
    summary = classification_report(y_test_data, y_predict_data, target_names=target_names, output_dict = True)
    print_summary(summary)

    #Multinomial Naive Bayes
    # print(f"Multinomial Naive Bayes model:")
    MNB_model = MultinomialNB()
    MNB_model.fit(x_train_data,y_train_data)
    y_predict_data = MNB_model.predict(x_test_data)
    summary = classification_report(y_test_data, y_predict_data, target_names=target_names, output_dict = True)
    print_summary(summary)
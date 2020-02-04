import urllib.request
import zipfile

import pandas as pd


# Dataset being adapted from Cortez, P. and Silva, A. “Using Data Mining to
# Predict Secondary School Student Performance.” In: A. Brito and J. Teixeira
# (eds), Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC
# 2008), pp. 5-12, Porto, Portugal, April, 2008, EUROSIS. (Resource suggested
# by Mark E. Fenner, in the book "Machine Learning with Python for Everyone")


def grab_student_numeric_discrete():
    # download zip file and unzip
    # unzipping unknown files can be a security hazard

    url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/00320/'
           + 'student.zip')

    urllib.request.urlretrieve(url, 'port_student.zip')
    zipfile.ZipFile('port_student.zip').extract('student-mat.csv')

    # preprocessing:
    df = pd.read_csv('student-mat.csv', sep=';')

    # Notes from the book (preparation for the exercises):
    # "g1 & g2 are highly correlated with g3; dropping them makes the problem
    # significantly harder we also remove all non-numeric columns and discretize
    # the final grade by 0-50-75-100 percentile which were determined by hand"

    df = df.drop(columns=['G1', 'G2']).select_dtypes(include=['number'])
    df['grade'] = pd.cut(df['G3'], [0, 11, 14, 20],
                         labels=['low', 'mid', 'high'],
                         include_lowest=True)

    df.drop(columns=['G3'], inplace=True)
    df.to_csv('portuguese_student_numeric_discrete.csv', index=False)


if __name__ == "__main__":
    grab_student_numeric_discrete()

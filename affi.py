import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation
from sklearn import preprocessing
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--path","-p")
parser.add_argument("--index","-i")

parsed = parser.parse_args()

path = parsed.path
index = parsed.index


def read_data(path,index):
	df=pd.read_csv(path,index_col=index)
	return df

def  lbl_encoder(df):
	label_encoder = preprocessing.LabelEncoder()
	for f in df.columns:
		if df[f].dtype == 'object':
			label_encoder.fit(list(df[f].values))
			df[f] = label_encoder.transform(list(df[f].values))
	return df		
				
def stnd_scaler(df):
	scaler = StandardScaler()
	df_std = scaler.fit_transform(df)
	return df_std
def affinity(df_std):
	AffPro = AffinityPropagation(max_iter=300, preference=-50, verbose=True)
	aa = AffPro.fit(df_std)
	return aa
def result_aff(df,aa):
	df_segm_aff = df.copy()
	df_segm_aff['Segment affinity'] = aa.labels_
	df_segm_analysis = df_segm_aff.groupby(['Segment affinity']).mean()
	print(df_segm_analysis)

def main():
	df = read_data(path,index)
	df = lbl_encoder(df)
	df_std = stnd_scaler(df)
	aa = affinity(df_std)
	result_aff(df,aa)

if __name__ == "__main__":
    main()

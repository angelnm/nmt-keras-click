import pandas as pd
import sys

#file_path='./Click_Europarl.csv'
#lan_pairs=['de_en', 'en_de', 'es_en', 'en_es', 'fr_en', 'en_fr']
#csv_file = pd.read_csv(file_path)


def data_pair_lan(pair):
	info = ""
	csv_pair_lan = csv_file.query('lenguas == "' + pair + '"')
	csv_query = csv_pair_lan.query('click_max == 0').iloc[0]
	info += "{} & ".format(csv_query['MAR'])
	info += "{} & ".format(csv_query['WSR'])
	csv_query = csv_pair_lan.query('click_max == 1').iloc[0]
	info += "{} & ".format(csv_query['MAR'])
	info += "{} & ".format(csv_query['WSR'])
	info += "{} & ".format(csv_query['WSR_red'])
	csv_query = csv_pair_lan.query('click_max == 5').iloc[0]
	info += "{} & ".format(csv_query['MAR'])
	info += "{} & ".format(csv_query['WSR'])
	info += "{}".format(csv_query['WSR_red'])
	print(info)

def data_WSR_MAR_uMAR(pair):
	info = {'WSR': "", 'MAR':"", 'uMAR':""}

	csv_pair_lan = csv_file.query('lenguas == "' + pair + '"')

	for key in info:
		csv_query = csv_pair_lan[key]
		for idx, value in enumerate(csv_query):
			info[key] += "({},{})".format(idx, value)


	print("WSR: " + info['WSR'])
	print("MAR: " + info['MAR'])
	print("uMAR: " + info['uMAR'])


def concatenate_csv(files_path):
	data_frames = []
	for file in files_path:
		df = pd.read_csv(file)
		#df = df.drop(columns=['id'])
		data_frames.append(df)

	df = pd.concat(data_frames)
	df = df.sort_values(by=['threshold'])
	df = df.drop_duplicates(subset='threshold')

	df.to_csv('./data.csv', index=False)



if __name__ == "__main__":
	concatenate_csv(sys.argv[1:])
	
#for pair in lan_pairs:
#	print('Language pair: {}'.format(pair))
#	data_WSR_MAR_uMAR(pair)

import json
import pandas as pd


name_list = []
gender_list = []
age_list = []

for user in range(1, 78):

    with open('./실험결과/인적사항/{}.json'.format(user), 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    name_list.append(json_data['Name'])
    gender_list.append(json_data['Gender'])
    age_list.append(json_data['Age'])

profile = [name_list, gender_list, age_list]
df_profile = pd.DataFrame(index=['user{}'.format(idx) for idx in range(1, 78)])
df_profile['Name'] = name_list
df_profile['Gender'] = gender_list
df_profile['Age'] = age_list
# print(df_profile)
df_profile.to_csv('./result/profile.csv', encoding='utf-8-sig')

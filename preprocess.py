import pandas as pd
import numpy as np
from utils import changemodel, changemodelname, changeyear, engine, guar

data = 'cars.csv'
df = pd.read_csv(data)

matchfuel={'디젤':0, 'LPG':1, '가솔린':2}
matchop={'무':0,'유':1}
insurinfo={'미등록':0,'등록':1}

df['이름'] = df['이름'].apply(changemodel).apply(changemodelname)
df['연식'] = df['연식'].apply(changeyear)
df['주행거리'] = df['주행거리'].apply(lambda x:x.replace(",","")[:-2].rstrip())
df['연료']=df['연료'].replace(matchfuel)
df['배기량']=df['배기량'].apply(engine)

guartable=df['보증정보'].apply(guar)
guartable.columns=['보증여부','보증기간','보증거리']
df=pd.concat([df,guartable],axis=1)
df=df.drop(['보증정보'],axis=1)

df=df.replace(matchop)

df['보험이력등록']=df['보험이력등록'].replace(insurinfo)
df['보험_내차피해(가격)']=df['보험_내차피해(가격)'].apply(lambda x: x.replace(",","") if isinstance(x,str) else np.nan)
df['보험_타차피해(가격)']=df['보험_타차피해(가격)'].apply(lambda x: x.replace(",","") if isinstance(x,str) else np.nan)

df['가격']=df['가격'].apply(lambda x:x.replace('만','0000').replace(',','')[:-1])
df['신차대비가격']=df['신차대비가격'].apply(lambda x: np.nan if x=='준비중' else x)

df=df.loc[:,['이름','연식', '주행거리', '연료', '배기량', '색상', '옵션_선루프', '옵션_파노라마선루프', 
    '옵션_열선앞', '옵션_열선뒤', '옵션_전방센서','옵션_후방센서', '옵션_전방캠', '옵션_후방캠', '옵션_어라운드뷰', '옵션_네비순정', 
    '보험이력등록','소유자변경횟수', '사고상세_전손', '사고상세_침수전손', '사고상세_침수분손', '사고상세_도난',
    '보험_내차피해(횟수)', '보험_내차피해(가격)', '사고상세_타차가해(횟수)', '보험_타차피해(가격)','가격','신차대비가격','링크']]
df.to_csv('cars_processed.csv')


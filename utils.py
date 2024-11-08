import pandas as pd
import numpy as np

def changemodel(str1):
    index1=str1.find(' ')
    index2=str1.find(' ',index1+1)
    index3=str1.find(' ',index2+1)
    return str1[:index3]

def changemodelname(str1):
    if '그랜져' in str1:
        str1=str1.replace("그랜져","그랜저")
        return str1
    else:
        return str1
    
def changeyear(string):
  year=int(string[:4])
  month=int(string[5:7])
  n=12*(2022-year)+month
  return n

def engine(str1):
    index=str1.find('cc')
    return str1[:index].replace(',','').rstrip()

def guar(str1):
    if str1=='만료' or str1=='불가':
        return pd.Series([0,0,0])
    elif str1=='정보없음':
        return pd.Series([np.nan,np.nan,np.nan])
    else:
        index=str1.find('/')
        if index==-1:
            time=0
        else:
            time=int(str1[:index-3].strip())
        km=str1[index+1:].replace(",","")[:-2].strip()
        return pd.Series([1,time,km])


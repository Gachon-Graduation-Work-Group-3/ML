import pandas as pd
import numpy as np
from datetime import timedelta

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
  
  if pd.isna(string):
        return np.nan  # NaN으로 반환
    
  # 문자열이 아닌 경우 처리
  if not isinstance(string, str):
        string = str(string)

  string = string[0:7]
  year=int(string[:4])
  month=int(string[5:7])
  n=(2024 + 12/12) - (year + month/12)
  return float(n)

def engine(x):
    if x is None:  # 기본값 처리
        return np.nan  # NaN으로 반환

    if not isinstance(x, str):  # 문자열이 아닌 경우 처리
        if pd.isna(x):
            return np.nan  # NaN으로 반환
        x = str(x)
    index=x.find('cc')
    return x[:index].replace(',','').rstrip()

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
    
def clean_distance(x):
    if pd.isna(x):  # NaN 값 처리
        return np.nan  # NaN으로 반환
    if not isinstance(x, str):  # 문자열이 아닌 경우 처리
        x = str(x)
    cleaned = x.replace(",", "").rstrip()[:-3]
    return float(cleaned)

def process_price(x):
    if pd.isna(x):  # NaN 값 처리
        return np.nan
    x = str(x)  # 문자열로 변환
    x = x.replace('만', '0000').replace(',', '')
    if x == '상담0000':  # '상담' 처리
        return np.nan
    return x

def parse_date(date_str):
    date_str = str(date_str)
    # 상대적인 날짜 (예: "5일전")
    if "일전" in date_str:
        days_ago = int(date_str.replace("일전", "").strip("[]").replace("'", ""))  # "5" 추출
        return pd.Timestamp.today() - timedelta(days=days_ago)
    
    # 절대적인 날짜 (예: "15.12.24")
    else:
        try:
            return pd.to_datetime(date_str.strip("[]").replace("'", ""), format="%y.%m.%d")
        except ValueError:
            return pd.NaT  # 오류 발생시 NaT 반환


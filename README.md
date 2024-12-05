# Variable
|Variable|Description|Type|
|:---:|:---:|:---:|
|new_price|신차 가격(만원)|Integer|
|**date**|**최초 등록일(현재-최초등록일)**|**Float**|
|**view**|**조회수**|**Integer**|
|age|연식(현재-출시일)|Float|
|km|주행거리|Integer|
|cc|배기량|Integer|
|high_out|최고출력|Float|
|fuel_eff|연비|Float|
|brand|브랜드|Categorical|

# Target Variable
|Variable|Description|Type|
|:---:|:---:|:---:|
|comp|신차대비가격(중고가격/신차가격*100)|Float|

# Performance
|Project|Accuracy|Train RMSE|Test RMSE|
|:---:|:---:|:---:|:---:|:---:|
|국산 중고 자동차 가격 예측 및영향요인 분석|X|0.0348|0.0790|
|**When Your Car**|**0.91**|**0.0325**|**0.0756**|

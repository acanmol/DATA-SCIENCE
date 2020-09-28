import numpy as np
import pandas as pan
from sklearn.linear_model import LogisticRegression
ds=pan.read_csv("heart_data.csv")
itemlist = []
for i in range(0,360):
    item = [ds.loc[i]['age'], ds.loc[i]['blood_pressure'], ds.loc[i]['cholestoral'],ds.loc[i]['max_heart_rate']]
    itemlist.append(item)

x=np.array(itemlist)

y=ds['heart_disease']
y=np.array(y)
#print(x)
log=LogisticRegression(C=1e5,solver='lbfgs',multi_class='multinomial',n_jobs=-1)
log.fit(x,y)
print(log.predict([[12,90,187,80]]))
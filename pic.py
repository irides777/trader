import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import *
objs = ['rb', 'bu', 'OI', 'i']
a, b, c = 0, 0, 0
fin = []
cha = []
for obj in objs:
    if os.path.exists(f'res/{obj}/trade/sum.csv'):
        fin.append(obj)
        data = pd.read_csv(f'res/{obj}/trade/sum.csv')
        a+=data.drl.sum()
        b+=data.pss.sum()
        c+=data['agg'].sum()
        cha.append((obj, data.pss.sum()-data.drl.sum()))
cha = sorted(cha, key=lambda x:x[1])
# data = data[data.date!=20220119]
a, b = [x[0] for x in cha], [x[1] for x in cha]
plt.subplots(figsize=(10,5))
plt.ylabel('pss-drl')
plt.bar(a, b)
plt.show()

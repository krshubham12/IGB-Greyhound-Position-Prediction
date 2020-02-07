
import pandas as pd
import urllib3
http = urllib3.PoolManager()

venue = ['Clonmel', 'Curraheen Park', 'Derry', 'Drumbo Park', 'Dundalk', 'Enniscorthy', 'Galway', 'Kilkenny', 'Lifford', 'Limerick', 'Longford', 'Mullingar', 'Newbridge', 'Shelbourne Park', 'Thurles Park', 'Tralee', 'Waterford', 'Youghal']
venue_code = ['CML','CRK','DRY','DBP','DLK','ECY','GLY','KKY','LFD','LMK','LGD','MGR','NWB','SPK','THR','TRL','WFD','YGL']

race_date = []
from datetime import timedelta, date
def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)

start_dt = date(2018, 1, 1)#year month day
end_dt = date(2018, 2, 1)
for dt in daterange(start_dt, end_dt):
    print(dt.strftime("%d")+'-'+dt.strftime("%b")+'-'+dt.strftime("%Y"))
    race_date.append(dt.strftime("%d")+'-'+dt.strftime("%b")+'-'+dt.strftime("%Y"))

cont_lis = []
raw_lis = []
Date = []
Location = []
test = 'Race 1'
for i in venue_code:
    for j in race_date:
        url = "https://www.igb.ie/results/view-results/?track="+i+"&date="+j
        response = http.request('GET', url)
        raw = response.data.decode('utf-8')
        raw_lis.append(raw)
        if test in raw:
            Date.append(j)
            Location.append(i)

raw_lis_fin = []
for i in raw_lis:
    if test in i:
        raw_lis_fin.append(i)

df = pd.DataFrame(list(zip(raw_lis_fin,Date,Location)),
                  columns=['rawdata','Date','Venue'])
df.to_csv('scrapped_data.csv',index=False)

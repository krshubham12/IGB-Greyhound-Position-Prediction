import pandas as pd
from bs4 import BeautifulSoup as bs

df = pd.read_csv('scrapped_data.csv')

def pre(raw):
    flat=[]
    soup = bs(raw['rawdata'])
    hs = soup.find_all('h4')
    for h in hs:
        inner_text = h.text
        flat.append(inner_text[-10:-2])

    x=0
    tr_data = []
    trs = soup.find_all('tr')
    for t in trs:
        if ' View' in t.text or 'Pos.' in t.text:
            continue
        if t.text[:2]==' 1':
            x+=1
        inner_text = t.text
        inner_text = inner_text+'\n'+flat[x-1][-3:]
        tr_data.append(inner_text)

    soup_a = bs(raw['rawdata'])
    ar = soup_a.find_all('a')
    test=[]
    for h in ar:
        inner_text = h.text
        test.append(inner_text)
    test = [i for i in test if i]
    test = test[38:]
    test = test[:-21]
    gr=[]
    si=[]
    dam=[]
    n=3
    final = [test[i * n:(i + 1) * n] for i in range((len(test) + n - 1) // n )]
    for i in final:
        gr.append(i[0])
        si.append(i[1])
        dam.append(i[2])

    Pos = []
    Prize= []
    Wt= []
    WinTime= []
    By= []
    Going= []
    EstTime= []
    SP= []
    Grade= []
    Comm= []
    Flat_s=[]
    Date= []
    Venue= []
    for i in tr_data:
        if len(i.split("\n"))==14:
            i=i.split("\n")
            Pos.append( i[0])
            Prize.append(i[3].replace('€',''))
            Wt.append(i[4])
            WinTime.append(i[5])
            By.append(i[6])
            Going.append(i[7])
            EstTime.append(i[8].replace('&nbsp',''))
            SP.append(i[9])
            Grade.append(i[10])
            Comm.append(i[11])
            Flat_s.append(i[13])
            Date.append(raw['Date'])
            Venue.append(raw['Venue'])
    dict = {'Pos.':Pos,'Greyhound':gr,'SIRE NAME':si,
        'DAM NAME':dam,'Prize':Prize,'Wt.':Wt,
        'Win Time':WinTime,'By':By,'Going':Going,
        'Est Time':EstTime,'SP.':SP,'Grade':Grade,
        'Comm.':Comm,'Flat':Flat_s,'Date':Date,'Venue':Venue}
    return dict


data = pd.DataFrame(pre(df.iloc[1]))
for i in range(2,len(df['rawdata'])):
    temp=pd.DataFrame(pre(df.iloc[i]))
    data=data.append(temp)
data.index=pd.RangeIndex(len(data.index))
data.to_csv('IGBfinaldata.csv',index=False)

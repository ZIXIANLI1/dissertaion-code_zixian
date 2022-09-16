# dissertaion-code_zixian
Appendices
#########################################################
# For the Chapter 2, we use the Python to write the code#
#########################################################
#Chapter 2 Pattern and bias recognition#
########################################
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
pip install plotly
pip show plotly
pip show show_solutions
import pandas as pd
import numpy as np
from scipy import stats 
# based on matplotlib
import matplotlib.pyplot as plt
# based on plotly
import plotly as py
import plotly.express as px
import plotly.graph_objects as go
py.offline.init_notebook_mode(connected =True)
from plotly.subplots import make_subplots
# based on seaborn
import seaborn as sns
#from show_solutions import show

data = pd.read_excel('MSCIACWI7.4.xlsx')
data.head(5)
# delete the useless columns
df = data.drop(columns = ['ISSUER_NAME','IVA_RATING_DATE','ISSUERID','ISSUER_TICKER','ISSUER_CUSIP','ISSUER_SEDOL','ISSUER_ISIN'])
df.head(5)
df.shape
# data types
df.dtypes
# data types statistics
pd.value_counts(df.dtypes.values)
# check whether there are missing values
df.isnull().sum()
# delecte the observations which contain the missing values
df.dropna(subset = ['ESG_SCORE','ESG_RATING','EVIC_EUR'],inplace=True)
df.shape

# the distribution of ESG scores
sns.distplot(df['ESG_SCORE'], kde=True)
# statistics
np.mean(df['ESG_SCORE'])
np.std(df['ESG_SCORE'])
#skew 
stats.skew(df['ESG_SCORE'])
#kurtosis
stats.kurtosis(df['ESG_SCORE'])
# breadth of ESG scoring
min(data['ESG_SCORE'])
max(data['ESG_SCORE'])
# the distribution of value of companies
sns.distplot(data['EVIC_EUR'], kde=True)
max(data['EVIC_EUR'])
min(data['EVIC_EUR'])
#delect the maximum 
max(df['EVIC_EUR'])
data['EVIC_EUR'].idxmax()
price = data['EVIC_EUR']
price.drop([997])
price.idxmax()
price.drop([997])
price.idxmax()
price.drop([997])
sns.distplot(price, kde=True)

plt.figure(figsize=(10, 6), dpi=200)
plt.title('Distribution of value of companies in MSCI ACWI',fontsize=20)
plt.boxplot(price,vert=False,showmeans=True,
            patch_artist = True,
            boxprops = {'color':'#00b4d8','facecolor':'#a2d2ff'})
#range
max(df['EVIC_EUR'])
min(df['EVIC_EUR'])

# how many countries include:54
df['ISSUER_CNTRY_DOMICILE'].nunique()
#ranking 10 countries
df['ISSUER_CNTRY_DOMICILE'].value_counts()

# ESG Rating distribution
data['ESG_RATING'].value_counts()
#explain the relationship between esg ratings and esg scores

# industries distribution
# how many industries include:54
df['IVA_INDUSTRY'].nunique()
#ranking 10 industries
df['IVA_INDUSTRY'].value_counts()


# The dynamism of ESG scoring
data2013 = pd.read_excel('ESG Ratings Timeseries Expanded 2013.xlsx')
df2013 = data2013['IVA_COMPANY_RATING'].value_counts().rename_axis('ESG_rating').reset_index(name='2013').set_index(["ESG_rating"])
df2013
data2014 = pd.read_excel('ESG Ratings Timeseries Expanded 2014.xlsx')
df2014 = data2014['IVA_COMPANY_RATING'].value_counts().rename_axis('ESG_rating').reset_index(name='2014').set_index(["ESG_rating"])
df2014
data2015 = pd.read_excel('ESG Ratings Timeseries Expanded 2015.xlsx')
df2015 = data2015['IVA_COMPANY_RATING'].value_counts().rename_axis('ESG_rating').reset_index(name='2015').set_index(["ESG_rating"])
df2015
data2016 = pd.read_excel('ESG Ratings Timeseries Expanded 2016.xlsx')
df2016 = data2016['IVA_COMPANY_RATING'].value_counts().rename_axis('ESG_rating').reset_index(name='2016').set_index(["ESG_rating"])
df2016
data2017 = pd.read_excel('ESG Ratings Timeseries Expanded 2017.xlsx')
df2017 = data2017['IVA_COMPANY_RATING'].value_counts().rename_axis('ESG_rating').reset_index(name='2017').set_index(["ESG_rating"])
df2017
data2018 = pd.read_excel('ESG Ratings Timeseries Expanded 2018.xlsx')
df2018 = data2018['IVA_COMPANY_RATING'].value_counts().rename_axis('ESG_rating').reset_index(name='2018').set_index(["ESG_rating"])
df2018
data2019 = pd.read_excel('ESG Ratings Timeseries Expanded 2019.xlsx')
df2019 = data2019['IVA_COMPANY_RATING'].value_counts().rename_axis('ESG_rating').reset_index(name='2019').set_index(["ESG_rating"])
df2019
data2020 = pd.read_excel('ESG Ratings Timeseries Expanded 2020.xlsx')
df2020 = data2020['IVA_COMPANY_RATING'].value_counts().rename_axis('ESG_rating').reset_index(name='2020').set_index(["ESG_rating"])
df2020
data2021 = pd.read_excel('ESG Ratings Timeseries Expanded 2021.xlsx')
df2021 = data2021['IVA_COMPANY_RATING'].value_counts().rename_axis('ESG_rating').reset_index(name='2021').set_index(["ESG_rating"])
df2021

df_dyna = pd.concat([df2013,df2014,df2015,df2016,df2017,df2018,df2019,df2020,df2021],axis =1 )
#delete the last two rows
df_dynamic = df_dyna.T
plt.figure(figsize=(20, 20), dpi=200)
df_dynamic.plot.bar(stacked=True, alpha=0.5) 

plt.figure(figsize=(6,9))
labels = [u'United States', u'Japan', u'China', u'United Kingdom' ,u'Canada', u'Other']
sizes =[60.63% 5.45% 4.14% 3.90% 3.18% 22.70%]
patches, text1, text2 = plt.pie(sizes,
                                labels = labels,
                                )
								












*****************************************************************
*For the Chapter 3-Chapter 5, we use the Stata to write the code*
**************************************************************************
*Chapter 3 Predictive power of ESG performance on stock price performance*
**************************************************************************
clear
use "D:\SOR\ISIN21179.dta" 
keep if _merge ==3
drop _merge
duplicates report fullTICKER year
*keep the companies whose data exist from 2009-2021
encode fullTICKER, gen(id)
tabulate ISSUER_ISIN,sort
bys id:g n=_n
count if n==13
*keep 634 firms which have 13 year's data
bysort id: gen N=_N
keep if N==13
*correct the debt data according to the external supervisor
gen NET_DEBT = -NetDebt
*transformation
gen ARP = AR/100
gen lnAR = ln(AR)
gen lnBVPS =ln(BookValuePerShare)
gen lnSE = ln(ShareholdersEquity)
gen lnTA = ln(TA)
gen lnMV = ln(MarketValue)
gen lnOM = ln(OperatingMargins)
gen lnROE = ln(RoE)
gen lnND = ln(NetDebt)
gen lnTAE = ln(TAEQUITY)
gen lnindustryESG = ln(INDUSTRY_ADJUSTED_SCORE_MEAN)
gen lnweightESG = ln(WEIGHTED_AVERAGE_SCORE_MEAN)
gen lnE = ln(ENVIRONMENTAL_PILLAR_SCORE_MEAN)
gen lnS = ln(SOCIAL_PILLAR_SCORE_MEAN)
gen lnG = ln(GOVERNANCE_PILLAR_SCORE_MEAN)

*standardize
by id: egen mean_AR=mean(lnAR)
by id: egen sd_AR = sd(lnAR)
gen stanAR = (lnAR-mean_AR)/sd_AR

by id: egen mean_BVPS=mean(lnBVPS)
by id: egen sd_BVPS = sd(lnBVPS)
gen stanBVPS = (lnBVPS-mean_BVPS)/sd_BVPS


by id: egen mean_TA=mean(lnTA)
by id: egen sd_TA = sd(lnTA)
gen stanTA = (lnTA-mean_TA)/sd_TA

by id: egen mean_ShareholdersEquity=mean(lnSE)
by id: egen sd_ShareholdersEquity = sd(lnSE)
gen stanSE = (lnSE-mean_ShareholdersEquity)/sd_ShareholdersEquity

by id: egen mean_MarketValue=mean(lnMV)
by id: egen sd_MarketValue = sd(lnMV)
gen stanMV= (lnMV-mean_MarketValue)/sd_MarketValue

by id: egen mean_IESG=mean(lnindustryESG)
by id: egen sd_IESG = sd(lnindustryESG)
gen stanIESG= (lnindustryESG-mean_IESG)/sd_IESG

*using "1" to represent Financial company, while using 2 to represent nonfinancial firms
gen Type =2
replace Type = 1 if INDUSTRYSector== "Financial"

*tell whether need to deal with outliers
summarize stanAR stanBVPS stanSE stanTA stanMV OperatingMargins RoE ///
AnnualReturns BookValuePerShare ShareholdersEquity stanIESG///
NET_DEBT TA TAEQUITY MarketValue INDUSTRY_ADJUSTED_SCORE_MEAN ///
WEIGHTED_AVERAGE_SCORE_MEAN ENVIRONMENTAL_PILLAR_SCORE_MEAN ///
SOCIAL_PILLAR_SCORE_MEAN GOVERNANCE_PILLAR_SCORE_MEAN
hist ARP
hist INDUSTRY_ADJUSTED_SCORE_MEAN
hist WEIGHTED_AVERAGE_SCORE_MEAN
hist ENVIRONMENTAL_PILLAR_SCORE_MEAN
hist SOCIAL_PILLAR_SCORE_MEAN
hist GOVERNANCE_PILLAR_SCORE_MEAN
hist lnAR
hist lnBVPS 
hist lnTA 
hist lnSE 
hist lnMV

*winsorize
ssc install winsor2
winsor2 AR stanAR stanBVPS stanIESG stanTA stanSE stanMV OperatingMargins RoE AnnualReturns AR BookValuePerShare ShareholdersEquity ///
NET_DEBT TA TAEQUITY MarketValue INDUSTRY_ADJUSTED_SCORE_MEAN ///
WEIGHTED_AVERAGE_SCORE_MEAN ENVIRONMENTAL_PILLAR_SCORE_MEAN ///
SOCIAL_PILLAR_SCORE_MEAN GOVERNANCE_PILLAR_SCORE_MEAN lnAR lnOM lnROE ///
lnBVPS lnSE lnND lnTA lnTAE lnMV lnE lnS lnG,replace cuts(1 99)

*statistical description
ssc install logout
cd D:\SOR\dissertaion
logout,save(table) excel fix(1) dec(3) replace: 
tabstat INDUSTRY_ADJUSTED_SCORE_MEAN ///
WEIGHTED_AVERAGE_SCORE_MEAN ///
ENVIRONMENTAL_PILLAR_SCORE_MEAN SOCIAL_PILLAR_SCORE_MEAN ///
GOVERNANCE_PILLAR_SCORE_MEAN AnnualReturns OperatingMargins ///
RoE BookValuePerShare lnND lnTAE lnMV lnSE lnTA, by (Type) stat(count mean sd min max) long

raSE = exp(lnSE_01-lnSE_02)
raBVPS= exp(lnBVPS_01-lnBVPS_02)

save as "D:\SOR\varfinal.dta" 
********************************************************************
*chpater4 Contribution of companies’ ESG performance to their value*
********************************************************************
*overall ESG
clear
use "D:\SOR\varfinal.dta" 
*Model 1
drop if year ==2009
xtreg ARP raBVPS stanTA raSE stanMV i.year if Type ==1,fe
predict e1,xb 
gen mse1 = (ARP-e1)*(ARP-e1) 
sum mse1
*0.072
xtreg ARP raBVPS stanTA raSE stanMV i.year if Type ==2 ,fe 
predict e2,xb 
gen mse2 = (ARP-e2)*(ARP-e2) 
sum mse2
*0.070

*Model 2
xtreg ARP INDUSTRY_ADJUSTED_SCORE_MEAN_01 raBVPS stanTA raSE stanMV i.year if Type ==1 ,fe
predict e3,xb 
gen mse3 = (ARP-e3)*(ARP-e3) 
sum mse3

xtreg ARP INDUSTRY_ADJUSTED_SCORE_MEAN_01 raBVPS stanTA raSE stanMV i.year if Type ==2 ,fe  
predict e4,xb 
gen mse4 = (ARP-e4)*(ARP-e4) 
sum mse4

*Model 3
clear
use "D:\SOR\varfinal.dta" 
drop if year == 2009 &2010&2011&2012
xtreg ARP AR1 AR2 AR3 AR4 INDUSTRY_ADJUSTED_SCORE_MEAN_01 raBVPS stanTA raSE stanMV i.year if Type ==1 ,fe
predict e7,xb 
gen mse7 = (ARP-e7)*(ARP-e7) 
sum mse7
*RMSE = 0.2546
xtreg ARP AR1 AR2 AR3 AR4 INDUSTRY_ADJUSTED_SCORE_MEAN_01 raBVPS stanTA raSE stanMV i.year if Type ==2 ,fe  
predict e8,xb 
gen mse8 = (ARP-e8)*(ARP-e8) 
sum mse8
*RMSE = 0.2437
*Separate E,S,G pillar

*Model4
clear
use "D:\SOR\varfinal.dta" 
drop if year ==2009
xtreg ARP var157 ///
SOCIAL_PILLAR_SCORE_MEAN_01 GOVERNANCE_PILLAR_SCORE_MEAN_01 raBVPS stanTA raSE stanMV i.year if Type ==1 ,fe
est store m16
predict e5,xb 
gen mse5 = (ARP-e5)*(ARP-e5) 
sum mse5
*0.061
xtreg ARP var157 ///
SOCIAL_PILLAR_SCORE_MEAN_01 GOVERNANCE_PILLAR_SCORE_MEAN_01 raBVPS stanTA raSE stanMV i.year if Type ==2 ,fe  
est store m17
predict e6,xb 
gen mse6 = (ARP-e6)*(ARP-e6) 
sum mse6
*0.056

*Model 5
clear
use "D:\SOR\varfinal.dta" 
drop if year == 2009 &2010&2011&2012
xtreg ARP AR1 AR2 AR3 AR4 var157 ///
SOCIAL_PILLAR_SCORE_MEAN_01 GOVERNANCE_PILLAR_SCORE_MEAN_01 raBVPS stanTA raSE stanMV i.year if Type ==1 ,fe
est store m18
predict e10,xb 0.2552
gen mse10 = (ARP-e10)*(ARP-e10) 
sum mse10

xtreg ARP AR1 AR2 AR3 AR4 var157 ///
SOCIAL_PILLAR_SCORE_MEAN_01 GOVERNANCE_PILLAR_SCORE_MEAN raBVPS stanTA raSE stanMV i.year if Type ==2 ,fe  
est store m19
predict e9,xb 
gen mse9 = (ARP-e9)*(ARP-e9) 
sum mse9
*RMSE = 0.2447
esttab m16 m17 m18 m19 using D:\SOR\毕业论文\regout8.rtf,b(%12.3f) ///
se(%12.3f) nogap compress drop (*.year ) ///
s(N r2 ar2) star(* 0.1 ** 0.05 ** 0.01)

*evaluation of models
*evaluation of Model 3
*top 10%
clear
use "D:\SOR\varfinal.dta" 
gen ARP2= -ARP
drop if year == 2009 &2010&2011&2012
xtile esgfz = ARP, nq(10)
xtreg ARP AR1 AR2 AR3 AR4 INDUSTRY_ADJUSTED_SCORE_MEAN raBVPS stanTA raSE stanMV i.year if esgfz == 10&Type ==1,fe
predict e11,xb 
mean(e11)
*0.182

mean(ARP2) if Type ==1

clear
use "D:\SOR\varfinal.dta" 
gen ARP2= -ARP
drop if year == 2009 &2010&2011&2012
xtile esgfz = ARP, nq(10)
xtreg ARP AR1 AR2 AR3 AR4 INDUSTRY_ADJUSTED_SCORE_MEAN lnBVPS lnTA lnSE lnMV ///
i.year if esgfz == 10&Type ==2,fe
predict e10,xb 
mean(e10)
*0.268
mean(ARP2) if Type ==2

*top 20%
clear
use "D:\SOR\varfinal.dta" 
drop if year == 2009 &2010&2011&2012
xtile esgfz = ARP, nq(5)
xtreg ARP AR1 AR2 AR3 AR4 INDUSTRY_ADJUSTED_SCORE_MEAN raBVPS stanTA raSE stanMV i.year if esgfz == 5&Type ==1,fe
predict e11,xb 
mean(e11)
*0.197

clear
use "D:\SOR\varfinal.dta" 
drop if year == 2009 &2010&2011&2012
xtile esgfz = ARP, nq(5)
xtreg ARP AR1 AR2 AR3 AR4 INDUSTRY_ADJUSTED_SCORE_MEAN raBVPS stanTA raSE stanMV i.year if esgfz == 5&Type ==2,fe
predict e11,xb 
mean(e11)

*top 30%
clear
use "D:\SOR\varfinal.dta" 
drop if year == 2009 &2010&2011&2012
xtile esgfz = ARP, nq(3)
xtreg ARP AR1 AR2 AR3 AR4 INDUSTRY_ADJUSTED_SCORE_MEAN raBVPS stanTA raSE stanMV i.year if esgfz == 3&Type ==1,fe
predict e11,xb 
mean(e11)
*0.129

clear
use "D:\SOR\varfinal.dta" 
drop if year == 2009 &2010&2011&2012
xtile esgfz = ARP, nq(3)
xtreg ARP AR1 AR2 AR3 AR4 var157 ///
SOCIAL_PILLAR_SCORE_MEAN_01 GOVERNANCE_PILLAR_SCORE_MEAN_01 raBVPS stanTA raSE stanMV i.year if esgfz == 3&Type ==2,fe
predict e11,xb 
mean(e11)
*0.129

*evaluation of Model 5
*top 10% financial group
clear
use "D:\SOR\varfinal.dta" 
drop if year == 2009 &2010&2011&2012
xtile esgfz = ARP, nq(10)
xtreg ARP AR1 AR2 AR3 AR4 var157 ///
SOCIAL_PILLAR_SCORE_MEAN_01 GOVERNANCE_PILLAR_SCORE_MEAN_01 raBVPS stanTA raSE stanMV i.year if esgfz == 10&Type ==1,fe
predict e11,xb 
mean(e11)
*0.181
mean(ARP) if Type ==1

*top 20% financial group
clear
use "D:\SOR\varfinal.dta" 
drop if year == 2009 &2010&2011&2012
xtile esgfz = ARP, nq(5)
xtreg ARP AR1 AR2 AR3 AR4 var157 ///
SOCIAL_PILLAR_SCORE_MEAN_01 GOVERNANCE_PILLAR_SCORE_MEAN_01 raBVPS stanTA raSE stanMV i.year if esgfz == 5&Type ==1,fe
predict e11,xb 
mean(e11)
*0.154

*top 30% financial group
clear
use "D:\SOR\varfinal.dta" 
drop if year == 2009 &2010&2011&2012
xtile esgfz = ARP, nq(4)
xtreg ARP AR1 AR2 AR3 AR4 var157 ///
SOCIAL_PILLAR_SCORE_MEAN_01 GOVERNANCE_PILLAR_SCORE_MEAN_01 raBVPS stanTA raSE stanMV i.year if esgfz == 4&Type ==1,fe
predict e11,xb 
mean(e11)
*0.141

*top 10% non-financial group
clear
use "D:\SOR\varfinal.dta" 
drop if year == 2009 &2010&2011&2012
xtile esgfz = ARP, nq(10)
xtreg ARP AR1 AR2 AR3 AR4 var157 ///
SOCIAL_PILLAR_SCORE_MEAN_01 GOVERNANCE_PILLAR_SCORE_MEAN_01 raBVPS stanTA raSE stanMV i.year if esgfz == 10&Type ==2,fe
predict e11,xb 
mean(e11)
*0.272
mean(ARP) if Type ==2

*top 20% non-financial group
clear
use "D:\SOR\varfinal.dta" 
drop if year == 2009 &2010&2011&2012
xtile esgfz = ARP, nq(5)
xtreg ARP AR1 AR2 AR3 AR4 var157 ///
SOCIAL_PILLAR_SCORE_MEAN_01 GOVERNANCE_PILLAR_SCORE_MEAN_01 raBVPS stanTA raSE stanMV i.year if esgfz == 5&Type ==2,fe
predict e11,xb 
mean(e11)
*0.197

*top 30% non-financial group
clear
use "D:\SOR\varfinal.dta" 
drop if year == 2009 &2010&2011&2012
xtile esgfz = ARP, nq(4)
xtreg ARP AR1 AR2 AR3 AR4 var157 ///
SOCIAL_PILLAR_SCORE_MEAN_01 GOVERNANCE_PILLAR_SCORE_MEAN_01 raBVPS stanTA raSE stanMV i.year if esgfz == 4&Type ==2,fe
predict e11,xb 
mean(e11)
*0.169

********************************************************************
*chpater4 Contribution of companies’ ESG performance to their value*
********************************************************************
*heterogeneity
*different levels of ESG score
clear
use "D:\SOR\VARadd.dta" 
xtile esgfz = INDUSTRY_ADJUSTED_SCORE_MEAN, nq(3)
*y is total return, x is ESG score* "1"represent top ESG score,3 represent lower ESG score
xtreg ARP INDUSTRY_ADJUSTED_SCORE_MEAN lnBVPS lnTA lnSE lnMV ///
i.year if esgfz == 1&Type ==1,fe
est store m1
xtreg ARP INDUSTRY_ADJUSTED_SCORE_MEAN lnBVPS lnTA lnSE lnMV ///
i.year if esgfz == 2&Type ==1,fe
est store m2
xtreg ARP INDUSTRY_ADJUSTED_SCORE_MEAN lnBVPS lnTA lnSE lnMV ///
i.year if esgfz == 3&Type ==1,fe
est store m3
xtreg ARP lnindustryESG lnBVPS lnTA lnSE lnMV ///
i.year if esgfz == 1&Type ==2,fe
est store m4
xtreg ARP lnindustryESG lnBVPS lnTA lnSE lnMV ///
i.year if esgfz == 2&Type ==2,fe
est store m5
xtreg ARP lnindustryESG lnBVPS lnTA lnSE lnMV ///
i.year if esgfz == 3&Type ==2,fe
est store m6

esttab m1 m2 m3 m4 m5 m6 using D:\SOR\毕业论文\regout1.rtf,b(%12.3f) ///
se(%12.3f) nogap compress drop (*.year ) ///
s(N r2 ar2) star(* 0.1 ** 0.05 ** 0.01)


*different industries
clear
use "D:\SOR\varfinal.dta" 
bysort INDUSTRYSector:xtreg ARP AR1 AR2 AR3 AR4 INDUSTRY_ADJUSTED_SCORE_MEAN_02 ///
raBVPS stanTA raSE stanMV i.year,fe

xtreg ARP AR1 AR2 AR3 AR4 INDUSTRY_ADJUSTED_SCORE_MEAN_02 ///
raBVPS stanTA raSE stanMV i.year,fe

bysort INDUSTRYSector:xtreg AR lnindustryESG ///
raBVPS stanTA raSE stanMV i.year if INDUSTRYSector == "Material",fe
est store m7
xtreg ARP AR1 AR2 AR3 AR4 lnindustryESG ///
raBVPS stanTA raSE stanMV i.year if INDUSTRYSector == "Energy",fe
est store m8
bysort INDUSTRYSector:xtreg AR lnindustryESG ///
lnND lnBVPS lnTA lnSE lnMV i.year if INDUSTRYSector == "Consumer, Non-cyclical",fe
est store m9

esttab m7 m8 m9 using D:\SOR\毕业论文\regout2.rtf,b(%12.3f) ///
se(%12.3f) nogap compress drop (*.year ) ///
s(N r2 ar2) star(* 0.1 ** 0.05 ** 0.01)

*different MarketValues
clear
use "D:\SOR\VARadd.dta" 
bys GICSSector year: egen MarketValue_mean=mean(MarketValue)
sort id year
gen MarketValuefz=1 if MarketValue>MarketValue_mean
replace MarketValuefz=0 if MarketValue<MarketValue_mean

xtreg AR INDUSTRY_ADJUSTED_SCORE_MEAN lnBVPS lnTA lnSE lnMV ///
i.year if MarketValuefz == 1,fe
est store m10
xtreg AR INDUSTRY_ADJUSTED_SCORE_MEAN lnBVPS lnTA lnSE lnMV ///
i.year if MarketValuefz == 0,fe
est store m11

esttab m10 m11 using D:\SOR\毕业论文\regout3.rtf,b(%12.3f) ///
se(%12.3f) nogap compress drop (*.year ) ///
s(N r2 ar2) star(* 0.1 ** 0.05 ** 0.01)
*developed country and emerging country*
*DM*
gen countryfz=1 if Country== "AU"| Country=="BE" | Country=="CA" | Country=="DK" | Country=="FI" | Country=="FR" | Country=="DE"| Country=="HK"| Country=="IE"| Country=="IT"| Country=="JP"| Country=="NL"| Country=="NZ" | ///
Country=="NO"| Country=="PT" | Country=="SG" | Country=="SP" | Country=="SW" | Country=="SE" | Country=="GB" | Country=="US"

*EM*
replace countryfz=0 if Country== "BR"| Country=="CL" | Country=="CN" | Country=="CO" | Country=="CS" | Country=="EG" | Country=="GR"| Country=="HU"| Country=="IN"| Country=="ID"| Country=="KR"| Country=="KW"| Country=="MY"| Country=="MX"| Country=="PE" | ///
Country=="PH"| Country=="PL" | Country=="QA" | Country=="SA" | Country=="ZA" | Country=="TW" | Country=="TH"| Country=="TR"| Country=="AE"

reg AR INDUSTRY_ADJUSTED_SCORE_MEAN lnBVPS lnTA lnSE lnMV ///
i.year if countryfz == 1
est store m12
reg AR INDUSTRY_ADJUSTED_SCORE_MEAN lnBVPS lnTA lnSE lnMV ///
i.year if countryfz == 0
est store m13
esttab m12 m13 using D:\SOR\毕业论文\regout4.rtf,b(%12.3f) ///
se(%12.3f) nogap compress drop (*.year ) ///
s(N r2 ar2) star(* 0.1 ** 0.05 ** 0.01)

xtreg AR lnindustryESG lnBVPS lnTA lnSE lnMV ///
i.year if countryfz == 1,fe
xtreg AR lnindustryESG lnBVPS lnTA lnSE lnMV ///
i.year if countryfz == 0,fe

*different periods

gen yearbe1 =0
replace yearbe1=1 if year<2016
xtreg AR INDUSTRY_ADJUSTED_SCORE_MEAN lnBVPS lnTA lnSE lnMV ///
i.year if yearbe1 == 1,fe 
est store m14
replace yearbe1=2 if year>2014
xtreg AR INDUSTRY_ADJUSTED_SCORE_MEAN lnBVPS lnTA lnSE lnMV ///
i.year if yearbe1 == 2,fe
est store m15
esttab m14 m15 using D:\SOR\毕业论文\regout5.rtf,b(%12.3f) ///
se(%12.3f) nogap compress drop (*.year ) ///
s(N r2 ar2) star(* 0.1 ** 0.05 ** 0.01)
*the influence of ESG on ROE
*ROE & INDUSTRY_ADJUSTED_SCORE_MEAN
xtreg RoE INDUSTRY_ADJUSTED_SCORE_MEAN if Type == 2,fe
xtreg RoE INDUSTRY_ADJUSTED_SCORE_MEAN i.year if Type == 2,fe
xtreg RoE INDUSTRY_ADJUSTED_SCORE_MEAN BookValuePerShare lnND lnSE lnTA ///
i.year if Type == 2,fe
*ROE & WEIGHTED_AVERAGE_SCORE_MEAN
xtreg RoE WEIGHTED_AVERAGE_SCORE_MEAN  if Type == 2,fe
xtreg RoE WEIGHTED_AVERAGE_SCORE_MEAN i.year if Type == 2,fe
xtreg RoE WEIGHTED_AVERAGE_SCORE_MEAN BookValuePerShare lnND lnSE lnTA ///
i.year if Type == 2,fe

** ***********************************************
*Causality between the International stock market*
**************************************************
clear
use "D:\SOR\毕业论文\chapter 2\final data.dta"

*unit root test(dfuller）
tostring Dates,replace
gen month=month(Dates)
gen date=date(accper,"YMD")
tsline SP500INDEX
tsline NIKKEI225 
tsline MSCICHINA
tsline dlnSP500INDEX
tsline dlnNIKKEI225
tsline dlnMSCICHINA

dfuller dlnMSCICHINA
dfuller dlnSP500INDEX
dfuller dlnNIKKEI225

pperron dlnMSCICHINA
pperron dlnSP500INDEX
pperron dlnNIKKEI225 
*determination of the lag order
varsoc dSP500INDEX dNIKKEI225 dMSCICHINA, maxlag(13)   
var dSP500INDEX dNIKKEI225 dMSCICHINA , lag(1/5)
est store var0 
varlmar
*test if VAR system is a stable process
var dlnSP500INDEX dlnNIKKEI225 dlnMSCICHINA , lag(1/5)
varstable, graph dlabel
* test the significance of the lag order
var dSP500INDEX dNIKKEI225 dMSCICHINA , lag(1/5)
varwle
var dSP500INDEX dNIKKEI225 dMSCICHINA, lag(1/5)
varnorm
*Granger's causality test
sum dSP500INDEX dNIKKEI225 dMSCICHINA
winsor2 dSP500INDEX dNIKKEI225 dMSCICHINA, replace cuts(1 99)
var dSP500INDEX dNIKKEI225 dMSCICHINA, lag(1/5)
vargranger
xcorr dSP500INDEX dNIKKEI225,table
xcorr dSP500INDEX dMSCICHINA, table
xcorr dNIKKEI225 dMSCICHINA, table
xcorr dSP500INDEX dNIKKEI225,name(SN)
xcorr dSP500INDEX dMSCICHINA, name(SM)
xcorr dNIKKEI225 dMSCICHINA, name(NM)
graph combine SN SM NM

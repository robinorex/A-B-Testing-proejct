import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv ('Sample - Superstore.csv',encoding='unicode_escape')
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',20)
print(df.head())
# print(df.info())
# filling null values
# df["Postal Code"].fillna(0, inplace=True)
# df["Postal Code"] = df["Postal Code"].astype(int)
# df.info()
# if df.duplicated().sum() >0 :
#     print("There's duplicates")
# else :
#     print("There is no duplicates")
#Customer Segmentation
# types_of_customers = df['Segment'].unique()
# print (types_of_customers)
number_of_customers = df['Segment'].value_counts().reset_index()
number_of_customers = number_of_customers.rename(columns={'Segment': 'Type Of Customer'})
print(number_of_customers)
# plt.pie(number_of_customers['count'], labels=number_of_customers['Type Of Customer'], autopct='%1.1f%%')
# plt.show()
sales_per_segment = df.groupby('Segment')['Sales'].sum().reset_index()
sales_per_segment = sales_per_segment.rename(columns={'Segment': 'Type Of Customer','Sales':'Total Sales'})
print("\n",sales_per_segment)
# plt.bar(sales_per_segment['Type Of Customer'], sales_per_segment['Total Sales'])
#find out the CAC customer acquisition cost
# plt.pie(sales_per_segment['Total Sales'], labels=sales_per_segment['Type Of Customer'], autopct='%1.1f%%')
# plt.show()
print(df.head(3))
customer_order_frequency = df.groupby(['Segment','Customer ID','Customer Name'])['Order ID'].count().reset_index()
customer_order_frequency.rename(columns={'Order ID' : 'Total Orders'},inplace=True)
print(customer_order_frequency)
repeat_customers = customer_order_frequency[customer_order_frequency['Total Orders']>=1]
repeat_customers_sorted = repeat_customers.sort_values(by="Total Orders",ascending=False)
print(repeat_customers_sorted.head(12).reset_index(drop=False))
customer_sales = df.groupby(['Segment','Customer ID','Customer Name'])['Sales'].sum().reset_index()
top_spenders = customer_sales.sort_values(by="Sales",ascending=False)
print(top_spenders.head(12))
shipping_mode = df["Ship Mode"].value_counts().reset_index()
shipping_mode = shipping_mode.rename(columns={'Ship Mode':'Method of Shipment','count':'Use Frequency'})
print(shipping_mode)
# plt.pie(shipping_mode['Use Frequency'],labels=shipping_mode['Method of Shipment'], autopct='%1.1f%%')
# plt.show()
state = df["State"].value_counts().reset_index()
state = state.rename(columns={"index":"State","State":"Number of Customers"})
print(state.head(20))
city = df["City"].value_counts().reset_index()
print("\n",city.head(24))
state_sales = pd.DataFrame(df.groupby(["State"])["Sales"].sum().reset_index())
top_sales = state_sales.sort_values(by='Sales', ascending=False)
print("\n",top_sales.head(20).reset_index(drop=True))
city_sales = pd.DataFrame(df.groupby(["City"])["Sales"].sum().reset_index())
top_city_sales = city_sales.sort_values(by='Sales', ascending=False)
print("\n",top_city_sales.head(20).reset_index(drop=True))
products = df['Category'].unique()
print(products)
product_sub_category = df['Sub-Category'].unique()
print(product_sub_category)
sub_category_count = df.groupby(["Category"])['Sub-Category'].nunique().reset_index()
sub_category_count = sub_category_count.sort_values(by="Sub-Category",ascending=False)
print("\n",sub_category_count)
sub_category_sales = df.groupby(['Category','Sub-Category'])['Sales'].sum().reset_index()
sub_category_sales = sub_category_sales.sort_values(by="Sales",ascending=False)
print("\n",sub_category_sales)
product_category_sales = df.groupby(["Category"])['Sales'].sum().reset_index()
product_category_sales = product_category_sales.sort_values(by="Sales",ascending=False)
print("\n",product_category_sales)
# plt.pie(product_category_sales['Sales'],labels=product_category_sales['Sales'], autopct='%1.1f%%')

sub_category_sales = sub_category_sales.sort_values(by='Sales', ascending = True)
# plt.barh(sub_category_sales['Sub-Category'], sub_category_sales['Sales'])
# plt.show()
df['Order Date']= pd.to_datetime(df['Order Date'])
yearly_sales = df.groupby(df['Order Date'].dt.year)['Sales'].sum().reset_index()
yearly_sales = yearly_sales.rename(columns={'Order Date':'Year','Sales':'Total Sales'})
print('\n',yearly_sales)
# plt.bar(yearly_sales['Year'],yearly_sales['Total Sales'])
# plt.plot(yearly_sales['Year'],yearly_sales['Total Sales'],marker='o')
# plt.show()
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
year_sales = df[df['Order Date'].dt.year == 2017]
quarterly_sales = year_sales.resample('QE', on='Order Date')['Sales'].sum()
quarterly_sales = quarterly_sales.reset_index()
quarterly_sales = quarterly_sales.rename(columns={'Order Date': 'Quarter', 'Sales': 'Total Sales'})
print(quarterly_sales)
year_sales = df[df['Order Date'].dt.year == 2016]
Monthly_sales = year_sales.resample('ME', on='Order Date')['Sales'].sum().reset_index()
Monthly_sales = Monthly_sales.rename(columns={'Order Date': 'Month', 'Sales': 'Total Sales'})
print(Monthly_sales)
# plt.plot(quarterly_sales['Quarter'],quarterly_sales['Total Sales'],marker='o',linestyle='--')
# plt.xticks(rotation=50)
# plt.show()
# plt.plot(Monthly_sales['Month'], Monthly_sales['Total Sales'], marker = 'o', linestyle = '--')
# plt.show()

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
all_state_mapping = {"Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
    "American Samoa": "AS",
    "Guam": "GU",
    "Northern Mariana Islands": "MP",
    "Puerto Rico": "PR",
    "United States Minor Outlying Islands": "UM",
    "U.S. Virgin Islands": "VI"}
df['Abbreviation']=df['State'].map(all_state_mapping)
sum_of_sales = df.groupby('State')['Sales'].sum().reset_index()
sum_of_sales['Abbreviation'] = sum_of_sales['State'].map(all_state_mapping)
# fig = go.Figure(data=go.Choropleth(locations=sum_of_sales["Abbreviation"],locationmode='USA-states',z=sum_of_sales['Sales'],
#                                    hoverinfo='location+z',showscale=True))
# fig.update_geos(projection_type='albers usa')
# fig.update_layout(geo_scope='usa',title='total sales by US states')
# fig.show()
import plotly.express as px
df_summary = df.groupby(['Category','Sub-Category'])['Sales'].sum().reset_index()
fig1=px.sunburst(df_summary,path=['Category','Sub-Category'],values='Sales')
fig1.show()
df_summary_tree = df.groupby(['Category','Ship Mode','Sub-Category'])['Sales'].sum().reset_index()
fig2 = px.treemap(df_summary_tree,path=['Category','Sub-Category'],values='Sales')
fig2.show()




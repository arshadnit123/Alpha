import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from joblib import dump, load

df=pd.read_csv('/Users/arshadahmed/Desktop/task2sigma/Alpha2.csv')

st.set_page_config(layout='wide',initial_sidebar_state='expanded')

st.title(' :shirt: :dress: :shoe: Alpha')

#************ Sidebar **************

st.sidebar.header("Filters")

region=st.sidebar.multiselect("Choose Region", df['region'].unique())
gender=st.sidebar.multiselect("Choose Gender", df['product_gender_target'].unique())



#Filtering data based on region and gender

filtered_df = df.copy()
if region:
    filtered_df = filtered_df[filtered_df['region'].isin(region)]
if gender:
    filtered_df = filtered_df[filtered_df['product_gender_target'].isin(gender)]

if region and gender:
    filtered_df = filtered_df[(filtered_df['region'].isin(region)) & (filtered_df['product_gender_target'].isin(gender))]

if not region and not gender:
    filtered_df = df.copy()



#******************************Prediction********************************


st.subheader("Predict Price for Me!")

c1, c2, c3, c4, c5, c6, c7, c8 = st.columns((8))

with c1:
    country_p=st.multiselect("Country", df['seller_country'].unique())

with c2:
    badge_p=st.multiselect("Seller Badge", df['seller_badge'].unique())

with c3:
    category_p=st.multiselect("Category", df['product_category'].unique())

with c4:
    clothing_p=st.multiselect("Clothing Type", df['clothing_type'].unique())

with c5:
    region_p=st.multiselect("Region", df['region'].unique())

with c6:
    color_p=st.multiselect("Color", df['product_color'].unique())

with c7:
    condition_p=st.multiselect("Condition", df['product_condition'].unique())

with c8:
    material_p=st.multiselect("Material", df['product_material'].unique())


input_data={
        'seller_country': [],
        'seller_badge': [],
        'product_category': [],
        'clothing_type': [],
        'region': [],
        'product_color': [],
        'product_condition': [],
        'product_material': []
    }

encoding_mappings=load('encoding_mappings.joblib')
voting_regressor=load('voting_regressor_model.joblib')

def encode_input(input_data, encoding_mappings):
    encoded_data={}
    for col, values in input_data.items():
        if col in encoding_mappings:
            encoded_col=col+'_e' 
            encoded_values=[encoding_mappings[col].get(val, 0) for val in values]
            encoded_data[encoded_col]=encoded_values
    return encoded_data

if not any([country_p, badge_p, category_p, clothing_p, region_p, color_p, condition_p, material_p]):
    st.warning("Please select one option for each category.")
else:
    input_data['seller_country'] = country_p
    input_data['seller_badge'] = badge_p
    input_data['product_category'] = category_p
    input_data['clothing_type'] = clothing_p
    input_data['region'] = region_p
    input_data['product_color'] = color_p
    input_data['product_condition'] = condition_p
    input_data['product_material'] = material_p

encoded_input=encode_input(input_data, encoding_mappings)
input_df=pd.DataFrame(encoded_input)

if not input_df.empty:
    predictions=voting_regressor.predict(input_df)
    st.write(f"Predicted Price: {predictions[0]:.2f}")
else:
    st.warning("Cannot Predict for Invalid input")


#************* Section 1 *****************

clothing_type_df=filtered_df['clothing_type'].value_counts().reset_index()
clothing_type_df.columns = ['Clothing Type', 'Sales']

product_category_df=filtered_df['product_category'].value_counts().reset_index()
product_category_df.columns = ['Category', 'Sales']


col1, col2 = st.columns((2))

with col1:
    st.subheader("Clothing Type")
    fig=px.bar(clothing_type_df, x="Clothing Type", y="Sales", template='seaborn')
    st.plotly_chart(fig, use_container_width=True, height=200)

with col2:
    st.subheader("Product Type")
    fig=px.pie(product_category_df, names="Category", values="Sales", hole=0.5)
    st.plotly_chart(fig, use_container_width=True)


cl1, cl2=st.columns(2)

with cl1:
    with st.expander("Clothing Type - ViewData"):
        st.write(clothing_type_df.style.background_gradient(cmap='Blues'))
        csv=clothing_type_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data", csv, "clothing_type.csv", "text/csv", help="Click here to download") 


with cl2:
    with st.expander("Product Type - ViewData"):
        st.write(product_category_df.style.background_gradient(cmap='Oranges'))
        csv=product_category_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data", csv, "product_type.csv", "text/csv", help="Click here to download") 


#******************** Section 2 ************************

st.subheader("Shipping Time")

average_shipping_time_df = df.groupby('region')['usually_ships_within'].mean().reset_index()
fig = px.violin(df, x='region', y='usually_ships_within', color='region', 
                title='Distribution of Shipping Times by Region',
                labels={'region': 'Region', 'usually_ships_within': 'Shipping Time (days)'},
                template='seaborn')

st.plotly_chart(fig, use_container_width=True)


with st.expander("Average Shipping Time - ViewData"):
    st.write(average_shipping_time_df.style.background_gradient(cmap='Blues'))
    csv=average_shipping_time_df .to_csv(index=False).encode('utf-8')
    st.download_button("Download Data", csv, "shipping_time.csv", "text/csv", help="Click here to download") 


#******************* Section 3 ******************************

chart1, chart2, chart3=st.columns((3))

colors_df= filtered_df['product_color'].value_counts().head(5).reset_index()
colors_df.columns = ['Color', 'Sales']

top_brands_df= filtered_df['brand_name'].value_counts().head(5).reset_index()
top_brands_df.columns = ['Brand', 'Sales']

materials_df= filtered_df['product_material'].value_counts().head(5).reset_index()
materials_df.columns = ['Material', 'Sales']


with chart1:
    st.subheader("Top 5 Colors")
    fig = px.histogram(colors_df, x='Color', y='Sales',
                   labels={'Color': 'Color', 'Sales': 'Number of Sales'},
                   template='seaborn')
    st.plotly_chart(fig, use_container_width=True)

with chart2:
    st.subheader("Top 5 Brands")
    fig = px.pie(top_brands_df, values='Sales', names='Brand', hole=0.5)
    st.plotly_chart(fig, use_container_width=True)

with chart3:
    st.subheader("Top 5 Materials")
    fig = px.line(materials_df, x='Material', y='Sales',
                   labels={'Material': 'Material', 'Sales': 'Number of Sales'},
                   template='seaborn')
    st.plotly_chart(fig, use_container_width=True)


ch1, ch2, ch3=st.columns((3))

with ch1:
    with st.expander("Top 5 Colors - ViewData"):
        st.write(colors_df.style.background_gradient(cmap='Blues'))
        csv=colors_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data", csv, "top5colors.csv", "text/csv", help="Click here to download") 


with ch2:
    with st.expander("Top 5 Brands - ViewData"):
        st.write(top_brands_df.style.background_gradient(cmap='Oranges'))
        csv=top_brands_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data", csv, "top5brands.csv", "text/csv", help="Click here to download") 

with ch3:
    with st.expander("Top 5 Materials - ViewData"):
        st.write(materials_df.style.background_gradient(cmap='Oranges'))
        csv=materials_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data", csv, "top5materials.csv", "text/csv", help="Click here to download") 



#**************Section 4************************

st.subheader("Europe Market Analysis")

df_europe = df[df['region'] == 'Europe']

average_price_per_country_brand = df_europe.groupby(['seller_country', 'brand_name'])['seller_price'].mean().reset_index()

top_10_countries = df_europe['seller_country'].value_counts().nlargest(10).index

top_brand_per_country = {}
for country in top_10_countries:
    top_brand_per_country[country] = df_europe[df_europe['seller_country'] == country]['brand_name'].value_counts().nlargest(1).index

fig = px.line(title='Average Price per Country and Brand in Europe')

for country in top_10_countries:
    for brand in top_brand_per_country[country]:
        data = average_price_per_country_brand[(average_price_per_country_brand['seller_country'] == country) & (average_price_per_country_brand['brand_name'] == brand)]
        if not data.empty:
            fig.add_scatter(x=data['seller_country'], y=data['seller_price'], mode='lines+markers', name=f'{country} - {brand}')


fig.update_xaxes(title='Country')
fig.update_yaxes(title='Average Price')
fig.update_layout(xaxis=dict(tickangle=45))


st.plotly_chart(fig, use_container_width=True)


with st.expander("Europe Market Analysis"):
    csv_data = average_price_per_country_brand.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Data as CSV", data=csv_data, file_name="europe_market_analysis.csv", mime="text/csv")

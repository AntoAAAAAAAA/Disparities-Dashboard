import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt 

# Set page layout
st.set_page_config(page_title="Sidebar Tab Dashboard", layout="wide")

# --- Initialize session state for tab navigation ---
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Home"

# --- Sidebar Buttons as Tabs ---
st.sidebar.title("Dashboard Sections")

def tab_button(label):
    if st.sidebar.button(label, use_container_width=True):
        st.session_state.current_tab = label

# Add more tab buttons here
tab_button("Home")
tab_button("Insurance Coverage")
tab_button("Mental Health Metrics")
tab_button("Physician Access")
tab_button('My Journey/Reflections')

# Optional: style selected tab
st.sidebar.markdown(f"**Selected: {st.session_state.current_tab}**")

# --- Main Content Area ---
st.title("Health Data Dashboard")

if st.session_state.current_tab == "Home":
    st.header("Welcome!")
    st.write("Select a section from the left to explore the data.")
    st.markdown("This project was undertaken to expand my data analysis skills  and learn how to make an app " \
    "using streamlit.")
    st.markdown(
    'Please find below the sources for all data referenced in this project:'
    )
    st.write('')
    st.write('')
    st.markdown(
    '**Demographics:**'
    )
    st.markdown(
    """
    - [Census 2020 Evaluation Estimates - 2010s County Detail](https://www.census.gov/programs-surveys/popest/technical-documentation/research/evaluation-estimates/2020-evaluation-estimates/2010s-county-detail.html)
    - [Census 2010-2020 Data Layout PDF](https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2010-2020/cc-est2020-alldata.pdf)
    """
    )
    st.markdown("**Insurance Coverage:**")
    st.markdown(
        """
    - [Estimates from ACS - Small Area Health Insurance Estimates (SAHIE)](https://www.census.gov/data/datasets/time-series/demo/sahie/estimates-acs.html)
    """
    )

    st.markdown("**Mental Health Coverage:**")
    st.markdown(
        """
    - [Texas County Health Rankings Data and Resources](https://www.countyhealthrankings.org/health-data/texas/data-and-resources)
    - [Texas Mental Health Data Dashboard](https://healthdata.dshs.texas.gov/dashboard/mental-health/mental-health#data-source)
    """
    )

    st.markdown("**Physician Access:**")
    st.markdown(
        """
    - [Primary Care Physicians Data - County Health Rankings](https://www.countyhealthrankings.org/health-data/community-conditions/health-infrastructure/clinical-care/primary-care-physicians?year=2025&state=48&tab=1)
    """
    )

    st.markdown("**Shapefiles and Population Data:**")
    st.markdown(
        """
    - [Census Shapefiles](https://www.census.gov/cgi-bin/geo/shapefiles/index.php)
    """
    )

    st.markdown("**Population Data:**")
    st.markdown(
        """
    - [Texas Demographic Center Population Estimates 2022](https://demographics.texas.gov/Estimates/2022/)
    """
    )
elif st.session_state.current_tab == "Insurance Coverage":
    st.header("Insurance Coverage")
    st.write("This section will display information regarding insurance coverage for racial/ethnic groups in different parts of Texas.")

    tab1, tab2 = st.tabs(['Bar Graphs', 'Heatmaps'])
    ### First plot 
    with tab1:
        data = pd.read_csv('data/sahie_2022.csv', low_memory=False)
        data['state_name'] = data['state_name'].str.strip() # remove the spaces 
        data['county_name'] = data['county_name'].str.strip()# remove the spaces 
        texas_data = data[data['state_name'] == 'Texas'].copy() #only Texas data
        texas_data.drop(columns='Unnamed: 25', inplace=True) #remove blank column on right 
        texas_data.dropna(axis=0, inplace=True) #remove rows with Na values 
        texas_data.reset_index(drop=True, inplace=True) #reset index 
        texas_data.rename(columns={'iprcat':'Income Category', 'PCTELIG':'Percent Uninsured for all income levels', 
                                'NUI':'Number Uninsured',
                                'NIC': 'Number Insured', 'nui_moe': 'Number Uninsured MOE', 
                                'nic_moe':'Number Insured MOE',
                                'pctelig_moe':'Percent Uninsured for all income levels MOE', 
                                'PCTLIIC':'Percent Insured for all income levels',
                                'pctliic_moe':'Percent Insured for all income levels MOE'}, inplace=True) #rename columns
        #make sure columns that have numbers are the right type, and not 'object'
        cols = texas_data.loc[:, 'statefips':'Percent Insured for all income levels MOE'].columns
        texas_data[cols] = texas_data[cols].apply(pd.to_numeric, errors='coerce')
        newdata = texas_data[texas_data['county_name'] != ''].copy() #remove rows with a blank county name 
        newdata = newdata[
            (newdata['agecat'] == 0) &
            (newdata['racecat'] == 0) &
            (newdata['sexcat'] == 0) &
            (newdata['Income Category'] == 0)
        ].copy()
        newdata['county_name'] = newdata['county_name'].str.replace(r'\s*county\s*','',case=False, regex=True) #remove 'county' 
        newdata.rename(columns={'county_name':'county'}, inplace=True) #rename column 
        #make sure column is the right type 
        newdata['Percent Uninsured for all income levels'] = newdata['Percent Uninsured for all income levels'].astype(dtype='float64')
        newdata.set_index('county', drop=True, inplace=True) #make 'county' the index 
        #create bar graph
        fig, ax = plt.subplots(figsize=(10,4))
        newdata.plot(kind='bar', y='Percent Uninsured for all income levels', ax=ax)
        ax.set_title("Percentage of Uninsured Individuals per County")
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=1, rotation = 90)
        ax.legend().set_visible(False)
        ax.set_ylabel('Percent Uninsured for all income levels')
        st.pyplot(fig)
        #identify the counties with highest and lowest percentages of uninsured individuals 
        highest = newdata['Percent Uninsured for all income levels'].max()
        lowest = newdata['Percent Uninsured for all income levels'].min()
        county_highest = newdata['Percent Uninsured for all income levels'].idxmax()
        county_lowest = newdata['Percent Uninsured for all income levels'].idxmin()
        this = (
            f'The highest percentage is {round(highest,2)}% and the lowest is {round(lowest, 3)}%. \n'
            f'The county with the highest percentage is {county_highest} and the lowest is {county_lowest}'
        )
        st.write(this) 

        ### Plot 2 

        st.divider()
        new_race_data = texas_data[texas_data['county_name'] == ''].copy()
        hmap = {0:'All Races', 1:'White only', 2:'Black or African American only', 3:'Hispanic or Latino (any race)',
                4:'American Indian and Alaska Native alone', 5:'Asian alone', 6:'Native Hawaiian and Other Pacific Islander alone',
                7:'Two or More Races, not Hispanic or Latino'}
        new_race_data['racecat'] = new_race_data['racecat'].map(hmap)
        new_race_data.rename(columns={'PCTUI':'Percent uninsured in demographic group for income category'}, inplace=True)
        new_race_data = new_race_data[(new_race_data['Income Category'] == 0) & (new_race_data['sexcat'] == 0) & (new_race_data['agecat'] == 0)].copy()
        new_race_data.reset_index(inplace=True, drop=True)
        new_race_data.set_index('racecat', inplace=True)
        new_race_data.drop(index='All Races', inplace=True)
        fig, ax = plt.subplots(figsize = (10,4))
        new_race_data.plot(kind='bar', y='Percent Uninsured for all income levels', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.legend().set_visible(False)
        ax.set_xlabel('Race Categories')
        ax.set_ylabel('Percent of Uninsured Individuals for all Income Levels')
        ax.set_title('Percent of Uninsured Texans by Race/Ethnicity')
        st.pyplot(fig)
        st.markdown(
        'In this figure, we see that African Americans alone, American Indian/Alaska Natives alone, '
        'and Pacific Islanders all make up a higher percentage of the uninsured population in Texas. '
        'When looking at ethnicity, individuals of Hispanic heritage show the highest proportion of the uninsured population.'
        )   

        ### Plot 3 

        st.divider()
        race_sex_data = texas_data[texas_data['county_name'] == ''].copy()
        # Map race categories to names
        race_map = {
            0: 'All Races', 1: 'White only', 2: 'Black or African American only',
            3: 'Hispanic or Latino (any race)', 4: 'American Indian and Alaska Native alone',
            5: 'Asian alone', 6: 'Native Hawaiian and Other Pacific Islander alone',
            7: 'Two or More Races, not Hispanic or Latino'
        }
        # Map sex categories to labels
        sex_map = {
            1: 'Male',
            2: 'Female'
        }
        # Apply the mappings
        race_sex_data['racecat'] = race_sex_data['racecat'].map(race_map)
        race_sex_data['sexcat'] = race_sex_data['sexcat'].map(sex_map)
        # Drop missing values in key columns
        plot_data = race_sex_data.dropna(subset=['racecat', 'sexcat', 'Percent Uninsured for all income levels'])
        # Create a pivot table for grouped bar chart
        pivot_data = plot_data.pivot_table(index='racecat', columns='sexcat', values='Percent Uninsured for all income levels')
        # Plot
        fig,ax = plt.subplots(figsize = (10,4))
        pivot_data.plot(kind='bar', figsize=(10,6), ax=ax)
        ax.set_xticklabels(ax.get_xticklabels() ,rotation=45, ha='right')
        ax.set_title('Percent Uninsured by Race and Sex (All Income Levels)')
        ax.set_ylabel('Percent Uninsured')
        ax.set_xlabel('Race Category')
        ax.legend(title='Sex')
        st.pyplot(fig)
        st.markdown(
        'We observe a similar breakdown of uninsured individuals by race, this time split by gender. '
        'According to the figure, there are no significant differences between male and female individuals. '
        'However, similar to the previous graph on this page, individuals of Hispanic ethnicity make up a larger proportion of the uninsured population. '
        'We also see that "White" and "Asian Only" are the two racial groups that make up the smallest proportion of the uninsured population. '
        'This data, along with the graph above, suggests a very surface-level correlation between being an ethnic or racial minority and being more likely '
        'to lack health insurance coverage.'
        )
    ### Plot 4

    with tab2:
        texas_data = texas_data[texas_data['county_name'] != '']
        unins_by_county_data = texas_data.groupby('county_name')['Percent Uninsured for all income levels'].mean().reset_index()
        unins_by_county_data.reset_index(inplace=True)
        demographics = pd.read_csv('/Users/antoantony/Library/CloudStorage/OneDrive-TheUniversityofTexasatAustin/Python/VS_Code/Data Analysis/Disparities Dashboard/demographics.csv')
        final = pd.merge(unins_by_county_data, demographics, left_on='county_name', right_on='COUNTYNAME',
                        how='left')
        final.drop(columns='Unnamed: 0', inplace=True)
        def label(input):
            if pd.isna(input):
                return 'Low % Minority'
            else:
                return 'High % Minority'
        final['% Minority'] = final['COUNTYNAME'].apply(label)
        import geopandas as gpd
        import pandas as pd
        import matplotlib.pyplot as plt
        # Load shapefile and filter for Texas
        tx_counties = gpd.read_file('/Users/antoantony/Library/CloudStorage/OneDrive-TheUniversityofTexasatAustin/Python/VS_Code/Data Analysis/Disparities Dashboard/texas_shapefile')
        tx_counties = tx_counties[tx_counties['STATEFP'] == '48']
        final['county_name'] = final['county_name'].str.replace(r'\s*county\s*','',case=False, regex=True) #remove 'county'
        final['COUNTYNAME'] = final['COUNTYNAME'].str.strip()
        final['county_name'] = final['county_name'].str.strip()
        tx_map = tx_counties.merge(final,left_on='NAME', right_on='county_name')
        # Percent Uninsured for all income levels
        # % Minority
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        tx_map.plot(column='% Minority', ax=axs[0], legend=True, cmap='coolwarm', edgecolor='black')
        axs[0].set_title('% Minority', fontsize=14)
        axs[0].axis('off')
        tx_map.plot(column='Percent Uninsured for all income levels', ax=axs[1], legend=True, cmap='viridis', edgecolor='black')
        axs[1].set_title('Percent Uninsured for all income levels', fontsize=14)
        axs[1].axis('off')
        plt.tight_layout()
        st.pyplot(fig)
        # Group and calculate mean
        st.dataframe(final.groupby('% Minority')['Percent Uninsured for all income levels'].mean())
        
        st.markdown(
        'The graph on the left shows counties categorized by whether they have high or low percentages of minority populations. '
        'This classification was made using a separate dataset that included the racial composition of each county. '
        'Each racial group’s percentage was converted into a Z-score. Counties with at least one racial group scoring above 1 standard deviation (Z-score > 1) '
        'were considered to have a high percentage of minorities. All other counties were considered to have a low percentage of minorities.'
        )
        st.markdown(
        'The graph on the right displays the percentage of uninsured individuals across different Texas counties.'
        )
        st.write(
        'Comparing these two graphs, we can observe a somewhat similar distribution between counties with higher percentages of uninsured individuals '
        'and counties with higher percentages of minorities. While no definitive conclusions can be drawn, there is some preliminary evidence suggesting '
        'that counties with larger minority populations may also have higher rates of uninsured individuals. This tentative finding aligns with the patterns '
        'observed in the bar graphs presented earlier in this section.'
        )
    ###Last Section: Dataframe 
    st.divider()

elif st.session_state.current_tab == "Mental Health Metrics":
    st.header("Mental Health Metrics")
    st.write("This section will visualize metrics such as provider ratios, distress levels, and suicide rates.")
    left = pd.read_excel('/Users/antoantony/Library/CloudStorage/OneDrive-TheUniversityofTexasatAustin/Python/VS_Code/Data Analysis/Disparities Dashboard/MentalHealthCoverage/2025_county_health_rankings_texas_data_-_v1.xlsx', sheet_name="Select Measure Data")

    tab1, tab2 = st.tabs(['Bar Graphs', 'Heatmap'])
    ## Plot 1 

    with tab1:
        left = left[['State', 'County','Average Number of Mentally Unhealthy Days','# Mental Health Providers', 'Mental Health Provider Rate', 
            'Mental Health Provider Ratio', '# Primary Care Physicians', 'Primary Care Physicians Rate', 
            'Primary Care Physicians Ratio', '# Uninsured', '% Uninsured' ]]
        right = pd.read_excel('/Users/antoantony/Library/CloudStorage/OneDrive-TheUniversityofTexasatAustin/Python/VS_Code/Data Analysis/Disparities Dashboard/MentalHealthCoverage/2025_county_health_rankings_texas_data_-_v1.xlsx', sheet_name='Additional Measure Data')
        right = right[['State', 'County','% Frequent Mental Distress', 'Suicide Rate (Age-Adjusted)', 'Other Primary Care Provider Ratio',
                'Population']]
        finaldata = pd.merge(left=left, right=right, how='inner', on=['State', 'County'])
        # read in demographics dataset
        demographics = pd.read_csv('/Users/antoantony/Library/CloudStorage/OneDrive-TheUniversityofTexasatAustin/Python/VS_Code/Data Analysis/Disparities Dashboard/demographics.csv')
        dem_final = demographics.copy()
        # remove the 'county' in each county name 
        dem_final['COUNTYNAME'] = dem_final['COUNTYNAME'].str.replace('County', '', case=False).str.strip()
        #merge mentalhealth data with demographics 
        merged_dem_data = pd.merge(left=finaldata, right=dem_final, left_on='County', right_on='COUNTYNAME', how='left')
        merged_dem_data.drop(columns='Unnamed: 0', inplace=True)
        # replace with a binary categorical value for later graphing
        def create(input):
            if pd.isna(input):
                return 'Low % Minority'
            else:
                return 'High % Minority'
        merged_dem_data['High or Low % Minority'] = merged_dem_data['COUNTYNAME'].apply(create)
        mh_and_demographic = merged_dem_data.copy()
        #function to turn ratios into decimals 
        def ratio(input):
            left, right = input.split(':')
            left = left.replace(',','')
            right = right.replace(',','')
            return round((float(right)/float(left) * 100),7)
        #take a copy 
        mh_ratio_cleaned = mh_and_demographic.copy()
        #clean up the Ratio column and remove any rows that contain NaN's in that column 
        mh_ratio_cleaned = mh_ratio_cleaned.dropna(subset=['Mental Health Provider Ratio']).copy()
        mh_ratio_cleaned['Mental Health Provider Ratio'] = mh_ratio_cleaned['Mental Health Provider Ratio'].astype(dtype=str)
        mh_ratio_cleaned['Mental Health Provider Ratio'] = mh_ratio_cleaned['Mental Health Provider Ratio'].apply(ratio)
        mh_ratio_cleaned = mh_ratio_cleaned[['County', 'Mental Health Provider Ratio', 'High or Low % Minority']] #Select relevant columns 
        mh_ratio_cleaned.dropna(inplace=True) #remove rows with NaN's 
        mh_ratio_cleaned.reset_index(drop=True, inplace=True) #reset the index 
        mh_ratio_cleaned.set_index('County', inplace=True)
        mh_ratio_cleaned.sort_values(by='Mental Health Provider Ratio', inplace=True)
        fig, ax = plt.subplots(figsize=(10,8))
        mh_ratio_cleaned.plot(y='Mental Health Provider Ratio', figsize=(20,6), ax=ax, kind='bar')
        ax.set_xticks(range(len(mh_ratio_cleaned.index)))
        ax.set_xticklabels(mh_ratio_cleaned.index, rotation=45, fontsize=5)
        st.pyplot(fig)

        st.markdown(
        'This figure shows the mental health provider ratios for each county. This ratio was originally in the form of ' \
        '"Patients:Mental Health Provider". This ratio was turned into a decimal for ease of graphing and visualization.'
        )

        ## Plot 2 

        st.divider()
        import plotly.express as px
        fig = px.bar(
            mh_ratio_cleaned.reset_index(),
            x='County',
            y='Mental Health Provider Ratio',
            color='High or Low % Minority',
            title='Mental Health Provider Ratio by County',
            height=600
        )
        fig.update_layout(xaxis_tickangle=90)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
        'This interactive figure also shows the mental health provider ratio for each county, while also indicating which '
        'counties are considered to have high or low percentages of minorities within their populations. '
        'This graph is not sufficient to draw conclusions about overall trends or whether there are significant differences '
        'between counties with high versus low minority populations.'
        )

        ## Random plot 
        st.divider()
        mh_ratio_grouped = mh_ratio_cleaned.groupby('High or Low % Minority')['Mental Health Provider Ratio'].mean()
        st.dataframe(mh_ratio_grouped)
        st.markdown(
        'This table shows that the average mental health provider ratio is lower in counties with a low percentage of minorities '
        'compared to those with a higher percentage. While this difference is not statistically significant, similar to previous findings, '
        'it offers a preliminary indication that there may be underlying factors contributing to a lower number of mental '
        'health providers per person in counties with fewer minorities.'
        )

        ## Plot 3 

        st.divider()
        mh_suicide_rate = mh_and_demographic[['County', 'Suicide Rate (Age-Adjusted)', 'High or Low % Minority']]
        mh_suicide_rate = mh_suicide_rate.dropna().copy()
        mh_suicide_rate.sort_values(by='Suicide Rate (Age-Adjusted)', inplace=True)
        mh_suicide_rate = mh_suicide_rate[mh_suicide_rate['Suicide Rate (Age-Adjusted)'] != 0]
        mh_suicide_rate.set_index('County', inplace=True)
        import plotly.express as px
        fig = px.bar(
        mh_suicide_rate.reset_index(),
        x='County',
        y='Suicide Rate (Age-Adjusted)',
        color='High or Low % Minority',
        title='Suicide Rate by County',
        height=600,
        color_discrete_map={
            'High % Minority': '#EF553B',  # red-orange
            'Low % Minority': '#636EFA'   # blue
        }
        )
        fig.update_layout(xaxis_tickangle=90)
        st.plotly_chart(fig, use_container_width=True)
        st.divider()

        ## Table for Plot 3 
        mh_suicide_grouped = mh_suicide_rate.groupby('High or Low % Minority')['Suicide Rate (Age-Adjusted)'].mean()
        st.dataframe(mh_suicide_grouped)

        st.markdown(
        'The graph and table above display the relationship between counties with high versus low minority populations and their age-adjusted suicide rates. '
        'From the table, we can see that the average suicide rates between these groups of counties are not significantly different. '
        'As such, no definitive conclusions can be drawn from this data.'
        )  

        ## Plot 4 
        mh_mental_distress = mh_and_demographic[['County', '% Frequent Mental Distress', 'High or Low % Minority']]
        mh_mental_distress = mh_mental_distress.dropna().copy()
        import plotly.express as px
        fig = px.bar(
            mh_mental_distress.reset_index(),
            x='County',
            y='% Frequent Mental Distress',
            color='High or Low % Minority',
            title='Mental Distress Rate per County',
            height=600
        )
        fig.update_layout(xaxis_tickangle=90)
        st.plotly_chart(fig)

        ## Table for Plot 4
        mh_mental_grouped = mh_mental_distress.groupby('High or Low % Minority')['% Frequent Mental Distress'].mean()
        mh_mental_grouped_median = mh_mental_distress.groupby('High or Low % Minority')['% Frequent Mental Distress'].median()
        st.dataframe(mh_mental_grouped)
        st.dataframe(mh_mental_grouped_median)

        st.markdown(
        'The graph and two tables above illustrate the relationship between counties with high or low percentages of minorities and the percentage '
        'of their populations experiencing frequent mental distress. Based on the tables, there is no significant difference in the mean or median percentage '
        'of frequent mental distress between the two groups. Once again, the results are inconclusive, and further analysis—potentially using different datasets—'
        'is needed to gain more insight.'
        )

    with tab2:
        ### Plot 5 

        therapists_data = pd.read_csv('/Users/antoantony/Library/CloudStorage/OneDrive-TheUniversityofTexasatAustin/Python/VS_Code/Data Analysis/Disparities Dashboard/MentalHealthCoverage/Clinician_Region.csv')
        #clean and merge the two datasets
        therapists_data['County'] = therapists_data['County'].str.replace('County', '', case=False).str.strip()
        therapist_merged = pd.merge(left=therapists_data, right=dem_final, left_on='County', right_on='COUNTYNAME',
                how='left')
        therapist_merged.drop(columns='Unnamed: 0', inplace=True)
        #Create new column that makes a binary categorical variable
        therapist_merged['High or Low % Minority'] = therapist_merged['COUNTYNAME'].apply(create)
        import geopandas as gpd
        import pandas as pd
        import matplotlib.pyplot as plt
        # Load shapefile and filter for Texas
        tx_counties = gpd.read_file('/Users/antoantony/Library/CloudStorage/OneDrive-TheUniversityofTexasatAustin/Python/VS_Code/Data Analysis/Disparities Dashboard/texas_shapefile')
        tx_counties = tx_counties[tx_counties['STATEFP'] == '48']
        # Prepare your dataset
        therapist_merged['County'] = therapist_merged['County'].str.strip()
        # Merge shapefile and your data
        tx_map = tx_counties.merge(therapist_merged, left_on='NAME', right_on='County')
        # Calculate psychiatrists per 100k population
        tx_map['Psychiatrists per 100k (2023)'] = tx_map['Psychiatrist 2023'] / tx_map['2023 Population'] * 100000
        # Plot side-by-side maps
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        tx_map.plot(column='High or Low % Minority', ax=axs[0], legend=True, cmap='coolwarm', edgecolor='black')
        axs[0].set_title('High vs Low % Minority', fontsize=14)
        axs[0].axis('off')
        tx_map.plot(column='Psychiatrists per 100k (2023)', ax=axs[1], legend=True, cmap='viridis', edgecolor='black')
        axs[1].set_title('Psychiatrists per 100k (2023)', fontsize=14)
        axs[1].axis('off')
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown(
        'The two maps above display mental health provider data alongside racial demographic information.'
        )

        st.markdown(
        'The map on the left shows which counties are considered to have high or low percentages of minority populations. '
        'The map on the right is a heatmap displaying the average ratio of psychiatrists per 100,000 people for each county.'
        )

        st.markdown(
        'Comparing these two maps by identifying overlapping regions can offer valuable insights. Notably, there are clusters of counties '
        'in southeast and east Texas that show higher psychiatrist ratios while also having higher percentages of minorities. '
        'The opposite pattern is also observed—and is more common—where counties, particularly in north and west Texas, '
        'have high minority populations but significantly lower psychiatrist ratios.'
        )

        st.divider()

elif st.session_state.current_tab == "Physician Access":
    import geopandas as gpd
    st.header("Physician Access")
    st.write("This section will visualize the amount of access different populations in Texas have to physicians.")
    
    data = pd.read_csv('/Users/antoantony/Library/CloudStorage/OneDrive-TheUniversityofTexasatAustin/Python/VS_Code/Data Analysis/Disparities Dashboard/PhysicianAccess/texas-2025-primary-care-physicians-place-sort.csv')
    # data[data.isnull().any(axis=1)]
    keep_row = data.iloc[0]
    data = data.dropna().copy()
    data.iloc[0] = keep_row
    # or 
    # filtered_data = pd.concat([keep_row, data.iloc[1:].dropna()])
    def ratio(input):
        left, right = input.split(':')
        left = left.replace(',','')
        right = right.replace(',','')
        return round((float(right)/float(left) * 100),4)
    data.rename(columns={'County Value**': 'Physician:Population'}, inplace=True)
    data['Physician:Population'] = data['Physician:Population'].apply(ratio) 
    data.drop(1, inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    tab1, tab2 = st.tabs(['Bar Graphs', 'Heatmap'])
    with tab1:
        ### Plot 1 
        data.sort_values('National Z-Score', inplace=True)
        def clean(input):
            if '^' in input:
                newinput = input.replace('^','')
                return newinput
            else:
                return input 
        data['County (new)'] = data['County'].apply(clean)
        data.reset_index(drop=True, inplace=True)
        anotherdata = data.copy()
        anotherdata.set_index('County (new)', inplace=True)
        fig,ax = plt.subplots(figsize=(10,8))
        data.plot(kind='bar', y='National Z-Score', figsize=(20,8), ax=ax, x='County (new)')
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=5)
        ax.set_xlabel('County Name')
        ax.set_ylabel('National Z-Score')
        st.pyplot(fig)

        ### Plot 2 
        st.divider()
        fig,ax = plt.subplots(figsize=(10,8))
        anotherdata.plot(kind='bar', y='Physician:Population', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=4)
        ax.set_xlabel('County Name')
        ax.set_ylabel('Physicians per Person (# of Physician/# of People)')
        ax.legend(['Physician:Population Ratio'])
        st.pyplot(fig)

        st.markdown(
        'The first bar graph displays the z-scores of Texas counties with respect to their physician-to-population ratios. '
        'These z-scores are "national," meaning they compare each county’s ratio to those of counties across the United States. '
        'The second bar graph shows the actual physician-to-population ratios for each county.'
        )

    with tab2:
        ### Plot 3 
        dems = pd.read_csv('/Users/antoantony/Library/CloudStorage/OneDrive-TheUniversityofTexasatAustin/Python/VS_Code/Data Analysis/Disparities Dashboard/demographics.csv')
        dems['COUNTYNAME'] = dems['COUNTYNAME'].str.replace('County', '',case=False).str.strip()
        graphdata = pd.merge(data, dems, left_on='County (new)', right_on='COUNTYNAME', how='left')
        def filter(input):
            if pd.isna(input):
                return 'Low % Minority'
            else:
                return 'High % Minority'
        graphdata['% Minority'] = graphdata['COUNTYNAME'].apply(filter)
        graphdata.set_index('County (new)', inplace=True)
        demographics = gpd.read_file('/Users/antoantony/Library/CloudStorage/OneDrive-TheUniversityofTexasatAustin/Python/VS_Code/Data Analysis/Disparities Dashboard/texas_shapefile.zip')
        tx_counties = demographics[demographics['STATEFP'] == '48']
        new_map = tx_counties.merge(graphdata, left_on='NAME', right_on='County (new)')
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        new_map.plot(column='Physician:Population', ax=axs[0], legend=True, cmap='coolwarm', edgecolor='black')
        axs[0].set_title('Physician Ratio per County', fontsize=14)
        axs[0].axis('off')
        new_map.plot(column='% Minority', ax=axs[1], legend=True, cmap='viridis', edgecolor='black')
        axs[1].set_title('High v Low Percentage Minority Counties', fontsize=14)
        axs[1].axis('off')
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown(
        'The two maps above include a heatmap of physician-to-population ratios alongside a map that indicates whether a county '
        'has a high or low percentage of minority residents. '
        'When comparing these maps, there do not appear to be any obvious trends or correlations between minority population percentages '
        'and physician-to-population ratios.'
        )

elif st.session_state.current_tab == 'My Journey/Reflections':
    st.subheader('Personal Reflection')
    st.markdown(
    "This project has proven to be incredibly difficult, but equally valuable to me. It is my second project, completed directly "
    "after finishing the Texas Public Health Blog. This experience has significantly expanded my proficiency with Jupyter Notebook "
    "and the use of Pandas in Python. I’ve learned many new skills, including how to apply functions to entire columns, clean data, "
    "set and reset indexes, split a single dataset into multiple datasets for graphing, use GeoPandas to create heatmaps, use Plotly "
    "to make interactive graphs, merge datasets, and use Streamlit to build this dashboard."
    )

    st.markdown(
        "This project felt overwhelming at first. The early stages began with creating three Jupyter Notebook files, which would later be "
        "translated into the three informational tabs seen on the left-hand sidebar. Finding the necessary datasets was made easier with the "
        "help of AI. Each notebook required significant time to refine its dataset, determine appropriate graphing methods, and resolve coding bugs. "
        "Repeating this process across three very different datasets was frustrating, but ultimately very rewarding. This process was especially impactful "
        "because I felt far more confident by the time I reached the second and third notebooks. By then, I had improved my logic, gained enough experience "
        "to code independently, and learned to make edits to datasets with the assurance that they would not cause errors later on."
    )

    st.markdown(
        "The Streamlit aspect of this project was the most rewarding. I used AI to create a basic template for the dashboard, then integrated the Jupyter "
        "Notebooks into different sections. One of the most challenging parts of this process was adjusting the graph code to be compatible with Streamlit, "
        "which requires a specific version of Matplotlib for proper display. In the end, learning to use Streamlit's tab creation system, button sidebars, "
        "titles, markdown text, and running the app through the terminal gave me valuable experience in software design."
    )

    st.markdown(
        "Overall, I consider this project a major success in terms of software development, but somewhat of a failure in terms of the data analysis itself. "
        "Unfortunately, I was not able to replicate the results of reputable research with the datasets and methods I used. I plan to revisit this and determine "
        "what went wrong, as well as explore why the data yielded insignificant differences. Further research could include more advanced analytical techniques "
        "(which I currently don’t know how to implement). I also believe many of the inconclusive results stem from the nature of the data itself. These large-scale, "
        "generalized datasets include many estimates, and it may simply be that there are no obvious trends to find. I’m aware of the shortcomings in this analysis, "
        "and I’m confident that I can improve my techniques and conduct more accurate analyses in the future."
    )




else:
    st.error("Unknown tab. Please select a valid one.")
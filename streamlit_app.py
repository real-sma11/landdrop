import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import os
from PIL import Image

# Load the CSV file
@st.cache_data
def load_data():
    return pd.read_csv('The Island - Genesis Land_ Snaphot List - List to Publish.csv')

def load_neighborhood_data():
    return pd.read_csv('Neighborhood.csv', index_col=0)  # Keep the first column as the index

df = load_data()
neighborhood_data = load_neighborhood_data()

# Calculate the "Total Units" column
df['Total Units'] = df[['6', '5', '4', '3']].sum(axis=1)

# App title
st.title("The Island: Wallet Probability Land Drop")

# Total units held across all wallets for each column
total_units_per_column = df[['6', '5', '4', '3']].sum()

# Number of unique wallets holding units in each column
unique_wallets_6 = df[df['6'] > 0]['Wallet'].nunique()
unique_wallets_5 = df[df['5'] > 0]['Wallet'].nunique()
unique_wallets_4 = df[df['4'] > 0]['Wallet'].nunique()
unique_wallets_3 = df[df['3'] > 0]['Wallet'].nunique()

# Total number of unique wallets
total_unique_wallets = df['Wallet'].nunique()

# Percentage of unique wallets owning units in each column
percent_wallets_6 = (unique_wallets_6 / total_unique_wallets) * 100
percent_wallets_5 = (unique_wallets_5 / total_unique_wallets) * 100
percent_wallets_4 = (unique_wallets_4 / total_unique_wallets) * 100
percent_wallets_3 = (unique_wallets_3 / total_unique_wallets) * 100

# Display the summary table
st.subheader("Summary of Chains Across All Wallets")
summary_data = {
    "Chain": ['6 (Rarest)', '5', '4', '3 (Most Common)'],
    "Total Held": [total_units_per_column['6'], total_units_per_column['5'], total_units_per_column['4'], total_units_per_column['3']],
    "Unique Wallets": [unique_wallets_6, unique_wallets_5, unique_wallets_4, unique_wallets_3],
    "% of Unique Wallets": [percent_wallets_6, percent_wallets_5, percent_wallets_4, percent_wallets_3]
}

summary_df = pd.DataFrame(summary_data)
st.table(summary_df.set_index('Chain'))

# Display total number of unique wallets
st.write(f"Total Unique Wallets: {total_unique_wallets}")

# Define bins and labels
bins = [0, 1, 2, 3, 4, 5, 10, df['Total Units'].max() + 1]  # Including all specific bins
labels = ['1', '2', '3', '4', '5', '6-10', '11+']

# Count wallets in each bin based on total units
wallet_counts_per_bin = pd.cut(df['Total Units'], bins=bins, labels=labels, include_lowest=True, right=True)

# Count unique wallets in each bin
wallet_counts_per_bin = wallet_counts_per_bin.value_counts().reindex(labels, fill_value=0)

st.subheader("Unique Wallet Distribution by Total Chains Held")
# Create columns for layout with adjusted widths
col1, col2 = st.columns([2, 1])  # Make the first column wider

# Plotting the pie chart in the first column
with col1:
    fig, ax = plt.subplots(figsize=(8, 8))  # Keep the pie chart size
    ax.pie(wallet_counts_per_bin, labels=wallet_counts_per_bin.index, autopct='%1.1f%%', startangle=90, counterclock=False)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

# Display the wallet counts per bin as a table in the second column
with col2:
    wallet_counts_df = wallet_counts_per_bin.reset_index()
    wallet_counts_df.columns = ['Bin', 'Wallets']  # Set custom column names
    st.table(wallet_counts_df.set_index('Bin'))  # Set 'Bin' as the index to hide the default index

# User input for wallet address
wallet_input = st.text_input("Enter Wallet Address:")

if wallet_input:
    # Filter data for the wallet address
    wallet_data = df[df['Wallet'].str.lower() == wallet_input.lower()]
    
    if not wallet_data.empty:
        st.subheader("Chains held by chain length")
        
        # Rename the columns as needed
        wallet_data.columns = ['Wallet', '6', '5', '4', '3', 'Total Chains']  # Adjust as necessary
        
        # Display the table with custom headers
        st.table(wallet_data.set_index('Wallet'))  # Set 'Wallet' as the index to hide the default index
        
        # Total units calculation
        total_units = wallet_data[['6', '5', '4', '3']].sum(axis=1).values[0]

        # Load and display the image
        st.title("Your Landdrop Possibilities")
        st.image('Landdrop01.png', caption='Land Drop Image', use_column_width=True)  # Adjust the caption as needed

        # Load and display the Neighborhood data
        neighborhood_data = load_neighborhood_data()
        st.subheader("Neighborhood Data")
        st.table(neighborhood_data)  # Display the Neighborhood data as a table

        # Visualizations
        st.subheader("Weights Sliders for Random Draw")
        st.write("The first slider adjusts the advantage for rarer units. The second slider illustrates how the chain size could impact the rarity of the land drop (When you run the random draw, the chain owned slider will not impact the results, rather the actual chain sizes in your wallet are used).")

        # Rarity multiplier slider
        rarity_multiplier = st.slider("Adjust the advantage that larger chain sizes have:", min_value=1, max_value=100, value=80)

        # Add a slider for units owned ('6' to '3')
        unit_multiplier = st.slider("Adjust Chain Size Owned Multiplier (6 = Full Weight, 3 = No Weight)", 
                                    min_value=3, max_value=6, value=6, step=1)

        # Define plot types for use in the selection process
        plot_types = ['Residential', 'Commercial', 'Industrial', 'Mixed Use', 'Legendary']

        # Visualization: Single Neighborhood Weights Chart
        st.subheader("Weights Visualization")

        # Calculate total units across neighborhoods for weighting
        total_units_neighborhoods = neighborhood_data['Totals'].sum()
        neighborhood_data['Weight'] = neighborhood_data['Totals'] / total_units_neighborhoods

        # Adjust weights based on both sliders (rarity and unit multiplier)
        column_based_multiplier = {
            '6': 1.0,  # 100% of slider impact for column '6'
            '5': 0.75,  # 75% of slider impact for column '5'
            '4': 0.5,  # 50% of slider impact for column '4'
            '3': 0  # No weighting for column '3' (just base random calculation)
        }

        # Apply rarity multiplier and unit multiplier to the neighborhood weights
        base_weights = neighborhood_data['Weight'].copy()

        # Calculate the minimum rarity value (most rare)
        min_rarity = min(base_weights)

        # Function to adjust weights
        def adjust_weights(weights, rarity_mult):
            if rarity_mult > 1:
                # Calculate rarity multiplier effect using reduced exponential scaling
                rarity_factor = (np.exp(rarity_mult / 22) - 1) * 0.5  # Reduced exponential scaling
                
                # Inverse rarity (higher for rarer neighborhoods)
                min_rarity = min(weights)
                inverse_rarity = (min_rarity / weights) ** 1.5  # Reduced power for less dramatic effect
                
                # Apply multipliers without adding 1
                adjusted = weights * (rarity_factor * inverse_rarity)
            else:
                adjusted = weights
            
            return adjusted

        # Function to adjust plot type weights
        def adjust_plot_weights(weights, rarity_mult):
            if rarity_mult > 1:
                # Calculate rarity multiplier effect using reduced exponential scaling
                rarity_factor = (np.exp(rarity_mult / 40) - 1) * 15 # Reduced exponential scaling
                
                # Inverse rarity (higher for rarer plot types)
                min_rarity = min(weights)
                inverse_rarity = (min_rarity / weights) ** 1.5  # Reduced power for less dramatic effect
                
                # Apply multipliers without adding 1
                adjusted = weights * (rarity_factor * inverse_rarity)
            else:
                adjusted = weights
            
            return adjusted

        # Calculate and store adjusted weights for visualization
        base_adjusted = adjust_weights(base_weights, rarity_multiplier)
        unit_factor = (unit_multiplier - 3) / 3  # Scales from 0 (at 3) to 1 (at 6)
        adjusted_weights_viz = base_weights + (base_adjusted * unit_factor)
        adjusted_weights_viz /= adjusted_weights_viz.sum()  # Normalize

        # Create columns for side-by-side layout
        col1, col2 = st.columns(2)

        # Plot the dynamically adjusted neighborhood weights
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('black')  # Set figure background to black
            ax.set_facecolor('black')  # Set plot area background to black
            ax.bar(neighborhood_data.index, adjusted_weights_viz, color='skyblue')
            ax.set_title('Adjusted Neighborhood Weights', color='white')
            ax.set_xlabel('Neighborhoods', color='white')
            ax.set_ylabel('Weight', color='white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.set_xticks(range(len(neighborhood_data.index)))
            ax.set_xticklabels(neighborhood_data.index, rotation=45, ha='right', color='white')
            plt.tight_layout()  # Adjust layout to prevent cutoff
            st.pyplot(fig)

        # Plot Type Weights Visualization

        # Calculate total number of plots across all neighborhoods
        total_plots = neighborhood_data['Totals'].sum()

        # Calculate the base weights for each plot type
        plot_type_base_weights = neighborhood_data[plot_types].sum() / total_plots

        # Calculate adjusted plot type weights for visualization
        base_adjusted_plot = adjust_plot_weights(plot_type_base_weights, rarity_multiplier)
        adjusted_plot_weights_viz = plot_type_base_weights + (base_adjusted_plot * unit_factor)
        adjusted_plot_weights_viz /= adjusted_plot_weights_viz.sum()  # Normalize

        # Plot the plot type weights dynamically
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('black')  # Set figure background to black
            ax.set_facecolor('black')  # Set plot area background to black
            ax.bar(plot_types, adjusted_plot_weights_viz, color='orange')
            ax.set_title('Adjusted Plot Type Weights', color='white')
            ax.set_xlabel('Plot Types', color='white')
            ax.set_ylabel('Weight', color='white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            plt.tight_layout()  # Adjust layout to prevent cutoff
            st.pyplot(fig)

        # Random plot selection per unit owned by wallet
        if st.button("Run Random Draw"):
            draw_results = []
            for column in ['6', '5', '4', '3']:
                units_owned = wallet_data[column].values[0]
                if units_owned > 0:
                    for _ in range(units_owned):
                        # Apply column-based multiplier
                        column_unit_factor = column_based_multiplier[column]
                        
                        # Adjust neighborhood weights
                        base_adjusted = adjust_weights(neighborhood_data['Weight'], rarity_multiplier)
                        adjusted_weights = neighborhood_data['Weight'] + (base_adjusted * column_unit_factor)
                        adjusted_weights = np.nan_to_num(adjusted_weights)  # Replace NaNs with 0
                        adjusted_weights = adjusted_weights / adjusted_weights.sum()  # Normalize
                        
                        # After normalizing neighborhood weights
                        #st.write("Sum of neighborhood weights:", adjusted_weights.sum())
                        #st.write("Any NaN in neighborhood weights?", np.isnan(adjusted_weights).any())
                        
                        # Select neighborhood
                        if not np.isclose(adjusted_weights.sum(), 1.0, rtol=1e-5) or np.isnan(adjusted_weights).any():
                            st.warning("Neighborhood weights are not properly normalized. Using equal probabilities.")
                            adjusted_weights = np.ones_like(adjusted_weights) / len(adjusted_weights)

                        selected_neighborhood_index = np.random.choice(neighborhood_data.index, p=adjusted_weights)
                        selected_neighborhood = neighborhood_data.loc[selected_neighborhood_index]
                        
                        # Adjust plot type weights
                        plot_weights = selected_neighborhood[plot_types] / selected_neighborhood['Totals']
                        base_adjusted_plot = adjust_plot_weights(plot_weights, rarity_multiplier)
                        adjusted_plot_weights = plot_weights + (base_adjusted_plot * column_unit_factor)
                        adjusted_plot_weights = np.nan_to_num(adjusted_plot_weights)  # Replace NaNs with 0
                        adjusted_plot_weights_sum = adjusted_plot_weights.sum()
                        if adjusted_plot_weights_sum > 0:
                            adjusted_plot_weights = adjusted_plot_weights / adjusted_plot_weights_sum  # Normalize
                        else:
                            adjusted_plot_weights = np.ones_like(adjusted_plot_weights) / len(adjusted_plot_weights)  # Equal probabilities if all are 0
                        
                        # After normalizing plot type weights
                        #st.write("Sum of plot type weights:", adjusted_plot_weights.sum())
                        #st.write("Any NaN in plot type weights?", np.isnan(adjusted_plot_weights).any())
                        
                        # Select plot type
                        if not np.isclose(adjusted_plot_weights.sum(), 1.0, rtol=1e-5) or np.isnan(adjusted_plot_weights).any():
                            st.warning("Plot type weights are not properly normalized. Using equal probabilities.")
                            adjusted_plot_weights = np.ones_like(adjusted_plot_weights) / len(adjusted_plot_weights)

                        selected_plot_type = np.random.choice(plot_types, p=adjusted_plot_weights)
                        
                        draw_results.append({
                            "Neighborhood": selected_neighborhood.name,
                            "Plot Type": selected_plot_type,
                            "Unit Owned": column
                        })

            results_df = pd.DataFrame(draw_results)
            
            # Display unique neighborhood images
            st.subheader("Drawn Neighborhoods")
            unique_neighborhoods = results_df['Neighborhood'].unique()
            
            # Create a container for the images
            image_container = st.container()
            
            # Use columns to create a flexible layout
            cols = image_container.columns(4)  # Adjust the number of columns as needed
            
            for idx, neighborhood in enumerate(unique_neighborhoods):
                image_path = f"{neighborhood}.png"
                if os.path.exists(image_path):
                    image = Image.open(image_path)
                    cols[idx % 4].image(image, use_column_width=True)
                else:
                    cols[idx % 4].write(f"Image not found for {neighborhood}")
            
            st.subheader("Random Draw Results")
            st.table(results_df)

    else:
        st.write("Wallet not found.")

# Add logging
st.write("Debug: Script execution completed")
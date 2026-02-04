import streamlit as st
import pandas as pd
import ast
import matplotlib.pyplot as plt
from pymatgen.core import Composition
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Tinderry Battery Tool",
    page_icon="🔋",
    layout="wide"
)

# --- LOGIC FUNCTIONS ---
@st.cache_data
def load_data():
    """Loads CSV files and performs basic filtering."""
    try:
        df_bat = pd.read_csv("all_batteries_by_ion.csv")
        df_bat = df_bat[df_bat["average_voltage"] >= 0]
        df_sus = pd.read_csv("sust_params_clean.csv")
        df_wgi = pd.read_csv("World Governance Index.csv")
        return df_bat, df_sus , df_wgi
    except FileNotFoundError as e:
        st.error(f"Error: File not found {e.filename}")
        return None, None , None

def total_element_masses(formula_anode, mass_anode, formula_cathode, mass_cathode, rename_anode_c=False):
    """Calculates mass per element for the anode-cathode pair."""
    try:
        comp_anode = Composition(formula_anode)
        comp_cathode = Composition(formula_cathode)
        
        total_molar_anode = comp_anode.weight
        total_molar_cathode = comp_cathode.weight
        
        total_masses = {}

        # Anode mass per element
        for el, el_index in comp_anode.element_composition.items():
            symbol = el.symbol
            # If flag active and element is C, rename to graphite
            if rename_anode_c and symbol == "C":
                symbol = "Graphite"
            
            total_masses[symbol] = (el_index * el.atomic_mass / total_molar_anode) * mass_anode

        # Cathode mass per element
        for el, el_index in comp_cathode.element_composition.items():
            symbol = el.symbol
            mass = (el_index * el.atomic_mass / total_molar_cathode) * mass_cathode
            total_masses[symbol] = total_masses.get(symbol, 0) + mass
            
        return total_masses
    except:
        return {}

    #################################
    #                               #
    #    INVERTED NORMALIZATION     #   ------> Using   n  =  max-a  /  max-min
    #                               #
    #################################

def inv_norm(column):
    normalized_column= (column.max()-column)/(column.max()-column.min())
    return normalized_column

def Score(R,Q,l):
    T=(R+Q)/(R+Q).sum()
    T1=R/R.min()+Q/Q.min()
    T2=(l*R+(1-l)*Q)/(l*R.max()+(1-l)*Q.max())
    S=(T*T1*T2)**(1/3)+(1/3)*(T+T1+T2)
    return S

if "filters_initialized" not in st.session_state:
    st.session_state["price_val"] = 150.0
    st.session_state["co2_val"] = 250.0
    st.session_state["energy_val"] = 3000.0
    st.session_state["water_val"] = 800.0
    st.session_state["stab_val"] = 0.0250
    st.session_state["dv_val"] = 0.40
    st.session_state["mis_val"] = 0.20
    st.session_state["filters_initialized"] = True

def reset_to_none():
    for key in ["price_val", "co2_val", "energy_val", "water_val", "stab_val", "dv_val", "mis_val"]:
        st.session_state[key] = None
def reset_to_default():
    st.session_state["price_val"] = 150.0
    st.session_state["co2_val"] = 250.0
    st.session_state["energy_val"] = 3000.0
    st.session_state["water_val"] = 800.0
    st.session_state["stab_val"] = 0.0250
    st.session_state["dv_val"] = 0.40
    st.session_state["mis_val"] = 0.20

l=0.05

# --- DATA LOADING & PREPARATION ---
df_batteries, df_sustainable, df_gov = load_data()

df_batteries["elements"] = df_batteries["elements"].apply(ast.literal_eval)


# --- SIDEBAR (INPUTS) ---
st.sidebar.header("🛠️ Parameters Configuration")

desired_energy = st.sidebar.number_input("Desired Energy (Wh)", value=1200.0, step=100.0)

# Ion Selection
ion_list = sorted(df_batteries["working_ion"].unique().tolist())
working_ion = st.sidebar.selectbox("Working Ion", ion_list, index=ion_list.index("Li"))

# Multiple element selection for anode
all_unique_elements = sorted(list(set(el for sublist in df_batteries["elements"] for el in sublist)))
selected_anode_elements = st.sidebar.multiselect(
    "Select Anode Materials", 
    options=all_unique_elements, 
    default=["C"]
)
with st.sidebar:
    st.markdown("### ⚖️ Ranking Weights")
    st.markdown("##### 0: Not important \n\n 5: Very important ")

    w_total_price = st.slider(
        "💰 Price Importance", 0, 5, 3,
        help="User-assigned weight for the monetary cost of battery materials (cathode + anode)"
    )

    w_sustainability = st.slider(
        "🍃 Sustainability Importance", 0, 5, 3,
        help="User-assigned weight for the environmental factors of battery materials (cathode + anode)"
    )

    with st.expander("🍃Subcategories", expanded=False):
        w_CO2_footprint = st.slider(
            "CO2 Footprint Importance", 0, 5, 3,
            help="Influence of the kilograms of CO₂ emitted during the production of battery materials (anode + cathode)."
        )

        w_energy_required = st.slider(
            "Energy Footprint. Importance", 0, 5, 3,
            help="Influence of the total energy consumed to produce the battery materials (anode + cathode)."
        )

        w_water_usage = st.slider(
            "Water Usage Importance", 0, 5, 3,
            help="Influence of the total water consumption required to obtain the battery materials (anode + cathode)."
        )

    w_durability = st.slider(
        "🔧 Durability Importance", 0, 5, 3,
        help="User-assigned weight for the technical performance of battery materials (cathode + anode)."
    )

    with st.expander("🔧Subcategories", expanded=False):
        w_stability = st.slider(
            "Stability Weight", 0, 5, 3,
            help="Influence of the thermodynamic stability of the battery materials; lower energy indicates higher stability."
        )

        w_max_delta_volume = st.slider(
            "Max Delta Volume Weight", 0, 5, 3,
            help="Influence of the maximum volume change of cathode during operation."
        )

        w_volume_mismatch = st.slider(
            "Volume mismatch weight", 0, 5, 3,
            help="Influence of the difference in maximum volume change between anode and cathode materials."
        )

total_sustainability_weight=w_CO2_footprint+w_energy_required+w_water_usage

w_co2=w_CO2_footprint/total_sustainability_weight
w_e=w_energy_required/total_sustainability_weight
w_h20=w_water_usage/total_sustainability_weight


total_durability_weight= w_stability +  w_max_delta_volume +w_volume_mismatch 
w_stab = w_stability/total_durability_weight
w_dv=w_max_delta_volume/total_durability_weight
w_mis= w_volume_mismatch/total_durability_weight


total_finals_weight=w_sustainability+w_durability+w_total_price
w_sust=w_sustainability/total_finals_weight
w_dur=w_durability/total_finals_weight
w_price=w_total_price/total_finals_weight

# --- MAIN BODY ---
st.title("🔋 Tinderry: the electrode matching app")

# Prepare sustainability dictionaries
price_dict = df_sustainable.set_index('Elements')['Price (USD/kg)'].to_dict()
CO2_dict = df_sustainable.set_index('Elements')['CO2 footprint (kg CO2/kg)'].to_dict()
energy_dict = df_sustainable.set_index('Elements')['Energy footprint (MJ/kg)'].to_dict()
water_dict = df_sustainable.set_index('Elements')['Water usage (L/kg)'].to_dict()
countries_dict = df_sustainable.set_index('Elements')['countries_list'].to_dict()
hhi_dict = df_sustainable.set_index('Elements')['HHI'].to_dict()

with st.expander("🔍 Filters", expanded=False):
    col_space, col_btn_def, col_btn_reset = st.columns([3, 1, 1])
    
    with col_btn_def:
        if st.button("⚙️ Default", width='stretch', help="Restore recomended values"):
            reset_to_default()
            st.rerun()

    with col_btn_reset:
        if st.button("🔓 Remove Filters", width='stretch', help="Remove all filters (show everything)"):
            reset_to_none()
            st.rerun()

    col1, col2 = st.columns(2)

    with col1:

        price_filter = st.number_input(
            "Max Price (USD)",
            min_value=0.0, step=100.0, format="%.2f", key="price_val",
            help="Maximum allowed monetary cost of battery materials (anode + cathode)."
        )

        CO2_filter = st.number_input(
            "Max CO₂ (kg)",
            min_value=0.0, step=10.0, format="%.2f", key="co2_val",
            help="Maximum allowed kilograms of CO₂ emitted during the production of battery materials (anode + cathode)."
        )

        energy_filter = st.number_input(
            "Max Energy (MJ)",
            min_value=0.0, step=100.0, format="%.2f", key="energy_val",
            help="Maximum allowed total energy consumed to produce the battery materials (anode + cathode)."
        )

        water_filter = st.number_input(
            "Max Water (L)",
            min_value=0.0, step=50.0, format="%.2f", key="water_val",
            help="Maximum allowed total water consumption required to obtain the battery materials (anode + cathode)."
        )

    with col2:
        stab_filter = st.number_input(
            "Max Stability (eV/atom)",
            min_value=0.0, step=0.001, format="%.4f", key="stab_val",
            help="Maximum allowed formation energy of battery materials; lower values indicate higher stability."
        )

        dv_filter = st.number_input(
            "Max Delta Volume",
            min_value=0.0, step=0.01, format="%.2f", key="dv_val",
            help="Maximum allowed volume change of cathode during operation."
        )

        mis_filter = st.number_input(
            "Max Volume Mismatch",
            min_value=0.0, step=0.01, format="%.2f", key="mis_val",
            help="Maximum allowed difference in volume change between anode and cathode materials."
        )
        
# Filter by Ion
df_selected_ion = df_batteries[df_batteries["working_ion"] == working_ion].copy()
if not selected_anode_elements:
    st.warning("Please select elements for the anode in the sidebar.")

else:
    df_anodes = df_selected_ion[
        df_selected_ion["elements"].apply(lambda e: set(e) == set(selected_anode_elements))
    ].copy()

    if df_anodes.empty:
        st.warning(f"No anodes found that contain the elements: {', '.join(selected_anode_elements)}.")
    else:
        # --- STEP 1: ANODE SELECTION ---
        st.subheader("1. Anode Selection")
        df_anodes["max_stability"] = df_anodes[["stability_charge", "stability_discharge"]].max(axis=1) 
        if "selected_anode_idx" not in st.session_state or st.session_state.selected_anode_idx not in df_anodes.index:
            st.session_state.selected_anode_idx = df_anodes.index[0]

        cols_anodes = st.columns(3) # Adjusts number according to how many anodes
        for i, (idx, row) in enumerate(df_anodes.iterrows()):
            label = f"{row['battery_formula']}\n({row['max_stability']:.3f} eV)"
            if cols_anodes[i % 3].button(label, key=f"btn_{idx}", width='stretch'):
                st.session_state.selected_anode_idx = idx

        # Extract data of anode via session_state
        selected_idx = st.session_state.selected_anode_idx
        v_anode = df_anodes.loc[selected_idx, "average_voltage"]
        cap_anode = df_anodes.loc[selected_idx, "capacity_grav"]
        f_anode = df_anodes.loc[selected_idx, "formula_charge"]
        name_anode = df_anodes.loc[selected_idx, "battery_formula"]
        delta_v_anode= df_anodes.loc[selected_idx, "max_delta_volume"]
        stab_value = df_anodes.loc[selected_idx, "max_stability"]


        if stab_value*1000 > 50:
            st.warning(
                f"ℹ️ **Stability note**\n\n"
                f"**{name_anode}** shows high instability "
                f"({stab_value*1000:.2f} meV/atom)"
            )
        else:
            st.success(f"Selected Anode: **{name_anode}**")

        # STEP 2: CATHODE CALCULATION

        # If selected anode is graphite
        is_graphite_anode = selected_anode_elements == ["C"]

        with st.spinner('Calculating compatible combinations...'):
            df_tmp = df_selected_ion[df_selected_ion["average_voltage"] > v_anode].copy()
            
            if df_tmp.empty:
                st.error("No compatible cathodes found with voltage higher than this anode.")
            else:
                # Calculations
                df_tmp["voltage_difference"] = df_tmp["average_voltage"] - v_anode
                df_tmp["mass_anode"] = desired_energy / (cap_anode * df_tmp["voltage_difference"])
                df_tmp["mass_cathode"] = desired_energy / (df_tmp["capacity_grav"] * df_tmp["voltage_difference"])
                df_tmp["total_mass"]=df_tmp["mass_anode"]+df_tmp["mass_cathode"]
                
                df_tmp["total_mass_elements"] = [
                total_element_masses(
                    f_anode, 
                    row["mass_anode"], 
                    row["formula_discharge"], 
                    row["mass_cathode"],
                    rename_anode_c=is_graphite_anode
                )
                for _, row in df_tmp.iterrows()
                    ]
                # --------CRITERIONS-------
                #Price
                df_tmp["total_price"] = df_tmp["total_mass_elements"].apply(lambda d: sum(m * price_dict.get(e, 0) for e, m in d.items()))
                #Sustainability
                df_tmp["total_CO2"] = df_tmp["total_mass_elements"].apply(lambda d: sum(m * CO2_dict.get(e, 0) for e, m in d.items()))
                df_tmp["total_energy"] = df_tmp["total_mass_elements"].apply(lambda d: sum(m * energy_dict.get(e, 0) for e, m in d.items()))
                df_tmp["total_water"] = df_tmp["total_mass_elements"].apply(lambda d: sum(m * water_dict.get(e, 0) for e, m in d.items()))
                # Durability score
                df_tmp["eff_stab"] = df_tmp[["stability_charge", "stability_discharge"]].max(axis=1)
                df_tmp["volume_mismatch"] = abs(delta_v_anode-df_tmp["max_delta_volume"])

                #Filter 
                if price_filter is not None:
                    df_tmp = df_tmp[df_tmp["total_price"] <= price_filter]

                if CO2_filter is not None:
                    df_tmp = df_tmp[df_tmp["total_CO2"] <= CO2_filter]

                if energy_filter is not None:
                    df_tmp = df_tmp[df_tmp["total_energy"] <= energy_filter]

                if water_filter is not None:
                    df_tmp = df_tmp[df_tmp["total_water"] <= water_filter]

                if stab_filter is not None:
                    df_tmp = df_tmp[df_tmp["eff_stab"] <= stab_filter]

                if dv_filter is not None:
                    df_tmp = df_tmp[df_tmp["max_delta_volume"] <= dv_filter]

                if mis_filter is not None:
                    df_tmp = df_tmp[df_tmp["volume_mismatch"] <= mis_filter]

    
                df_tmp["energy_density"] =  desired_energy/ df_tmp["total_mass"]

                #################################
                #                               #
                #    INVERTED NORMALIZATION     #   ------> Using   n  =  max-a  /  max-min
                #                               #
                #################################

                #DURABILTY
                df_tmp["norm_stab"] = inv_norm(df_tmp["eff_stab"])
                df_tmp["norm_mismatch"] =inv_norm(df_tmp["volume_mismatch"])
                df_tmp["norm_vol"] = inv_norm(df_tmp["max_delta_volume"])
                #SUSTAINABILITY
                df_tmp["norm_CO2"] =inv_norm(df_tmp["total_CO2"])
                df_tmp["norm_energy"] = inv_norm(df_tmp["total_energy"])
                df_tmp["norm_water"] = inv_norm(df_tmp["total_water"])
                #PRICE
                df_tmp["norm_price"] = inv_norm(df_tmp["total_price"])

                df_tmp["R"]=  (
                    (df_tmp["norm_stab"] * w_stab*w_dur +df_tmp["norm_vol"] * w_dv*w_dur +df_tmp["norm_mismatch"] * w_mis*w_dur) + 
                     (df_tmp["norm_CO2"]* w_co2*w_sust +df_tmp["norm_energy"] * w_e*w_sust +df_tmp["norm_water"]* w_h20*w_sust) +
                    df_tmp["norm_price"]*w_price
                )

                df_tmp["Q"]= (
                    (df_tmp["norm_stab"] ** (w_stab*w_dur) +df_tmp["norm_vol"] ** (w_dv*w_dur) +df_tmp["norm_mismatch"] ** (w_mis*w_dur)) + 
                    (df_tmp["norm_CO2"]** (w_co2**w_sust) +df_tmp["norm_energy"] **( w_e*w_sust) +df_tmp["norm_water"]** (w_h20*w_sust)) + 
                    df_tmp["norm_price"]**w_price
                )

                df_tmp["S"]= Score(df_tmp["R"],df_tmp["Q"],l)


                # --- RESULTS DISPLAY 
                df_sorted = df_tmp.sort_values(by="S", ascending=False).reset_index(drop=True)
                df_sorted["Rank"] = (
                    df_sorted["S"].index +1
                )
                total_cathodes = len(df_sorted)

                st.subheader(f"2. Showing {total_cathodes} cathodes matching with {name_anode}")


                # Show table and keep user selection
                selection_event = st.dataframe(
                    df_sorted[[ "battery_id","Rank","formula_discharge", "S",
                               "mass_cathode","mass_anode","total_mass", "energy_density","total_price","norm_price", 
                               "total_CO2", "total_water", "total_energy","max_delta_volume",
                               "volume_mismatch","eff_stab"]],
                    column_config={
                        "battery_id": None,
                        "Rank": "Rank",
                        "formula_discharge": "Formula",
                        "S": "Overall Score",
                        "total_mass":None,
                        "mass_cathode": "Cathode mass (kg)",
                        "mass_anode": "Anode mass (kg)",
                        "energy_density" : "E/electrodes mass (Wh/kg)",
                        "total_price": "Cost (USD)",
                        "norm_price":None,
                        "total_CO2": "CO2 fooprint(kg)",
                        "total_water": "Water usage(L)",
                        "total_energy": "Energy footprint(MJ)",
                        "max_delta_volume": "Max delta volume",
                        "eff_stab" : "Stability (eV/atom)",
                        "volume_mismatch":" Volume mismatch"

                    },
                    hide_index=True, width='stretch',
                    on_select="rerun", # <--- Enables interactivity
                    selection_mode="single-row"
                )

                # Logic for selection of what is shown in 3rd step
                if selection_event.selection.rows:
                    selected_row_index = selection_event.selection.rows[0]
                    selected_bat_formula = df_sorted.iloc[selected_row_index]["formula_discharge"]
                else:
                    # If nothing is selected, selects first entry by default
                    if len(df_sorted) > 0:
                        selected_bat_formula = df_sorted.iloc[0]["formula_discharge"]
                    else:
                        st.info("⚠️ No results to display. Please adjust the anode material filters.")
                        st.stop()

                st.subheader(f"3.  Showing details for: **{selected_bat_formula}**")


                row_data = df_sorted[df_sorted["formula_discharge"] == selected_bat_formula].iloc[0]
                mass_dict = row_data["total_mass_elements"]

                breakdown_data = []
                missing_info = [] # List for missing information

                for el, mass in mass_dict.items():
                    # Verify if the data exists in the dictionaries (and is not NaN)
                    p = price_dict.get(el); p = p if p != 0 else pd.NA
                    c = CO2_dict.get(el); c = c if c != 0 else pd.NA
                    e = energy_dict.get(el); e = e if e != 0 else pd.NA
                    w = water_dict.get(el); w = w if w != 0 else pd.NA

                    # save value if it is NaN or None
                    current_missing = []
                    if pd.isna(p): current_missing.append("Price")
                    if pd.isna(c): current_missing.append("CO2 footprint")
                    if pd.isna(e): current_missing.append("Energy")
                    if pd.isna(w): current_missing.append("Water usage")
                    
                    if current_missing:
                        missing_info.append(f"**{el}**: {', '.join(current_missing)}")

                    breakdown_data.append({
                        "Element": el,
                        "Mass (kg)": mass,
                        "Cost ($)": mass * (p if pd.notna(p) else 0),
                        "CO2 (kg)": mass * (c if pd.notna(c) else 0),
                        "Energy (MJ)": mass * (e if pd.notna(e) else 0),
                        "Water (L)": mass * (w if pd.notna(w) else 0),
                        "HHI": hhi_dict.get(el)
                    })

                df_breakdown = pd.DataFrame(breakdown_data).set_index("Element")

                # --- Missing information warning ---
                if missing_info:
                    items_text = "\n".join([f"- {item}" for item in missing_info])
                    st.info(" ℹ️ **Information notice**\n\n"
                    "The following elements do not have data source -> "
                    "Calculations for this variables are shown as zero:\n"
                    f"{items_text}")



                st.markdown("### 📊 Detailed Breakdown")
                
                col0, col1= st.columns(2)

                with col0:
                    radar_df = pd.DataFrame(dict(
                        r=[row_data["norm_price"], row_data["norm_CO2"],row_data["norm_energy"],row_data["norm_water"],row_data["norm_stab"],row_data["norm_vol"],row_data["norm_mismatch"]],
                        theta=['Cost','CO2', 'Energy footprint','Water Usage','Stability','Δ Volume','mismatch']))
                    
                    fig_radar = px.line_polar(radar_df, r='r', theta='theta', line_close=True,title=f"🌐 Overall Performance Profile<br><sup>By score - higher is better</sup>")
                    fig_radar.update_traces(fill='toself')
                    fig_radar.update_polars(radialaxis=dict(tickfont=dict(color="black")))
                    st.plotly_chart(fig_radar, width='stretch',)
                    

                with col1:
                    fig_mass = px.pie(
                        df_breakdown, 
                        values="Mass (kg)", 
                        names=df_breakdown.index,
                        title=f"⚖️ Total Mass {row_data['total_mass']:.2f} kg",
                        hole=0.4,
                        color_discrete_sequence=px.colors.sequential.Purples_r
                    )

                    fig_mass.update_traces(
                        texttemplate='%{label}<br>%{value:.2f} kg',
                        textfont_size=16,  
                        hovertemplate='<b>%{label}</b><br>%{value:.2f} kg<br><b>%{percent}</b><extra></extra>',  
                        hoverlabel=dict(font_size=20)  
                    )
                    st.plotly_chart(fig_mass, width='stretch')

                col2, col3= st.columns(2)
                with col2:
                    fig_cost = px.pie(
                        df_breakdown, 
                        values="Cost ($)", 
                        names=df_breakdown.index,
                        title=f"💰 Cost Breakdown ${row_data['total_price']:.2f}",
                        hole=0.4,
                        color_discrete_sequence=px.colors.sequential.Greens_r
                    )
                    fig_cost.update_traces(
                        texttemplate='%{label}<br>$%{value:.2f}',
                        textfont_size=16,
                        hovertemplate='<b>%{label}</b><br>$%{value:.2f}<br><b>%{percent}</b><extra></extra>',
                        hoverlabel=dict(font_size=20)
                    )
                    st.plotly_chart(fig_cost, width='stretch')

                with col3:
                    fig_co2 = px.pie(
                        df_breakdown, 
                        values="CO2 (kg)", 
                        names=df_breakdown.index,
                        title=f"🌫️ CO2 Footprint {row_data['total_CO2']:.2f} kg",
                        hole=0.4,
                        color_discrete_sequence=px.colors.sequential.Reds_r
                    )
                    fig_co2.update_traces(
                        texttemplate='%{label}<br>%{value:.2f} kg',
                        textfont_size=16,
                        hovertemplate='<b>%{label}</b><br>%{value:.2f} kg<br><b>%{percent}</b><extra></extra>',
                        hoverlabel=dict(font_size=20)
                    )
                    st.plotly_chart(fig_co2, width='stretch')

                col4, col5 = st.columns(2)

                with col4:
                    fig_energy = px.pie(
                        df_breakdown, 
                        values="Energy (MJ)", 
                        names=df_breakdown.index,
                        title=f"⚡ Energy Footprint {row_data['total_energy']:.2f} MJ",
                        hole=0.4,
                        color_discrete_sequence=px.colors.sequential.YlOrBr_r
                    )
                    fig_energy.update_traces(
                        texttemplate='%{label}<br>%{value:.2f} MJ',
                        textfont_size=16,
                        hovertemplate='<b>%{label}</b><br>%{value:.2f} MJ<br><b>%{percent}</b><extra></extra>',
                        hoverlabel=dict(font_size=20)
                    )
                    st.plotly_chart(fig_energy, width='stretch')

                with col5:
                    fig_water = px.pie(
                        df_breakdown, 
                        values="Water (L)", 
                        names=df_breakdown.index,
                        title=f"💧 Water Usage {row_data['total_water']:.2f} L",
                        hole=0.4,
                        color_discrete_sequence=px.colors.sequential.Blues_r
                    )
                    fig_water.update_traces(
                        texttemplate='%{label}<br>%{value:.2f} L',
                        textfont_size=16,
                        hovertemplate='<b>%{label}</b><br>%{value:.2f} L<br><b>%{percent}</b><extra></extra>',
                        hoverlabel=dict(font_size=20)
                    )
                    st.plotly_chart(fig_water, width='stretch')
                                # --- COUNTRIES BREAKDOWN ---

                # Compute countries first
                all_countries = set()
                countries_pills = {}
                for el in mass_dict:
                    raw = countries_dict.get(el)
                    cleaned = raw.replace("nan", "None")
                    parsed = ast.literal_eval(cleaned)
                    countries = [c.strip() for c in parsed if c is not None and c.strip()]
                    all_countries.update(countries)
                    countries_pills[el] = countries

                countries_list = sorted(all_countries)

                with st.expander("🌍 Countries information", expanded=True):
                    col0, col1 = st.columns([1, 1.5])
                    with col0:
                        
                        for el, countries in countries_pills.items():
                            if countries:
                                pills = " ".join(
                                    f'<span style="background-color:#4a3f5c; color:#ffffff; padding:5px 14px; border-radius:12px; font-size:16px; margin-right:6px;">{c}</span>'
                                    for c in countries
                                )
                                st.markdown(f'''
                                    <div style="margin-top:15px; margin-bottom:5px;">
                                        <span style="font-size:20px; font-weight:bold;">{el}</span>
                                    </div>
                                    <div style="display:flex; flex-wrap:wrap; gap:6px;">
                                        {pills}
                                    </div>
                                ''', unsafe_allow_html=True)
                        st.divider()        
                        df_hhi_plot = df_breakdown.reset_index()
                        df_hhi_plot = df_hhi_plot[df_hhi_plot["HHI"].notna()]

                        if not df_hhi_plot.empty:
                            fig_hhi = px.bar(
                                df_hhi_plot,
                                x="HHI",
                                y="Element",
                                orientation="h",
                                title="Market Concentration (HHI)",
                                color="HHI",
                                color_continuous_scale=["#2ecc71", "#f1c40f", "#e74c3c"]
                            )
                            #Unconcentrated (< 0.15)
                            fig_hhi.add_vrect(x0=0, x1=0.15, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Low", annotation_position="top left")
                            #Moderately Concentrated (0.15 - 0.25)
                            fig_hhi.add_vrect(x0=0.15, x1=0.25, fillcolor="yellow", opacity=0.1, line_width=0, annotation_text="Mid", annotation_position="top left")
                            # Highly Concentrated (> 0.25)
                            max_hhi = max(df_hhi_plot["HHI"].max(), 0.4) 
                            fig_hhi.add_vrect(x0=0.25, x1=max_hhi, fillcolor="red", opacity=0.1, line_width=0, annotation_text="High", annotation_position="top left")
                            # --- ---
                            fig_hhi.add_vline(x=0.15, line_dash="dash", line_color="gray", opacity=0.5)
                            fig_hhi.add_vline(x=0.25, line_dash="dash", line_color="gray", opacity=0.5)

                            fig_hhi.update_layout(
                                height=350, 
                                margin=dict(l=10, r=10, t=50, b=10),
                                showlegend=False,
                                coloraxis_showscale=False,
                                yaxis={'categoryorder':'total ascending'},
                                yaxis_title=None,
                                xaxis_title="HHI Score",
                                xaxis=dict(range=[0, max_hhi + 0.05]) 
                            )
                            
                            st.plotly_chart(fig_hhi, width='stretch')
                        

                    with col1:
                        df_gov_filtered = df_gov[df_gov["Country"].isin(countries_list)]

                        fig_gov = px.bar(
                            df_gov_filtered.melt(id_vars="Country", value_vars=["Political Stability", "Regulatory Quality", "Rule of law", "Control of corruption"], var_name="Category", value_name="Index"),
                            x="Index", y="Country", color="Category", orientation="h", barmode="group",
                            title="🏛️ Governance Indices",
                            #height=len(countries_list) * 65
                        )
                        fig_gov.update_layout(
                            yaxis_tickfont=dict(size=16),
                            xaxis_tickfont=dict(size=16),
                            legend_font=dict(size=14),
                            title_font=dict(size=18),
                            title_x=0.2,
                            yaxis_title=None
                        )
                        st.plotly_chart(fig_gov,width='stretch')


st.markdown("---")
st.caption("Developed by GREENANO Alumni for Battery Sustainability Research.")
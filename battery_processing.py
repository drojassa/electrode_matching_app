import pandas as pd
import ast 
from pymatgen.core import Composition
from pymatgen.core import Element
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

#Function to obtain the mass of each element per battery (anode+cathode)
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

  
#Multiply each mass of each element with their respective price
def quantity_by_element(element_masses, quantity_dict):
    return {
        elem: mass * quantity_dict.get(elem, 0)
        for elem, mass in element_masses.items()
    }

    #################################
    #                               #
    #    INVERTED NORMALIZATION     #   ------> Using   n  =  max-a  /  max-min
    #                               #
    #################################

def inv_norm(column):
    normalized_column= (column.max()-column)/(column.max()-column.min())
    return normalized_column

l=0.99

def Score(R,Q,l):
    T=(R+Q)/(R+Q).sum()
    T1=R/R.min()+Q/Q.min()
    T2=(l*R+(1-l)*Q)/(l*R.max()+(1-l)*Q.max())
    S=(T*T1*T2)**(1/3)+(1/3)*(T+T1+T2)
    return S

#Load CSV of electrodes
df_batteries = pd.read_csv("all_batteries_by_ion.csv")
df_batteries = df_batteries[df_batteries["average_voltage"] >= 0] #Filter negative voltage
df_batteries = df_batteries.drop(columns=["material_ids", "fields_not_requested"])
df_batteries["elements"] = df_batteries["elements"].apply(ast.literal_eval) #Converts string to list


#Load Sustainable parameters
df_sustainable= pd.read_csv("Sustainable_Parameters.csv")
price_dict = df_sustainable.set_index('Elements')['Price (USD/kg)'].to_dict() #Obtain prices as dictionary
CO2_dict= df_sustainable.set_index('Elements')['CO2 footprint (kg CO2/kg)'].to_dict()
energy_dict= df_sustainable.set_index('Elements')['Energy footprint (MJ/kg)'].to_dict()
water_dict= df_sustainable.set_index('Elements')['Water usage (L/kg)'].to_dict()

#User inputs
DesiredEnergy= 1.2e3 #Watt hour
DesiredIon='Li' #input('Enter ion: ')
MaterialAnodeList = ['C'] #Graphite

#USER INPUT Defining weights for durability (from 0 to 1)
w_stab= 1/3
w_dv= 1/3
w_mis=1/3

#User input sustainability weights
w_co2=1/3
w_e=1/3
w_h20=1/3

#User input final weights
w_price=3/9
w_dur=4/9
w_sust=2/9


                #Filter 
price_filter =150.0

CO2_filter =250.0

energy_filter =3000.0

water_filter =800.0

stab_filter =0.025

dv_filter =0.4

mis_filter =0.2
                    

df_selected_ion= df_batteries[df_batteries["working_ion"] == DesiredIon] #Dataframe with Desired Ion
df_selected_ion = df_selected_ion.copy() #Copy of dataframe to avoid warnings


print(f"Number of Electrodes with {DesiredIon}: {len(df_selected_ion)}")  

#This dataframe shows what are the possible anodes with the specified material (Ex:graphite is C)
df_anodes = df_selected_ion[
    df_selected_ion["elements"].apply(
        lambda elems: set(elems).issubset(set(MaterialAnodeList))
    )
]
print('THE LIST OF THE ANODES IS:','\n', df_anodes)

anode_indices = df_anodes.index.tolist()
dfs_by_anode = {}

#For each anode do the calculation of Delta V, mass anode and mass cathode
for idx in anode_indices:
    #Active flag
    is_graphite_anode = MaterialAnodeList == ["C"] 

    VoltageAnode = df_anodes.loc[idx, "average_voltage"]
    Grav_Cap_Anode= df_anodes.loc[idx, "capacity_grav"]
    formula_anode = df_anodes.loc[idx, "formula_charge"]
    delta_v_anode= df_anodes.loc[idx, "max_delta_volume"]

    df_tmp = df_selected_ion.copy()
    df_tmp = df_tmp[df_tmp["average_voltage"] > VoltageAnode]

    #Calculate voltage difference (V)
    df_tmp["voltage_difference"] = (df_tmp["average_voltage"] - VoltageAnode)

    #Calculate mass of anode (kg)
    df_tmp["mass_anode"] = DesiredEnergy/(Grav_Cap_Anode*df_tmp["voltage_difference"])

    #Calculate mass of cathode (kg)
    df_tmp["mass_cathode"] = DesiredEnergy/(df_tmp["capacity_grav"]*df_tmp["voltage_difference"])

    #Applies the function defined above to calculate the mass of each element in the battery
    df_tmp["total_mass_elements"] = [
        total_element_masses(formula_anode, row["mass_anode"], row["formula_discharge"], row["mass_cathode"],rename_anode_c=is_graphite_anode)
        for _, row in df_tmp.iterrows()
    ]

    #-------------Obtain durability criterion for each cathode-----------------

    df_tmp["effective_stability"] = df_tmp[["stability_charge", "stability_discharge"]].max(axis=1)
    df_tmp["volume_mismatch"] = abs(delta_v_anode-df_tmp["max_delta_volume"])
    #  We already have  df_tmp["max_delta_volume"]

    #-------------Obtain sustainability criterion for each cathode--------------
    #CO2 footprint
    df_tmp["CO2"] = df_tmp["total_mass_elements"].apply(lambda d: quantity_by_element(d, CO2_dict))
    df_tmp["total_CO2"] = df_tmp["CO2"].apply(lambda d: sum(d.values()))
    #Energy
    df_tmp["energy"] = df_tmp["total_mass_elements"].apply(lambda d: quantity_by_element(d, energy_dict))
    df_tmp["total_energy"] = df_tmp["energy"].apply(lambda d: sum(d.values()))
    #Water
    df_tmp["water"] = df_tmp["total_mass_elements"].apply(lambda d: quantity_by_element(d, water_dict))
    df_tmp["total_water"] = df_tmp["water"].apply(lambda d: sum(d.values()))

    #----------------Obtain price criterion--------------------

    df_tmp["prices"] = df_tmp["total_mass_elements"].apply(lambda d: quantity_by_element(d, price_dict))
    df_tmp["total_price"] = df_tmp["prices"].apply(lambda d: sum(d.values()))

    #################################
    #                               #
    #    INVERTED NORMALIZATION     #   ------> Using   n  =  max-a  /  max-min
    #                               #
    #################################

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
        df_tmp = df_tmp[df_tmp["effective_stability"] <= stab_filter]

    if dv_filter is not None:
        df_tmp = df_tmp[df_tmp["max_delta_volume"] <= dv_filter]

    if mis_filter is not None:
        df_tmp = df_tmp[df_tmp["volume_mismatch"] <= mis_filter]

    #DURABILTY
    df_tmp["norm_stab"] = inv_norm(df_tmp["effective_stability"])
    df_tmp["norm_mismatch"] =inv_norm(df_tmp["volume_mismatch"])
    df_tmp["norm_vol"] = inv_norm(df_tmp["max_delta_volume"])
    #SUSTAINABILITY
    df_tmp["norm_CO2"] =inv_norm(df_tmp["total_CO2"])
    df_tmp["norm_energy"] = inv_norm(df_tmp["total_energy"])
    df_tmp["norm_water"] = inv_norm(df_tmp["total_water"])
    #PRICE
    df_tmp["norm_price"] = inv_norm(df_tmp["total_price"])
    df_tmp["energy_density"] =  DesiredEnergy/ (df_tmp["mass_anode"]+df_tmp["mass_cathode"])

    #R calculation
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

    df_sorted = df_tmp.sort_values(by="S", ascending=False).reset_index(drop=True)

    dfs_by_anode[idx] = df_sorted


df_anode_case = dfs_by_anode[2]
print(df_anode_case.head(10))


df_bat = df_anode_case.head(5)
labels = df_bat["formula_discharge"].tolist()
categories = ["norm_price","norm_CO2","norm_energy","norm_water","norm_stab","norm_vol","norm_mismatch"]

N = len(categories)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]

# ── RADAR ──
fig, ax = plt.subplots(figsize=(7,7), subplot_kw={"polar": True})
for i, row in df_bat.iterrows():
    vals = row[categories].tolist() + [row[categories[0]]]
    ax.plot(angles, vals, "o-", label=labels[df_bat.index.get_loc(i)], markersize=3)
    ax.fill(angles, vals, alpha=0.07)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(["Price","CO₂ footprint","Energy footprint","Water usage","Thermodynamical stability","Delta volume","Volume mismatch"])
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

# Guardar y mostrar
plt.savefig('grafico_radar.png', dpi=300, bbox_inches='tight')
plt.show()


# ── BARRAS HORIZONTALES (Overall Score) ──
df_s = df_bat[["formula_discharge", "S"]].sort_values("S")
fig, ax = plt.subplots(figsize=(8,4))

colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_s)))
ax.barh(df_s["formula_discharge"], df_s["S"], color=colors)
ax.set_xlabel("Overall Score")
ax.set_xlim(2.0, 2.2)
ax.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
# Guardar y mostrar
plt.savefig('barras_score.png', dpi=300, bbox_inches='tight')
plt.show()


# ── BARRAS HORIZONTALES (Energy Density) ──
df_s = df_bat[["formula_discharge", "energy_density"]].sort_values("energy_density")
fig, ax = plt.subplots(figsize=(8,4))

colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(df_s)))
ax.barh(df_s["formula_discharge"], df_s["energy_density"], color=colors)
ax.set_xlabel("Energy per mass of electrodes (Wh/kg)")
ax.set_xlim(200, 450)
ax.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
# Guardar y mostrar
plt.savefig('barras_energy_density.png', dpi=300, bbox_inches='tight')
plt.show()
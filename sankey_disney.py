import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

#SANKEY DIAGRAM CON EL INCOME STATEMENT DE DISNEY A 3Q22

#Definimos las variables

label = ['Revenue', 'Media & Entertainment', 'Parks', 'Linear Networks', 'DTC', 'Park & Experiences', 
        'Content Sales Licensing', 'Consumer Products']
source = [0, 1, 1, 1, 0, 2, 2]
target = [1, 3, 4, 6, 2, 5, 7]
value = [12.7, 6.3, 4.8, 1.6, 7.4, 6.1, 1.3]

#Data a dict, dict a sankey

link = dict(source = source, target = target, value = value)
node = dict(label = label, pad=70, thickness=25)
data = go.Sankey(link = link, node=node, valuesuffix = 'B', arrangement = "freeform")

#Dibujamos el sankey diagram

fig = go.Figure(data)
fig.update_layout(title = 'Disney Q4 FY2022 Income Statement (in Billions)', 
                font = dict(size=18, color = 'white'), paper_bgcolor = 'black',  plot_bgcolor='black', 
                hovermode = "x unified")
fig.show()

######################



# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import timedelta, datetime, date
from scipy.optimize import fsolve
import requests

### Funciones
def cargaObs(serie_id,timestart,timeend):
    response = requests.get(
        'https://alerta.ina.gob.ar/a6/obs/puntual/series/'+str(serie_id)+'/observaciones',
        params={'timestart':timestart,'timeend':timeend},
        headers={'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IlNhbnRpYWdvIEd1aXp6YXJkaSIsImlhdCI6MTUxNjIzOTAyMn0.YjqQYMCh4AIKSsSEq-QsTGz3Q4WOS5VE-CplGQdInfQ'},)
    json_response = response.json()
    df_obs_i = pd.DataFrame.from_dict(json_response,orient='columns')
    df_obs_i = df_obs_i[['timestart','valor']]
    df_obs_i = df_obs_i.rename(columns={'timestart':'fecha'})

    df_obs_i['fecha'] = pd.to_datetime(df_obs_i['fecha'])
    df_obs_i['valor'] = df_obs_i['valor'].astype(float)

    df_obs_i = df_obs_i.sort_values(by='fecha')
    df_obs_i.set_index(df_obs_i['fecha'], inplace=True)
    
    df_obs_i.index = df_obs_i.index.tz_convert(None)#("America/Argentina/Buenos_Aires")
    df_obs_i.index = df_obs_i.index - timedelta(hours=3)

    df_obs_i['fecha'] = df_obs_i.index
    df_obs_i = df_obs_i.reset_index(drop=True)
    return df_obs_i

def Plotea(dfplot,variables,labels,nombres):
    # Grafico de Niveles
    for i,var in enumerate(variables):
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(dfplot.index, dfplot[var],'-',label=nombres[i],linewidth=2)
        plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.5)
        plt.tick_params(axis='both', labelsize=16)
        plt.xlabel('Fecha', size=18)
        plt.ylabel(labels[i], size=18)
        plt.legend(prop={'size':16},loc=2,ncol=2 )
        plt.show()
        plt.close()

def descript_estadisticos(df,variable):
    '''
    Las estadísticas descriptivas incluyen aquellas que resumen 
    la tendencia central, la dispersión y la forma de la distribución 
    de un conjunto de datos, excluyendo los valores de NaN.
    '''
    print('Variable: ',variable)
    media = df[variable].mean()
    mediana = df[variable].median()
    moda = df[variable].mode()
    print("""
        Media: %d
        Mediana: %d
        Moda: %d
    """ % (media,mediana,moda))
    print(df.describe())

def CreaVariablesTemporales(df):
    df.insert(0, 'year', df.index.year)
    df.insert(1, 'month', df.index.month)
    df.insert(2, 'day', df.index.day)
    df.insert(3, 'yrDay', df.index.dayofyear)
    print(df.head(2))

def PloteaXano(df,var_x,var_y,var_hue):
    sns.lineplot(data=df, x=var_x, y=var_y, hue=var_hue)

    plt.grid(True,axis='y', which='both', color='0.75', linestyle='-.',linewidth=0.3)
    plt.tick_params(axis='y', labelsize=14)
    plt.tick_params(axis='x', labelsize=14)
    plt.xlabel('Día del Año', size=18)
    plt.ylabel('Caudal [m'+r'$^3$'+'/s]', size=18)
    plt.show()
    plt.close()

def PlotVarMaxMedMin_Anual(df,variable):
    df_anual = df.groupby(["year"]).agg({ variable: ["max","mean","min"]}).reset_index()
    df_anual.set_index(df_anual['year'], inplace=True)
    del df_anual['year']
    df_anual.columns = ['_'.join(col) for col in df_anual.columns.values]
    df_anual['Caudal_mean'] = df_anual['Caudal_mean'] - df_anual['Caudal_min']
    df_anual['Caudal_max'] = df_anual['Caudal_max'] - df_anual['Caudal_mean']

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)
    df_anual.plot(kind='bar', stacked=True, color=['skyblue', 'cornflowerblue','darkblue' ], ax=ax)

    plt.grid(True,axis='y', which='both', color='0.75', linestyle='-.',linewidth=0.3)
    plt.tick_params(axis='y', labelsize=14)
    plt.tick_params(axis='x', labelsize=9)
    plt.xlabel('Año', size=18)
    plt.ylabel('Caudal [m'+r'$^3$'+'/s]', size=18)
    plt.legend(['Caudal Mínimo','Caudal Medio','Caudal Máximo'],prop={'size':16},loc=0,ncol=1)
    plt.show()
    plt.close()

def PlotVarMaxMedMin_Mensual(df,variable):
    df_mensual = df.groupby(["month"]).agg({ variable: ["max","mean","min"]}).reset_index()
    df_mensual.set_index(df_mensual['month'], inplace=True)
    del df_mensual['month']
    df_mensual.columns = ['_'.join(col) for col in df_mensual.columns.values]
    df_mensual['Caudal_mean'] = df_mensual['Caudal_mean'] - df_mensual['Caudal_min']
    df_mensual['Caudal_max'] = df_mensual['Caudal_max'] - df_mensual['Caudal_mean']

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)
    df_mensual.plot(kind='bar', stacked=True, color=['skyblue', 'cornflowerblue','darkblue' ], ax=ax)

    plt.grid(True,axis='y', which='both', color='0.75', linestyle='-.',linewidth=0.3)
    plt.tick_params(axis='y', labelsize=14)
    plt.tick_params(axis='x', labelsize=14,rotation=0)
    plt.xlabel('Mes', size=18)
    plt.ylabel('Caudal [m'+r'$^3$'+'/s]', size=18)
    plt.legend(['Caudal Mínimo','Caudal Medio','Caudal Máximo'],prop={'size':16},loc=0,ncol=1)
    plt.show()
    plt.close()

def HistoVariable(df,variable,round_val = 100):
    print('Análisis de frecuencia')
    # round_val se usa para redondear el label de las barras
    from matplotlib.ticker import PercentFormatter

    num_of_bins = round(5*np.log10(32608)+1)

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)

    n, bins, patches = ax.hist(df0[variable], edgecolor='black', weights=np.ones_like(df[variable])*100 / len(df[variable]), bins=num_of_bins, rwidth=0.9,color='#607c8e')
    ax.yaxis.set_major_formatter(PercentFormatter())

    bins = [round(item/round_val)*round_val for item in bins]
    plt.xticks(bins)
    plt.tick_params(axis='y', labelsize=14)
    plt.tick_params(axis='x', labelsize=12,rotation=45)
    plt.ylabel('Frecuencia de aparición', size=18)
    plt.xlabel('Caudal [m'+r'$^3$'+'/s]', size=18)
    plt.grid(axis='y', alpha=0.75, linewidth=0.3)
    plt.show()

def CreaDFMaxAnual(df,variable):
    #Crea una nueva tabla con los maximos anuales
    df_maxAnual = df[['year',variable]].groupby(['year']).max()
    print(df_maxAnual.head(2))
    return df_maxAnual

def PlotMaxAnual(df,varaible):
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df.index, df[variable])
    ax.plot(df.index, df[variable],'-',linewidth=0.5)
    plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.5)
    plt.tick_params(axis='both', labelsize=16)
    plt.xlabel('Año', size=18)
    plt.ylabel('Caudal Máximo Anual [m'+r'$^3$'+'/s]', size=18)
    #plt.legend(prop={'size':16},loc=2,ncol=2 )
    plt.show()
    plt.close()

### Arranca:
actualiza_serie = False  #'API'

Plot = False

LP3 = True
EV1 = False

## Carga Datos
if actualiza_serie:    #Carga Datos desde la API:
    desde = datetime(1900,1,1)
    hasta = datetime(2023,1,10)
    f_inicio_srt = desde.strftime("%Y-%m-%d")
    f_fin_srt = hasta.strftime("%Y-%m-%d")
    df_Parana = cargaObs(29,f_inicio_srt,f_fin_srt)

    df_Parana.set_index(df_Parana['fecha'].dt.round('5min') , inplace=True)
    del df_Parana['fecha']
    print(df_Parana.head()) 

    df_Parana = df_Parana.rename(columns={'valor':'nivel'})
    df_Parana.to_pickle('Serie_Parana_last.pkl')

df0 = pd.read_pickle('Serie_Parana_last.pkl')

## Calcula Caudales con la HQ generada por PHC
def curvaHQ(h):
    return 2.7392 * pow(h,5) - 14.527 * pow(h,4) - 6.2064 * pow(h,3) + 390.16 * pow(h, 2) + 2108.3 * h + 8794.2
df0['Caudal'] = df0.apply(lambda row : curvaHQ(row['nivel']), axis = 1).round(2)

# Crea columna con el año, el mes, el dia y el dia del año
CreaVariablesTemporales(df0)

# Selecciona la variable con la que arma los graficos
variable = 'Caudal'

# Calcula valores estadisticos de las disintas variables en el DF
descript_estadisticos(df0,variable)

if Plot:
    # Análisis de la Serie Temporal
        
    # Plot de la serie completa
    Plotea(df0,['nivel','Caudal'],['Nivel [m]','Caudal [m'+r'$^3$'+'/s]'],['Paraná','Paraná'])

    # Plot de la serie por año
    PloteaXano(df0,var_x="yrDay",var_y=variable,var_hue='year')

    PlotVarMaxMedMin_Anual(df0,variable)

    PlotVarMaxMedMin_Mensual(df0,variable)

    HistoVariable(df0,variable,round_val = 100)


###########     Análisis de Extremos    ###########
''' La magnitud de un evento extremo está inversamente relacionada con su frecuencia de ocurrencia, es decir,
    los eventos más severos ocurren con menor frecuencia que los eventos moderados.
    
    Para poder realizar una previsión de ocurrencia de estos eventos nos basamos en el análisis de frecuencia 
    de información hidrológica. Esto es relacionar la magnitud de los eventos extremos con su frecuencia de 
    ocurrencia mediante el uso de distribuciones de probabilidad.

    Para que este análisis sea válido utilizamos las siguientes hipótesis:
        • La información hidrológica analizada es independiente
        • Está idénticamente distribuida
        • El sistema hidrológico que la produce se considera estocástico, independiente del espacio y del tiempo
    
    La información hidrológica empleada debe seleccionarse cuidadosamente de tal manera que se satisfagan 
    las suposiciones de independencia y de distribución idéntica.

    En este trabajo, esto se lleva a cabo seleccionando el máximo anual de la variable que está siendo analizada, 
    con la expectativa de que las observaciones sucesivas de esta variable de un año a otro sean independientes.
    En este capítulo se busca hallar la función de distribución de probabilidades que mejor ajuste la serie.
'''
# Datos y Metodología
''' Una serie de valor extremo incluye el valor máximo o mínimo que ocurre en cada uno de los intervalos de tiempo 
    de igual longitud del registro. La longitud del intervalo de tiempo usualmente y en este ejercicio se toma como 
    un año, y una serie seleccionada de esta manera se conoce como una serie anual. Si se utilizan los valores 
    máximos anuales es una serie anual máxima.
''' 

df_filt = df0[df0.index < datetime(2023,1,1)].copy()    # Filtra los del ultimo año porque no esta completo
df_maxAnual = CreaDFMaxAnual(df_filt,variable)
print(df_maxAnual.tail(2))

if Plot:
    PlotMaxAnual(df_maxAnual,'Caudal')

''' Para el análisis de eventos extremos utilizamos dos funciones de distribución de probabilidad. 
    Aplicamos distintos métodos de bondad de ajuste a estas distribuciones de probabilidad. 

    Las funciones utilizadas son:
        • EV1, Gumbel
        • LP-III, Log-Pearson 
''' 

def Fuc_EV1(df,variable):
    print('----------------------------------------')
    print('Función de distribución Gumbel - EV1 /n')

    ''' F(x) = exp[-exp(- (y - u)/ alfa)]
        
        Es una distribución biparámetrica cuyos parámetros son u, que es la moda de
        la distribución (punto de máxima densidad de probabilidad) y alfa. Los mismos los
        estimamos mediante dos métodos: 
    '''
    # valor medio y desvio estandar
    v_medio = df_maxAnual_EV1['Caudal'].mean()
    desvio_std = df_maxAnual_EV1['Caudal'].std()
    print ('Media: ',round(v_medio,2),'	Desvio std: ',round(desvio_std,2))

    print()
    print('Parametros:')
    print('Metodo de los momentos')
    alfa_mo = np.sqrt(6)*desvio_std/np.pi   # alfa
    moda_mo = v_medio - 0.5772*alfa_mo      # moda
    print ('Alfa: ',alfa_mo)
    print('Moda: ', moda_mo)
    print()

    print('Metodo Máxima Verosimilitud') # utiliza un solver para encontrar alfa
    def buscaObjet(alfa_mv, *data):                 						#funcion
        lst_variable, v_medio = data										#datos de entrada
        aux_amv1= (lst_variable*np.exp(-lst_variable/alfa_mv)).sum()
        aux_amv2= (np.exp(-lst_variable/alfa_mv)).sum()
        return alfa_mv - (v_medio - (aux_amv1/aux_amv2))					#iguala a cero
    
    var_list = df_maxAnual_EV1['Caudal'].to_numpy()
    data = (var_list, v_medio)								                #datos de entrada
    alfa_mv0 = 2000															#valor de arranque
    x_opt = fsolve(buscaObjet, alfa_mv0, args=data)							#llama al solver
    alfa_mv = x_opt[0]
    
    #moda
    aux_mmv= (np.exp(-df_maxAnual_EV1['Caudal']/alfa_mv)).sum()
    N = len(df_maxAnual_EV1)															#Longitud de la muestra
    moda_mv = -alfa_mv*np.log(aux_mmv/N)
    print ('Alfa: ',alfa_mv)
    print('Moda: ', moda_mv)
    print()

    ### 
    df_maxAnual_EV1_sort = df_maxAnual_EV1.sort_values(by=['Caudal'],ascending=True)
    df_maxAnual_EV1_sort['yr'] = df_maxAnual_EV1_sort.index
    df_maxAnual_EV1_sort = df_maxAnual_EV1_sort.reset_index(drop=True)

    print(df_maxAnual_EV1_sort.head(2))

    df_maxAnual_EV1_sort['i'] = df_maxAnual_EV1_sort.index + 1												# Posicion i
    df_maxAnual_EV1_sort['Fi'] = (df_maxAnual_EV1_sort['i']-0.44)/(N+0.12)									# Fi=(i-0,44)/(N+0,12)
    df_maxAnual_EV1_sort['y'] = -np.log(-np.log(df_maxAnual_EV1_sort['Fi']))								# y=-ln(-ln(F))
    df_maxAnual_EV1_sort['tri'] = 1/(1-df_maxAnual_EV1_sort['Fi'])
    #df_maxAnual_EV1_sort['tri'] = (N + 1)/(N+1-df_maxAnual_EV1_sort['i'])

    print(df_maxAnual_EV1_sort)


    #Método MO
    df_maxAnual_EV1_sort['Var_teo_mo'] = moda_mo + alfa_mo * df_maxAnual_EV1_sort['y']

    #Método de MV
    df_maxAnual_EV1_sort['Var_teo_mv'] = moda_mv + alfa_mv * df_maxAnual_EV1_sort['y']										#xt = u +α.yt
    
    #Variable asociadas a distintos períodos de retorno
    df_Tr_EV1 = pd.DataFrame(columns=('Tr',))
    df_Tr_EV1['Tr'] = [2,5,10,20,30,40,50,100,200,500]
    df_Tr_EV1['Tr_c'] = 1/(1-np.exp((-1/df_Tr_EV1['Tr'])))
    df_Tr_EV1['y'] = -np.log(np.log(df_Tr_EV1['Tr_c']/(df_Tr_EV1['Tr_c']-1)))	    #ACA CREO QUE HAY QUE LLAMAR A Tr_c y no Tr
    df_Tr_EV1['Var_ev1_mo'] = moda_mo + alfa_mo * df_Tr_EV1['y']				#xt = u +α.yt 	Pt = f(T)
    df_Tr_EV1['Var_ev1_mv'] = moda_mv + alfa_mv * df_Tr_EV1['y']				#xt = u +α.yt 	Pt = f(T)
    #df_Tr.to_excel('EV1_CaudalesTr.xlsx',index=False)

    # Grafico
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(df_Tr_EV1['Tr_c'], df_Tr_EV1['Var_ev1_mo'], label = 'MO', linewidth=2) # Tr_c
    ax.plot(df_Tr_EV1['Tr_c'], df_Tr_EV1['Var_ev1_mv'], label = 'MV', linewidth=2)

    ax.scatter(x=df_maxAnual_EV1_sort['tri'],y=df_maxAnual_EV1_sort['Caudal'],label='Obs',s=4, c='red')    # tri
    

    plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.5)
    plt.tick_params(axis='both', labelsize=16)
    plt.xlabel('Tr [Años]', size=18)
    plt.ylabel('Caudal [m'+r'$^3$'+'/s]', size=18)
    plt.legend(prop={'size':16},loc=2,ncol=1 )
    plt.show()
    plt.close()

    print()
    print('Test de Bondad de Ajuste de Kolmogorov-Smirnov / EVI')
    df_test = df_maxAnual_EV1_sort[['i','yr','Caudal']].copy()
    df_test.loc[:,'Fi'] = df_test['i']/df_test['i'].max()

    #Método MO
    df_test['F_pi_mo'] = np.exp(-np.exp(-(df_test['Caudal']-moda_mo)/alfa_mo))
    df_test['Dif_F_mo'] = np.absolute(df_test['Fi']-df_test['F_pi_mo'])
    df_test['Dif2_F_mo'] = np.square(df_test['Dif_F_mo'])

    #Método de MV
    df_test['F_pi_mv'] = np.exp(-np.exp(-(df_test['Caudal']-moda_mv)/alfa_mv))
    df_test['Dif_F_mv'] = np.absolute(df_test['Fi']-df_test['F_pi_mv'])
    df_test['Dif2_F_mv'] = np.square(df_test['Dif_F_mv'])

    #Criterios de aprobación:
    max_dif_mo = df_test['Dif_F_mo'].max()
    max_dif_mv = df_test['Dif_F_mv'].max()
    D_n5 = 1.36/np.sqrt(N)
    if max_dif_mo<D_n5:
        print ('Aprueba por mo. ',round(max_dif_mo,3),' < ',round(D_n5,3))
    else:
        print ('NO aprueba por mo. ',round(max_dif_mo,3),' > ',round(D_n5,3))
    if max_dif_mv<D_n5:
        print ('Aprueba por mv. ',round(max_dif_mv,3),' < ',round(D_n5,3))
    else:
        print ('NO aprueba por mv. ',round(max_dif_mv,3),' > ',round(D_n5,3))

    return df_maxAnual_EV1_sort, df_Tr_EV1

def Fuc_LP3(df,variable):
    print('----------------------------------------')
    print('Función de distribución Log-Pearson III  /n')
    df_maxAnual_LP3_sort = df.sort_values(by=[variable],ascending=True)
    df_maxAnual_LP3_sort['yr'] = df_maxAnual_LP3_sort.index
    df_maxAnual_LP3_sort = df_maxAnual_LP3_sort.reset_index(drop=True)
    print(df_maxAnual_LP3_sort.head(2))

    # parámetros
    print('Estimación de parámetros Log-Pearson III mediante kt')
    df_maxAnual_LP3_sort.insert(0, 'i', df_maxAnual_LP3_sort.index + 1)
    df_maxAnual_LP3_sort['ln_var'] = np.log(df_maxAnual_LP3_sort[variable])		# y = ln(x)

    v_med = df_maxAnual_LP3_sort['ln_var'].mean()			    #Media Geometrica de y
    desv_std = df_maxAnual_LP3_sort['ln_var'].std()				#Desvío Estándar de y
    Cs =  df_maxAnual_LP3_sort['ln_var'].skew()					#Coeficiente de Asimetría de y
    print ('Media: ',round(v_med,2),'	Desvio std: ',round(desv_std,2),'	Coef de Asim: ',round(Cs,5))

    N = len(df_maxAnual_LP3_sort)								#Longitud de la serie
    df_maxAnual_LP3_sort['Fi'] = df_maxAnual_LP3_sort['i']/(N + 1)  # posiciones de graficación de Weibul

    # Caudales asociadas a distintos períodos de retorno
    ind = [2,5,10,20,30,40,50,100,200,500]
    df_Tr_LP3 = pd.DataFrame(index = ind)
    df_Tr_LP3.index.rename('Tr', inplace=True)
    df_Tr_LP3['Tr_c'] = 1/(1-np.exp((-1/df_Tr_LP3.index)))
    df_Tr_LP3['p'] = 1/df_Tr_LP3['Tr_c']

    df_Tr_LP3['w'] = np.nan
    df_Tr_LP3['z'] = np.nan
    for index, row in df_Tr_LP3.T.iteritems():
        if row['p'] <= 0.5:
            w = np.sqrt(np.log(1/np.square(row['p'])))
            z = w-(2.515517+0.802853*w+0.010328*np.square(w))/(1+1.432788*w+0.189269*np.square(w)+0.001308*np.power(w,3))
            df_Tr_LP3.loc[index,'w'] = w
            df_Tr_LP3.loc[index,'z'] = z
        else:
            w = np.sqrt(np.log(1/np.square(1-row['p'])))
            z = -(w-(2.515517+0.802853*w+0.010328*np.square(w))/(1+1.432788*w+0.189269*np.square(w)+0.001308*np.power(w,3)))
            df_Tr_LP3.loc[index,'w'] = w
            df_Tr_LP3.loc[index,'z'] = z

    df_Tr_LP3['Kt'] = df_Tr_LP3['z']+(np.power(df_Tr_LP3['z'],2)-1)*np.divide(Cs,6.0)+np.divide(1,3.0)*(np.power(df_Tr_LP3['z'],3.0)-6*df_Tr_LP3['z'])*np.power(np.divide(Cs,6.0),2)-(np.power(df_Tr_LP3['z'],2)-1)*np.power(np.divide(Cs,6.0),3)+df_Tr_LP3['z']*np.power(np.divide(Cs,6.0),4)+np.divide(1,3)*np.power(np.divide(Cs,6.0),5)
    df_Tr_LP3['ln_pt'] = v_med+desv_std*df_Tr_LP3['Kt']
    df_Tr_LP3['Q_tr'] = np.exp(df_Tr_LP3['ln_pt'])

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(df_Tr_LP3['Tr_c'], df_Tr_LP3['Q_tr'], label = 'LP3', linewidth=2)

    plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.5)
    plt.tick_params(axis='both', labelsize=16)
    plt.xlabel('Tr [Años]', size=18)
    plt.ylabel('Caudal [m'+r'$^3$'+'/s]', size=18)
    plt.legend(prop={'size':16},loc=2,ncol=1 )
    plt.show()
    plt.close()

    print()
    print('Test de Bondad de Ajuste de Kolmogorov-Smirnov')
    df_maxAnual_LP3_sort['Kt_obs'] = (df_maxAnual_LP3_sort['ln_var']-v_med)/desv_std

    from scipy.optimize import fsolve
    def buscaObjet(p, *data):											#funcion
        v_med, desv_std, Cs, Kt_obs = data								#datos de entrada
        if p <= 0.5:
            w = np.sqrt(np.log(1/np.square(p)))
            z = w-(2.515517+0.802853*w+0.010328*np.square(w))/(1+1.432788*w+0.189269*np.square(w)+0.001308*np.power(w,3))
        else:
            w = np.sqrt(np.log(1/np.square(1-p)))
            z = -(w-(2.515517+0.802853*w+0.010328*np.square(w))/(1+1.432788*w+0.189269*np.square(w)+0.001308*np.power(w,3)))
            
        Kt = z+(np.power(z,2)-1)*np.divide(Cs,6.0)+np.divide(1,3.0)*(np.power(z,3.0)-6*z)*np.power(np.divide(Cs,6.0),2)-(np.power(z,2)-1)*np.power(np.divide(Cs,6.0),3)+z*np.power(np.divide(Cs,6.0),4)+np.divide(1,3)*np.power(np.divide(Cs,6.0),5)
        return Kt - Kt_obs												#iguala a cero
    
    for index, row in df_maxAnual_LP3_sort.T.iteritems():
        data = (v_med, desv_std, Cs, row['Kt_obs'])							#datos de entrada
        p0 = row['Fi']														#valor de arranque
        p_opt = fsolve(buscaObjet, p0, args=data)							#llama al solver
        prob = p_opt[0]
        df_maxAnual_LP3_sort.loc[index,'p'] = prob
    
    df_maxAnual_LP3_sort['Ti'] = 1/(df_maxAnual_LP3_sort['p'])
    #df_maxAnual_LP3_sort.to_excel('Q_Tr_LP3.xlsx',index=True)    
    return df_Tr_LP3
   

if EV1:
    df_maxAnual_EV1 = df_maxAnual.copy()
    df_maxAnual_EV1_sort, df_Tr_EV1 = Fuc_EV1(df_maxAnual_EV1,'Caudal')

if LP3:
    df_maxAnual_LP3 = df_maxAnual.copy()
    df_Tr_LP3 = Fuc_LP3(df_maxAnual_LP3,'Caudal')
    


quit()
    
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(1, 1, 1)

ax.plot(df_Tr_EV1['Tr_c'], df_Tr_EV1['Var_ev1_mo'], label = 'MO', linewidth=2) # Tr_c
ax.plot(df_Tr_EV1['Tr_c'], df_Tr_EV1['Var_ev1_mv'], label = 'MV', linewidth=2)
ax.plot(df_Tr_LP3['Tr_c'], df_Tr_LP3['Q_tr'], label = 'LP3', linewidth=2)
ax.scatter(x=df_maxAnual_EV1_sort['tri'],y=df_maxAnual_EV1_sort['Caudal'],label='Obs',s=4, c='red')    # tri


plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.5)
plt.tick_params(axis='both', labelsize=16)
plt.xlabel('Tr [Años]', size=18)
plt.ylabel('Caudal [m'+r'$^3$'+'/s]', size=18)
plt.legend(prop={'size':16},loc=2,ncol=1 )
plt.show()
plt.close()
    




quit()

#	Medidas de posición: cuartiles y percentiles
'''
El concepto es igual al de mediana, salvo que aquí la división ya no es en el 50%. 
El 25% de las observaciones es menor que el primer cuartil. 
Los cuartiles abarcan el 25%, 50% y 75% de las observaciones. 
Los percentiles son una generalización con cualquier porcentaje.
Ejemplo: ¿Cuál es la nota que tiene como mínimo el 10% más nota de la clase?
Este enunciado nos pide calcular el percentil 90. 
Mencionar que existen distintos tipos de interpolación para este cálculo. 
En la referencia podemos consultar cual nos conviene más'''
p90 = df["nota"].quantile(0.9)

#	Medidas de dispersión: desviación típica, rango, IQR, coeficiente de variación
'''La desviación típica mide la dispersión de los datos respecto a la media. 
Se trata de la raíz cuadrada de la varianza, que en sí misma no es una medida de dispersión. 
Para calcular la desviación típica usamos std y var para la varianza. 
(ddof=0 es necesario si quieres seguir la definición de desviación típica y varianza de algunas bibliografías, 
la razón es que hay un parámetro de ajuste que Pandas pone a 1, pero otras librerías ponen a 0). 
En Excel es la diferencia que hay entre DESVEST.M (ddof=1) y DESVEST.P (ddof=0).'''

std = df["nota"].std(ddof=0)
var = df["nota"].var(ddof=0)
assert(np.sqrt(var) == std)

#	El rango es la diferencia entre el máximo y el mínimo y el rango intercuartílico o IQR es la diferencia entre el tercer y el primer cuartil.
rango = df["nota"].max() - df["nota"].min()
iqr = df["nota"].quantile(0.75) - df["nota"].quantile(0.25)

'''El coeficiente de variación es una medida que sirve para comparar entre dos muestras, cuál varía más y cuál es más estable. 
Es una simple división, de la desviación típica sobre la media, sin embargo, SciPy nos ofrece una función ya preparada.'''

import scipy.stats as ss

cv = df["nota"].std(ddof=0) / df["nota"].mean()
cv2 = ss.variation(df["nota"])
assert(cv == cv2)

#	Medidas de asimetría
'''Para saber si los datos estan repartidos de forma simétrica existen varios coeficientes: Pearson, Fisher, Bowley-Yule, etc
Para no liarnos demasiado, podemos usar la función skew de SciPy.
Para valores cercanos a 0, la variable es simétrica. Si es positiva tiene cola a la derecha y si es negativa tiene cola a la izquierda.'''
asimetria = ss.skew(df["nota"])
























'''
df1[n_est0] = df1[n_est0].replace(np.nan,-5)
Plot_1_var(df1,n_est0)
df1[n_est0] = df1[n_est0].replace(-5,np.nan)

'''
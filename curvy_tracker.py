

import pdblp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize  # optimization function
from pandas.tseries.offsets import BDay
from pandas.tseries.offsets import BMonthEnd
from tqdm import tqdm  # this will just make thing a bit prettier



class CurvyTrade(object):
    """
    This class calculates trackers for FX carry trade and curvy trade. 
    Enters in a 1 month fwd position and rebalances monthly. MtM is daily.
    """

    # Available currencies
    # TODO add 'BRL', 'CLP', 'COP', 'CZK', 'HUF', 'KRW', 'MXN', 'PHP', 'PLN', 'SGD', 'TRY', 'TWD', 'ZAR'
    currency_list = ['AUD', 'CAD', 'CHF', 'DEM', 'GBP', 'JPY', 'NOK', 'NZD', 'SEK']

    # BBG connection
    con = pdblp.BCon(debug=True, port=8194, timeout=5000)
    con.debug = False
    con.start()

    # tickers, start date, end date
    ticker_spot_bbg = [c + ' Curncy' for c in currency_list]
    bbg_field_last = ['PX_LAST']
    bbg_field_bid = ['PX_BID']
    bbg_field_ask = ['PX_ASK']
    ini_date_bbg = '19890101'  # extended period for vol calculation
    # data_ini = '19910101'
    end_date_bbg = '20190702'
    # data_fim = '20190527'

    # TODO delete these dictionaries??
    # FX_ticker = dict(zip(currency_list, ticker_spot_bbg))
    # ticker_FX = dict(zip(ticker_spot_bbg, currency_list))

    # TODO outliers
    # ### Spots e fwd points
    #
    # We consider all G10 currencies vis-a-vis the US dollar, including the Australian dollar, the British pound, the Canadian dollar, the euro (Deutsche mark before 1999), the Japanese yen, the New Zealand dollar, the Norwegian krone, the Swiss franc and the Swedish krona.
    #
    # We use monthly end-of-period data, with the sample spanning the period from January 1991 to December 2015.

    # TODO allow for today as end_date

    fwd_dict = {'1W BGN Curncy': 7.0,
                '2W BGN Curncy': 14.0,
                '3W BGN Curncy': 21.0,
                '1M BGN Curncy': 31.0,
                '2M BGN Curncy': 61.0,
                '3M BGN Curncy': 91.0,
                '4M BGN Curncy': 121.0,
                '5M BGN Curncy': 151.0,
                '6M BGN Curncy': 181.0}




    def __init__(self, start_date='19890101', end_date='20190702'):
        # BBG connection
        self.con = pdblp.BCon(debug=True, port=8194, timeout=5000)
        self.con.debug = False
        self.con.start()
        # calendars
        self.daily_calendar = self._build_calendars()
        # tickers
        self.df_tickers = self._build_df_tickers()
        self.ticker_fwdpts_1m = list(self.df_tickers['fwdpts'])
        self.ticker_scale = self.df_tickers['scale']
        self.ticker_inverse = self.df_tickers['inverse']
        # TODO get bid and ask data
        self.spot_last = self._get_spot_data(self.bbg_field_last)
        self.fwdpts_last = self._get_fwdpts_data(self.bbg_field_last)
        self.fwd_last = self.spot_last + self.fwdpts_last / self.ticker_scale
        self.spot_last_USDXXX = self.spot_last ** self.ticker_inverse
        self.fwd_last_USDXXX = self.fwd_last ** self.ticker_inverse
        self.fwd_discount_last = (self.fwd_last_USDXXX / self.spot_last_USDXXX - 1) * 12

    @staticmethod
    def _build_calendars(start_date='1991-01-01', end_date='2019-07-02'):
        # TODO organize dates (these dates are hardcoded for now). Eg: start=pd.to_datetime(self.initial_date)
        daily_calendar = pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), freq=BDay())
        return daily_calendar

    def _build_df_tickers(self):
        cod_fwdpts = [c + '1M BGN Curncy' for c in self.currency_list]
        df_tickers = pd.DataFrame(index=self.currency_list, data=self.ticker_spot_bbg, columns=['spot'])
        df_tickers['fwdpts'] = cod_fwdpts
        # only for DEM, replace BGN for CMPN
        df_tickers.loc['DEM', 'fwdpts'] = df_tickers.loc['DEM', 'fwdpts'].replace('BGN', 'CMPN')
        # scale column
        df_scale = self.con.ref(list(self.ticker_spot_bbg), 'FWD_SCALE').set_index(keys='ticker', drop=True)
        df_scale.index = [x.replace(' Curncy', '') for x in df_scale.index]
        df_tickers['scale'] = 10 ** df_scale['value']
        # inverse column
        df_inverse = self.con.ref(self.ticker_spot_bbg, 'INVERSE_QUOTED').set_index(keys='ticker', drop=True)
        df_inverse.index = [x.replace(' Curncy', '') for x in df_inverse.index]
        df_inverse.loc[df_inverse['value'] == "Y", 'value'] = -1
        df_inverse.loc[df_inverse['value'] == "N", 'value'] = 1
        df_tickers['inverse'] = df_inverse['value']
        # Forward curve tickers
        # TODO for DEM, replace BGN for CMPN
        df_tickers['1w'] = [c + '1W BGN Curncy' for c in self.currency_list]
        df_tickers['2w'] = [c + '2W BGN Curncy' for c in self.currency_list]
        df_tickers['3w'] = [c + '3W BGN Curncy' for c in self.currency_list]
        df_tickers['1m'] = [c + '1M BGN Curncy' for c in self.currency_list]
        df_tickers['2m'] = [c + '2M BGN Curncy' for c in self.currency_list]
        df_tickers['3m'] = [c + '3M BGN Curncy' for c in self.currency_list]
        df_tickers['4m'] = [c + '4M BGN Curncy' for c in self.currency_list]
        df_tickers['5m'] = [c + '5M BGN Curncy' for c in self.currency_list]
        df_tickers['6m'] = [c + '6M BGN Curncy' for c in self.currency_list]
        return df_tickers

    def _get_spot_data(self, bbg_field=['PX_LAST']):
        spot_last = self.con.bdh(self.ticker_spot_bbg, bbg_field, self.ini_date_bbg, self.end_date_bbg, elms=[("periodicitySelection", 'DAILY')])
        spot_last.columns = spot_last.columns.droplevel(1)
        spot_last.fillna(method='ffill', inplace=True)
        spot_last.columns = [x.replace(' Curncy', '') for x in spot_last.columns]
        spot_last.sort_index(axis=1, ascending=True, inplace=True)
        return spot_last

    def _get_fwdpts_data(self, bbg_field=['PX_LAST']):
        fwd_pts_last = self.con.bdh(self.ticker_fwdpts_1m, bbg_field, self.ini_date_bbg, self.end_date_bbg, elms=[("periodicitySelection", 'DAILY')])
        fwd_pts_last.columns = fwd_pts_last.columns.droplevel(1)
        fwd_pts_last.fillna(method='ffill', inplace=True)
        fwd_pts_last.columns = [x.replace('1M BGN Curncy', '') for x in fwd_pts_last.columns]
        fwd_pts_last.columns = [x.replace('1M CMPN Curncy', '') for x in fwd_pts_last.columns]
        fwd_pts_last.sort_index(axis=1, ascending=True, inplace=True)
        return fwd_pts_last

    def plot_fwd_discount(self, figure_size):
        self.fwd_discount_last.multiply(100).plot(figsize=figure_size, title='Forward premium (annual rate)')
        return plt.show()

    # Weighting function
    # Equal weight
    # wgt = 1/k
    @staticmethod
    def _ranking_to_wgt(df_ranking, k=1):
        assert df_ranking.isnull().sum().sum() == 0, 'There is NaN in ranking DataFrame!! Check this!!'
        unit_wgt = 1.0 / k
        up_limit = len(df_ranking.columns) - k
        down_limit = k
        df_signals = (df_ranking > up_limit).multiply(1) - (df_ranking <= down_limit).multiply(1)
        weights = df_signals * unit_wgt
        return weights, df_signals

    # Equal vol
    # wgt_vol_adjusted = wgt/vol
    @staticmethod
    def _ranking_to_equalvolwgt(df_ranking, df_vols, target_vol=0.06, k=1):
        assert df_ranking.isnull().sum().sum() == 0, 'There is NaN in ranking DataFrame!! Check this!!'
        assert df_vols.isnull().sum().sum() == 0, 'There is NaN in vols DataFrame!! Check this!!'
        unit_wgt = 1.0 / k
        up_limit = len(df_ranking.columns) - k
        down_limit = k
        df_signals = (df_ranking > up_limit).multiply(1) - (df_ranking <= down_limit).multiply(1)
        weights = df_signals * unit_wgt * target_vol / df_vols
        return weights, df_signals


    # #### Forward Discount ranking

    # In[21]:


    # Inicializando DataFrame
    calendar = fwd_discount_last.index
    lst_currencies = df_tickers.index
    ranks_discount = pd.DataFrame(index=calendar, columns=lst_currencies)

    # In[22]:


    for d in fwd_discount_last.index:
        available_FX = list(fwd_discount_last.loc[d].dropna(how='any').index)
        ranks_discount.loc[d, available_FX] = fwd_discount_last.loc[d, available_FX].rank(ascending=True)

    # In[23]:


    ranks_discount.tail()

    # #### Vol ranking

    # In[24]:


    # Getting spot data before 1991 (extended period)
    vol_data_ini = '19890101'
    spot_last_extended = con.bdh(ticker_spot_bbg, bbg_field_last, vol_data_ini, end_date_bbg, elms=[("periodicitySelection", "DAILY")])
    spot_last_extended.columns = spot_last_extended.columns.droplevel(1)
    spot_last_extended.fillna(method='ffill', inplace=True)
    spot_last_extended.columns = [x.replace(' Curncy', '') for x in spot_last_extended.columns]
    spot_last_extended.sort_index(axis=1, ascending=True, inplace=True)
    spot_last_extended.head()

    # In[25]:


    # Converting in USDXXX (FX/USD) format
    spot_last_USDXXX_extended = spot_last_extended ** df_tickers['inverse']
    spot_last_USDXXX_extended.head()

    # #### EWMA Vol (Lambda = 0.97)

    # In[26]:


    ### EWM (Pandas)
    vol_calendar = spot_last_USDXXX_extended.index
    df_vols = pd.DataFrame(index=vol_calendar, columns=lst_currencies)
    ewma_lambda = 0.94
    df_vols = spot_last_USDXXX_extended.pct_change(1).ewm(com=None, span=None, halflife=None, alpha=(1.0 - ewma_lambda),
                                                          min_periods=0, adjust=True).var() * 252
    df_vols = df_vols ** 0.5
    df_vols.plot(figsize=(12, 8))

    # In[27]:


    ### EWMA
    # vol_calendar = spot_last_USDXXX_extended.index
    # df_vols = pd.DataFrame(index=vol_calendar, columns=lst_currencies)
    # ewma_lambda = 0.94
    # df_vols.loc[vol_calendar[0]] = spot_last_USDXXX_extended.pct_change(1).var()
    # for d, dm1 in zip(vol_calendar[1:], vol_calendar[:-1]):
    #     df_vols.loc[d] = (np.log((spot_last_USDXXX_extended.loc[d] / spot_last_USDXXX_extended.loc[dm1]).astype(float))**2)* (1.0 - ewma_lambda) + ewma_lambda * df_vols.loc[dm1]
    # df_vols = (df_vols * 252) ** 0.5
    # df_vols.plot(figsize=(12,8))


    # In[28]:


    # Monthly DataFrame
    df_vols_monthly = pd.DataFrame(index=calendar, columns=lst_currencies)
    for d in calendar:
        df_vols_monthly.loc[d] = df_vols.loc[d]
    df_vols_monthly.plot(figsize=(12, 8), title='Annualized Vol (EWMA)')


    # # P&L of traditional carry trade

    # In[29]:


    def pnl_fwd_dsc(df_forwards, df_spots, df_ranking, k):
        # d=day, tm1=t menos 1
        TR_index = pd.DataFrame(index=df_ranking.index, columns=['TR_index'])
        holdings = pd.DataFrame(index=df_ranking.index, columns=df_ranking.columns)
        d_ini = TR_index.index[0]
        TR_index.loc[d_ini] = 100.0
        [weights, signals] = ranking_to_wgt(df_ranking, k)
        # Holdings have inverted signal, as FX is USDXXX and negative signal is long XXX/Short USD.
        holdings.loc[d_ini] = ((100.0 * weights.loc[d_ini]) / df_forwards.loc[d_ini]).multiply(
            -1)  # df_forwards should be in USDXXX
        # tm1: t minus 1
        for d, tm1 in zip(df_ranking.index[1:], df_ranking.index[:-1]):
            pnl = (holdings.loc[tm1] * (df_spots.loc[d] - df_forwards.loc[tm1])).sum()
            TR_index.loc[d] = TR_index.loc[tm1] + pnl
            holdings.loc[d] = (TR_index.loc[d].values * weights.loc[d] / df_forwards.loc[d]).multiply(-1)
        return TR_index, holdings, weights, signals


    # ex: pnl_fwd_dsc(fwd_last_USDXXX, spot_last_USDXXX, ranks_discount, k=1)


    # In[30]:


    [TR_index1, holdings1, weights1, signals1] = pnl_fwd_dsc(fwd_last_USDXXX, spot_last_USDXXX, ranks_discount, k=1)
    [TR_index2, holdings2, weights2, signals2] = pnl_fwd_dsc(fwd_last_USDXXX, spot_last_USDXXX, ranks_discount, k=2)
    [TR_index3, holdings3, weights3, signals3] = pnl_fwd_dsc(fwd_last_USDXXX, spot_last_USDXXX, ranks_discount, k=3)
    [TR_index4, holdings4, weights4, signals4] = pnl_fwd_dsc(fwd_last_USDXXX, spot_last_USDXXX, ranks_discount, k=4)

    # In[31]:


    TR_dsc = pd.DataFrame(index=TR_index1.index, columns=['k1', 'k2', 'k3', 'k4'])
    TR_dsc['k1'] = TR_index1
    TR_dsc['k2'] = TR_index2
    TR_dsc['k3'] = TR_index3
    TR_dsc['k4'] = TR_index4
    TR_dsc.plot(figsize=(15, 10), title='Carry Trade', logy=True)


    # # P&L of equal vol carry trade

    # In[32]:


    def pnl_equal_vol(df_forwards, df_spots, df_ranking, df_vol, target_vol, k):
        # d=day, tm1=t menos 1
        TR_index = pd.DataFrame(index=df_ranking.index, columns=['TR_index'])
        holdings = pd.DataFrame(index=df_ranking.index, columns=df_ranking.columns)
        d_ini = TR_index.index[0]
        TR_index.loc[d_ini] = 100.0
        [weights, signals] = ranking_to_equalvolwgt(df_ranking, df_vol, target_vol, k)
        # Holdings have inverted signal, as FX is USDXXX and negative signal is long XXX/Short USD.
        holdings.loc[d_ini] = ((100.0 * weights.loc[d_ini]) / df_forwards.loc[d_ini]).multiply(
            -1)  # df_forwards should be in USDXXX

        for d, tm1 in zip(df_ranking.index[1:], df_ranking.index[:-1]):
            pnl = (holdings.loc[tm1] * (df_spots.loc[d] - df_forwards.loc[tm1])).sum()
            TR_index.loc[d] = TR_index.loc[tm1] + pnl
            holdings.loc[d] = (TR_index.loc[d].values * weights.loc[d] / df_forwards.loc[d]).multiply(-1)
        return TR_index, holdings, weights, signals


    # ex: pnl_fwd_dsc(fwd_last_USDXXX, spot_last_USDXXX, ranks_discount, k=1)


    # In[33]:


    tgt_vol = 0.09  # vol que deixa o retorno acumulado parecido com o carry trade normal (peso igual)
    [TR_dsc_ev1, holdings_dsc_ev1, weights_dsc_ev1, signals_dsc_ev1] = pnl_equal_vol(fwd_last_USDXXX, spot_last_USDXXX,
                                                                                     ranks_discount, df_vols_monthly,
                                                                                     target_vol=tgt_vol, k=1)
    [TR_dsc_ev2, holdings_dsc_ev2, weights_dsc_ev2, signals_dsc_ev2] = pnl_equal_vol(fwd_last_USDXXX, spot_last_USDXXX,
                                                                                     ranks_discount, df_vols_monthly,
                                                                                     target_vol=tgt_vol, k=2)
    [TR_dsc_ev3, holdings_dsc_ev3, weights_dsc_ev3, signals_dsc_ev3] = pnl_equal_vol(fwd_last_USDXXX, spot_last_USDXXX,
                                                                                     ranks_discount, df_vols_monthly,
                                                                                     target_vol=tgt_vol, k=3)
    [TR_dsc_ev4, holdings_dsc_ev4, weights_dsc_ev4, signals_dsc_ev4] = pnl_equal_vol(fwd_last_USDXXX, spot_last_USDXXX,
                                                                                     ranks_discount, df_vols_monthly,
                                                                                     target_vol=tgt_vol, k=4)

    # In[34]:


    TR_dsc_ev = pd.DataFrame(index=TR_dsc_ev1.index, columns=['k1', 'k2', 'k3', 'k4'])
    TR_dsc_ev['k1'] = TR_dsc_ev1
    TR_dsc_ev['k2'] = TR_dsc_ev2
    TR_dsc_ev['k3'] = TR_dsc_ev3
    TR_dsc_ev['k4'] = TR_dsc_ev4
    TR_dsc_ev.plot(figsize=(15, 10), title='Equal vol Carry Trade', logy=True)

    # ### Tables 04 and 05

    # In[35]:


    iterables = [['Forward discount', 'Curvature'], range(1, 5)]
    idx = pd.MultiIndex.from_product(iterables, names=['Strategy', 'k'])
    table04_Funding = pd.DataFrame(index=idx, columns=df_tickers.index)
    table05_Investing = pd.DataFrame(index=idx, columns=df_tickers.index)

    # In[36]:


    # table04 Funding Currencies
    table04_Funding.loc[('Forward discount', 1)] = (signals1 < -0.01).sum()
    table04_Funding.loc[('Forward discount', 2)] = (signals2 < -0.01).sum()
    table04_Funding.loc[('Forward discount', 3)] = (signals3 < -0.01).sum()
    table04_Funding.loc[('Forward discount', 4)] = (signals4 < -0.01).sum()
    # table05 Investing Currencies
    table05_Investing.loc[('Forward discount', 1)] = (signals1 > 0.01).sum()
    table05_Investing.loc[('Forward discount', 2)] = (signals2 > 0.01).sum()
    table05_Investing.loc[('Forward discount', 3)] = (signals3 > 0.01).sum()
    table05_Investing.loc[('Forward discount', 4)] = (signals4 > 0.01).sum()

    # In[37]:




    # In[38]:


    # ### Curves

    # In[39]:


    bbgcurve_dict = {'AUD': 'YCSW0001 Index',
                     'CAD': 'YCSW0004 Index',
                     'CHF': 'YCSW0021 Index',
                     'DEM': 'YCGT0054 Index',
                     'GBP': 'YCSW0022 Index',
                     'JPY': 'YCSW0013 Index',
                     'NOK': 'YCSW0016 Index',
                     'NZD': 'YCSW0015 Index',
                     'SEK': 'YCSW0020 Index',
                     'USD': 'YCSW0023 Index', }

    tenor_dict = {'1D': 1.0 / 360,
                  '2D': 2.0 / 360,
                  '1W': 7.0 / 360,
                  '1M': 1.0 / 12,
                  '2M': 2. / 12,
                  '3M': 3.0 / 12,
                  '4M': 4.0 / 12,
                  '5M': 5.0 / 12,
                  '6M': 0.5,
                  '1Y': 1.0,
                  '18M': 1.5,
                  '2Y': 2.0,
                  '3Y': 3.0,
                  '4Y': 4.0,
                  '5Y': 5.0,
                  '6Y': 6.0,
                  '7Y': 7.0,
                  '8Y': 8.0,
                  '9Y': 9.0,
                  '10Y': 10.0,
                  '11Y': 11.0,
                  '12Y': 12.0,
                  '13Y': 13.0,
                  '14Y': 14.0,
                  '15Y': 15.0,
                  '20Y': 20.0,
                  '25Y': 25.0,
                  '30Y': 30.0,
                  '35Y': 35.0,
                  '40Y': 40.0,
                  '45Y': 45.0,
                  '50Y': 50.0}


    def converte_tenors(tenors):
        return [tenor_dict[tenor] * 12 for tenor in tenors]


    def metadados_curva_bbg(ticker_moeda='CHF'):
        ticker_curva = bbgcurve_dict[ticker_moeda]
        ticker_membros = list(con.bulkref(ticker_curva, 'CURVE_MEMBERS')['value'])
        prazos = list(con.bulkref(ticker_curva, 'CURVE_TERMS')['value'])
        df_curve_aux = pd.DataFrame(data=list(zip(prazos, ticker_membros)), columns=['tenor', 'ticker'])
        df_curve_aux['years'] = converte_tenors(list(df_curve_aux.tenor))
        return df_curve_aux


    # In[40]:


    # Exemplo de um DataFrame de metadados da curva da Suica.
    metadados_curva_bbg().head()

    # In[41]:


    lista_moedas_aux = list(bbgcurve_dict.keys())
    # Arruma o primeiro df para receber Append.
    moeda1 = lista_moedas_aux[0]
    df_metadados_curva = metadados_curva_bbg(moeda1)
    idx2 = pd.MultiIndex.from_product(iterables=[[moeda1], df_metadados_curva['years']])
    df_metadados_curva.set_index(idx2, drop=True, verify_integrity=True, inplace=True)
    # Baixa os outros dados e depois 'append' na df_metadados_curva.
    for moeda in lista_moedas_aux[1:]:
        df_metadados_aux = metadados_curva_bbg(moeda)
        idx2 = pd.MultiIndex.from_product(iterables=[[moeda], df_metadados_aux['years']])
        df_metadados_aux.set_index(idx2, drop=True, verify_integrity=True, inplace=True)
        df_metadados_curva = df_metadados_curva.append(other=df_metadados_aux, verify_integrity=True)

    # In[42]:


    # Cria dicionario p/ conversao de ticker p/ ano
    ticker_year_dict = dict(zip(df_metadados_curva['ticker'], df_metadados_curva['years']))

    # In[43]:


    df_metadados_curva.loc[(slice(None), 5), :]

    # In[44]:




    # In[46]:


    def consulta_curva_bbg(lista_tickers=['US00O/N  Index', 'USDR2T   Curncy'], dt_ini=ini_date_bbg, dt_fim=end_date_bbg):
        # Consulta
        df_curva_bbg_aux = con.bdh(lista_tickers, bbg_field_last, ini_date_bbg, end_date_bbg,
                                   elms=[("periodicitySelection", "MONTHLY")])
        # Tratamento
        df_curva_bbg_aux.columns = df_curva_bbg_aux.columns.droplevel(level='field')
        df_curva_bbg_aux.columns = [ticker_year_dict[ticker] for ticker in df_curva_bbg_aux.columns]
        df_curva_bbg_aux.sort_index(axis=1, inplace=True)
        df_curva_bbg_aux.fillna(method='ffill', inplace=True)
        return df_curva_bbg_aux


    # In[47]:


    lista_moedas_aux = list(bbgcurve_dict.keys())
    # Arruma o primeiro df para receber Append.
    # Parametros
    moeda1 = lista_moedas_aux[0]
    lista_tickers1 = list(df_metadados_curva.loc[moeda1, 'ticker'].values)
    # Consulta
    df_curva_bbg = consulta_curva_bbg(lista_tickers1)
    # Tratamento
    idx3 = pd.MultiIndex.from_product(iterables=[[moeda1], df_curva_bbg.index.values])
    df_curva_bbg.set_index(idx3, drop=True, verify_integrity=True, inplace=True)
    # Baixa os outros dados e depois 'append' na df_curva_bbg.
    for moeda in lista_moedas_aux[1:]:
        # Parametros
        lista_tickers = list(df_metadados_curva.loc[moeda, 'ticker'].values)
        # Consulta
        df_curva_bbg_auxiliar = consulta_curva_bbg(lista_tickers)
        # Tratamento
        idx3 = pd.MultiIndex.from_product(iterables=[[moeda], df_curva_bbg_auxiliar.index.values])
        df_curva_bbg_auxiliar.set_index(idx3, drop=True, verify_integrity=True, inplace=True)
        # Junta os dados
        df_curva_bbg = df_curva_bbg.append(other=df_curva_bbg_auxiliar, verify_integrity=True)

    # In[48]:


    df_curva_bbg.tail()


    # # Nelson-Siegel

    # In[49]:


    # A funcao recebe um DataFrame de betas b1,b2,b3 (Lambda eh hardcoded) e retorna uma soma de erros quadrados ponderados
    def NSiegel(array_betas, df_curve):
        b1 = array_betas[0]
        b2 = array_betas[1]
        b3 = array_betas[2]
        years = np.array(df_curve.columns.values)
        int_rates = df_curve.values[0]
        #     _lambda = df_parameters['lambda']
        _lambda = 0.0609  # hardcoded
        #     df_y_est = pd.DataFrame(index=tau, columns=['y_est','errors'])
        sum_squared_errors = 0.0
        for tau, rate in zip(years, int_rates):
            _aux = ((1 - math.exp(-_lambda * tau)) / (_lambda * tau))
            y_tau = b1 + b2 * _aux + b3 * (_aux - math.exp(-_lambda * tau))
            factor = 1.0  # to improve...
            sum_squared_errors = sum_squared_errors + ((rate - y_tau) ** 2) * factor
        return sum_squared_errors


    # In[50]:


    # Example of df_curve_input layout
    df_curve_input_ex = df_curva_bbg.loc[('USD', '1991-01-31'), :].dropna(how='all', axis=1)


    # In[51]:


    # listas p/ rodar NSiegel
    calendar_curves = list(df_curva_bbg.index.get_level_values(1).unique())
    # lista_moedas_aux
    # DataFrame p/ guardar betas
    idx4 = pd.MultiIndex.from_product(iterables=[lista_moedas_aux, calendar_curves])
    df_betas = pd.DataFrame(index=idx4, columns=['b1', 'b2', 'b3'])
    beta0 = np.ones(3)  # b1, b2 e b3

    for d in tqdm(calendar_curves, 'Nelson-Siegel Fitting'):
        for moeda in lista_moedas_aux:
            df_curve_input = df_curva_bbg.loc[(moeda, str(d)), :].dropna(how='all', axis=1)  # quando d é str, axis=1
            #         beta0 = np.ones(3) #b1, b2 e b3
            res = minimize(fun=NSiegel,
                           x0=beta0,
                           args=df_curve_input,
                           method='SLSQP')
            if res.success:
                df_betas.loc[moeda, d] = res.x
    df_betas

    # In[52]:


    df_curvature_all_tenors = pd.DataFrame(index=calendar_curves, columns=lista_moedas_aux)
    for moeda in lista_moedas_aux:
        df_curvature_all_tenors[moeda] = df_betas.loc[(moeda), 'b3']
    df_curvature_all_tenors.plot(figsize=(15, 10), title='Ficou RUIM usando todos os tenors')

    # ### Outliers coming from NOK and SEK, specially from short tenors (Sept 1992)

    # In[53]:


    var = df_betas.loc[('NOK'), 'b3']['1992']

    # In[54]:


    var = df_curva_bbg.xs('1992-09-30', level=1).T

    # ### New Nelson-Siegel, starting from 3 month tenor...

    # In[55]:


    # listas p/ rodar NSiegel
    calendar_curves = list(df_curva_bbg.index.get_level_values(1).unique())
    # lista_moedas_aux
    # DataFrame p/ guardar betas
    idx4 = pd.MultiIndex.from_product(iterables=[lista_moedas_aux, calendar_curves])
    df_betas1 = pd.DataFrame(index=idx4, columns=['b1', 'b2', 'b3'])
    beta0 = np.ones(3)  # b1, b2 e b3

    for d in tqdm(calendar_curves, 'Nelson-Siegel Fitting'):
        for moeda in lista_moedas_aux:
            df_curve_input = df_curva_bbg.loc[(moeda, str(d)), :].dropna(how='all', axis=1)  # quando d é str, axis=1
            df_curve_input = df_curve_input.loc[:, df_curve_input.columns >= 0.25 * 12]
            #         beta0 = np.ones(3) #b1, b2 e b3
            res = minimize(fun=NSiegel,
                           x0=beta0,
                           args=df_curve_input,
                           method='SLSQP')
            if res.success:
                df_betas1.loc[moeda, d] = res.x

    # In[56]:


    df_curvature_all_tenors1 = pd.DataFrame(index=calendar_curves, columns=lista_moedas_aux)
    for moeda in lista_moedas_aux:
        df_curvature_all_tenors1[moeda] = df_betas1.loc[(moeda), 'b3']
    df_curvature_all_tenors1.plot(figsize=(15, 10), title='Better if start from 3Mo')

    # In[57]:


    cond1 = df_curvature_all_tenors1.columns != 'NOK'
    # cond2 = df_betas_all_tenors1.columns != 'SEK'
    cond = cond1  # & cond2
    df_curvature_all_tenors.loc[:, cond].plot(figsize=(15, 10), title='ALL TENORS')

    # In[58]:


    df_curvature_all_tenors1.loc[:, cond].plot(figsize=(15, 10), title='From 3Mo')

    # In[59]:


    df_curvature_all_tenors1.loc['1993':, cond].plot(figsize=(15, 10), title='From 3Mo (since 1993)')

    # In[60]:


    df_level_all_tenors1 = pd.DataFrame(index=calendar_curves, columns=lista_moedas_aux)
    for moeda in lista_moedas_aux:
        df_level_all_tenors1[moeda] = df_betas1.loc[(moeda), 'b1']
    df_level_all_tenors1.plot(figsize=(15, 10), title="Level (beta1)")

    # In[61]:


    df_slope_all_tenors1 = pd.DataFrame(index=calendar_curves, columns=lista_moedas_aux)
    for moeda in lista_moedas_aux:
        df_slope_all_tenors1[moeda] = df_betas1.loc[(moeda), 'b2']
    df_slope_all_tenors1.plot(figsize=(15, 10), title="Slope (beta2)")

    # ### Starting from 3 months looks better

    # In[62]:


    Stdev_b3 = pd.DataFrame(df_curvature_all_tenors.std(), columns=['All tenors'])
    Stdev_b3['From 3Mo on'] = df_curvature_all_tenors1.std()
    Stdev_b3['Dif'] = Stdev_b3['From 3Mo on'] - Stdev_b3['All tenors']
    Stdev_b3

    # ### Signals from Curvature

    # In[63]:


    # Inicializando DataFrame
    # Parametros
    # calendar_curves
    lst_currencies = list(df_tickers.index)
    relative_curvature = pd.DataFrame(index=calendar_curves, columns=lst_currencies)
    ranks_curvature = pd.DataFrame(index=calendar_curves, columns=lst_currencies)

    # #### Relative Curvature:
    # Curvature loading from a country - US curvature (beta3)

    # In[64]:


    relative_curvature = df_curvature_all_tenors1[lst_currencies]
    for moeda in lst_currencies:
        relative_curvature.loc[:, moeda] = relative_curvature.loc[:, moeda] - df_curvature_all_tenors1.loc[:, 'USD']

    # In[65]:


    for d in calendar_curves:
        available_FX = list(relative_curvature.loc[d].dropna(how='any').index)
        ranks_curvature.loc[d, available_FX] = relative_curvature.loc[d, available_FX].rank(ascending=True)

    # In[66]:


    ranks_curvature.tail()

    # ### P&L of curvature strategy

    # In[67]:


    [TR_index1_c, holdings1_c, weights1_c, signals1_c] = pnl_fwd_dsc(fwd_last_USDXXX, spot_last_USDXXX, ranks_curvature,
                                                                     k=1)
    [TR_index2_c, holdings2_c, weights2_c, signals2_c] = pnl_fwd_dsc(fwd_last_USDXXX, spot_last_USDXXX, ranks_curvature,
                                                                     k=2)
    [TR_index3_c, holdings3_c, weights3_c, signals3_c] = pnl_fwd_dsc(fwd_last_USDXXX, spot_last_USDXXX, ranks_curvature,
                                                                     k=3)
    [TR_index4_c, holdings4_c, weights4_c, signals4_c] = pnl_fwd_dsc(fwd_last_USDXXX, spot_last_USDXXX, ranks_curvature,
                                                                     k=4)

    # In[68]:


    TR_curvy = pd.DataFrame(index=TR_index1_c.index, columns=['k1', 'k2', 'k3', 'k4'])
    TR_curvy['k1'] = TR_index1_c
    TR_curvy['k2'] = TR_index2_c
    TR_curvy['k3'] = TR_index3_c
    TR_curvy['k4'] = TR_index4_c
    TR_curvy.plot(figsize=(15, 10), title='CURVY Trade')

    # In[69]:


    TR_dsc.plot(figsize=(15, 10), title='CARRY Trade')

    # In[70]:


    TR_curvy.plot(figsize=(15, 10), title='CURVY Trade (log scale)', logy=True)

    # In[71]:


    TR_dsc.plot(figsize=(15, 10), title='CARRY Trade(log scale)', logy=True)

    # # P&L of equal vol curvy trade

    # In[72]:


    # using pnl_equal_vol() function
    tgt_vol_curvy = 0.09
    [TR_crv_ev1, holdings_crv_ev1, weights_crv_ev1, signals_crv_ev1] = pnl_equal_vol(fwd_last_USDXXX, spot_last_USDXXX,
                                                                                     ranks_curvature, df_vols_monthly,
                                                                                     tgt_vol_curvy, k=1)
    [TR_crv_ev2, holdings_crv_ev2, weights_crv_ev2, signals_crv_ev2] = pnl_equal_vol(fwd_last_USDXXX, spot_last_USDXXX,
                                                                                     ranks_curvature, df_vols_monthly,
                                                                                     tgt_vol_curvy, k=2)
    [TR_crv_ev3, holdings_crv_ev3, weights_crv_ev3, signals_crv_ev3] = pnl_equal_vol(fwd_last_USDXXX, spot_last_USDXXX,
                                                                                     ranks_curvature, df_vols_monthly,
                                                                                     tgt_vol_curvy, k=3)
    [TR_crv_ev4, holdings_crv_ev4, weights_crv_ev4, signals_crv_ev4] = pnl_equal_vol(fwd_last_USDXXX, spot_last_USDXXX,
                                                                                     ranks_curvature, df_vols_monthly,
                                                                                     tgt_vol_curvy, k=4)

    # In[73]:


    TR_crv_ev = pd.DataFrame(index=TR_dsc_ev1.index, columns=['k1', 'k2', 'k3', 'k4'])
    TR_crv_ev['k1'] = TR_crv_ev1
    TR_crv_ev['k2'] = TR_crv_ev2
    TR_crv_ev['k3'] = TR_crv_ev3
    TR_crv_ev['k4'] = TR_crv_ev4
    TR_crv_ev.plot(figsize=(15, 10), title='Equal vol Curvy Trade', logy=True)

    # ### Signals and P&L for Level

    # In[74]:


    # Inicializando DataFrame
    # Parametros
    # calendar_curves
    lst_currencies = list(df_tickers.index)
    relative_level = pd.DataFrame(index=calendar_curves, columns=lst_currencies)
    ranks_level = pd.DataFrame(index=calendar_curves, columns=lst_currencies)
    relative_level = df_level_all_tenors1[lst_currencies]
    for moeda in lst_currencies:
        relative_level.loc[:, moeda] = relative_level.loc[:, moeda] - df_level_all_tenors1.loc[:, 'USD']
    for d in calendar_curves:
        available_FX = list(relative_level.loc[d].dropna(how='any').index)
        ranks_level.loc[d, available_FX] = relative_level.loc[d, available_FX].rank(ascending=True)

    # In[75]:


    [TR_index1_l, holdings1_l, weights1_l, signals1_l] = pnl_fwd_dsc(fwd_last_USDXXX, spot_last_USDXXX, ranks_level, k=1)
    [TR_index2_l, holdings2_l, weights2_l, signals2_l] = pnl_fwd_dsc(fwd_last_USDXXX, spot_last_USDXXX, ranks_level, k=2)
    [TR_index3_l, holdings3_l, weights3_l, signals3_l] = pnl_fwd_dsc(fwd_last_USDXXX, spot_last_USDXXX, ranks_level, k=3)
    [TR_index4_l, holdings4_l, weights4_l, signals4_l] = pnl_fwd_dsc(fwd_last_USDXXX, spot_last_USDXXX, ranks_level, k=4)

    # In[76]:


    TR_level = pd.DataFrame(index=TR_index1_c.index, columns=['k1', 'k2', 'k3', 'k4'])
    TR_level['k1'] = TR_index1_l
    TR_level['k2'] = TR_index2_l
    TR_level['k3'] = TR_index3_l
    TR_level['k4'] = TR_index4_l
    TR_level.plot(figsize=(15, 10), title='LEVEL Trade')

    # In[77]:


    TR_level.plot(figsize=(15, 10), title='LEVEL Trade (log scale)', logy=True)

    # ### Signals and P&L for Slope

    # In[78]:


    # Inicializando DataFrame
    # Parametros
    # calendar_curves
    lst_currencies = list(df_tickers.index)
    relative_slope = pd.DataFrame(index=calendar_curves, columns=lst_currencies)
    ranks_slope = pd.DataFrame(index=calendar_curves, columns=lst_currencies)
    relative_slope = df_slope_all_tenors1[lst_currencies]
    for moeda in lst_currencies:
        relative_slope.loc[:, moeda] = relative_slope.loc[:, moeda] - df_slope_all_tenors1.loc[:, 'USD']
    for d in calendar_curves:
        available_FX = list(relative_slope.loc[d].dropna(how='any').index)
        ranks_slope.loc[d, available_FX] = relative_slope.loc[d, available_FX].rank(ascending=True)

    # In[79]:


    [TR_index1_s, holdings1_s, weights1_s, signals1_s] = pnl_fwd_dsc(fwd_last_USDXXX, spot_last_USDXXX, ranks_slope, k=1)
    [TR_index2_s, holdings2_s, weights2_s, signals2_s] = pnl_fwd_dsc(fwd_last_USDXXX, spot_last_USDXXX, ranks_slope, k=2)
    [TR_index3_s, holdings3_s, weights3_s, signals3_s] = pnl_fwd_dsc(fwd_last_USDXXX, spot_last_USDXXX, ranks_slope, k=3)
    [TR_index4_s, holdings4_s, weights4_s, signals4_s] = pnl_fwd_dsc(fwd_last_USDXXX, spot_last_USDXXX, ranks_slope, k=4)

    # In[80]:


    TR_slope = pd.DataFrame(index=TR_index1_c.index, columns=['k1', 'k2', 'k3', 'k4'])
    TR_slope['k1'] = TR_index1_s
    TR_slope['k2'] = TR_index2_s
    TR_slope['k3'] = TR_index3_s
    TR_slope['k4'] = TR_index4_s
    TR_slope.plot(figsize=(15, 10), title='SLOPE Trade')

    # In[81]:


    TR_slope.plot(figsize=(15, 10), title='SLOPE Trade (log scale)', logy=True)

    # ### Tables 04 and 05

    # In[82]:


    # table04 Funding Currencies
    table04_Funding.loc[('Curvature', 1)] = (signals1_c < -0.01).sum()
    table04_Funding.loc[('Curvature', 2)] = (signals2_c < -0.01).sum()
    table04_Funding.loc[('Curvature', 3)] = (signals3_c < -0.01).sum()
    table04_Funding.loc[('Curvature', 4)] = (signals4_c < -0.01).sum()
    # table05 Investing Currencies
    table05_Investing.loc[('Curvature', 1)] = (signals1_c > 0.01).sum()
    table05_Investing.loc[('Curvature', 2)] = (signals2_c > 0.01).sum()
    table05_Investing.loc[('Curvature', 3)] = (signals3_c > 0.01).sum()
    table05_Investing.loc[('Curvature', 4)] = (signals4_c > 0.01).sum()

    # In[83]:


    table04_Funding

    # In[84]:


    table05_Investing

    # ### Table 03

    # In[85]:


    from scipy.stats import skew
    from scipy.stats import kurtosis


    # In[86]:


    def GetPerformanceTable(IndexSeries, freq='Daily'):
        adju_factor = 252.0
        if freq == 'Monthly':
            adju_factor = 12.0
        elif freq == 'Weekly':
            adju_factor = 52.0

        Table = pd.Series(index=['Excess Return', 'Volatility', 'Sharpe', 'Sortino', 'Max Drawdown',
                                 'Max Drawdown in Vol Terms', '5th percentile in Vol Terms',
                                 '10th percentile in Vol Terms'])

        CleanIndexSeries = IndexSeries.dropna().sort_index()

        ER_index = pd.Series(index=CleanIndexSeries.index)
        ER_index[CleanIndexSeries.index[0]] = 100.0
        for d, d_minus_1 in zip(ER_index.index[1:], ER_index.index[:-1]):
            ER = CleanIndexSeries[d] / CleanIndexSeries[d_minus_1] - 1.0
            ER_index[d] = ER_index[d_minus_1] * (1 + ER)

        Retornos = CleanIndexSeries.pct_change(1)
        Retorno_medio = CleanIndexSeries.pct_change(1).mean()
        Table['Excess Return'] = (CleanIndexSeries[-1] / CleanIndexSeries[0]) ** (
                    adju_factor / (len(CleanIndexSeries) - 1.0)) - 1
        Table['Volatility'] = (np.log(ER_index).diff(1).dropna()).std() * np.sqrt(adju_factor)
        Table['Sharpe'] = Table['Excess Return'] / Table['Volatility']
        Table['Sortino'] = Table['Excess Return'] / (np.sqrt(adju_factor) * (
        np.log(ER_index).diff(1).dropna()[np.log(ER_index).diff(1).dropna() < 0.0]).std())
        Table['Skewness'] = skew(Retornos.dropna())
        Table['Kurtosis'] = kurtosis(Retornos.dropna())
        Table['Max Drawdown'] = max_dd(ER_index)
        Table['Max Drawdown in Vol Terms'] = max_dd(ER_index) / Table['Volatility']
        Table['5th percentile in Vol Terms'] = (ER_index.pct_change(1).dropna()).quantile(q=0.05) / Table['Volatility']
        Table['10th percentile in Vol Terms'] = (ER_index.pct_change(1).dropna()).quantile(q=0.1) / Table['Volatility']
        return Table


    def max_dd(ser):
        max2here = ser.expanding(min_periods=1).max()
        dd2here = ser / max2here - 1.0
        return dd2here.min()


    # In[87]:


    idx5 = pd.MultiIndex.from_product(
        iterables=[['Forward discount', 'Equal vol carry', 'Level', 'Slope', 'Curvature', 'Equal vol curvy'], range(1, 5)],
        names=['Sorting', 'k'])
    colunas = ['Mean annual', 'Stdev annual', 'Skewness monthly', 'Kurtosis monthly', 'Sharpe ratio']
    table03 = pd.DataFrame(index=idx5, columns=colunas)

    # In[88]:


    table03

    # In[89]:


    mean_annual_curvy = lambda k: GetPerformanceTable(TR_curvy["k" + str(k)], freq='Monthly')['Excess Return'] * 100
    mean_annual_crv_ev = lambda k: GetPerformanceTable(TR_crv_ev["k" + str(k)], freq='Monthly')['Excess Return'] * 100
    mean_annual_dsc = lambda k: GetPerformanceTable(TR_dsc["k" + str(k)], freq='Monthly')['Excess Return'] * 100
    mean_annual_dsc_ev = lambda k: GetPerformanceTable(TR_dsc_ev["k" + str(k)], freq='Monthly')['Excess Return'] * 100
    mean_annual_level = lambda k: GetPerformanceTable(TR_level["k" + str(k)], freq='Monthly')['Excess Return'] * 100
    mean_annual_slope = lambda k: GetPerformanceTable(TR_slope["k" + str(k)], freq='Monthly')['Excess Return'] * 100

    # In[90]:


    vol_annual_curvy = lambda k: GetPerformanceTable(TR_curvy["k" + str(k)], freq='Monthly')['Volatility'] * 100
    vol_annual_crv_ev = lambda k: GetPerformanceTable(TR_crv_ev["k" + str(k)], freq='Monthly')['Volatility'] * 100
    vol_annual_dsc = lambda k: GetPerformanceTable(TR_dsc["k" + str(k)], freq='Monthly')['Volatility'] * 100
    vol_annual_dsc_ev = lambda k: GetPerformanceTable(TR_dsc_ev["k" + str(k)], freq='Monthly')['Volatility'] * 100
    vol_annual_level = lambda k: GetPerformanceTable(TR_level["k" + str(k)], freq='Monthly')['Volatility'] * 100
    vol_annual_slope = lambda k: GetPerformanceTable(TR_slope["k" + str(k)], freq='Monthly')['Volatility'] * 100

    # In[91]:


    sharpe_annual_curvy = lambda k: GetPerformanceTable(TR_curvy["k" + str(k)], freq='Monthly')['Sharpe']
    sharpe_annual_crv_ev = lambda k: GetPerformanceTable(TR_crv_ev["k" + str(k)], freq='Monthly')['Sharpe']
    sharpe_annual_dsc = lambda k: GetPerformanceTable(TR_dsc["k" + str(k)], freq='Monthly')['Sharpe']
    sharpe_annual_dsc_ev = lambda k: GetPerformanceTable(TR_dsc_ev["k" + str(k)], freq='Monthly')['Sharpe']
    sharpe_annual_level = lambda k: GetPerformanceTable(TR_level["k" + str(k)], freq='Monthly')['Sharpe']
    sharpe_annual_slope = lambda k: GetPerformanceTable(TR_slope["k" + str(k)], freq='Monthly')['Sharpe']

    # In[92]:


    skew_annual_curvy = lambda k: GetPerformanceTable(TR_curvy["k" + str(k)], freq='Monthly')['Skewness']
    skew_annual_crv_ev = lambda k: GetPerformanceTable(TR_crv_ev["k" + str(k)], freq='Monthly')['Skewness']
    skew_annual_dsc = lambda k: GetPerformanceTable(TR_dsc["k" + str(k)], freq='Monthly')['Skewness']
    skew_annual_dsc_ev = lambda k: GetPerformanceTable(TR_dsc_ev["k" + str(k)], freq='Monthly')['Skewness']
    skew_annual_level = lambda k: GetPerformanceTable(TR_level["k" + str(k)], freq='Monthly')['Skewness']
    skew_annual_slope = lambda k: GetPerformanceTable(TR_slope["k" + str(k)], freq='Monthly')['Skewness']

    # In[93]:


    kurt_annual_curvy = lambda k: GetPerformanceTable(TR_curvy["k" + str(k)], freq='Monthly')['Kurtosis']
    kurt_annual_crv_ev = lambda k: GetPerformanceTable(TR_crv_ev["k" + str(k)], freq='Monthly')['Kurtosis']
    kurt_annual_dsc = lambda k: GetPerformanceTable(TR_dsc["k" + str(k)], freq='Monthly')['Kurtosis']
    kurt_annual_dsc_ev = lambda k: GetPerformanceTable(TR_dsc_ev["k" + str(k)], freq='Monthly')['Kurtosis']
    kurt_annual_level = lambda k: GetPerformanceTable(TR_level["k" + str(k)], freq='Monthly')['Kurtosis']
    kurt_annual_slope = lambda k: GetPerformanceTable(TR_slope["k" + str(k)], freq='Monthly')['Kurtosis']

    # In[94]:


    table03.loc[('Curvature',), 'Mean annual'] = [mean_annual_curvy(k) for k in range(1, 5)]
    table03.loc[('Equal vol curvy',), 'Mean annual'] = [mean_annual_crv_ev(k) for k in range(1, 5)]
    table03.loc[('Forward discount',), 'Mean annual'] = [mean_annual_dsc(k) for k in range(1, 5)]
    table03.loc[('Equal vol carry',), 'Mean annual'] = [mean_annual_dsc_ev(k) for k in range(1, 5)]
    table03.loc[('Level',), 'Mean annual'] = [mean_annual_level(k) for k in range(1, 5)]
    table03.loc[('Slope',), 'Mean annual'] = [mean_annual_slope(k) for k in range(1, 5)]
    table03

    # In[95]:


    table03.loc[('Curvature',), 'Stdev annual'] = [vol_annual_curvy(k) for k in range(1, 5)]
    table03.loc[('Equal vol curvy',), 'Stdev annual'] = [vol_annual_crv_ev(k) for k in range(1, 5)]
    table03.loc[('Forward discount',), 'Stdev annual'] = [vol_annual_dsc(k) for k in range(1, 5)]
    table03.loc[('Equal vol carry',), 'Stdev annual'] = [vol_annual_dsc_ev(k) for k in range(1, 5)]
    table03.loc[('Level',), 'Stdev annual'] = [vol_annual_level(k) for k in range(1, 5)]
    table03.loc[('Slope',), 'Stdev annual'] = [vol_annual_slope(k) for k in range(1, 5)]

    # In[96]:


    table03.loc[('Curvature',), 'Sharpe ratio'] = [sharpe_annual_curvy(k) for k in range(1, 5)]
    table03.loc[('Equal vol curvy',), 'Sharpe ratio'] = [sharpe_annual_crv_ev(k) for k in range(1, 5)]
    table03.loc[('Forward discount',), 'Sharpe ratio'] = [sharpe_annual_dsc(k) for k in range(1, 5)]
    table03.loc[('Equal vol carry',), 'Sharpe ratio'] = [sharpe_annual_dsc_ev(k) for k in range(1, 5)]
    table03.loc[('Level',), 'Sharpe ratio'] = [sharpe_annual_level(k) for k in range(1, 5)]
    table03.loc[('Slope',), 'Sharpe ratio'] = [sharpe_annual_slope(k) for k in range(1, 5)]

    # In[97]:


    table03.loc[('Curvature',), 'Skewness monthly'] = [skew_annual_curvy(k) for k in range(1, 5)]
    table03.loc[('Equal vol curvy',), 'Skewness monthly'] = [skew_annual_crv_ev(k) for k in range(1, 5)]
    table03.loc[('Forward discount',), 'Skewness monthly'] = [skew_annual_dsc(k) for k in range(1, 5)]
    table03.loc[('Equal vol carry',), 'Skewness monthly'] = [skew_annual_dsc_ev(k) for k in range(1, 5)]
    table03.loc[('Level',), 'Skewness monthly'] = [skew_annual_level(k) for k in range(1, 5)]
    table03.loc[('Slope',), 'Skewness monthly'] = [skew_annual_slope(k) for k in range(1, 5)]

    # In[98]:


    table03.loc[('Curvature',), 'Kurtosis monthly'] = [kurt_annual_curvy(k) for k in range(1, 5)]
    table03.loc[('Equal vol curvy',), 'Kurtosis monthly'] = [kurt_annual_crv_ev(k) for k in range(1, 5)]
    table03.loc[('Forward discount',), 'Kurtosis monthly'] = [kurt_annual_dsc(k) for k in range(1, 5)]
    table03.loc[('Equal vol carry',), 'Kurtosis monthly'] = [kurt_annual_dsc_ev(k) for k in range(1, 5)]
    table03.loc[('Level',), 'Kurtosis monthly'] = [kurt_annual_level(k) for k in range(1, 5)]
    table03.loc[('Slope',), 'Kurtosis monthly'] = [kurt_annual_slope(k) for k in range(1, 5)]

    # In[99]:


    table03

    # # Dados p/ MtM diario

    # In[100]:


    # In[101]:



    # In[102]:


    # In[103]:


    def eomonth(date_):
        return (date_ - pd.Timedelta(1, unit='d')) + BMonthEnd(1)


    # In[104]:


    def consulta_fwdcurve_bbg(
            lista_tickers=['AUD2W BGN Curncy',
                           'AUD3M BGN Curncy',
                           'AUD2M BGN Curncy',
                           'AUD6M BGN Curncy',
                           'AUD4M BGN Curncy',
                           'AUD1M BGN Curncy',
                           'AUD3W BGN Curncy',
                           'AUD5M BGN Curncy',
                           'AUD1W BGN Curncy'],
            dt_ini=ini_date_bbg, dt_fim=end_date_bbg):
        # Consulta
        df_curva_bbg_aux = con.bdh(lista_tickers, bbg_field_last, ini_date_bbg, end_date_bbg, elms=[("periodicitySelection", "DAILY")])
        # Tratamento
        df_curva_bbg_aux.columns = df_curva_bbg_aux.columns.droplevel(level='field')
        df_curva_bbg_aux.columns = [fwd_dict[ticker[3:]] for ticker in df_curva_bbg_aux.columns]
        df_curva_bbg_aux[0] = 0.0
        df_curva_bbg_aux.sort_index(axis=1, inplace=True)
        df_curva_bbg_aux.fillna(method='ffill', inplace=True)
        # Insere colunas
        for t in range(1, 7):
            df_curva_bbg_aux[t] = float('NaN')
        for t in range(8, 21):
            df_curva_bbg_aux[t] = float('NaN')
        for t in range(22, 31):
            df_curva_bbg_aux[t] = float('NaN')
        for t in range(32, 61):
            df_curva_bbg_aux[t] = float('NaN')
        for t in range(62, 91):
            df_curva_bbg_aux[t] = float('NaN')
        for t in range(92, 121):
            df_curva_bbg_aux[t] = float('NaN')
        for t in range(122, 151):
            df_curva_bbg_aux[t] = float('NaN')
        for t in range(152, 181):
            df_curva_bbg_aux[t] = float('NaN')
        df_curva_bbg_aux.sort_index(axis=1, inplace=True)
        # Interpola
        df_curva_bbg_aux.interpolate(method='linear', axis=1, inplace=True)
        return df_curva_bbg_aux


    # In[105]:


    # DataFrame for daily fwd points
    daily_calendar = pd.date_range(start=pd.to_datetime('1991-01-01'), end=pd.to_datetime('2019-07-02'), freq=BDay())
    idx6 = pd.MultiIndex.from_product(iterables=[currency_list, daily_calendar])
    df_daily_fwdpts = pd.DataFrame(data=None, index=idx6, columns=[float(x) for x in range(0, 182)])

    # In[106]:


    # Building DataFrame with data from bbg
    take_second = lambda s1, s2: s2
    for currency in tqdm(currency_list, 'Daily forward points...'):
        list_to_work = list(df_tickers.loc[currency, {'1w', '2w', '3w', '1m', '2m', '3m', '4m', '5m', '6m'}])
        # Consulta
        df_daily_fwd_aux = consulta_fwdcurve_bbg(list_to_work)
        # Tratamento
        idx7 = pd.MultiIndex.from_product(iterables=[[currency], df_daily_fwdpts.loc[currency].index.values])
        df_daily_fwd_aux = df_daily_fwdpts.loc[currency].combine(other=df_daily_fwd_aux, func=take_second)
        df_daily_fwd_aux.set_index(idx7, drop=True, inplace=True)
        df_daily_fwdpts.loc[currency] = df_daily_fwd_aux
    df_daily_fwdpts.fillna(method='ffill', inplace=True)
    df_daily_fwdpts.head()

    # In[107]:


    # scale adjusted fwd pts
    df_daily_fwdpts_scale_adj = df_daily_fwdpts.copy(deep=True)
    for currency in currency_list:
        df_daily_fwdpts_scale_adj.loc[currency] = (
                    df_daily_fwdpts_scale_adj.loc[currency] / df_tickers.loc[currency, 'scale']).values

    # In[108]:


    # Daily data for spot
    spot_daily_last = con.bdh(ticker_spot_bbg, bbg_field_last, ini_date_bbg, end_date_bbg, elms=[("periodicitySelection", "DAILY")])
    spot_daily_last.columns = spot_daily_last.columns.droplevel(1)
    spot_daily_last.fillna(method='ffill', inplace=True)
    spot_daily_last.columns = [x.replace(' Curncy', '') for x in spot_daily_last.columns]
    spot_daily_last.sort_index(axis=1, ascending=True, inplace=True)
    spot_daily_last.head()

    # In[109]:


    # DataFrame for daily fwds (outright)
    df_daily_spot = pd.DataFrame(data=None, index=idx6, columns=[float(x) for x in range(0, 182)])
    df_daily_fwds = pd.DataFrame(data=None, index=idx6, columns=[float(x) for x in range(0, 182)])
    for currency in currency_list:
        df_daily_spot.loc[(currency,), 0] = spot_daily_last.loc[df_daily_spot.loc[(currency,), 0].index, currency].values
    df_daily_spot.fillna(method='ffill', axis=1, inplace=True)

    # In[110]:


    # Spot + fwd pts
    df_daily_fwds = df_daily_spot + df_daily_fwdpts_scale_adj

    # In[111]:


    df_daily_fwds.head()

    # #### Currencies in USDXXX terms (FX/USD)

    # In[112]:


    # Currencies in USDXXX terms (FX/USD)
    df_daily_spotxxx = df_daily_spot.copy(deep=True)
    df_daily_fwdsxxx = df_daily_fwds.copy(deep=True)
    for currency in currency_list:
        df_daily_spotxxx.loc[currency] = (df_daily_spot.loc[currency] ** df_tickers.loc[currency, 'inverse']).values
        df_daily_fwdsxxx.loc[currency] = (df_daily_fwds.loc[currency] ** df_tickers.loc[currency, 'inverse']).values

    # In[113]:


    df_daily_spotxxx.head()

    # In[114]:


    df_daily_fwdsxxx.head()

    # #### P&L calculation

    # In[115]:


    # price change of a forward position
    fwd_chg_1d = df_daily_fwdsxxx - df_daily_fwdsxxx.shift(1, axis=0).shift(-1, axis=1)

    # In[116]:


    # DataFrame with price changes of holdings (eg: price change of a 14 days fwd)
    # number of days to maturity (forward)
    hds_pxchg = pd.DataFrame(data=None, index=daily_calendar, columns=currency_list)
    hds_pxchg['Maturity'] = daily_calendar.copy(True)
    hds_pxchg['Maturity'] = hds_pxchg['Maturity'].apply(eomonth)
    td = (hds_pxchg['Maturity'] - hds_pxchg['Maturity'].index)
    hds_pxchg['days'] = td.astype('timedelta64[D]')
    # corresponding price change of positions
    fwd_chg_1d_aux = fwd_chg_1d.unstack(level=-2)
    for dt, ndays in tqdm(zip(hds_pxchg.index, hds_pxchg['days']), 'building'):
        hds_pxchg.loc[dt, currency_list] = fwd_chg_1d_aux.loc[dt, ndays]
    hds_pxchg.loc['1991-01-01'] = np.NaN  # erasing first row
    hds_pxchg.head(20)


    # In[117]:


    def calc_daily_pnl(holdings, hds_pxchg, daily_calendar):
        hds_daily = pd.DataFrame(data=None, index=daily_calendar, columns=currency_list)
        hds_daily = hds_daily.combine(other=holdings, func=take_second)
        hds_daily.fillna(method='ffill', inplace=True)
        daily_tr_index = pd.DataFrame(data=None, index=daily_calendar, columns=currency_list)
        daily_tr_index = 100.0  # initial point
        daily_pnl = (hds_daily[currency_list].shift(1) * hds_pxchg[currency_list]).sum(axis=1)
        daily_tr_index = daily_tr_index + daily_pnl.cumsum()
        return daily_tr_index


    # In[118]:


    # Carry trade daily
    TR_dsc_daily = pd.DataFrame(index=daily_calendar, columns=['k1', 'k2', 'k3', 'k4'])
    TR_dsc_daily['k1'] = calc_daily_pnl(holdings1, hds_pxchg, daily_calendar)
    TR_dsc_daily['k2'] = calc_daily_pnl(holdings2, hds_pxchg, daily_calendar)
    TR_dsc_daily['k3'] = calc_daily_pnl(holdings3, hds_pxchg, daily_calendar)
    TR_dsc_daily['k4'] = calc_daily_pnl(holdings4, hds_pxchg, daily_calendar)
    TR_dsc_daily.plot(figsize=(15, 10), title='Carry trade daily MtM (log)', logy=True)

    # In[119]:


    TR_dsc_ev_daily = pd.DataFrame(index=daily_calendar, columns=['k1', 'k2', 'k3', 'k4'])
    TR_dsc_ev_daily['k1'] = calc_daily_pnl(holdings_dsc_ev1, hds_pxchg, daily_calendar)
    TR_dsc_ev_daily['k2'] = calc_daily_pnl(holdings_dsc_ev2, hds_pxchg, daily_calendar)
    TR_dsc_ev_daily['k3'] = calc_daily_pnl(holdings_dsc_ev3, hds_pxchg, daily_calendar)
    TR_dsc_ev_daily['k4'] = calc_daily_pnl(holdings_dsc_ev4, hds_pxchg, daily_calendar)
    TR_dsc_ev_daily.plot(figsize=(15, 10), title='Equal vol Carry Trade daily MtM (log)', logy=True)

    # In[120]:


    # Curvy trade daily
    TR_curvy_daily = pd.DataFrame(index=daily_calendar, columns=['k1', 'k2', 'k3', 'k4'])
    TR_curvy_daily['k1'] = calc_daily_pnl(holdings1_c, hds_pxchg, daily_calendar)
    TR_curvy_daily['k2'] = calc_daily_pnl(holdings2_c, hds_pxchg, daily_calendar)
    TR_curvy_daily['k3'] = calc_daily_pnl(holdings3_c, hds_pxchg, daily_calendar)
    TR_curvy_daily['k4'] = calc_daily_pnl(holdings4_c, hds_pxchg, daily_calendar)
    TR_curvy_daily.plot(figsize=(15, 10), title='Curvy trade daily MtM (log)', logy=True)

    # In[121]:


    # Equal vol curvy trade daily
    TR_crv_ev_daily = pd.DataFrame(index=daily_calendar, columns=['k1', 'k2', 'k3', 'k4'])
    TR_crv_ev_daily['k1'] = calc_daily_pnl(holdings_crv_ev1, hds_pxchg, daily_calendar)
    TR_crv_ev_daily['k2'] = calc_daily_pnl(holdings_crv_ev2, hds_pxchg, daily_calendar)
    TR_crv_ev_daily['k3'] = calc_daily_pnl(holdings_crv_ev3, hds_pxchg, daily_calendar)
    TR_crv_ev_daily['k4'] = calc_daily_pnl(holdings_crv_ev4, hds_pxchg, daily_calendar)
    TR_crv_ev_daily.plot(figsize=(15, 10), title='Equal vol Curvy Trade daily MtM (log)', logy=True)

    # In[122]:


    TR_level_daily = pd.DataFrame(index=daily_calendar, columns=['k1', 'k2', 'k3', 'k4'])
    TR_level_daily['k1'] = calc_daily_pnl(holdings1_l, hds_pxchg, daily_calendar)
    TR_level_daily['k2'] = calc_daily_pnl(holdings2_l, hds_pxchg, daily_calendar)
    TR_level_daily['k3'] = calc_daily_pnl(holdings3_l, hds_pxchg, daily_calendar)
    TR_level_daily['k4'] = calc_daily_pnl(holdings4_l, hds_pxchg, daily_calendar)
    TR_level_daily.plot(figsize=(15, 10), title='LEVEL trade daily MtM (log)', logy=True)

    # In[123]:


    TR_slope_daily = pd.DataFrame(index=daily_calendar, columns=['k1', 'k2', 'k3', 'k4'])
    TR_slope_daily['k1'] = calc_daily_pnl(holdings1_s, hds_pxchg, daily_calendar)
    TR_slope_daily['k2'] = calc_daily_pnl(holdings2_s, hds_pxchg, daily_calendar)
    TR_slope_daily['k3'] = calc_daily_pnl(holdings3_s, hds_pxchg, daily_calendar)
    TR_slope_daily['k4'] = calc_daily_pnl(holdings4_s, hds_pxchg, daily_calendar)
    TR_slope_daily.plot(figsize=(15, 10), title='SLOPE trade daily MtM (log)', logy=True)




import pdblp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime
from scipy.optimize import minimize  # optimization function
from pandas.tseries.offsets import BDay, BMonthBegin
from pandas.tseries.offsets import BMonthEnd
from tqdm import tqdm  # this will just make thing a bit prettier
from scipy.stats import skew
from scipy.stats import kurtosis
import time
from itertools import combinations, product



class CurvyTrade:
    """
    This class calculates trackers for FX carry trade and curvy trade. 
    Enters in a 1 month fwd position and rebalances monthly. MtM is daily.
    """

    # TODO DONE put a default folder for data as attribute
    # TODO DONE build a USD index
    # TODO DONE and corrected: study disparity btw monthly and daily
    # TODO DONE allow k greater than 4
    # TODO betas of strategies Vs USD index
    # TODO Breakdown of interest return and spot return
    # TODO start date indicator


    # TODO refactor code on strategies, so it can be standized by number
    # Available currencies
    # TODO DONE add 'BRL', 'CLP', 'COP', 'HUF', 'KRW', 'MXN', 'PHP', 'PLN', 'SGD', 'TRY', 'TWD', 'ZAR'
    currency_list =         ['AUD', 'CAD', 'CHF', 'DEM', 'EUR', 'GBP', 'JPY', 'NOK', 'NZD', 'SEK', 'BRL', 'CLP', 'COP', 'MXN',
                             'CZK', 'HUF', 'PLN', 'TRY', 'RUB', 'INR', 'HKD', 'KRW', 'SGD', 'TWD', 'ZAR']
    currency_list_curve =   ['AUD', 'CAD', 'CHF', 'DEM', 'EUR', 'GBP', 'JPY', 'NOK', 'NZD', 'SEK', 'USD', 'BRL', 'CLP', 'COP', 'MXN',
                             'CZK', 'HUF', 'PLN', 'TRY', 'RUB', 'INR', 'HKD', 'KRW', 'SGD', 'TWD', 'ZAR']
    currency_list_DM =      ['AUD', 'CAD', 'CHF', 'DEM', 'EUR', 'GBP', 'JPY', 'NOK', 'NZD', 'SEK']

    # REMINDER
    # Dates: start and end dates are used for strategy period. Price/market data will start from vol_date (hardcoded)

    # Default data folder
    data_folder = 'data'
    # BBG connection
    con = pdblp.BCon(debug=True, port=8194, timeout=5000)
    con.debug = False
    con.start()

    # tickers, start date, end date
    bbg_field_last = ['PX_LAST']
    bbg_field_bid = ['PX_LAST']
    bbg_field_ask = ['PX_LAST']
    bbg_periodicity_spot = 'DAILY'
    bbg_periodicity_interest_curve = 'DAILY'
    # ini_date_bbg = '19890101'  # extended period for vol calculation
    # data_ini = '19910101'
    # end_date_bbg = '20190702'
    # data_fim = '20190527'

    # TODO outliers
    # BRL: start from 1999-05
    # TRY: from 2001-08??
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
                '6M BGN Curncy': 181.0,
                '1W CMPN Curncy': 7.0,
                '2W CMPN Curncy': 14.0,
                '3W CMPN Curncy': 21.0,
                '1M CMPN Curncy': 31.0,
                '2M CMPN Curncy': 61.0,
                '3M CMPN Curncy': 91.0,
                '4M CMPN Curncy': 121.0,
                '5M CMPN Curncy': 151.0,
                '6M CMPN Curncy': 181.0
                }

    dict_fwdpts_source =  {'AUD': 'BGN',
                           'CAD': 'BGN',
                           'CHF': 'BGN',
                           'DEM': 'CMPN',
                           'EUR': 'BGN',
                           'GBP': 'BGN',
                           'JPY': 'BGN',
                           'NOK': 'BGN',
                           'NZD': 'BGN',
                           'SEK': 'BGN',
                           'BRL': 'BGN',
                           'CLP': 'BGN',
                           'COP': 'BGN',
                           'MXN': 'BGN',
                           'CZK': 'BGN',
                           'HUF': 'BGN',
                           'PLN': 'BGN',
                           'TRY': 'BGN',
                           'RUB': 'BGN',
                           'INR': 'BGN',
                           'HKD': 'BGN',
                           'KRW': 'BGN',
                           'PHP': 'BGN',
                           'SGD': 'BGN',
                           'TWD': 'BGN',
                           'ZAR': 'BGN'}

    dict_FX_NDF = {'AUD': 'AUD',
                   'CAD': 'CAD',
                   'CHF': 'CHF',
                   'DEM': 'DEM',
                   'EUR': 'EUR',
                   'GBP': 'GBP',
                   'JPY': 'JPY',
                   'NOK': 'NOK',
                   'NZD': 'NZD',
                   'SEK': 'SEK',
                   'BRL': 'BCN',
                   'CLP': 'CHN',
                   'COP': 'CLN',
                   'MXN': 'MXN',
                   'CZK': 'CZK',
                   'HUF': 'HUF',
                   'PLN': 'PLN',
                   'TRY': 'TRY',
                   'RUB': 'RUB',
                   'INR': 'IRN',
                   'HKD': 'HKD',
                   'KRW': 'KWN',
                   'PHP': 'PPN',
                   'SGD': 'SGD',
                   'TWD': 'NTN',
                   'ZAR': 'ZAR'
                  }


    def __init__(self, start_date='1991-01-01', end_date='2019-07-02', k_max=4):
        # Start/End date of strategy
        time_start = time.time()
        self.total_strategy_number = 11
        self.k_max = k_max
        self.double_sorting_groups = 3
        self.double_sorting_subgroups = 2
        self.signal_moving_avg = 5
        self.k_range, self.k_list = self._build_k_list()
        self.ini_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        # Initial date for getting prices/mkt data
        self.ini_date_prices = pd.to_datetime('1989-01-01')
        # Dates in BBG format (string)
        self.ini_date_bbg = datetime.strftime(self.ini_date, '%Y%m%d')
        self.end_date_bbg = datetime.strftime(self.end_date, '%Y%m%d')
        self.ini_date_prices_bbg = datetime.strftime(self.ini_date_prices, '%Y%m%d')

        # Calendars
        # Celendar for strategy
        self.daily_calendar = self._build_bday_calendars(self.ini_date, self.end_date)
        self.monthly_calendar = self._build_monthly_calendars(self.ini_date, self.end_date)
        # Calendar for prices / mkt data
        self.daily_calendar_extended = self._build_bday_calendars(self.ini_date_prices, self.end_date)
        # self.list_daily_pnl_dates = pd.DataFrame(index=self.daily_calendar).loc['1991-01-31':].index


    def run_constructor_in_parts1(self):
        time_start = time.time()
        # Tickers
        self.ticker_spot_bbg = [c + ' Curncy' for c in self.currency_list]
        # Conversion dictionaries
        self.dict_FX_spot = dict(zip(self.currency_list, self.ticker_spot_bbg))
        self.dict_spot_FX = dict(zip(self.ticker_spot_bbg, self.currency_list))
        self.dict_NDF_FX = dict(zip(self.dict_FX_NDF.values(), self.dict_FX_NDF.keys()))
        self.df_strategy = self.build_strategy_table()
        try:
            # Try to read df_tickers.xlsx, otherwise get from BBG and write to Excel
            df_tickers = pd.read_excel(self.data_folder + '\\' + 'df_tickers.xlsx')
            self.df_tickers = df_tickers
        except FileNotFoundError:
            self.df_tickers = self._build_df_tickers()
            self.df_tickers.to_excel(self.data_folder + '\\' + 'df_tickers.xlsx')
        self.ticker_fwdpts_1m = list(self.df_tickers['fwdpts'])
        self.ticker_scale = self.df_tickers['scale']
        self.ticker_inverse = self.df_tickers['inverse']
        try:
            # Try to read df_tickers_curve.xlsx, otherwise get from BBG and write to Excel
            df_tickers_curve = pd.read_excel(self.data_folder + '\\' + 'df_tickers_curve.xlsx', index_col=[0,1])
            self.df_tickers_curve = df_tickers_curve
        except FileNotFoundError:
            self.df_tickers_curve = self._build_df_tickers_curve()
            self.df_tickers_curve.to_excel(self.data_folder + '\\' + 'df_tickers_curve.xlsx')
        self.ticker_year_dict = self._build_dict_ticker_year()
        print('tickers OK - ', time.time() - time_start, ' seconds')
        time_start = time.time()

        # Prices
        # Last price
        time_start = time.time()
        try:
            # Try to read spot_last.xlsx, otherwise get from BBG and write to Excel
            df_spot_last = pd.read_excel(self.data_folder + '\\' + 'spot_last.xlsx')
            self.spot_last = df_spot_last
            # Adjusting DEM and EUR
            self.spot_last.loc['1998-12':]['DEM'] = np.NaN
            self.spot_last.loc[:'1998-12']['EUR'] = np.NaN
        except FileNotFoundError:
            self.spot_last = self._get_spot_data(self.bbg_field_last)
            # Adjusting DEM and EUR
            self.spot_last.loc['1998-12':]['DEM'] = np.NaN
            self.spot_last.loc[:'1998-12']['EUR'] = np.NaN
            self.spot_last.to_excel(self.data_folder + '\\' + 'spot_last.xlsx')
        print('spot_last OK - ', time.time() - time_start, ' seconds')
        # Bid and Ask of spot price
        time_start = time.time()
        try:
            # Try to read spot_bid.xlsx, otherwise get from BBG and write to Excel
            df_spot_bid = pd.read_excel(self.data_folder + '\\' + 'spot_bid.xlsx')
            df_spot_ask = pd.read_excel(self.data_folder + '\\' + 'spot_ask.xlsx')
            self.spot_bid = df_spot_bid
            self.spot_ask = df_spot_ask
        except FileNotFoundError:
            self.spot_bid = self._get_spot_data(self.bbg_field_bid)
            self.spot_ask = self._get_spot_data(self.bbg_field_ask)
            self.adjust_spot_bid_ask()
            self.spot_bid.to_excel(self.data_folder + '\\' + 'spot_bid.xlsx')
            self.spot_ask.to_excel(self.data_folder + '\\' + 'spot_ask.xlsx')
        print('spot_bid and ask OK - ', time.time() - time_start, ' seconds')
        # Fwd points last
        time_start = time.time()
        try:
            # Try to read fwdpts_last.xlsx, otherwise get from BBG and write to Excel
            df_fwdpts_last = pd.read_excel(self.data_folder + '\\' + 'fwdpts_last.xlsx')
            self.fwdpts_last = df_fwdpts_last
            # Adjusting DEM and EUR
            self.fwdpts_last.loc['1998-12':]['DEM'] = np.NaN
            self.fwdpts_last.loc[:'1998-12']['EUR'] = np.NaN
        except FileNotFoundError:
            self.fwdpts_last = self._get_fwdpts_data(self.bbg_field_last)
            # Adjusting DEM and EUR
            self.fwdpts_last.loc['1998-12':]['DEM'] = np.NaN
            self.fwdpts_last.loc[:'1998-12']['EUR'] = np.NaN
            self.fwdpts_last.to_excel(self.data_folder + '\\' + 'fwdpts_last.xlsx')
        print('fwdpts_last OK - ', time.time() - time_start, ' seconds')
        # Fwd points bid and ask
        time_start = time.time()
        try:
            # Try to read fwdpts_bid.xlsx, otherwise get from BBG and write to Excel
            # Try to read fwdpts_ask.xlsx, otherwise get from BBG and write to Excel
            df_fwdpts_bid = pd.read_excel(self.data_folder + '\\' + 'fwdpts_bid.xlsx')
            df_fwdpts_ask = pd.read_excel(self.data_folder + '\\' + 'fwdpts_ask.xlsx')
            self.fwdpts_bid = df_fwdpts_bid
            self.fwdpts_ask = df_fwdpts_ask
        except FileNotFoundError:
            self.fwdpts_bid = self._get_fwdpts_data(self.bbg_field_bid)
            self.fwdpts_ask = self._get_fwdpts_data(self.bbg_field_ask)
            self.adjust_fwdpts_bid_ask()
            self.fwdpts_bid.to_excel(self.data_folder + '\\' + 'fwdpts_bid.xlsx')
            self.fwdpts_ask.to_excel(self.data_folder + '\\' + 'fwdpts_ask.xlsx')
        print('fwdpts_bid and ask OK - ', time.time() - time_start, ' seconds')
        # Outright forward, last price
        time_start = time.time()
        self.fwd_last = self._get_fwd_outright(self.spot_last, self.fwdpts_last, self.ticker_scale)
        print('fwd_last OK - ', time.time() - time_start, ' seconds')
        # Outright forward, bid price
        time_start = time.time()
        self.fwd_bid = self._get_fwd_outright(self.spot_bid, self.fwdpts_bid, self.ticker_scale)
        print('fwd_bid OK - ', time.time() - time_start, ' seconds')
        # Outright forward, ask price
        time_start = time.time()
        self.fwd_ask = self._get_fwd_outright(self.spot_ask, self.fwdpts_ask, self.ticker_scale)
        print('fwd_ask OK - ', time.time() - time_start, ' seconds')
        # prices in XXXUSD terms
        time_start = time.time()
        self.spot_last_XXXUSD = self.spot_last ** self.ticker_inverse
        self.spot_bid_XXXUSD = self.spot_bid ** self.ticker_inverse
        self.spot_ask_XXXUSD = self.spot_ask ** self.ticker_inverse
        self.fwd_last_XXXUSD = self.fwd_last ** self.ticker_inverse
        self.fwd_bid_XXXUSD = self.fwd_bid ** self.ticker_inverse
        self.fwd_ask_XXXUSD = self.fwd_ask ** self.ticker_inverse
        self.fwd_discount_last = (self.spot_last_XXXUSD / self.fwd_last_XXXUSD - 1) * 12 # signal fwd premium
        self.tradeable_fx = (~np.isnan(self.fwd_discount_last))
        print('fwd_discount_last OK - ', time.time() - time_start, ' seconds')
        time_start = time.time()
        # self.vols = self._ewma_vol(ewma_lambda=0.94) #  discontinued after version 8
        self.vols = self._std_vol(window=126)
        self.adjust_vol()
        self.vols.to_excel(self.data_folder + '\\' + 'vols.xlsx')
        print('vol EWMA OK - ', time.time() - time_start, ' seconds')
        self.currency_list_EM = self.build_currency_list_EM()
        time_start = time.time()
        try:
            # Try to read USD_index.xlsx, otherwise calculate it and write to Excel
            df_USD_index = pd.read_excel(self.data_folder + '\\' + 'USD_index.xlsx')
            self.USD_index = df_USD_index
        except FileNotFoundError:
            self.USD_index = self.run_USD_index_all()
            self.USD_index.to_excel(self.data_folder + '\\' + 'USD_index.xlsx')
        print('USD index OK - ', time.time() - time_start, ' seconds')

    def run_constructor_in_parts2(self):
        time_start = time.time()
        try:
            # Try to read interest_curve.xlsx, otherwise get from BBG and write to Excel
            df_interest_curve = pd.read_excel(self.data_folder + '\\' + 'interest_curve.xlsx', index_col=[0,1])
            self.interest_curve = df_interest_curve
        except FileNotFoundError:
            self.interest_curve = self._get_all_interest_curves()
            if 'BRL' in self.currency_list:
                self.BZ_curve_data() # includes Bz curves
            if 'DE<' in self.currency_list:
                self.DEM_curve_data() # includes DEM curves
            self.interest_curve.to_excel(self.data_folder + '\\' + 'interest_curve.xlsx')
        print('interest_curve OK - ', time.time() - time_start, ' seconds')
        self.calendar_curves = list(self.interest_curve.index.get_level_values(1).unique()) # available dates (curves)


    def run_constructor_in_parts3(self):
        time_start = time.time()
        # self.nsiegel_betas_all_tenors = self._run_NSiegel_fitting()
        try:
            # Try to read nsiegel_betas_all_tenors.xlsx, otherwise get from BBG and write to Excel
            df_nsiegel_betas = pd.read_excel(self.data_folder + '\\' + 'nsiegel_betas_all_tenors.xlsx', index_col=[0, 1])
            self.nsiegel_betas_all_tenors = df_nsiegel_betas
        except FileNotFoundError:
            self.nsiegel_betas_all_tenors = self._run_NSiegel_fitting_sequential()
            self.nsiegel_betas_all_tenors.to_excel(self.data_folder + '\\' + 'nsiegel_betas_all_tenors.xlsx')
        print('NS-all tenors OK - ', time.time() - time_start, ' seconds')

    def run_constructor_in_parts4(self):
        time_start = time.time()
        try:
            # Try to read nsiegel_betas_3month.xlsx, otherwise get from BBG and write to Excel
            df_nsiegel_betas = pd.read_excel(self.data_folder + '\\' + 'nsiegel_betas_3month.xlsx', index_col=[0,1])
            self.nsiegel_betas_3month = df_nsiegel_betas
        except FileNotFoundError:
            self.nsiegel_betas_3month = self._run_NSiegel_fitting_sequential(tenors_greater_than_n_years=0.25)
            # filling NaN for BRL series
            if 'BRL' in self.currency_list:
                nsiegel_betas_3month_aux = self.nsiegel_betas_3month.loc['BRL'].fillna(method='ffill', limit=3)
                dates_list = self.nsiegel_betas_3month.loc['BRL'].index
                for dt in dates_list:
                    self.nsiegel_betas_3month.loc['BRL', dt] = nsiegel_betas_3month_aux.loc[dt]
            if 'DEM' in self.currency_list:
                nsiegel_betas_3month_aux = self.nsiegel_betas_3month.loc['DEM'].fillna(method='ffill', limit=3)
                dates_list = self.nsiegel_betas_3month.loc['DEM'].index
                for dt in dates_list:
                    self.nsiegel_betas_3month.loc['DEM', dt] = nsiegel_betas_3month_aux.loc[dt]
            self.nsiegel_betas_3month.to_excel(self.data_folder + '\\' + 'nsiegel_betas_3month.xlsx')
        print('NS-from 3 month tenor OK - ', time.time() - time_start, ' seconds')

    def run_constructor_in_parts5(self):
        time_start = time.time()
        # self.curvature_all_tenors = self._get_df_specific_beta(df_betas=self.nsiegel_betas_all_tenors, beta='b3')
        self.curvature_from_three_month_tenor = self._get_df_specific_beta(df_betas=self.nsiegel_betas_3month, beta='b3')
        self.level_from_three_month_tenor = self._get_df_specific_beta(df_betas=self.nsiegel_betas_3month, beta='b1')
        self.slope_from_three_month_tenor = self._get_df_specific_beta(df_betas=self.nsiegel_betas_3month, beta='b2')
        print('betas OK - ', time.time() - time_start, ' seconds')
        time_start = time.time()
        self.relative_level = self._relative_level()  # signal relative level (N-Siegel)
        # Adjusting DEM and EUR
        self.relative_level.loc['1998-12':]['DEM'] = np.NaN
        self.relative_level.loc[:'1998-12']['EUR'] = np.NaN
        print('relative level OK - ', time.time() - time_start, ' seconds')
        time_start = time.time()
        self.relative_slope = self._relative_slope()  # signal relative slope (N-Siegel)
        # Adjusting DEM and EUR
        self.relative_slope.loc['1998-12':]['DEM'] = np.NaN
        self.relative_slope.loc[:'1998-12']['EUR'] = np.NaN
        self.relative_curvature = self._relative_curvature() # signal relative curvature (N-Siegel)
        # Adjusting DEM and EUR
        self.relative_curvature.loc['1998-12':]['DEM'] = np.NaN
        self.relative_curvature.loc[:'1998-12']['EUR'] = np.NaN
        # self.stdev_curvature = self.standard_deviation_curvature()
        try:
            # Try to read daily_fwdpts.xlsx, otherwise get from BBG and write to Excel
            df_daily_fwdpts = pd.read_csv(self.data_folder + '\\' + 'daily_fwdpts.csv', index_col=[0, 1], header=0, names=[float(x) for x in range(0, 182)])
            df_daily_fwdpts.index = pd.MultiIndex.from_product(iterables=[self.currency_list, self.daily_calendar])
            # df_daily_fwdpts = pd.read_excel(self.data_folder + '\\' + 'daily_fwdpts.xlsx', index_col=[0,1])
            self.daily_fwdpts = df_daily_fwdpts
        except FileNotFoundError:
            self.daily_fwdpts = self._get_daily_fwdpts_curve_data() # scale adjusted fwd pts
            self.daily_fwdpts.to_excel(self.data_folder + '\\' + 'daily_fwdpts.xlsx')
            self.daily_fwdpts.to_csv(self.data_folder + '\\' + 'daily_fwdpts.CSV')
        print('daily fwdpts OK - ', time.time() - time_start, ' seconds')
        time_start = time.time()
        self.daily_fwds = self._get_daily_fwd_all_tenors(self.daily_fwdpts) # all tenors (FX spot + forwards)
        self.daily_fwds_XXXUSD = self._get_daily_fwd_XXXUSD(self.daily_fwds)
        try:
            # Try to read px_change_matrix.xlsx, otherwise calculate and write to Excel
            df_px_change_matrix = pd.read_excel(self.data_folder + '\\' + 'px_change_matrix.xlsx')
            self.px_change_matrix = df_px_change_matrix
        except FileNotFoundError:
            self.px_change_matrix = self._price_change_matrix()
            self.px_change_matrix.to_excel(self.data_folder + '\\' + 'px_change_matrix.xlsx')
        print('daily prices OK - ', time.time() - time_start, ' seconds')

        # Implementation of rankings
        time_start = time.time()
        try:
            self.fwd_discount_rank          = pd.read_excel(self.data_folder + '\\' + 'fwd_discount_rank.xlsx')
            self.relative_level_rank        = pd.read_excel(self.data_folder + '\\' + 'relative_level_rank.xlsx')
            self.relative_slope_rank        = pd.read_excel(self.data_folder + '\\' + 'relative_slope_rank.xlsx')
            self.relative_curvature_rank    = pd.read_excel(self.data_folder + '\\' + 'relative_curvature_rank.xlsx')
        except FileNotFoundError:
            self.fwd_discount_rank = self._simple_ranking(df_signals=self.fwd_discount_last)
            self.relative_level_rank = self._simple_ranking(df_signals=self.relative_level)
            self.relative_slope_rank = self._simple_ranking(df_signals=self.relative_slope)
            self.relative_curvature_rank = self._simple_ranking(df_signals=self.relative_curvature)
            self.fwd_discount_rank.to_excel(self.data_folder + '\\' + 'fwd_discount_rank.xlsx')
            self.relative_level_rank.to_excel(self.data_folder + '\\' + 'relative_level_rank.xlsx')
            self.relative_slope_rank.to_excel(self.data_folder + '\\' + 'relative_slope_rank.xlsx')
            self.relative_curvature_rank.to_excel(self.data_folder + '\\' + 'relative_curvature_rank.xlsx')
        print('ranks OK - ', time.time() - time_start, ' seconds')

    def build_strategy_table(self):
        list_strat_number = list(range(1, self.total_strategy_number + 1))
        df_strategy = pd.DataFrame(data=None, index=list_strat_number)
        strategy_name_dict = {1: 'Carry trade',
                              2: 'Curvature',
                              3: 'Level',
                              4: 'Slope',
                              5: 'Carry (EV)',
                              6: 'Curvature (EV)',
                              7: 'Level (EV)',
                              8: 'Slope (EV)',
                              9: 'Carry-Curvy DS',
                              10: 'Curvy-Carry DS',
                              11: 'Carry to Vol'}

        df_strategy['Name'] = [strategy_name_dict[x] for x in list_strat_number]
        df_strategy['TR'] = ['TR_df_' + "{:02}".format(x) for x in list_strat_number]
        df_strategy['holdings'] = ['holdings_df_' + "{:02}".format(x) for x in list_strat_number]
        df_strategy['weights'] = ['weights_df_' + "{:02}".format(x) for x in list_strat_number]
        df_strategy['long_short_signals'] = ['long_short_signals_df_' + "{:02}".format(x) for x in list_strat_number]
        df_strategy['TR_df_daily'] = ['TR_df_daily_' + "{:02}".format(x) for x in list_strat_number]
        return df_strategy


    def run_strategy01(self):
        time_start = time.time()
        # Traditional Carry Trade
        strategy_n = 1
        name_start = self.data_folder + "\\" + "{:02}".format(strategy_n) + '_'
        strategy_criteria = self.fwd_discount_last
        try:
            # Try to read results from Excel files, otherwise run and write to Excel
            df_excel_tr         = pd.read_excel(name_start + 'TR.xlsx')
            df_excel_holdings   = pd.read_excel(name_start + 'Holdings.xlsx', index_col=[0,1])
            df_excel_weights    = pd.read_excel(name_start + 'Weights.xlsx', index_col=[0,1])
            df_excel_ls_signals = pd.read_excel(name_start + 'LS_Signals.xlsx', index_col=[0,1])
            df_excel_ir         = pd.read_excel(name_start + 'IR.xlsx')
            df_excel_sr         = pd.read_excel(name_start + 'SR.xlsx')
            df_excel_cost       = pd.read_excel(name_start + 'Cost.xlsx')
            df_excel_pnlfx      = pd.read_excel(name_start + 'PctPnlFX.xlsx', index_col=[0,1])
            df_excel_irfx       = pd.read_excel(name_start + 'PctIRFX.xlsx', index_col=[0,1])
            df_excel_srfx       = pd.read_excel(name_start + 'PctSRFX.xlsx', index_col=[0,1])
            df_excel_costfx     = pd.read_excel(name_start + 'PctCostFX.xlsx', index_col=[0,1])
            dt_excel_daily_pnl  = pd.read_excel(name_start + 'TR_Daily.xlsx')
            self.TR_df_01                    = df_excel_tr
            self.holdings_df_01              = df_excel_holdings
            self.weights_df_01               = df_excel_weights
            self.long_short_signals_df_01    = df_excel_ls_signals
            self.ir_01                       = df_excel_ir
            self.sr_01                       = df_excel_sr
            self.cost_01                     = df_excel_cost
            self.pct_pnlfx_01                = df_excel_pnlfx
            self.pct_irfx_01                 = df_excel_irfx
            self.pct_srfx_01                 = df_excel_srfx
            self.pct_costfx_01               = df_excel_costfx
            self.TR_df_daily_01              = dt_excel_daily_pnl
        except FileNotFoundError:
            # this function already writes results do excel (just need strategy_n): self.run_equal_vol_monthly_strategy
            [self.TR_df_01, self.holdings_df_01, self.weights_df_01, self.long_short_signals_df_01, self.ir_01, self.sr_01,
             self.cost_01, self.pct_pnlfx_01, self.pct_irfx_01, self.pct_srfx_01, self.pct_costfx_01, self.blotter_01] = \
                self.run_default_monthly_strategy(df_signals=strategy_criteria, func_weight=self._ranking_to_wgt, strategy_n=strategy_n)
            # self.TR_df_daily_01 = self.run_daily_pnl(self.holdings_df_01, strategy_n)
            [self.TR_df_daily_01, self.TR_df_daily_fx_01] = self.calculate_daily_pnl(df_blotter=self.blotter_01)
            (self.TR_df_daily_01.loc['1991-01-31':]+100).to_excel(name_start + 'TR_Daily.xlsx')
            (self.TR_df_daily_fx_01 + 100).to_excel(name_start + 'TR_DailyFX.xlsx')
        print('Carry trade OK - ', time.time() - time_start, ' seconds')

    def run_strategy02(self):
        time_start = time.time()
        # Traditional CURVY Trade (Nelson-Siegel curvature)
        strategy_n = 2
        name_start = self.data_folder + "\\" + "{:02}".format(strategy_n) + '_'
        strategy_criteria = self.relative_curvature
        try:
            # Try to read results from Excel files, otherwise run and write to Excel
            df_excel_tr         = pd.read_excel(name_start + 'TR.xlsx')
            df_excel_holdings   = pd.read_excel(name_start + 'Holdings.xlsx', index_col=[0,1])
            df_excel_weights    = pd.read_excel(name_start + 'Weights.xlsx', index_col=[0,1])
            df_excel_ls_signals = pd.read_excel(name_start + 'LS_Signals.xlsx', index_col=[0,1])
            df_excel_ir         = pd.read_excel(name_start + 'IR.xlsx')
            df_excel_sr         = pd.read_excel(name_start + 'SR.xlsx')
            df_excel_cost       = pd.read_excel(name_start + 'Cost.xlsx')
            df_excel_pnlfx      = pd.read_excel(name_start + 'PctPnlFX.xlsx', index_col=[0,1])
            df_excel_irfx       = pd.read_excel(name_start + 'PctIRFX.xlsx', index_col=[0,1])
            df_excel_srfx       = pd.read_excel(name_start + 'PctSRFX.xlsx', index_col=[0,1])
            df_excel_costfx     = pd.read_excel(name_start + 'PctCostFX.xlsx', index_col=[0,1])
            dt_excel_daily_pnl  = pd.read_excel(name_start + 'TR_Daily.xlsx')
            self.TR_df_02                    = df_excel_tr
            self.holdings_df_02              = df_excel_holdings
            self.weights_df_02               = df_excel_weights
            self.long_short_signals_df_02    = df_excel_ls_signals
            self.ir_02                       = df_excel_ir
            self.sr_02                       = df_excel_sr
            self.cost_02                     = df_excel_cost
            self.pct_pnlfx_02                = df_excel_pnlfx
            self.pct_irfx_02                 = df_excel_irfx
            self.pct_srfx_02                 = df_excel_srfx
            self.pct_costfx_02               = df_excel_costfx
            self.TR_df_daily_02              = dt_excel_daily_pnl
        except FileNotFoundError:
            # this function already writes results do excel (just need strategy_n): self.run_equal_vol_monthly_strategy
            [self.TR_df_02, self.holdings_df_02, self.weights_df_02, self.long_short_signals_df_02, self.ir_02, self.sr_02,
             self.cost_02, self.pct_pnlfx_02, self.pct_irfx_02, self.pct_srfx_02, self.pct_costfx_02, self.blotter_02] = \
                self.run_default_monthly_strategy(df_signals=strategy_criteria, func_weight=self._ranking_to_wgt, strategy_n=strategy_n)
            # self.TR_df_daily_02 = self.run_daily_pnl(self.holdings_df_02, strategy_n)
            [self.TR_df_daily_02, self.TR_df_daily_fx_02] = self.calculate_daily_pnl(df_blotter=self.blotter_02)
            (self.TR_df_daily_02.loc['1991-01-31':]+100).to_excel(name_start + 'TR_Daily.xlsx')
            (self.TR_df_daily_fx_02 + 100).to_excel(name_start + 'TR_DailyFX.xlsx')
        print('Curvy trade OK - ', time.time() - time_start, ' seconds')

    def run_strategy03(self):
        time_start = time.time()
        # Traditional level Trade (Nelson-Siegel level)
        strategy_n = 3
        name_start = self.data_folder + "\\" + "{:02}".format(strategy_n) + '_'
        strategy_criteria = self.relative_level
        try:
            # Try to read results from Excel files, otherwise run and write to Excel
            df_excel_tr         = pd.read_excel(name_start + 'TR.xlsx')
            df_excel_holdings   = pd.read_excel(name_start + 'Holdings.xlsx', index_col=[0,1])
            df_excel_weights    = pd.read_excel(name_start + 'Weights.xlsx', index_col=[0,1])
            df_excel_ls_signals = pd.read_excel(name_start + 'LS_Signals.xlsx', index_col=[0,1])
            df_excel_ir         = pd.read_excel(name_start + 'IR.xlsx')
            df_excel_sr         = pd.read_excel(name_start + 'SR.xlsx')
            df_excel_cost       = pd.read_excel(name_start + 'Cost.xlsx')
            df_excel_pnlfx      = pd.read_excel(name_start + 'PctPnlFX.xlsx', index_col=[0,1])
            df_excel_irfx       = pd.read_excel(name_start + 'PctIRFX.xlsx', index_col=[0,1])
            df_excel_srfx       = pd.read_excel(name_start + 'PctSRFX.xlsx', index_col=[0,1])
            df_excel_costfx     = pd.read_excel(name_start + 'PctCostFX.xlsx', index_col=[0,1])
            dt_excel_daily_pnl  = pd.read_excel(name_start + 'TR_Daily.xlsx')
            self.TR_df_03                    = df_excel_tr
            self.holdings_df_03              = df_excel_holdings
            self.weights_df_03               = df_excel_weights
            self.long_short_signals_df_03    = df_excel_ls_signals
            self.ir_03                       = df_excel_ir
            self.sr_03                       = df_excel_sr
            self.cost_03                     = df_excel_cost
            self.pct_pnlfx_03                = df_excel_pnlfx
            self.pct_irfx_03                 = df_excel_irfx
            self.pct_srfx_03                 = df_excel_srfx
            self.pct_costfx_03               = df_excel_costfx
            self.TR_df_daily_03              = dt_excel_daily_pnl
        except FileNotFoundError:
            # this function already writes results do excel (just need strategy_n): self.run_equal_vol_monthly_strategy
            [self.TR_df_03, self.holdings_df_03, self.weights_df_03, self.long_short_signals_df_03, self.ir_03, self.sr_03,
             self.cost_03, self.pct_pnlfx_03, self.pct_irfx_03, self.pct_srfx_03, self.pct_costfx_03, self.blotter_03] = \
                self.run_default_monthly_strategy(df_signals=strategy_criteria, func_weight=self._ranking_to_wgt, strategy_n=strategy_n)
            # self.TR_df_daily_03 = self.run_daily_pnl(self.holdings_df_03, strategy_n)
            [self.TR_df_daily_03, self.TR_df_daily_fx_03] = self.calculate_daily_pnl(df_blotter=self.blotter_03)
            (self.TR_df_daily_03.loc['1991-01-31':]+100).to_excel(name_start + 'TR_Daily.xlsx')
            (self.TR_df_daily_fx_03 + 100).to_excel(name_start + 'TR_DailyFX.xlsx')
        print('Level trade OK - ', time.time() - time_start, ' seconds')

    def run_strategy04(self):
        time_start = time.time()
        # Traditional slope Trade (Nelson-Siegel slope)
        strategy_n = 4
        name_start = self.data_folder + "\\" + "{:02}".format(strategy_n) + '_'
        strategy_criteria = self.relative_slope
        try:
            # Try to read results from Excel files, otherwise run and write to Excel
            df_excel_tr         = pd.read_excel(name_start + 'TR.xlsx')
            df_excel_holdings   = pd.read_excel(name_start + 'Holdings.xlsx', index_col=[0,1])
            df_excel_weights    = pd.read_excel(name_start + 'Weights.xlsx', index_col=[0,1])
            df_excel_ls_signals = pd.read_excel(name_start + 'LS_Signals.xlsx', index_col=[0,1])
            df_excel_ir         = pd.read_excel(name_start + 'IR.xlsx')
            df_excel_sr         = pd.read_excel(name_start + 'SR.xlsx')
            df_excel_cost       = pd.read_excel(name_start + 'Cost.xlsx')
            df_excel_pnlfx      = pd.read_excel(name_start + 'PctPnlFX.xlsx', index_col=[0,1])
            df_excel_irfx       = pd.read_excel(name_start + 'PctIRFX.xlsx', index_col=[0,1])
            df_excel_srfx       = pd.read_excel(name_start + 'PctSRFX.xlsx', index_col=[0,1])
            df_excel_costfx     = pd.read_excel(name_start + 'PctCostFX.xlsx', index_col=[0,1])
            dt_excel_daily_pnl  = pd.read_excel(name_start + 'TR_Daily.xlsx')
            self.TR_df_04                    = df_excel_tr
            self.holdings_df_04              = df_excel_holdings
            self.weights_df_04               = df_excel_weights
            self.long_short_signals_df_04    = df_excel_ls_signals
            self.ir_04                       = df_excel_ir
            self.sr_04                       = df_excel_sr
            self.cost_04                     = df_excel_cost
            self.pct_pnlfx_04                = df_excel_pnlfx
            self.pct_irfx_04                 = df_excel_irfx
            self.pct_srfx_04                 = df_excel_srfx
            self.pct_costfx_04               = df_excel_costfx
            self.TR_df_daily_04              = dt_excel_daily_pnl
        except FileNotFoundError:
            # this function already writes results do excel (just need strategy_n): self.run_equal_vol_monthly_strategy
            [self.TR_df_04, self.holdings_df_04, self.weights_df_04, self.long_short_signals_df_04, self.ir_04, self.sr_04,
             self.cost_04, self.pct_pnlfx_04, self.pct_irfx_04, self.pct_srfx_04, self.pct_costfx_04, self.blotter_04] = \
                self.run_default_monthly_strategy(df_signals=strategy_criteria, func_weight=self._ranking_to_wgt, strategy_n=strategy_n)
            # self.TR_df_daily_04 = self.run_daily_pnl(self.holdings_df_04, strategy_n)
            [self.TR_df_daily_04, self.TR_df_daily_fx_04] = self.calculate_daily_pnl(df_blotter=self.blotter_04)
            (self.TR_df_daily_04.loc['1991-01-31':]+100).to_excel(name_start + 'TR_Daily.xlsx')
            (self.TR_df_daily_fx_04 + 100).to_excel(name_start + 'TR_DailyFX.xlsx')
        print('Slope trade OK - ', time.time() - time_start, ' seconds')

    def run_strategy05(self):
        time_start = time.time()
        # Equal volatility Carry Trade (each FX position is sized to have the same vol)
        # Target vol used is 9%. A good guess to get strategy volatility that is similar to traditional carry trade.
        strategy_n = 5
        self.carry_trade_target_vol = 0.09
        name_start = self.data_folder + "\\" + "{:02}".format(strategy_n) + '_'
        strategy_criteria = self.fwd_discount_last
        try:
            # Try to read results from Excel files, otherwise run and write to Excel
            df_excel_tr         = pd.read_excel(name_start + 'TR.xlsx')
            df_excel_holdings   = pd.read_excel(name_start + 'Holdings.xlsx', index_col=[0,1])
            df_excel_weights    = pd.read_excel(name_start + 'Weights.xlsx', index_col=[0,1])
            df_excel_ls_signals = pd.read_excel(name_start + 'LS_Signals.xlsx', index_col=[0,1])
            df_excel_ir         = pd.read_excel(name_start + 'IR.xlsx')
            df_excel_sr         = pd.read_excel(name_start + 'SR.xlsx')
            df_excel_cost       = pd.read_excel(name_start + 'Cost.xlsx')
            df_excel_pnlfx      = pd.read_excel(name_start + 'PctPnlFX.xlsx', index_col=[0,1])
            df_excel_irfx       = pd.read_excel(name_start + 'PctIRFX.xlsx', index_col=[0,1])
            df_excel_srfx       = pd.read_excel(name_start + 'PctSRFX.xlsx', index_col=[0,1])
            df_excel_costfx     = pd.read_excel(name_start + 'PctCostFX.xlsx', index_col=[0,1])
            dt_excel_daily_pnl  = pd.read_excel(name_start + 'TR_Daily.xlsx')
            self.TR_df_05                    = df_excel_tr
            self.holdings_df_05              = df_excel_holdings
            self.weights_df_05               = df_excel_weights
            self.long_short_signals_df_05    = df_excel_ls_signals
            self.ir_05                       = df_excel_ir
            self.sr_05                       = df_excel_sr
            self.cost_05                     = df_excel_cost
            self.pct_pnlfx_05                = df_excel_pnlfx
            self.pct_irfx_05                 = df_excel_irfx
            self.pct_srfx_05                 = df_excel_srfx
            self.pct_costfx_05               = df_excel_costfx
            self.TR_df_daily_05              = dt_excel_daily_pnl
        except FileNotFoundError:
            # this function already writes results do excel (just need strategy_n): self.run_equal_vol_monthly_strategy
            [self.TR_df_05, self.holdings_df_05, self.weights_df_05, self.long_short_signals_df_05, self.ir_05, self.sr_05,
             self.cost_05, self.pct_pnlfx_05, self.pct_irfx_05, self.pct_srfx_05, self.pct_costfx_05, self.blotter_05] = \
                self.run_equal_vol_monthly_strategy(df_signals=strategy_criteria, target_vol=self.carry_trade_target_vol, strategy_n=strategy_n)
            # self.TR_df_daily_05 = self.run_daily_pnl(self.holdings_df_05, strategy_n)
            [self.TR_df_daily_05, self.TR_df_daily_fx_05] = self.calculate_daily_pnl(df_blotter=self.blotter_05)
            (self.TR_df_daily_05.loc['1991-01-31':]+100).to_excel(name_start + 'TR_Daily.xlsx')
            (self.TR_df_daily_fx_05 + 100).to_excel(name_start + 'TR_DailyFX.xlsx')
        print('Equal vol Carry trade OK - ', time.time() - time_start, ' seconds')

    def run_strategy06(self):
        time_start = time.time()
        # Equal vol CURVY Trade
        strategy_n = 6
        self.curvy_trade_target_vol = 0.09
        name_start = self.data_folder + "\\" + "{:02}".format(strategy_n) + '_'
        strategy_criteria = self.relative_curvature
        try:
            # Try to read results from Excel files, otherwise run and write to Excel
            df_excel_tr         = pd.read_excel(name_start + 'TR.xlsx')
            df_excel_holdings   = pd.read_excel(name_start + 'Holdings.xlsx', index_col=[0,1])
            df_excel_weights    = pd.read_excel(name_start + 'Weights.xlsx', index_col=[0,1])
            df_excel_ls_signals = pd.read_excel(name_start + 'LS_Signals.xlsx', index_col=[0,1])
            df_excel_ir         = pd.read_excel(name_start + 'IR.xlsx')
            df_excel_sr         = pd.read_excel(name_start + 'SR.xlsx')
            df_excel_cost       = pd.read_excel(name_start + 'Cost.xlsx')
            df_excel_pnlfx      = pd.read_excel(name_start + 'PctPnlFX.xlsx', index_col=[0,1])
            df_excel_irfx       = pd.read_excel(name_start + 'PctIRFX.xlsx', index_col=[0,1])
            df_excel_srfx       = pd.read_excel(name_start + 'PctSRFX.xlsx', index_col=[0,1])
            df_excel_costfx     = pd.read_excel(name_start + 'PctCostFX.xlsx', index_col=[0,1])
            dt_excel_daily_pnl  = pd.read_excel(name_start + 'TR_Daily.xlsx')
            self.TR_df_06                    = df_excel_tr
            self.holdings_df_06              = df_excel_holdings
            self.weights_df_06               = df_excel_weights
            self.long_short_signals_df_06    = df_excel_ls_signals
            self.ir_06                       = df_excel_ir
            self.sr_06                       = df_excel_sr
            self.cost_06                     = df_excel_cost
            self.pct_pnlfx_06                = df_excel_pnlfx
            self.pct_irfx_06                 = df_excel_irfx
            self.pct_srfx_06                 = df_excel_srfx
            self.pct_costfx_06               = df_excel_costfx
            self.TR_df_daily_06              = dt_excel_daily_pnl
        except FileNotFoundError:
            # this function already writes results do excel (just need strategy_n): self.run_equal_vol_monthly_strategy
            [self.TR_df_06, self.holdings_df_06, self.weights_df_06, self.long_short_signals_df_06, self.ir_06, self.sr_06,
             self.cost_06, self.pct_pnlfx_06, self.pct_irfx_06, self.pct_srfx_06, self.pct_costfx_06, self.blotter_06] = \
                self.run_equal_vol_monthly_strategy(df_signals=strategy_criteria, target_vol=self.carry_trade_target_vol, strategy_n=strategy_n)
            # self.TR_df_daily_06 = self.run_daily_pnl(self.holdings_df_06, strategy_n)
            [self.TR_df_daily_06, self.TR_df_daily_fx_06] = self.calculate_daily_pnl(df_blotter=self.blotter_06)
            (self.TR_df_daily_06.loc['1991-01-31':]+100).to_excel(name_start + 'TR_Daily.xlsx')
            (self.TR_df_daily_fx_06 + 100).to_excel(name_start + 'TR_DailyFX.xlsx')
        print('Equal vol Curvy trade OK - ', time.time() - time_start, ' seconds')

    def run_strategy07(self):
        time_start = time.time()
        # Equal vol level Trade
        strategy_n = 7
        self.level_trade_target_vol = 0.09
        name_start = self.data_folder + "\\" + "{:02}".format(strategy_n) + '_'
        strategy_criteria = self.relative_level
        try:
            # Try to read results from Excel files, otherwise run and write to Excel
            df_excel_tr         = pd.read_excel(name_start + 'TR.xlsx')
            df_excel_holdings   = pd.read_excel(name_start + 'Holdings.xlsx', index_col=[0,1])
            df_excel_weights    = pd.read_excel(name_start + 'Weights.xlsx', index_col=[0,1])
            df_excel_ls_signals = pd.read_excel(name_start + 'LS_Signals.xlsx', index_col=[0,1])
            df_excel_ir         = pd.read_excel(name_start + 'IR.xlsx')
            df_excel_sr         = pd.read_excel(name_start + 'SR.xlsx')
            df_excel_cost       = pd.read_excel(name_start + 'Cost.xlsx')
            df_excel_pnlfx      = pd.read_excel(name_start + 'PctPnlFX.xlsx', index_col=[0,1])
            df_excel_irfx       = pd.read_excel(name_start + 'PctIRFX.xlsx', index_col=[0,1])
            df_excel_srfx       = pd.read_excel(name_start + 'PctSRFX.xlsx', index_col=[0,1])
            df_excel_costfx     = pd.read_excel(name_start + 'PctCostFX.xlsx', index_col=[0,1])
            dt_excel_daily_pnl  = pd.read_excel(name_start + 'TR_Daily.xlsx')
            self.TR_df_07                    = df_excel_tr
            self.holdings_df_07              = df_excel_holdings
            self.weights_df_07               = df_excel_weights
            self.long_short_signals_df_07    = df_excel_ls_signals
            self.ir_07                       = df_excel_ir
            self.sr_07                       = df_excel_sr
            self.cost_07                     = df_excel_cost
            self.pct_pnlfx_07                = df_excel_pnlfx
            self.pct_irfx_07                 = df_excel_irfx
            self.pct_srfx_07                 = df_excel_srfx
            self.pct_costfx_07               = df_excel_costfx
            self.TR_df_daily_07              = dt_excel_daily_pnl
        except FileNotFoundError:
            # this function already writes results do excel (just need strategy_n): self.run_equal_vol_monthly_strategy
            [self.TR_df_07, self.holdings_df_07, self.weights_df_07, self.long_short_signals_df_07, self.ir_07, self.sr_07,
             self.cost_07, self.pct_pnlfx_07, self.pct_irfx_07, self.pct_srfx_07, self.pct_costfx_07, self.blotter_07] = \
                self.run_equal_vol_monthly_strategy(df_signals=strategy_criteria, target_vol=self.carry_trade_target_vol, strategy_n=strategy_n)
            # self.TR_df_daily_07 = self.run_daily_pnl(self.holdings_df_07, strategy_n)
            [self.TR_df_daily_07, self.TR_df_daily_fx_07] = self.calculate_daily_pnl(df_blotter=self.blotter_07)
            (self.TR_df_daily_07.loc['1991-01-31':]+100).to_excel(name_start + 'TR_Daily.xlsx')
            (self.TR_df_daily_fx_07 + 100).to_excel(name_start + 'TR_DailyFX.xlsx')
        print('Equal vol Level trade OK - ', time.time() - time_start, ' seconds')

    def run_strategy08(self):
        time_start = time.time()
        # Equal vol slope Trade
        strategy_n = 8
        self.slope_trade_target_vol = 0.09
        name_start = self.data_folder + "\\" + "{:02}".format(strategy_n) + '_'
        strategy_criteria = self.relative_slope
        try:
            # Try to read results from Excel files, otherwise run and write to Excel
            df_excel_tr         = pd.read_excel(name_start + 'TR.xlsx')
            df_excel_holdings   = pd.read_excel(name_start + 'Holdings.xlsx', index_col=[0,1])
            df_excel_weights    = pd.read_excel(name_start + 'Weights.xlsx', index_col=[0,1])
            df_excel_ls_signals = pd.read_excel(name_start + 'LS_Signals.xlsx', index_col=[0,1])
            df_excel_ir         = pd.read_excel(name_start + 'IR.xlsx')
            df_excel_sr         = pd.read_excel(name_start + 'SR.xlsx')
            df_excel_cost       = pd.read_excel(name_start + 'Cost.xlsx')
            df_excel_pnlfx      = pd.read_excel(name_start + 'PctPnlFX.xlsx', index_col=[0,1])
            df_excel_irfx       = pd.read_excel(name_start + 'PctIRFX.xlsx', index_col=[0,1])
            df_excel_srfx       = pd.read_excel(name_start + 'PctSRFX.xlsx', index_col=[0,1])
            df_excel_costfx     = pd.read_excel(name_start + 'PctCostFX.xlsx', index_col=[0,1])
            dt_excel_daily_pnl  = pd.read_excel(name_start + 'TR_Daily.xlsx')
            self.TR_df_08                    = df_excel_tr
            self.holdings_df_08              = df_excel_holdings
            self.weights_df_08               = df_excel_weights
            self.long_short_signals_df_08    = df_excel_ls_signals
            self.ir_08                       = df_excel_ir
            self.sr_08                       = df_excel_sr
            self.cost_08                     = df_excel_cost
            self.pct_pnlfx_08                = df_excel_pnlfx
            self.pct_irfx_08                 = df_excel_irfx
            self.pct_srfx_08                 = df_excel_srfx
            self.pct_costfx_08               = df_excel_costfx
            self.TR_df_daily_08              = dt_excel_daily_pnl
        except FileNotFoundError:
            # this function already writes results do excel (just need strategy_n): self.run_equal_vol_monthly_strategy
            [self.TR_df_08, self.holdings_df_08, self.weights_df_08, self.long_short_signals_df_08, self.ir_08, self.sr_08,
             self.cost_08, self.pct_pnlfx_08, self.pct_irfx_08, self.pct_srfx_08, self.pct_costfx_08, self.blotter_08] = \
                self.run_equal_vol_monthly_strategy(df_signals=strategy_criteria, target_vol=self.carry_trade_target_vol, strategy_n=strategy_n)
            # self.TR_df_daily_08 = self.run_daily_pnl(self.holdings_df_08, strategy_n)
            [self.TR_df_daily_08, self.TR_df_daily_fx_08] = self.calculate_daily_pnl(df_blotter=self.blotter_08)
            (self.TR_df_daily_08.loc['1991-01-31':]+100).to_excel(name_start + 'TR_Daily.xlsx')
            (self.TR_df_daily_fx_08 + 100).to_excel(name_start + 'TR_DailyFX.xlsx')
        print('Equal vol Slope trade OK - ', time.time() - time_start, ' seconds')

    def run_constructor_in_parts6(self):
        print('Starting strategies')
        [getattr(self, 'run_strategy' + "{:02}".format(strategy_n))() for strategy_n in range(1, self.total_strategy_number + 1)]


    def _build_k_list(self):
        k_range = range(1, self.k_max + 1)
        k_list = ['k' + str(i) for i in k_range]
        return k_range, k_list

    @staticmethod
    def _build_bday_calendars(start_date, end_date):
        # TODO DONE: work with all business days, repeating prices (ffill) and rebalancing beggining of month.
        daily_calendar = pd.date_range(start=start_date, end=end_date, freq=BDay())
        return daily_calendar

    @staticmethod
    def _build_monthly_calendars(start_date, end_date):
        # TODO DONE: work with all business days, repeating prices (ffill) and rebalancing beggining of month.
        monthly_calendar = pd.date_range(start=start_date, end=end_date, freq=BMonthEnd())
        return monthly_calendar

    def _build_df_tickers(self):
        cod_fwdpts = [self.dict_FX_NDF[c] + '1M BGN Curncy' for c in self.currency_list]
        df_tickers = pd.DataFrame(index=self.currency_list, data=self.ticker_spot_bbg, columns=['spot'])
        df_tickers['fwdpts'] = cod_fwdpts
        # only for DEM, replace BGN for CMPN
        try:
            df_tickers.loc['DEM', 'fwdpts'] = df_tickers.loc['DEM', 'fwdpts'].replace('BGN', 'CMPN')
        except KeyError:
            print('Warning: DEM is not in currency list.')
        # scale column
        df_scale = self.con.ref(list(self.ticker_spot_bbg), 'FWD_SCALE').set_index(keys='ticker', drop=True)
        df_scale.index = [x.replace(' Curncy', '') for x in df_scale.index]
        df_tickers['scale'] = 10 ** df_scale['value']
        # inverse column
        df_inverse = self.con.ref(self.ticker_spot_bbg, 'INVERSE_QUOTED').set_index(keys='ticker', drop=True)
        df_inverse.index = [x.replace(' Curncy', '') for x in df_inverse.index]
        df_inverse.loc[df_inverse['value'] == "Y", 'value'] = 1
        df_inverse.loc[df_inverse['value'] == "N", 'value'] = -1
        df_tickers['inverse'] = df_inverse['value']
        # Forward curve tickers
        df_tickers['1w'] = [self.dict_FX_NDF[c] + '1W ' + self.dict_fwdpts_source[c] + ' Curncy' for c in self.currency_list]
        df_tickers['2w'] = [self.dict_FX_NDF[c] + '2W ' + self.dict_fwdpts_source[c] + ' Curncy' for c in self.currency_list]
        df_tickers['3w'] = [self.dict_FX_NDF[c] + '3W ' + self.dict_fwdpts_source[c] + ' Curncy' for c in self.currency_list]
        df_tickers['1m'] = [self.dict_FX_NDF[c] + '1M ' + self.dict_fwdpts_source[c] + ' Curncy' for c in self.currency_list]
        df_tickers['2m'] = [self.dict_FX_NDF[c] + '2M ' + self.dict_fwdpts_source[c] + ' Curncy' for c in self.currency_list]
        df_tickers['3m'] = [self.dict_FX_NDF[c] + '3M ' + self.dict_fwdpts_source[c] + ' Curncy' for c in self.currency_list]
        df_tickers['4m'] = [self.dict_FX_NDF[c] + '4M ' + self.dict_fwdpts_source[c] + ' Curncy' for c in self.currency_list]
        df_tickers['5m'] = [self.dict_FX_NDF[c] + '5M ' + self.dict_fwdpts_source[c] + ' Curncy' for c in self.currency_list]
        df_tickers['6m'] = [self.dict_FX_NDF[c] + '6M ' + self.dict_fwdpts_source[c] + ' Curncy' for c in self.currency_list]
        return df_tickers

    def _get_spot_data(self, bbg_field=['PX_LAST']):
        spot_last = self.con.bdh(self.ticker_spot_bbg, bbg_field, self.ini_date_prices_bbg, self.end_date_bbg, elms=[("periodicitySelection", self.bbg_periodicity_spot)])
        spot_last.columns = spot_last.columns.droplevel(1)
        spot_last.fillna(method='ffill', inplace=True)
        spot_last.columns = [x.replace(' Curncy', '') for x in spot_last.columns]
        spot_last.sort_index(axis=1, ascending=True, inplace=True)
        # DataFrame with extended calendar
        spot_last_extended_calendar = pd.DataFrame(data=None, index=self.daily_calendar_extended, columns=self.currency_list)
        _take_second = lambda s1, s2: s2
        spot_last_extended_calendar = spot_last_extended_calendar.combine(spot_last, func=_take_second)
        spot_last_extended_calendar.fillna(method='ffill', inplace=True)
        return spot_last_extended_calendar

    def adjust_spot_bid_ask(self):
        fx_side = self.df_tickers['inverse']
        _bid = self.spot_bid.copy(deep=True)
        _ask = self.spot_ask.copy(deep=True)
        for curncy in self.currency_list:
            if fx_side.loc[curncy] == -1:
                self.spot_bid[curncy] = _ask[curncy]
                self.spot_ask[curncy] = _bid[curncy]
        self.spot_last.loc['1998-12':]['DEM'] = np.NaN
        self.spot_last.loc[:'1998-12']['EUR'] = np.NaN


    def adjust_fwdpts_bid_ask(self):
        fx_side = self.df_tickers['inverse']
        _bid = self.fwdpts_bid.copy(deep=True)
        _ask = self.fwdpts_ask.copy(deep=True)
        for curncy in self.currency_list:
            if fx_side.loc[curncy] == -1:
                self.fwdpts_bid[curncy] = _ask[curncy]
                self.fwdpts_ask[curncy] = _bid[curncy]
        self.fwdpts_last.loc['1998-12':]['DEM'] = np.NaN
        self.fwdpts_last.loc[:'1998-12']['EUR'] = np.NaN

    def adjust_vol(self):
        if 'DEM' in self.currency_list:
            self.vols['DEM'].loc['1998-12':] = np.NaN
            lst_DEM_vol = float(self.vols['DEM'].dropna()[-1])
            self.vols['EUR'].loc['1999-01':'1999-06-25'] = lst_DEM_vol


    def _get_fwdpts_data(self, bbg_field=['PX_LAST']):
        fwd_pts_last = self.con.bdh(self.ticker_fwdpts_1m, bbg_field, self.ini_date_prices_bbg, self.end_date_bbg, elms=[("periodicitySelection", self.bbg_periodicity_spot)])
        fwd_pts_last.columns = fwd_pts_last.columns.droplevel(1)
        fwd_pts_last.fillna(method='ffill', inplace=True)
        fwd_pts_last.columns = [self.dict_NDF_FX[x.replace('1M BGN Curncy', '').replace('1M CMPN Curncy', '')] for x in fwd_pts_last.columns]
        fwd_pts_last.sort_index(axis=1, ascending=True, inplace=True)
        # DataFrame with extended calendar
        fwd_pts_last_extended_calendar = pd.DataFrame(data=None, index=self.daily_calendar_extended, columns=self.currency_list)
        _take_second = lambda s1, s2: s2
        fwd_pts_last_extended_calendar = fwd_pts_last_extended_calendar.combine(fwd_pts_last, func=_take_second)
        fwd_pts_last_extended_calendar.fillna(method='ffill', inplace=True)
        return fwd_pts_last_extended_calendar

    @staticmethod
    def _get_fwd_outright(spot, fwd_pts, fwdpts_scale):
        fwd_outright = spot + fwd_pts / fwdpts_scale
        fwd_outright.fillna(method='ffill', inplace=True)
        return fwd_outright

    def plot_fwd_discount(self, figure_size):
        self.fwd_discount_last.multiply(100).plot(figsize=figure_size, title='Forward premium (annual rate)')
        plt.show()

    # Weighting function
    # Equal weight
    # wgt = 1/k
    @staticmethod
    def _ranking_to_wgt(df_ranking, k=1):
        df_signals = pd.DataFrame(data=None, index=df_ranking.index, columns=df_ranking.columns)
        weights = pd.DataFrame(data=None, index=df_ranking.index, columns=df_ranking.columns)
        unit_wgt = 1.0 / k
        down_limit = k
        for d in df_ranking.index:
            n = df_ranking.loc[d].max()
            if n >= k * 2:
                up_limit = df_ranking.loc[d].max() - k
                df_signals.loc[d] = (df_ranking.loc[d] > up_limit).multiply(1) - (df_ranking.loc[d] <= down_limit).multiply(1)
                weights.loc[d] = df_signals.loc[d] * unit_wgt
            else:
                # Do not trade if there is not enough assets to trade long/short
                df_signals.loc[d] = 0
                weights.loc[d] = 0

        return weights, df_signals

    def _same_wgt(self, df_ranking, k=1):
        df_signals = (-np.isnan(self.fwd_discount_last)).multiply(1)
        for fx in self.currency_list:
            if fx != self.currency_strat12:
                df_signals[fx] = 0
            if fx == 'DEM':
                df_signals.loc['1998-11-30', 'DEM'] = 0
        weights = df_signals

        return weights, df_signals

    @staticmethod
    def _ranking_to_wgt_double_sorting(df_ranking, df_ranking_2, groups=3, subgroups=2):
        df_signals = pd.DataFrame(data=None, index=df_ranking.index, columns=df_ranking.columns)
        weights = pd.DataFrame(data=None, index=df_ranking.index, columns=df_ranking.columns)

        for d in df_ranking.index:
            # Filters for buying FX
            # First filter: divide first criteria in 3 groups (high, medium, low)
            up_limit = df_ranking.loc[d].max() * (1 - 1 / groups) # Ranking is ascending, so up limit needs to filter from 2/3 to 3/3.
            filter1 = (df_ranking.loc[d] > up_limit) # Series of true or false
            # Second filter: divide high and low blocks in two subgroups: high and low (eg. high-high, high-low)
            rank_2 = df_ranking_2.loc[d, filter1].rank(ascending=True)
            up_limit_2 = rank_2.max() / subgroups
            filter2 = (rank_2 > up_limit_2)
            filter2 = filter2.loc[filter2].index
            n_assets = len(filter2)
            df_signals.loc[d] = 0
            df_signals.loc[d, filter2] = 1
            # Filters for selling FX
            # First filter: divide first criteria in 3 groups (high, medium, low)
            down_limit = df_ranking.loc[d].max() * (1 / groups)
            filter1 = (df_ranking.loc[d] <= down_limit)  # Series of true or false
            # Second filter: divide high and low blocks in two subgroups: high and low (eg. high-high, high-low)
            rank_2 = df_ranking_2.loc[d, filter1].rank(ascending=True)
            # down_limit_2 = rank_2.max() / subgroups
            # filter2 = (rank_2 <= down_limit_2)
            filter2 = (rank_2 <= n_assets)
            filter2 = filter2.loc[filter2].index
            df_signals.loc[d, filter2] = -1
            # Weight calculation follows normal strategy approach (same weight for each FX)
            if n_assets == 0:
                unit_wgt = 0.0
            else:
                unit_wgt = 1.0 / n_assets
            weights.loc[d] = df_signals.loc[d] * unit_wgt
        return weights, df_signals

    # Equal vol
    # wgt_vol_adjusted = wgt/vol
    @staticmethod
    def _ranking_to_equalvolwgt(df_ranking, df_vols, target_vol=0.06, k=1):
        df_signals = pd.DataFrame(data=None, index=df_ranking.index, columns=df_ranking.columns)
        weights = pd.DataFrame(data=None, index=df_ranking.index, columns=df_ranking.columns)
        unit_wgt = 1.0 / k
        down_limit = k
        for d in df_ranking.index:
            up_limit = df_ranking.loc[d].max() - k
            df_signals.loc[d] = (df_ranking.loc[d] > up_limit).multiply(1) - (df_ranking.loc[d] <= down_limit).multiply(1)
            weights.loc[d] = df_signals.loc[d] * unit_wgt * target_vol / df_vols.loc[d].fillna(value=1)

        return weights, df_signals

    def _simple_ranking(self, df_signals):
        """
        Ranks elements of a matrix of signals. The highest signal receives the highest rank.
        Ranks are given for each line.
        Parameters
        ----------
        :type df_signals: Pandas.DataFrame
        Example
        -------
        >> _simple_ranking(fwd_discount_last)
        >> _simple_ranking(curvature_last)
        """
        # DataFrame with daily calendar
        ranked_matrix = pd.DataFrame(data=np.NaN, index=self.daily_calendar, columns=self.currency_list)

        for d in self.daily_calendar:
            tradeable_FX = self.tradeable_fx.loc[d]
            available_FX = list(df_signals.loc[d, tradeable_FX].dropna(how='any').index)
            ranked_matrix.loc[d, available_FX] = df_signals.loc[d, available_FX].rank(ascending=True)
        # ranked_matrix.fillna(method='ffill', axis=0, inplace=True)
        # ranked_matrix.dropna(axis=0, how='any', inplace=True)
        return ranked_matrix

    def _ewma_vol(self, ewma_lambda=0.94):
        df_vols = pd.DataFrame(data=None, index=self.daily_calendar_extended, columns=self.currency_list)
        df_vols = self.spot_last_XXXUSD.pct_change(1).ewm(com=None, span=None, halflife=None, alpha=(1.0 - ewma_lambda),
                                                          min_periods=0, adjust=True).var() * 252
        df_vols = df_vols ** 0.5
        return df_vols

    def _std_vol(self, window=126):
        df_vols = self.spot_last_XXXUSD.pct_change(1).rolling(window=window).var() * 252
        df_vols = df_vols ** 0.5
        return df_vols

    def plot_vols(self, figure_size):
        self.vols.multiply(100).plot(figsize=figure_size, title='Vols (annualized 252 bdays)')
        plt.show()

    def _run_default_monthly_strategy(self, df_ranking, func_weight, k):
        # df_forwards should be in XXXUSD
        # d=day, tm1=t minus 1
        TR_index = pd.DataFrame(index=self.monthly_calendar, columns=['TR_index'])
        holdings = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list)
        # weights and signals are calculated daily, but this functions should only return the ones used in strategy.
        weights_used = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list)
        signals_used = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list)
        traded_forward = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list)
        settle_spot = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list)
        traded_spot = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list)
        cost_index = pd.DataFrame(index=self.monthly_calendar, columns=['index']) # cost in units of USD
        cost_of_settle = pd.DataFrame(index=self.monthly_calendar, columns=['index']) # cost in units of USD
        cost_of_settle_fx_pct = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list) # cost in % for each currency
        cost_index_fx = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list) # cost in units of USD for each currency
        cost_pct = pd.DataFrame(index=self.monthly_calendar, columns=['index']) # cost in %
        cost_fx_pct = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list) # cost in % for each currency
        pnl_fx_pct = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list)  # pnl in % for each currency
        interest_return_index = pd.DataFrame(index=self.monthly_calendar, columns=['index'])
        interest_return_fx_pct = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list)
        spot_return_index = pd.DataFrame(index=self.monthly_calendar, columns=['index'])
        spot_return_index_fx = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list)
        spot_return_fx_pct = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list)
        # build a trade blotter
        lst_blotter_columns = ['trade_dt', 'maturity', 'holding', 'fx', 'traded_px', 'eval_date', 'price_mtm', 'pnl']
        blotter_df = pd.DataFrame(data=None, columns=lst_blotter_columns)
        def _build_ticket_dict(t_trade_dt, t_maturity, t_holding, t_fx, t_traded_px):
            t_eval_dt = t_trade_dt
            t_price_mtm = t_traded_px
            t_pnl = 0.0
            t_values = [t_trade_dt, t_maturity, t_holding, t_fx, t_traded_px, t_eval_dt, t_price_mtm, t_pnl]
            dict_ticket = dict(zip(lst_blotter_columns, t_values))
            return dict_ticket

        d_ini = self.monthly_calendar[0]
        TR_index.loc[d_ini] = 100.0
        [weights, signals] = func_weight(df_ranking, k)  # Daily data
        weights_used.loc[d_ini, self.currency_list] = weights.loc[d_ini - BDay(1), self.currency_list]
        signals_used.loc[d_ini, self.currency_list] = signals.loc[d_ini - BDay(1), self.currency_list]
        traded_forward.loc[d_ini, self.currency_list] = self.fwd_pricer(trade_date=d_ini, df_trading_side=weights_used.loc[d_ini, self.currency_list])
        traded_spot.loc[d_ini, self.currency_list] = self.spot_pricer(trade_date=d_ini, df_trading_side=weights_used.loc[d_ini, self.currency_list].multiply(-1)) # Get spot quote for the same side as forward (used in spot return)
        # holdings.loc[d_ini] = ((100.0 * weights.loc[d_ini - BDay(1)]) / df_forwards.loc[d_ini])  # fwd should be XXXUSD
        holdings.loc[d_ini, self.currency_list] = ((100.0 * weights_used.loc[d_ini]) / traded_forward.loc[d_ini])  # fwd should be XXXUSD

        cost_of_trading_fx = holdings.loc[d_ini, self.currency_list] * self.fwd_last_XXXUSD.loc[d_ini, self.currency_list] - weights_used.loc[d_ini, self.currency_list] * TR_index.loc[d_ini]
        cost_index.loc[d_ini] = cost_of_trading_fx.sum() # cost is positive (for comparative reason: eg. plotting with TR_index)
        cost_pct.loc[d_ini] = cost_index.loc[d_ini] / TR_index.loc[d_ini]
        cost_index_fx.loc[d_ini] = cost_of_trading_fx
        cost_fx_pct.loc[d_ini] = cost_of_trading_fx / float(TR_index.loc[d_ini])
        interest_return_index.loc[d_ini] = 100.0
        spot_return_index.loc[d_ini] = 100.0
        cost_of_settle.loc[d_ini] = 0.0
        spot_return_index_fx.loc[d_ini] = 100.0
        # writting trades into blotter
        for curncy in self.currency_list:
            curncy_holding = holdings.loc[d_ini, curncy]
            if (curncy_holding > 0.0 or curncy_holding < 0.0):
                trade_ticket_dict = _build_ticket_dict(t_trade_dt=d_ini, t_maturity=self.monthly_calendar[1], t_holding=curncy_holding, t_fx=curncy, t_traded_px=traded_forward.loc[d_ini, curncy])
                blotter_df = blotter_df.append(trade_ticket_dict, ignore_index=True)

        # tm1: t minus 1
        for d, tm1 in zip(self.monthly_calendar[1:], self.monthly_calendar[:-1]):
            settle_spot.loc[d, self.currency_list] = self.spot_pricer(trade_date=d, df_trading_side=weights_used.loc[tm1, self.currency_list])
            # pnl = (holdings.loc[tm1] * (df_spots.loc[d] - df_forwards.loc[tm1])).sum()
            # concept: return of a long forward position
            # fwd_ask[tm1] = spot_ask[tm1] + fwd_pts_ask[tm1]
            # total return = holdings[tm1] * (spot_bid[d] - fwd_ask[tm1])
            # interest return = holdings[tm1] * (spot_ask[tm1] - fwd_ask[tm1])  or  holdings[tm1] * (0 - fwd_pts_ask[tm1])
            # spot return = holdings[tm1] * (spot_bid[d] - spot_ask[tm1])

            pnl = (holdings.loc[tm1, self.currency_list] * (settle_spot.loc[d, self.currency_list] - traded_forward.loc[tm1, self.currency_list])).sum() # pnl in units of USD
            pnl_fx = (holdings.loc[tm1, self.currency_list] * (settle_spot.loc[d, self.currency_list] - traded_forward.loc[tm1, self.currency_list]))
            pnl_fx_pct.loc[d, self.currency_list] = pnl_fx / float(TR_index.loc[tm1]) # pnl in %, for each FX
            interest_return_fx = (holdings.loc[tm1, self.currency_list] * (traded_spot.loc[tm1, self.currency_list] - traded_forward.loc[tm1, self.currency_list])) # pnl in units of USD
            interest_return_index.loc[d] = interest_return_index.loc[tm1] + interest_return_fx.sum() # pnl in units of USD
            interest_return_fx_pct.loc[d] = interest_return_fx / float(TR_index.loc[tm1]) # pnl in %, for each FX
            spot_return_fx = (holdings.loc[tm1, self.currency_list] * (settle_spot.loc[d, self.currency_list] - traded_spot.loc[tm1, self.currency_list]))  # pnl in units of USD
            spot_return_index.loc[d] = spot_return_index.loc[tm1] + spot_return_fx.sum()
            spot_return_index_fx.loc[d] = spot_return_index_fx.loc[tm1] + spot_return_fx.loc[self.currency_list]
            spot_return_fx_pct.loc[d] = spot_return_fx.loc[self.currency_list] / float(TR_index.loc[tm1])
            cost_of_settle_fx = (holdings.loc[tm1, self.currency_list] * (settle_spot.loc[d, self.currency_list] - self.spot_last_XXXUSD.loc[d, self.currency_list])).multiply(-1)
            cost_of_settle.loc[d] = cost_of_settle.loc[tm1] + cost_of_settle_fx.sum() # in units of USD (cost is positive)
            cost_of_settle_fx_pct.loc[d] = cost_of_settle_fx / float(TR_index.loc[tm1])  # pnl in %
            # writting trades into blotter
            # settlement trades
            for curncy in self.currency_list:
                curncy_holding = holdings.loc[tm1, curncy]
                if (curncy_holding > 0.0 or curncy_holding < 0.0):
                    trade_ticket_dict = _build_ticket_dict(t_trade_dt=d,
                                                           t_maturity=d,
                                                           t_holding=curncy_holding * -1.0,
                                                           t_fx=curncy,
                                                           t_traded_px=settle_spot.loc[d, curncy])
                    blotter_df = blotter_df.append(trade_ticket_dict, ignore_index=True)

            TR_index.loc[d] = TR_index.loc[tm1] + pnl
            dm1 = d - BDay(1)
            # Adding 1 day delay to weights and signals used for trading
            weights_used.loc[d, self.currency_list] = weights.loc[dm1, self.currency_list]
            signals_used.loc[d, self.currency_list] = signals.loc[dm1, self.currency_list]
            traded_forward.loc[d, self.currency_list] = self.fwd_pricer(trade_date=d, df_trading_side=weights_used.loc[d, self.currency_list])
            traded_spot.loc[d, self.currency_list] = self.spot_pricer(trade_date=d, df_trading_side=weights_used.loc[d, self.currency_list].multiply(-1))  # Get spot quote for the same side as forward (used in spot return)
            holdings.loc[d] = (TR_index.loc[d].values * weights_used.loc[d] / traded_forward.loc[d])

            cost_of_trading_fx = holdings.loc[d, self.currency_list] * self.fwd_last_XXXUSD.loc[d, self.currency_list] - weights_used.loc[d, self.currency_list] * TR_index.loc[d]
            cost_index.loc[d] = cost_index.loc[tm1] + cost_of_settle_fx.sum() + cost_of_trading_fx.sum() # cost is positive (for comparative reason, eg. plotting with TR_index)
            cost_pct.loc[d] = (cost_index.loc[d] - cost_index.loc[tm1]) / TR_index.loc[d]
            cost_index_fx.loc[d, self.currency_list] = cost_index_fx.loc[tm1, self.currency_list] + cost_of_settle_fx.loc[self.currency_list] + cost_of_trading_fx.loc[self.currency_list]
            cost_fx_pct.loc[d] = (cost_of_trading_fx + cost_of_settle_fx)/ float(TR_index.loc[d])

            # writting trades into blotter
            # new forward trades
            for curncy in self.currency_list:
                curncy_holding = holdings.loc[d, curncy]
                if (curncy_holding > 0.0 or curncy_holding < 0.0):
                    trade_ticket_dict = _build_ticket_dict(t_trade_dt=d,
                                                           t_maturity=d + BMonthEnd(1),
                                                           t_holding=curncy_holding,
                                                           t_fx=curncy,
                                                           t_traded_px=traded_forward.loc[d, curncy])
                    blotter_df = blotter_df.append(trade_ticket_dict, ignore_index=True)

        return TR_index, holdings, weights_used, signals_used, interest_return_index, spot_return_index, cost_index, pnl_fx_pct, interest_return_fx_pct, spot_return_fx_pct, cost_fx_pct, blotter_df

    def fwd_pricer(self, trade_date, df_trading_side):
        traded_fwd_price = pd.Series(data=None, index=df_trading_side.index)
        # Buy at ask price
        traded_tickers = df_trading_side > 0.0
        traded_fwd_price[traded_tickers] = self.fwd_ask_XXXUSD.loc[trade_date, traded_tickers]
        # Sell at bid price
        traded_tickers = df_trading_side < 0.0
        traded_fwd_price[traded_tickers] = self.fwd_bid_XXXUSD.loc[trade_date, traded_tickers]
        return traded_fwd_price

    def spot_pricer(self, trade_date, df_trading_side):
        traded_spot_price = pd.Series(data=None, index=df_trading_side.index)
        # inverted signal, as spot should be traded in order to unwind an existing forward position
        # Buy spot at ask price (if existing forward is short)
        traded_tickers = df_trading_side < 0.0
        traded_spot_price[traded_tickers] = self.spot_ask_XXXUSD.loc[trade_date, traded_tickers]
        # Sell spot at bid price (if existing forward is long)
        traded_tickers = df_trading_side > 0.0
        traded_spot_price[traded_tickers] = self.spot_bid_XXXUSD.loc[trade_date, traded_tickers]
        return traded_spot_price

    def _run_default_monthly_strategy_double_sorting(self, df_forwards, df_spots, df_ranking1, df_ranking2, func_weight, groups, subgroups):
        # df_forwards should be in XXXUSD
        # d=day, tm1=t minus 1
        TR_index = pd.DataFrame(index=self.monthly_calendar, columns=['TR_index'])
        holdings = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list)
        # weights and signals are calculated daily, but this functions should only return the ones used in strategy.
        weights_used = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list)
        signals_used = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list)
        traded_forward = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list)
        settle_spot = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list)
        traded_spot = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list)
        cost_index = pd.DataFrame(index=self.monthly_calendar, columns=['index']) # cost in units of USD
        cost_of_settle = pd.DataFrame(index=self.monthly_calendar, columns=['index']) # cost in units of USD
        cost_of_settle_fx_pct = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list) # cost in % for each currency
        cost_index_fx = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list) # cost in units of USD for each currency
        cost_pct = pd.DataFrame(index=self.monthly_calendar, columns=['index']) # cost in %
        cost_fx_pct = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list) # cost in % for each currency
        pnl_fx_pct = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list)  # pnl in % for each currency
        interest_return_index = pd.DataFrame(index=self.monthly_calendar, columns=['index'])
        interest_return_fx_pct = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list)
        spot_return_index = pd.DataFrame(index=self.monthly_calendar, columns=['index'])
        spot_return_index_fx = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list)
        spot_return_fx_pct = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list)
        # build a trade blotter
        lst_blotter_columns = ['trade_dt', 'maturity', 'holding', 'fx', 'traded_px', 'eval_date', 'price_mtm', 'pnl']
        blotter_df = pd.DataFrame(data=None, columns=lst_blotter_columns)
        def _build_ticket_dict(t_trade_dt, t_maturity, t_holding, t_fx, t_traded_px):
            t_eval_dt = t_trade_dt
            t_price_mtm = t_traded_px
            t_pnl = 0.0
            t_values = [t_trade_dt, t_maturity, t_holding, t_fx, t_traded_px, t_eval_dt, t_price_mtm, t_pnl]
            dict_ticket = dict(zip(lst_blotter_columns, t_values))
            return dict_ticket

        d_ini = self.monthly_calendar[0]
        TR_index.loc[d_ini] = 100.0
        [weights, signals] = func_weight(df_ranking1, df_ranking2, groups, subgroups)  # Daily data
        weights_used.loc[d_ini, self.currency_list] = weights.loc[d_ini - BDay(1), self.currency_list]
        signals_used.loc[d_ini, self.currency_list] = signals.loc[d_ini - BDay(1), self.currency_list]
        traded_forward.loc[d_ini, self.currency_list] = self.fwd_pricer(trade_date=d_ini, df_trading_side=weights_used.loc[d_ini, self.currency_list])
        traded_spot.loc[d_ini, self.currency_list] = self.spot_pricer(trade_date=d_ini, df_trading_side=weights_used.loc[d_ini, self.currency_list].multiply(-1)) # Get spot quote for the same side as forward (used in spot return)
        # holdings.loc[d_ini] = ((100.0 * weights_used.loc[d_ini]) / traded_forward.loc[d_ini])  # fwd should be XXXUSD
        holdings.loc[d_ini, self.currency_list] = ((100.0 * weights_used.loc[d_ini]) / traded_forward.loc[d_ini])  # fwd should be XXXUSD

        cost_of_trading_fx = holdings.loc[d_ini, self.currency_list] * self.fwd_last_XXXUSD.loc[d_ini, self.currency_list] - weights_used.loc[d_ini, self.currency_list] * TR_index.loc[d_ini]
        cost_index.loc[d_ini] = cost_of_trading_fx.sum() # cost is positive (for comparative reason: eg. plotting with TR_index)
        cost_pct.loc[d_ini] = cost_index.loc[d_ini] / TR_index.loc[d_ini]
        cost_index_fx.loc[d_ini, self.currency_list] = cost_of_trading_fx.loc[self.currency_list]
        cost_fx_pct.loc[d_ini] = cost_of_trading_fx / float(TR_index.loc[d_ini])
        interest_return_index.loc[d_ini] = 100.0
        spot_return_index.loc[d_ini] = 100.0
        cost_of_settle.loc[d_ini] = 0.0
        spot_return_index_fx.loc[d_ini] = 100.0
        # writting trades into blotter
        for curncy in self.currency_list:
            curncy_holding = holdings.loc[d_ini, curncy]
            if (curncy_holding > 0.0 or curncy_holding < 0.0):
                trade_ticket_dict = _build_ticket_dict(t_trade_dt=d_ini, t_maturity=self.monthly_calendar[1], t_holding=curncy_holding, t_fx=curncy, t_traded_px=traded_forward.loc[d_ini, curncy])
                blotter_df = blotter_df.append(trade_ticket_dict, ignore_index=True)

        # tm1: t minus 1
        for d, tm1 in zip(self.monthly_calendar[1:], self.monthly_calendar[:-1]):
            settle_spot.loc[d, self.currency_list] = self.spot_pricer(trade_date=d, df_trading_side=weights_used.loc[tm1, self.currency_list])
            # pnl = (holdings.loc[tm1] * (df_spots.loc[d] - df_forwards.loc[tm1])).sum()
            # concept: return of a long forward position
            # fwd_ask[tm1] = spot_ask[tm1] + fwd_pts_ask[tm1]
            # total return = holdings[tm1] * (spot_bid[d] - fwd_ask[tm1])
            # interest return = holdings[tm1] * (spot_ask[tm1] - fwd_ask[tm1])  or  holdings[tm1] * (0 - fwd_pts_ask[tm1])
            # spot return = holdings[tm1] * (spot_bid[d] - spot_ask[tm1])

            pnl = (holdings.loc[tm1, self.currency_list] * (settle_spot.loc[d, self.currency_list] - traded_forward.loc[tm1, self.currency_list])).sum() # pnl in units of USD
            pnl_fx = (holdings.loc[tm1, self.currency_list] * (settle_spot.loc[d, self.currency_list] - traded_forward.loc[tm1, self.currency_list]))
            pnl_fx_pct.loc[d, self.currency_list] = pnl_fx / float(TR_index.loc[tm1]) # pnl in %, for each FX
            interest_return_fx = (holdings.loc[tm1, self.currency_list] * (traded_spot.loc[tm1, self.currency_list] - traded_forward.loc[tm1, self.currency_list])) # pnl in units of USD
            interest_return_index.loc[d] = interest_return_index.loc[tm1] + interest_return_fx.sum() # pnl in units of USD
            interest_return_fx_pct.loc[d] = interest_return_fx / float(TR_index.loc[tm1]) # pnl in %, for each FX
            spot_return_fx = (holdings.loc[tm1, self.currency_list] * (settle_spot.loc[d, self.currency_list] - traded_spot.loc[tm1, self.currency_list]))  # pnl in units of USD
            spot_return_index.loc[d] = spot_return_index.loc[tm1] + spot_return_fx.sum()
            spot_return_index_fx.loc[d] = spot_return_index_fx.loc[tm1] + spot_return_fx.loc[self.currency_list]
            spot_return_fx_pct.loc[d] = spot_return_fx.loc[self.currency_list] / float(TR_index.loc[tm1])
            cost_of_settle_fx = (holdings.loc[tm1, self.currency_list] * (settle_spot.loc[d, self.currency_list] - self.spot_last_XXXUSD.loc[d, self.currency_list])).multiply(-1)
            cost_of_settle.loc[d] = cost_of_settle.loc[tm1] + cost_of_settle_fx.sum() # in units of USD (cost is positive)
            cost_of_settle_fx_pct.loc[d] = cost_of_settle_fx / float(TR_index.loc[tm1])  # pnl in %
            # writting trades into blotter
            # settlement trades
            for curncy in self.currency_list:
                curncy_holding = holdings.loc[tm1, curncy]
                if (curncy_holding > 0.0 or curncy_holding < 0.0):
                    trade_ticket_dict = _build_ticket_dict(t_trade_dt=d,
                                                           t_maturity=d,
                                                           t_holding=curncy_holding * -1.0,
                                                           t_fx=curncy,
                                                           t_traded_px=settle_spot.loc[d, curncy])
                    blotter_df = blotter_df.append(trade_ticket_dict, ignore_index=True)

            TR_index.loc[d] = TR_index.loc[tm1] + pnl
            dm1 = d - BDay(1)
            # Adding 1 day delay to weights and signals used for trading
            weights_used.loc[d, self.currency_list] = weights.loc[dm1, self.currency_list]
            signals_used.loc[d, self.currency_list] = signals.loc[dm1, self.currency_list]
            traded_forward.loc[d, self.currency_list] = self.fwd_pricer(trade_date=d, df_trading_side=weights_used.loc[d, self.currency_list])
            traded_spot.loc[d, self.currency_list] = self.spot_pricer(trade_date=d, df_trading_side=weights_used.loc[d, self.currency_list].multiply(-1))  # Get spot quote for the same side as forward (used in spot return)
            holdings.loc[d] = (TR_index.loc[d].values * weights_used.loc[d] / traded_forward.loc[d])

            cost_of_trading_fx = holdings.loc[d, self.currency_list] * self.fwd_last_XXXUSD.loc[d, self.currency_list] - weights_used.loc[d, self.currency_list] * TR_index.loc[d]
            cost_index.loc[d] = cost_index.loc[tm1] + cost_of_settle_fx.sum() + cost_of_trading_fx.sum() # cost is positive (for comparative reason, eg. plotting with TR_index)
            cost_pct.loc[d] = (cost_index.loc[d] - cost_index.loc[tm1]) / TR_index.loc[d]
            cost_index_fx.loc[d, self.currency_list] = cost_index_fx.loc[tm1, self.currency_list] + cost_of_settle_fx.loc[self.currency_list] + cost_of_trading_fx.loc[self.currency_list]
            cost_fx_pct.loc[d] = (cost_of_trading_fx + cost_of_settle_fx)/ float(TR_index.loc[d])
            # writting trades into blotter
            # new forward trades
            for curncy in self.currency_list:
                curncy_holding = holdings.loc[d, curncy]
                if (curncy_holding > 0.0 or curncy_holding < 0.0):
                    trade_ticket_dict = _build_ticket_dict(t_trade_dt=d,
                                                           t_maturity=d + BMonthEnd(1),
                                                           t_holding=curncy_holding,
                                                           t_fx=curncy,
                                                           t_traded_px=traded_forward.loc[d, curncy])
                    blotter_df = blotter_df.append(trade_ticket_dict, ignore_index=True)

        return TR_index, holdings, weights_used, signals_used, interest_return_index, spot_return_index, cost_index, pnl_fx_pct, interest_return_fx_pct, spot_return_fx_pct, cost_fx_pct, blotter_df

    def run_default_monthly_strategy(self, df_signals, func_weight, strategy_n=0):
        """
        Run strategy for k=1 to k=4 and returns a DataFrame with 4 series
        """
        df_signals_smooth = df_signals.rolling(self.signal_moving_avg).mean()
        df_ranking = self._simple_ranking(df_signals_smooth)
        # TR_df
        TR_df = pd.DataFrame(data=None, index=self.monthly_calendar, columns=self.k_list)
        IR_df = pd.DataFrame(data=None, index=self.monthly_calendar, columns=self.k_list)
        SR_df = pd.DataFrame(data=None, index=self.monthly_calendar, columns=self.k_list)
        Cost_df = pd.DataFrame(data=None, index=self.monthly_calendar, columns=self.k_list)
        # holdings_df, weights_df, long_short_signals_df
        iterables_k = [self.k_list, self.monthly_calendar]
        idx = pd.MultiIndex.from_product(iterables=iterables_k)
        holdings_df = pd.DataFrame(data=None, index=idx, columns=self.currency_list)
        weights_df = pd.DataFrame(data=None, index=idx, columns=self.currency_list)
        long_short_signals_df = pd.DataFrame(data=None, index=idx, columns=self.currency_list)
        pct_pnl_fx_df = pd.DataFrame(data=None, index=idx, columns=self.currency_list)
        Pct_IR_fx_df = pd.DataFrame(data=None, index=idx, columns=self.currency_list)
        Pct_SR_fx_df = pd.DataFrame(data=None, index=idx, columns=self.currency_list)
        Pct_Cost_fx_df = pd.DataFrame(data=None, index=idx, columns=self.currency_list)

        for k in self.k_range:
            [TR_index_k, holdings_k, weights_k, signals_k, IR_index_k, SR_index_k, Cost_index_k, Pct_pnlFX_k, Pct_IR_fx_k, Pct_SR_fx_k, Pct_Cost_fx_k, Blotter_df] = \
                self._run_default_monthly_strategy(df_ranking=df_ranking, func_weight=func_weight, k=k)

            TR_df['k' + str(k)] = TR_index_k
            holdings_df.loc['k' + str(k)] = holdings_k.values
            weights_df.loc['k' + str(k)] = weights_k.values
            long_short_signals_df.loc['k' + str(k)] = signals_k.values
            IR_df['k' + str(k)] = IR_index_k
            SR_df['k' + str(k)] = SR_index_k
            Cost_df['k' + str(k)] = Cost_index_k
            pct_pnl_fx_df.loc['k' + str(k)] = Pct_pnlFX_k.values
            Pct_IR_fx_df.loc['k' + str(k)] = Pct_IR_fx_k.values
            Pct_SR_fx_df.loc['k' + str(k)] = Pct_SR_fx_k.values
            Pct_Cost_fx_df.loc['k' + str(k)] = Pct_Cost_fx_k.values
            if k == 1:
                Blotter = Blotter_df.copy(deep=True)
                Blotter['k'] = 1
            else:
                Blotter_df['k'] = k
                Blotter = Blotter.append(Blotter_df, ignore_index=True)

        #Write results in 'data' folder as Excel file. File will be name according with strategy number
        name_start = self.data_folder + "\\" + "{:02}".format(strategy_n) + '_'
        TR_df.to_excel(name_start + 'TR.xlsx')
        holdings_df.to_excel(name_start + 'Holdings.xlsx')
        weights_df.to_excel(name_start + 'Weights.xlsx')
        long_short_signals_df.to_excel(name_start + 'LS_Signals.xlsx')
        IR_df.to_excel(name_start + 'IR.xlsx') # interest return
        SR_df.to_excel(name_start + 'SR.xlsx') # spot return
        Cost_df.to_excel(name_start + 'Cost.xlsx')  # Cost index
        pct_pnl_fx_df.to_excel(name_start + 'PctPnlFX.xlsx')  # total return by fx
        Pct_IR_fx_df.to_excel(name_start + 'PctIRFX.xlsx')  # Interest return by fx
        Pct_SR_fx_df.to_excel(name_start + 'PctSRFX.xlsx')  # spot return by fx
        Pct_Cost_fx_k.to_excel(name_start + 'PctCostFX.xlsx')  # Cost by fx
        Blotter.to_excel(name_start + 'Blotter.xlsx')  # Blotter

        return TR_df, holdings_df, weights_df, long_short_signals_df, IR_df, SR_df, Cost_df, pct_pnl_fx_df, Pct_IR_fx_df, Pct_SR_fx_df, Pct_Cost_fx_k, Blotter

    def run_default_monthly_strategy_double_sorting(self, df_signals1, df_signals2, func_weight, strategy_n=0, groups=3, subgroups=2):
        """
        Run strategy for k=1 to k=4 and returns a DataFrame with 4 series
        """
        df_signals1_smooth = df_signals1.rolling(self.signal_moving_avg).mean()
        df_signals2_smooth = df_signals2.rolling(self.signal_moving_avg).mean()

        df_ranking1 = self._simple_ranking(df_signals1_smooth)
        df_ranking2 = self._simple_ranking(df_signals2_smooth)

        # TR_df
        TR_df = pd.DataFrame(data=None, index=self.monthly_calendar, columns=self.k_list)
        IR_df = pd.DataFrame(data=None, index=self.monthly_calendar, columns=self.k_list)
        SR_df = pd.DataFrame(data=None, index=self.monthly_calendar, columns=self.k_list)
        Cost_df = pd.DataFrame(data=None, index=self.monthly_calendar, columns=self.k_list)
        # holdings_df, weights_df, long_short_signals_df
        iterables_k = [self.k_list, self.monthly_calendar]
        idx = pd.MultiIndex.from_product(iterables=iterables_k)
        holdings_df = pd.DataFrame(data=None, index=idx, columns=self.currency_list)
        weights_df = pd.DataFrame(data=None, index=idx, columns=self.currency_list)
        long_short_signals_df = pd.DataFrame(data=None, index=idx, columns=self.currency_list)
        pct_pnl_fx_df = pd.DataFrame(data=None, index=idx, columns=self.currency_list)
        Pct_IR_fx_df = pd.DataFrame(data=None, index=idx, columns=self.currency_list)
        Pct_SR_fx_df = pd.DataFrame(data=None, index=idx, columns=self.currency_list)
        Pct_Cost_fx_df = pd.DataFrame(data=None, index=idx, columns=self.currency_list)

        [TR_index_k, holdings_k, weights_k, signals_k, IR_index_k, SR_index_k, Cost_index_k, Pct_pnlFX_k, Pct_IR_fx_k, Pct_SR_fx_k, Pct_Cost_fx_k, Blotter_df] = \
            self._run_default_monthly_strategy_double_sorting(self.fwd_last_XXXUSD, self.spot_last_XXXUSD, df_ranking1, df_ranking2, func_weight, groups, subgroups)
        for k in self.k_range:
            TR_df['k' + str(k)] = TR_index_k
            holdings_df.loc['k' + str(k)] = holdings_k.values
            weights_df.loc['k' + str(k)] = weights_k.values
            long_short_signals_df.loc['k' + str(k)] = signals_k.values
            IR_df['k' + str(k)] = IR_index_k
            SR_df['k' + str(k)] = SR_index_k
            Cost_df['k' + str(k)] = Cost_index_k
            pct_pnl_fx_df.loc['k' + str(k)] = Pct_pnlFX_k.values
            Pct_IR_fx_df.loc['k' + str(k)] = Pct_IR_fx_k.values
            Pct_SR_fx_df.loc['k' + str(k)] = Pct_SR_fx_k.values
            Pct_Cost_fx_df.loc['k' + str(k)] = Pct_Cost_fx_k.values
            if k == 1:
                Blotter = Blotter_df.copy(deep=True)
                Blotter['k'] = 1
            else:
                Blotter_df['k'] = k
                Blotter = Blotter.append(Blotter_df, ignore_index=True)

        #Write results in 'data' folder as Excel file. File will be name according with strategy number
        name_start = self.data_folder + "\\" + "{:02}".format(strategy_n) + '_'
        TR_df.to_excel(name_start + 'TR.xlsx')
        holdings_df.to_excel(name_start + 'Holdings.xlsx')
        weights_df.to_excel(name_start + 'Weights.xlsx')
        long_short_signals_df.to_excel(name_start + 'LS_Signals.xlsx')
        IR_df.to_excel(name_start + 'IR.xlsx') # interest return
        SR_df.to_excel(name_start + 'SR.xlsx') # spot return
        Cost_df.to_excel(name_start + 'Cost.xlsx')  # Cost index
        pct_pnl_fx_df.to_excel(name_start + 'PctPnlFX.xlsx')  # total return by fx
        Pct_IR_fx_df.to_excel(name_start + 'PctIRFX.xlsx')  # Interest return by fx
        Pct_SR_fx_df.to_excel(name_start + 'PctSRFX.xlsx')  # spot return by fx
        Pct_Cost_fx_k.to_excel(name_start + 'PctCostFX.xlsx')  # Cost by fx
        Blotter.to_excel(name_start + 'Blotter.xlsx')  # Blotter

        return TR_df, holdings_df, weights_df, long_short_signals_df, IR_df, SR_df, Cost_df, pct_pnl_fx_df, Pct_IR_fx_df, Pct_SR_fx_df, Pct_Cost_fx_k, Blotter

    @staticmethod
    def _plot_total_return(total_return_df, chart_title, figure_size, log_y=True):
        total_return_df.plot(figsize=figure_size, title=chart_title, logy=log_y)
        plt.show()

    def plot_default_strategy_return_v1(self, chart_title, figure_size, log_y=True):
        self._plot_total_return(self.TR_df, chart_title, figure_size, log_y)
        plt.show()

    def plot_default_strategy_return(self, chart_title, figure_size, log_y=True):
        self.TR_df.plot(figsize=figure_size, title=chart_title, logy=log_y)
        plt.show()

    def _run_equal_vol_monthly_strategy(self, df_ranking, df_vol, target_vol=0.06, k=1):
        # df_forwards should be in XXXUSD
        # d=day, tm1=t minus 1
        TR_index = pd.DataFrame(index=self.monthly_calendar, columns=['TR_index'])
        holdings = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list)
        # weights and signals are calculated daily, but this functions should only return the ones used in strategy.
        weights_used = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list)
        signals_used = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list)
        traded_forward = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list)
        settle_spot = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list)
        traded_spot = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list)
        cost_index = pd.DataFrame(index=self.monthly_calendar, columns=['index']) # cost in units of USD
        cost_of_settle = pd.DataFrame(index=self.monthly_calendar, columns=['index']) # cost in units of USD
        cost_of_settle_fx_pct = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list) # cost in % for each currency
        cost_index_fx = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list) # cost in units of USD for each currency
        cost_pct = pd.DataFrame(index=self.monthly_calendar, columns=['index']) # cost in %
        cost_fx_pct = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list) # cost in % for each currency
        pnl_fx_pct = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list)  # pnl in % for each currency
        interest_return_index = pd.DataFrame(index=self.monthly_calendar, columns=['index'])
        interest_return_fx_pct = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list)
        spot_return_index = pd.DataFrame(index=self.monthly_calendar, columns=['index'])
        spot_return_index_fx = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list)
        spot_return_fx_pct = pd.DataFrame(index=self.monthly_calendar, columns=self.currency_list)
        # build a trade blotter
        lst_blotter_columns = ['trade_dt', 'maturity', 'holding', 'fx', 'traded_px', 'eval_date', 'price_mtm', 'pnl']
        blotter_df = pd.DataFrame(data=None, columns=lst_blotter_columns)
        def _build_ticket_dict(t_trade_dt, t_maturity, t_holding, t_fx, t_traded_px):
            t_eval_dt = t_trade_dt
            t_price_mtm = t_traded_px
            t_pnl = 0.0
            t_values = [t_trade_dt, t_maturity, t_holding, t_fx, t_traded_px, t_eval_dt, t_price_mtm, t_pnl]
            dict_ticket = dict(zip(lst_blotter_columns, t_values))
            return dict_ticket

        d_ini = self.monthly_calendar[0]
        TR_index.loc[d_ini] = 100.0
        [weights, signals] = self._ranking_to_equalvolwgt(df_ranking, df_vol, target_vol, k)
        weights_used.loc[d_ini, self.currency_list] = weights.loc[d_ini - BDay(1), self.currency_list]
        signals_used.loc[d_ini, self.currency_list] = signals.loc[d_ini - BDay(1), self.currency_list]
        traded_forward.loc[d_ini, self.currency_list] = self.fwd_pricer(trade_date=d_ini, df_trading_side=weights_used.loc[d_ini, self.currency_list])
        traded_spot.loc[d_ini, self.currency_list] = self.spot_pricer(trade_date=d_ini, df_trading_side=weights_used.loc[d_ini, self.currency_list].multiply(-1)) # Get spot quote for the same side as forward (used in spot return)
        # holdings.loc[d_ini] = ((100.0 * weights.loc[d_ini - BDay(1)]) / df_forwards.loc[d_ini])  # fwd should be XXXUSD
        holdings.loc[d_ini, self.currency_list] = ((100.0 * weights_used.loc[d_ini]) / traded_forward.loc[d_ini])  # fwd should be XXXUSD

        cost_of_trading_fx = holdings.loc[d_ini, self.currency_list] * self.fwd_last_XXXUSD.loc[d_ini, self.currency_list] - weights_used.loc[d_ini, self.currency_list] * TR_index.loc[d_ini]
        cost_index.loc[d_ini] = cost_of_trading_fx.sum() # cost is positive (for comparative reason: eg. plotting with TR_index)
        cost_pct.loc[d_ini] = cost_index.loc[d_ini] / TR_index.loc[d_ini]
        cost_index_fx.loc[d_ini, self.currency_list] = cost_of_trading_fx.loc[self.currency_list]
        cost_fx_pct.loc[d_ini] = cost_of_trading_fx / float(TR_index.loc[d_ini])
        interest_return_index.loc[d_ini] = 100.0
        spot_return_index.loc[d_ini] = 100.0
        cost_of_settle.loc[d_ini] = 0.0
        spot_return_index_fx.loc[d_ini] = 100.0
        # writting trades into blotter
        for curncy in self.currency_list:
            curncy_holding = holdings.loc[d_ini, curncy]
            if (curncy_holding > 0.0 or curncy_holding < 0.0):
                trade_ticket_dict = _build_ticket_dict(t_trade_dt=d_ini, t_maturity=self.monthly_calendar[1], t_holding=curncy_holding, t_fx=curncy, t_traded_px=traded_forward.loc[d_ini, curncy])
                blotter_df = blotter_df.append(trade_ticket_dict, ignore_index=True)

        # tm1: t minus 1
        for d, tm1 in zip(self.monthly_calendar[1:], self.monthly_calendar[:-1]):
            settle_spot.loc[d, self.currency_list] = self.spot_pricer(trade_date=d, df_trading_side=weights_used.loc[tm1, self.currency_list])
            # pnl = (holdings.loc[tm1] * (df_spots.loc[d] - df_forwards.loc[tm1])).sum()
            # concept: return of a long forward position
            # fwd_ask[tm1] = spot_ask[tm1] + fwd_pts_ask[tm1]
            # total return = holdings[tm1] * (spot_bid[d] - fwd_ask[tm1])
            # interest return = holdings[tm1] * (spot_ask[tm1] - fwd_ask[tm1])  or  holdings[tm1] * (0 - fwd_pts_ask[tm1])
            # spot return = holdings[tm1] * (spot_bid[d] - spot_ask[tm1])

            pnl = (holdings.loc[tm1, self.currency_list] * (settle_spot.loc[d, self.currency_list] - traded_forward.loc[tm1, self.currency_list])).sum() # pnl in units of USD
            pnl_fx = (holdings.loc[tm1, self.currency_list] * (settle_spot.loc[d, self.currency_list] - traded_forward.loc[tm1, self.currency_list]))
            pnl_fx_pct.loc[d, self.currency_list] = pnl_fx / float(TR_index.loc[tm1]) # pnl in %, for each FX
            interest_return_fx = (holdings.loc[tm1, self.currency_list] * (traded_spot.loc[tm1, self.currency_list] - traded_forward.loc[tm1, self.currency_list])) # pnl in units of USD
            interest_return_index.loc[d] = interest_return_index.loc[tm1] + interest_return_fx.sum() # pnl in units of USD
            interest_return_fx_pct.loc[d] = interest_return_fx / float(TR_index.loc[tm1]) # pnl in %, for each FX
            spot_return_fx = (holdings.loc[tm1, self.currency_list] * (settle_spot.loc[d, self.currency_list] - traded_spot.loc[tm1, self.currency_list]))  # pnl in units of USD
            spot_return_index.loc[d] = spot_return_index.loc[tm1] + spot_return_fx.sum()
            spot_return_index_fx.loc[d] = spot_return_index_fx.loc[tm1] + spot_return_fx.loc[self.currency_list]
            spot_return_fx_pct.loc[d] = spot_return_fx.loc[self.currency_list] / float(TR_index.loc[tm1])
            cost_of_settle_fx = (holdings.loc[tm1, self.currency_list] * (settle_spot.loc[d, self.currency_list] - self.spot_last_XXXUSD.loc[d, self.currency_list])).multiply(-1)
            cost_of_settle.loc[d] = cost_of_settle.loc[tm1] + cost_of_settle_fx.sum() # in units of USD (cost is positive)
            cost_of_settle_fx_pct.loc[d] = cost_of_settle_fx / float(TR_index.loc[tm1])  # pnl in %
            # writting trades into blotter
            # settlement trades
            for curncy in self.currency_list:
                curncy_holding = holdings.loc[tm1, curncy]
                if (curncy_holding > 0.0 or curncy_holding < 0.0):
                    trade_ticket_dict = _build_ticket_dict(t_trade_dt=d,
                                                           t_maturity=d,
                                                           t_holding=curncy_holding * -1.0,
                                                           t_fx=curncy,
                                                           t_traded_px=settle_spot.loc[d, curncy])
                    blotter_df = blotter_df.append(trade_ticket_dict, ignore_index=True)

            TR_index.loc[d] = TR_index.loc[tm1] + pnl
            dm1 = d - BDay(1)
            # Adding 1 day delay to weights and signals used for trading
            weights_used.loc[d, self.currency_list] = weights.loc[dm1, self.currency_list]
            signals_used.loc[d, self.currency_list] = signals.loc[dm1, self.currency_list]
            traded_forward.loc[d, self.currency_list] = self.fwd_pricer(trade_date=d, df_trading_side=weights_used.loc[d, self.currency_list])
            traded_spot.loc[d, self.currency_list] = self.spot_pricer(trade_date=d, df_trading_side=weights_used.loc[d, self.currency_list].multiply(-1))  # Get spot quote for the same side as forward (used in spot return)
            holdings.loc[d] = (TR_index.loc[d].values * weights_used.loc[d] / traded_forward.loc[d])

            cost_of_trading_fx = holdings.loc[d, self.currency_list] * self.fwd_last_XXXUSD.loc[d, self.currency_list] - weights_used.loc[d, self.currency_list] * TR_index.loc[d]
            cost_index.loc[d] = cost_index.loc[tm1] + cost_of_settle_fx.sum() + cost_of_trading_fx.sum() # cost is positive (for comparative reason, eg. plotting with TR_index)
            cost_pct.loc[d] = (cost_index.loc[d] - cost_index.loc[tm1]) / TR_index.loc[d]
            cost_index_fx.loc[d, self.currency_list] = cost_index_fx.loc[tm1, self.currency_list] + cost_of_settle_fx.loc[self.currency_list] + cost_of_trading_fx.loc[self.currency_list]
            cost_fx_pct.loc[d] = (cost_of_trading_fx + cost_of_settle_fx)/ float(TR_index.loc[d])
            # writting trades into blotter
            # new forward trades
            for curncy in self.currency_list:
                curncy_holding = holdings.loc[d, curncy]
                if (curncy_holding > 0.0 or curncy_holding < 0.0):
                    trade_ticket_dict = _build_ticket_dict(t_trade_dt=d,
                                                           t_maturity=d + BMonthEnd(1),
                                                           t_holding=curncy_holding,
                                                           t_fx=curncy,
                                                           t_traded_px=traded_forward.loc[d, curncy])
                    blotter_df = blotter_df.append(trade_ticket_dict, ignore_index=True)

        return TR_index, holdings, weights_used, signals_used, interest_return_index, spot_return_index, cost_index, pnl_fx_pct, interest_return_fx_pct, spot_return_fx_pct, cost_fx_pct, blotter_df

    def run_equal_vol_monthly_strategy(self, df_signals, target_vol=0.06, strategy_n=0):
        """
        Run strategy for k=1 to k=4 and returns a DataFrame with 4 series

        Parameters
        ----------
        df_signals : Pandas.DataFrame
            DataFrame with signals (eg. fwd discount, curvature, level, slope)
        target_vol  : float
            Target volatility of each FX position. Used in _ranking_to_equalvolwgt() function.
        strategy_n : int
            Stragegy number is used to name Excel files that are written to save results in disk.
        Notes
        ----------
        A 9% volatility should make an equal vol strategy with similar volatility of a normal carry trade strategy.

        """
        df_signals_smooth = df_signals.rolling(self.signal_moving_avg).mean()
        df_ranking = self._simple_ranking(df_signals_smooth)
        # TR_df
        TR_df = pd.DataFrame(data=None, index=self.monthly_calendar, columns=self.k_list)
        IR_df = pd.DataFrame(data=None, index=self.monthly_calendar, columns=self.k_list)
        SR_df = pd.DataFrame(data=None, index=self.monthly_calendar, columns=self.k_list)
        Cost_df = pd.DataFrame(data=None, index=self.monthly_calendar, columns=self.k_list)
        # holdings_df, weights_df, long_short_signals_df
        iterables_k = [self.k_list, self.monthly_calendar]
        idx = pd.MultiIndex.from_product(iterables=iterables_k)
        holdings_df = pd.DataFrame(data=None, index=idx, columns=self.currency_list)
        weights_df = pd.DataFrame(data=None, index=idx, columns=self.currency_list)
        long_short_signals_df = pd.DataFrame(data=None, index=idx, columns=self.currency_list)
        pct_pnl_fx_df = pd.DataFrame(data=None, index=idx, columns=self.currency_list)
        Pct_IR_fx_df = pd.DataFrame(data=None, index=idx, columns=self.currency_list)
        Pct_SR_fx_df = pd.DataFrame(data=None, index=idx, columns=self.currency_list)
        Pct_Cost_fx_df = pd.DataFrame(data=None, index=idx, columns=self.currency_list)

        for k in self.k_range:
            [TR_index_k, holdings_k, weights_k, signals_k, IR_index_k, SR_index_k, Cost_index_k, Pct_pnlFX_k, Pct_IR_fx_k, Pct_SR_fx_k, Pct_Cost_fx_k, Blotter_df] = \
                self._run_equal_vol_monthly_strategy(df_ranking=df_ranking, df_vol=self.vols, target_vol=target_vol, k=k)
            TR_df['k' + str(k)] = TR_index_k
            holdings_df.loc['k' + str(k)] = holdings_k.values
            weights_df.loc['k' + str(k)] = weights_k.values
            long_short_signals_df.loc['k' + str(k)] = signals_k.values
            IR_df['k' + str(k)] = IR_index_k
            SR_df['k' + str(k)] = SR_index_k
            Cost_df['k' + str(k)] = Cost_index_k
            pct_pnl_fx_df.loc['k' + str(k)] = Pct_pnlFX_k.values
            Pct_IR_fx_df.loc['k' + str(k)] = Pct_IR_fx_k.values
            Pct_SR_fx_df.loc['k' + str(k)] = Pct_SR_fx_k.values
            Pct_Cost_fx_df.loc['k' + str(k)] = Pct_Cost_fx_k.values
            if k == 1:
                Blotter = Blotter_df.copy(deep=True)
                Blotter['k'] = 1
            else:
                Blotter_df['k'] = k
                Blotter = Blotter.append(Blotter_df, ignore_index=True)

        #Write results in 'data' folder as Excel file. File will be name according with strategy number
        name_start = self.data_folder + "\\" + "{:02}".format(strategy_n) + '_'
        TR_df.to_excel(name_start + 'TR.xlsx')
        holdings_df.to_excel(name_start + 'Holdings.xlsx')
        weights_df.to_excel(name_start + 'Weights.xlsx')
        long_short_signals_df.to_excel(name_start + 'LS_Signals.xlsx')
        IR_df.to_excel(name_start + 'IR.xlsx') # interest return
        SR_df.to_excel(name_start + 'SR.xlsx') # spot return
        Cost_df.to_excel(name_start + 'Cost.xlsx')  # Cost index
        pct_pnl_fx_df.to_excel(name_start + 'PctPnlFX.xlsx')  # total return by fx
        Pct_IR_fx_df.to_excel(name_start + 'PctIRFX.xlsx')  # Interest return by fx
        Pct_SR_fx_df.to_excel(name_start + 'PctSRFX.xlsx')  # spot return by fx
        Pct_Cost_fx_k.to_excel(name_start + 'PctCostFX.xlsx')  # Cost by fx
        Blotter.to_excel(name_start + 'Blotter.xlsx')  # Blotter

        return TR_df, holdings_df, weights_df, long_short_signals_df, IR_df, SR_df, Cost_df, pct_pnl_fx_df, Pct_IR_fx_df, Pct_SR_fx_df, Pct_Cost_fx_k, Blotter


    # Curvy trade data
    bbgcurve_dict = {'AUD': 'YCSW0001 Index',
                     'CAD': 'YCSW0004 Index',
                     'CHF': 'YCSW0021 Index',
                     'DEM': 'YCGT0054 Index',
                     'EUR': 'YCGT0054 Index',
                     'GBP': 'YCSW0022 Index',
                     'JPY': 'YCSW0013 Index',
                     'NOK': 'YCSW0016 Index',
                     'NZD': 'YCSW0015 Index',
                     'SEK': 'YCSW0020 Index',
                     'USD': 'YCSW0023 Index',
                     'ZAR': 'YCSW0018 Index',
                     'CZK': 'YCSW0037 Index',
                     'TWD': 'YCSW0041 Index',
                     'SGD': 'YCSW0044 Index',
                     'PLN': 'YCSW0048 Index',
                     'KRW': 'YCSW0057 Index',
                     'PHP': 'YCSW0081 Index',
                     'MXN': 'YCSW0083 Index',
                     'BRL': 'YCSW0089 Index',
                     'HUF': 'YCSW0124 Index',
                     'TRY': 'YCSW0164 Index',
                     'COP': 'YCSW0329 Index',
                     'CLP': 'YCSW0193 Index',
                     'INR': 'YCSW0046 Index',
                     'RUB': 'YCSW0179 Index',
                     'HKD': 'YCSW0010 Index',
                     }

    tenor_dict = {'1D':  1.0 / 360.0,
                  '2D':  2.0 / 360.0,
                  '1W':  7.0 / 360.0,
                  '10D': 10.0 / 360.0,
                  '2W':  14.0 / 360.0,
                  '20D': 20.0 / 360.0,
                  '1M':  1.0 / 12.0,
                  '2M':  2.0 / 12.0,
                  '3M':  3.0 / 12.0,
                  '4M':  4.0 / 12.0,
                  '5M':  5.0 / 12.0,
                  '6M':  0.5,
                  '7M':  7.0 / 12.0,
                  '8M':  8.0 / 12.0,
                  '9M':  9.0 / 12.0,
                  '10M': 10.0 / 12.0,
                  '11M': 11.0 / 12.0,
                  '1Y':  1.0,
                  '13M': 13.0 / 12.0,
                  '14M': 14.0 / 12.0,
                  '18M': 1.5,
                  '2Y':  2.0,
                  '3Y':  3.0,
                  '4Y':  4.0,
                  '5Y':  5.0,
                  '6Y':  6.0,
                  '7Y':  7.0,
                  '8Y':  8.0,
                  '9Y':  9.0,
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

    def _convert_tenors(self, tenors):
        return [self.tenor_dict[tenor] * 12 for tenor in tenors]

    def _metadata_curve_bbg(self, currency='CHF'):
        """
        Return a DataFrame with tenors and tickers of a interest rate curve.
        :param currency: Currency. Example: AUD (Australia).
        :type currency: str
        :return: DataFrame with tenors and tickers.
        :rtype: pd.DataFrame
        """
        curve_ticker = self.bbgcurve_dict[currency]
        member_ticker = list(self.con.bulkref(curve_ticker, 'CURVE_MEMBERS')['value'])
        tenor_string = list(self.con.bulkref(curve_ticker, 'CURVE_TERMS')['value'])
        df_curve_aux = pd.DataFrame(data=list(zip(tenor_string, member_ticker)), columns=['tenor', 'ticker'])
        df_curve_aux['months'] = self._convert_tenors(list(df_curve_aux.tenor))
        return df_curve_aux

    def _build_df_tickers_curve(self):
        # Preparing DataFrame to Append data from other currencies of currency_list_curve.
        currency_1 = self.currency_list_curve[0]
        df_tickers_curve = self._metadata_curve_bbg(currency_1)
        idx = pd.MultiIndex.from_product(iterables=[[currency_1], df_tickers_curve['months']])
        df_tickers_curve.set_index(idx, drop=True, verify_integrity=True, inplace=True)
        # Append other FX data to df_tickers_curve
        for currency in self.currency_list_curve[1:]:
            df_metadata_aux = self._metadata_curve_bbg(currency)
            idx = pd.MultiIndex.from_product(iterables=[[currency], df_metadata_aux['months']])
            df_metadata_aux.set_index(idx, drop=True, verify_integrity=True, inplace=True)
            df_tickers_curve = df_tickers_curve.append(other=df_metadata_aux, verify_integrity=True)
        return df_tickers_curve

    def _build_dict_ticker_year(self):
        # Dictionary for tenor BBG ticker to years (float)
        return dict(zip(self.df_tickers_curve['ticker'], self.df_tickers_curve['months']))

    def _get_interest_rate_curve_bbg(self, curve_ticker_list=['US00O/N Index', 'USDR2T Curncy']):
        # Data from BBG
        df_curve_bbg_aux = self.con.bdh(curve_ticker_list, self.bbg_field_last, self.ini_date_bbg, self.end_date_bbg,
                                   elms=[("periodicitySelection", self.bbg_periodicity_interest_curve)])
        # Data handling
        df_curve_bbg_aux.columns = df_curve_bbg_aux.columns.droplevel(level='field')
        df_curve_bbg_aux.columns = [self.ticker_year_dict[ticker] for ticker in df_curve_bbg_aux.columns]
        df_curve_bbg_aux.sort_index(axis=1, inplace=True)
        df_curve_bbg_aux.fillna(method='ffill', inplace=True)
        # Fitting into daily calendar
        df_curve_bbg_daily = pd.DataFrame(data=np.NaN, index=self.daily_calendar, columns=df_curve_bbg_aux.columns)
        take_second = lambda s1, s2: s2
        df_curve_bbg_daily = df_curve_bbg_daily.combine(other=df_curve_bbg_aux, func=take_second)
        df_curve_bbg_daily.fillna(method='ffill', inplace=True)

        return df_curve_bbg_daily

    def _get_all_interest_curves(self):
        # Preparing DataFrame to Append data from other currencies of currency_list_curve.
        currency_1 = self.currency_list_curve[0]
        tickers_list_1 = list(self.df_tickers_curve.loc[currency_1, 'ticker'].values)
        df_interest_curve = self._get_interest_rate_curve_bbg(tickers_list_1)
        idx = pd.MultiIndex.from_product(iterables=[[currency_1], df_interest_curve.index.values])
        df_interest_curve.set_index(idx, drop=True, verify_integrity=True, inplace=True)
        # Append other FX data to df_interest_curve
        for currency in self.currency_list_curve[1:]:
            tickers_list = list(self.df_tickers_curve.loc[currency, 'ticker'].values)
            df_curve_bbg_aux = self._get_interest_rate_curve_bbg(tickers_list)
            idx = pd.MultiIndex.from_product(iterables=[[currency], df_curve_bbg_aux.index.values])
            df_curve_bbg_aux.set_index(idx, drop=True, verify_integrity=True, inplace=True)
            # Append data
            df_interest_curve = df_interest_curve.append(other=df_curve_bbg_aux, verify_integrity=True)
        return df_interest_curve

    def _get_one_curve(self, currency, date, tenors_greater_than_n_years=0.0):
        # series_size = len(self.interest_curve.loc[(currency, date), :])
        # nan_count = self.interest_curve.loc[(currency, date), :].isna().count()
        # if nan_count < series_size:
        # df_curve = self.interest_curve.loc[(currency, date), :].dropna(how='all', axis=1)
        try:
            df_curve = self.interest_curve.loc[(currency, date)]
            df_curve = df_curve.dropna(how='all')
            df_curve = df_curve.loc[df_curve.index >= tenors_greater_than_n_years * 12]
        except:
            df_curve = None
        # else:
        #     df_curve = None
        return df_curve


    @staticmethod
    def NSiegel(array_betas, df_curve):
        """
        Returns the sum of squared errors of N-Siegel curve Vs actual curve, given b1, b2 and b3.

        Parameters
        ----------
        :param array_betas: b1, b2 and b3 are level, slope and curvature parameters in Nelson Siegel curve.
        :type array_betas: float
        :param df_curve: actual curve.
        :type df_curve: pd.DataFrame
        :return: Sum of squared errors.
        :rtype: float

        Notes
        ----------
        Example of df_curve_input layout:
        df_curve_input_ex = df_interest_curve.loc[('USD', '1991-01-31'), :].dropna(how='all', axis=1)
        """

        b1 = array_betas[0]
        b2 = array_betas[1]
        b3 = array_betas[2]
        years = np.array(df_curve.index.values)
        int_rates = df_curve.values

        #     _lambda = df_parameters['lambda']
        _lambda = 0.0609  # hardcoded
        #     df_y_est = pd.DataFrame(index=tau, columns=['y_est','errors'])
        sum_squared_errors = 0.0
        for tau, rate in zip(years, int_rates):
            _aux = ((1 - math.exp(-_lambda * tau)) / (_lambda * tau))
            y_tau = b1 + b2 * _aux + b3 * (_aux - math.exp(-_lambda * tau))
            # TODO Factor could be changed to make a weighted optimization
            factor = 1.0  # to improve...
            sum_squared_errors = sum_squared_errors + ((rate - y_tau) ** 2) * factor
        return sum_squared_errors

    def _run_NSiegel_fitting(self, tenors_greater_than_n_years=0.0):
        # Run Nelson-Siegel fitting only for available dates
        # calendar_curves = list(self.interest_curve.index.get_level_values(1).unique())
        idx = pd.MultiIndex.from_product(iterables=[self.currency_list_curve, self.calendar_curves])
        df_betas = pd.DataFrame(data=np.NaN, index=idx, columns=['b1', 'b2', 'b3'])
        beta0 = np.ones(3)  # b1, b2 e b3

        for d in tqdm(self.calendar_curves, 'Nelson-Siegel Fitting'):
            for currency in self.currency_list_curve:
                df_curve_input = self._get_one_curve(currency, d, tenors_greater_than_n_years)
                if df_curve_input is None:
                    print(currency, ' ', d, 'ERROR IN THS CURVE')
                    continue
                try:
                    res = minimize(fun=self.NSiegel,
                               x0=beta0,
                               args=df_curve_input,
                               method='SLSQP')
                    if res.success:
                        df_betas.loc[currency, d] = res.x
                except:
                    print(currency, ' ', d, 'ERROR IN THS CURVE')
                    continue
        return df_betas

    def _get_df_curvature_all_tenors(self):
        df_curvature_all_tenors = pd.DataFrame(index=self.interest_curve.index.get_level_values(1).unique(), columns=self.currency_list_curve)
        for currency in self.currency_list_curve:
            df_curvature_all_tenors[currency] = self.nsiegel_betas_all_tenors.loc[(currency), 'b3']
        return df_curvature_all_tenors

    def _get_df_specific_beta(self, df_betas, beta='b3'):
        df_betas_result = pd.DataFrame(index=self.daily_calendar, columns=self.currency_list_curve)
        for currency in self.currency_list_curve:
            df_betas_result[currency] = df_betas.loc[(currency), beta]
        return df_betas_result

    def _get_df_curvature_from_three_month_tenor(self):
        df_curvature_from_three_month_tenor = pd.DataFrame(index=self.interest_curve.get_level_values(1).unique(), columns=self.currency_list_curve)
        for currency in self.currency_list_curve:
            df_curvature_from_three_month_tenor[currency] = self.nsiegel_betas_3month.loc[(currency), 'b3']
        return df_curvature_from_three_month_tenor

    def _get_df_level_from_three_month_tenor(self):
        df_level_from_three_month_tenor = pd.DataFrame(index=self.interest_curve.get_level_values(1).unique(), columns=self.currency_list_curve)
        for currency in self.currency_list_curve:
            df_level_from_three_month_tenor[currency] = self.nsiegel_betas_3month.loc[(currency), 'b1']
        return df_level_from_three_month_tenor

    def _get_df_slope_from_three_month_tenor(self):
        df_slope_from_three_month_tenor = pd.DataFrame(index=self.interest_curve.get_level_values(1).unique(), columns=self.currency_list_curve)
        for currency in self.currency_list_curve:
            df_slope_from_three_month_tenor[currency] = self.nsiegel_betas_3month.loc[(currency), 'b2']
        return df_slope_from_three_month_tenor

    def plot_curvature_all_tenors(self, figure_size, chart_title="Not so good usging all tenors"):
        # Outliers coming from NOK and SEK, specially from short tenors (Sept 1992)
        self.curvature_all_tenors.plot(figsize=figure_size, title=chart_title)
        plt.show()

    def plot_curvature_from_three_month(self, figure_size, chart_title='Better if start from 3Mo'):
        self.curvature_from_three_month_tenor.plot(figsize=figure_size, title=chart_title)
        plt.show()

    def plot_level_from_three_month(self, figure_size, chart_title='Level (beta1)'):
        self.level_from_three_month_tenor.plot(figsize=figure_size, title=chart_title)
        plt.show()

    def plot_slope_from_three_month(self, figure_size, chart_title='Slope (beta2)'):
        self.slope_from_three_month_tenor.plot(figsize=figure_size, title=chart_title)
        plt.show()

    def standard_deviation_curvature(self):
        stdev_b3 = pd.DataFrame(self.curvature_all_tenors.std(), columns=['All tenors'])
        stdev_b3['From 3Mo on'] = self.curvature_from_three_month_tenor.std()
        stdev_b3['Dif'] = stdev_b3['From 3Mo on'] - stdev_b3['All tenors']
        return stdev_b3

    def _relative_curvature(self):
        df_relative_curvature = pd.DataFrame(index=self.calendar_curves, columns=self.currency_list)
        df_relative_curvature = self.curvature_from_three_month_tenor[self.currency_list]
        for currency in self.currency_list:
            df_relative_curvature.loc[:, currency] = self.curvature_from_three_month_tenor.loc[:, currency] - self.curvature_from_three_month_tenor.loc[:, 'USD']
        return df_relative_curvature

    def _relative_level(self):
        df_relative_level = pd.DataFrame(index=self.calendar_curves, columns=self.currency_list)
        df_relative_level = self.level_from_three_month_tenor[self.currency_list]
        for currency in self.currency_list:
            df_relative_level.loc[:, currency] = self.level_from_three_month_tenor.loc[:, currency] - self.level_from_three_month_tenor.loc[:, 'USD']
        return df_relative_level

    def _relative_slope(self):
        df_relative_slope = pd.DataFrame(index=self.calendar_curves, columns=self.currency_list)
        df_relative_slope = self.slope_from_three_month_tenor[self.currency_list]
        for currency in self.currency_list:
            df_relative_slope.loc[:, currency] = self.slope_from_three_month_tenor.loc[:, currency] - self.slope_from_three_month_tenor.loc[:, 'USD']
        return df_relative_slope

    def run_table_four_five(self):
        time_start = time.time()
        iterables_strat = [['Forward discount', 'Curvature'], self.k_range]
        idx = pd.MultiIndex.from_product(iterables_strat, names=['Strategy', 'k'])
        table04_Funding = pd.DataFrame(data=None, index=idx, columns=self.currency_list)
        table05_Investing = pd.DataFrame(data=None, index=idx, columns=self.currency_list)

        for i in self.k_range:
            # table04 Funding Currencies
            table04_Funding.loc[('Forward discount', i)] = (self.long_short_signals_df_01.loc['k' + str(i)] < -0.01).sum()
            table04_Funding.loc[('Curvature', i)] = (self.long_short_signals_df_02.loc['k' + str(i)] < -0.01).sum()
            # table05 Investing Currencies
            table05_Investing.loc[('Forward discount', i)] = (self.long_short_signals_df_01.loc['k' + str(i)] > 0.01).sum()
            table05_Investing.loc[('Curvature', i)] = (self.long_short_signals_df_02.loc['k' + str(i)] > 0.01).sum()

        self.table04_Funding = table04_Funding
        self.table05_Investing = table05_Investing
        print('tables 4 and 5 OK - ', time.time() - time_start, ' seconds')

    def GetPerformanceTable(self, IndexSeries, freq='Daily'):
        adju_factor = 252.0
        if freq == 'Monthly':
            adju_factor = 12.0
        elif freq == 'Weekly':
            adju_factor = 52.0

        Table = pd.Series(index=['Excess Return', 'Volatility', 'Sharpe', 'Sortino', 'Max Drawdown',
                                 'Max Drawdown in Vol Terms', '5th percentile in Vol Terms',
                                 '10th percentile in Vol Terms'])

        CleanIndexSeries = IndexSeries.copy(deep=True)
        for d, d_minus_1 in zip(CleanIndexSeries.index[1:], CleanIndexSeries.index[:-1]):
            if (CleanIndexSeries.loc[d] / CleanIndexSeries.loc[d_minus_1] -1) != 0:
                CleanIndexSeries = CleanIndexSeries.loc[d_minus_1:]
                break

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
        Table['Max Drawdown'] = self.max_dd(ER_index)
        Table['Max Drawdown in Vol Terms'] = self.max_dd(ER_index) / Table['Volatility']
        Table['5th percentile in Vol Terms'] = (ER_index.pct_change(1).dropna()).quantile(q=0.05) / Table['Volatility']
        Table['10th percentile in Vol Terms'] = (ER_index.pct_change(1).dropna()).quantile(q=0.1) / Table['Volatility']
        return Table

    @staticmethod
    def max_dd(ser):
        max2here = ser.expanding(min_periods=1).max()
        dd2here = ser / max2here - 1.0
        return dd2here.min()

    def build_performance_df_monthly(self):
        freq='Monthly'
        column_names = ['Excess Return', 'Volatility', 'Sharpe', 'Sortino', 'Skewness', 'Kurtosis', 'Max Drawdown',
                      'Max Drawdown in Vol Terms', '5th percentile in Vol Terms', '10th percentile in Vol Terms']
        list_of_strategy_names = list(self.df_strategy['Name'])
        idx = pd.MultiIndex.from_product(
            iterables=[list_of_strategy_names, self.k_range], names=['Sorting', 'k'])

            # iterables=[['Carry trade', 'Carry (EV)', 'Level', 'Level (EV)', 'Slope', 'Slope (EV)', 'Curvature',
            #             'Curvature (EV)', 'Carry-Curvy DS'],
            #            self.k_range],
            # names=['Sorting', 'k'])

        performance_df_monthly = pd.DataFrame(data=None, index=idx, columns=column_names)

        for i in range(1, self.total_strategy_number + 1):
            for k in self.k_range:
                TR_string = self.df_strategy.loc[i, 'TR']
                performance_df_monthly.loc[(list_of_strategy_names[i], k)] = self.GetPerformanceTable(getattr(self, TR_string)['k' + str(k)], freq=freq)

        return performance_df_monthly

    def build_performance_df(self, freq='Daily'):
        column_names = ['Excess Return', 'Volatility', 'Sharpe', 'Sortino', 'Skewness', 'Kurtosis', 'Max Drawdown',
                      'Max Drawdown in Vol Terms', '5th percentile in Vol Terms', '10th percentile in Vol Terms']
        list_of_strategy_names = list(self.df_strategy['Name'])
        idx = pd.MultiIndex.from_product(
            iterables=[list_of_strategy_names, self.k_range], names=['Sorting', 'k'])

            # iterables=[['Carry trade', 'Carry (EV)', 'Level', 'Level (EV)', 'Slope', 'Slope (EV)', 'Curvature',
            #             'Curvature (EV)', 'Carry-Curvy DS'],
            #            self.k_range],
            # names=['Sorting', 'k'])

        performance_df = pd.DataFrame(data=None, index=idx, columns=column_names)

        for i in range(1, self.total_strategy_number + 1):
            for k in self.k_range:
                TR_string = self.df_strategy.loc[i, 'TR']
                performance_df.loc[(list_of_strategy_names[i], k)] = self.GetPerformanceTable(getattr(self, TR_string)['k' + str(k)], freq=freq)

        return performance_df

    def table_three(self):
        freq = 'Monthly'
        list_of_strategy_names = list(self.df_strategy['Name'])
        idx = pd.MultiIndex.from_product(iterables=[list_of_strategy_names, self.k_range], names=['Sorting', 'k'])
        # idx = pd.MultiIndex.from_product(
        #     iterables=[['Carry trade', 'Carry (EV)', 'Level', 'Level (EV)', 'Slope', 'Slope (EV)', 'Curvature', 'Curvature (EV)'],
        #                self.k_range],
        #     names=['Sorting', 'k'])
        column_names = ['Mean annual', 'Stdev annual', 'Skewness monthly', 'Kurtosis monthly', 'Sharpe ratio']
        dict_columns = {'Mean annual': 'Excess Return',
                        'Stdev annual': 'Volatility',
                        'Skewness monthly': 'Skewness',
                        'Kurtosis monthly': 'Kurtosis',
                        'Sharpe ratio': 'Sharpe'}
        dict_mult = {'Excess Return': 100.0,
                     'Volatility': 100.0,
                     'Skewness': 1.0,
                     'Kurtosis': 1.0,
                     'Sharpe': 1.0}
        table03 = pd.DataFrame(index=idx, columns=column_names)
        performance_data = lambda k, n, field: self.GetPerformanceTable(getattr(self, self.df_strategy.loc[n, 'TR'])['k' + str(k)], freq=freq)[field] * dict_mult[field]
        for strategy_name in list_of_strategy_names:
            for column_name in column_names:
                field = dict_columns[column_name]
                n = self.df_strategy[self.df_strategy['Name'] == strategy_name].index[0]
                table03.loc[(strategy_name,), column_name] = [performance_data(k, n, field) for k in self.k_range]
        return table03

    def table_three_daily(self):
        freq = 'Daily'
        list_of_strategy_names = list(self.df_strategy['Name'])
        idx = pd.MultiIndex.from_product(iterables=[list_of_strategy_names, self.k_range], names=['Sorting', 'k'])
        # idx = pd.MultiIndex.from_product(
        # iterables=[['Carry trade', 'Carry (EV)', 'Level', 'Level (EV)', 'Slope', 'Slope (EV)', 'Curvature', 'Curvature (EV)'],
        #                self.k_range],
        # names=['Sorting', 'k'])
        column_names = ['Mean annual', 'Stdev annual', 'Skewness monthly', 'Kurtosis monthly', 'Sharpe ratio']
        dict_columns = {'Mean annual': 'Excess Return',
                        'Stdev annual': 'Volatility',
                        'Skewness monthly': 'Skewness',
                        'Kurtosis monthly': 'Kurtosis',
                        'Sharpe ratio': 'Sharpe'}
        dict_mult = {'Excess Return': 100.0,
                     'Volatility': 100.0,
                     'Skewness': 1.0,
                     'Kurtosis': 1.0,
                     'Sharpe': 1.0}
        table03_daily = pd.DataFrame(index=idx, columns=column_names)
        performance_data = lambda k, n, field: self.GetPerformanceTable(getattr(self, self.df_strategy.loc[n, 'TR_df_daily'])['k' + str(k)], freq=freq)[field] * dict_mult[field]
        for strategy_name in list_of_strategy_names:
            for column_name in column_names:
                field = dict_columns[column_name]
                n = self.df_strategy[self.df_strategy['Name'] == strategy_name].index[0]
                table03_daily.loc[(strategy_name,), column_name] = [performance_data(k, n, field) for k in self.k_range]
        return table03_daily


    def table_three_daily_period(self, start='1991', end='2019'):
        freq = 'Daily'
        list_of_strategy_names = list(self.df_strategy['Name'])
        idx = pd.MultiIndex.from_product(iterables=[list_of_strategy_names, self.k_range], names=['Sorting', 'k'])
        # idx = pd.MultiIndex.from_product(
        # iterables=[['Carry trade', 'Carry (EV)', 'Level', 'Level (EV)', 'Slope', 'Slope (EV)', 'Curvature', 'Curvature (EV)'],
        #        self.k_range],
        # names=['Sorting', 'k'])
        column_names = ['Mean annual', 'Stdev annual', 'Skewness monthly', 'Kurtosis monthly', 'Sharpe ratio']
        dict_columns = {'Mean annual': 'Excess Return',
                        'Stdev annual': 'Volatility',
                        'Skewness monthly': 'Skewness',
                        'Kurtosis monthly': 'Kurtosis',
                        'Sharpe ratio': 'Sharpe'}
        # dict_mult = {'Excess Return': 100.0,
        #                 'Volatility': 100.0,
        #                 'Skewness': 1.0,
        #                 'Kurtosis': 1.0,
        #                 'Sharpe': 1.0}
        table03_daily = pd.DataFrame(index=idx, columns=column_names)
        # performance_data = lambda k, n, field: self.GetPerformanceTable(getattr(self, self.df_strategy.loc[n, 'TR_df_daily']).loc[start:end, 'k' + str(k)], freq=freq)[field] * dict_mult[field]
        for strategy_name in list_of_strategy_names:
            for column_name in column_names:
                field = dict_columns[column_name]
                n = self.df_strategy[self.df_strategy['Name'] == strategy_name].index[0]
                # table03_daily.loc[(strategy_name,), column_name] = [self.performance_data(k, n, field) for k in self.k_range]
                table03_daily.loc[(strategy_name,), column_name] = [self.performance_data(k, n, field, start, end, freq) for k in self.k_range]
        return table03_daily

    def performance_data(self, k, n, field, start, end, freq):
        dict_mult = {'Excess Return': 100.0,
                     'Volatility': 100.0,
                     'Skewness': 1.0,
                     'Kurtosis': 1.0,
                     'Sharpe': 1.0}
        try:
            result = self.GetPerformanceTable(getattr(self, self.df_strategy.loc[n, 'TR_df_daily']).loc[start:end, 'k' + str(k)], freq=freq)[field] * dict_mult[field]
        except:
            result = np.NaN
        return result

    def table_three_monthly_period(self, start='1991', end='2019'):
        freq = 'Monthly'
        list_of_strategy_names = list(self.df_strategy['Name'])
        idx = pd.MultiIndex.from_product(iterables=[list_of_strategy_names, self.k_range], names=['Sorting', 'k'])
        # idx = pd.MultiIndex.from_product(
        # iterables=[['Carry trade', 'Carry (EV)', 'Level', 'Level (EV)', 'Slope', 'Slope (EV)', 'Curvature', 'Curvature (EV)'],
        #        self.k_range],
        # names=['Sorting', 'k'])
        column_names = ['Mean annual', 'Stdev annual', 'Skewness monthly', 'Kurtosis monthly', 'Sharpe ratio']
        dict_columns = {'Mean annual': 'Excess Return',
                        'Stdev annual': 'Volatility',
                        'Skewness monthly': 'Skewness',
                        'Kurtosis monthly': 'Kurtosis',
                        'Sharpe ratio': 'Sharpe'}
        dict_mult = {'Excess Return': 100.0,
                        'Volatility': 100.0,
                        'Skewness': 1.0,
                        'Kurtosis': 1.0,
                        'Sharpe': 1.0}
        table03_monthly = pd.DataFrame(index=idx, columns=column_names)
        performance_data = lambda k, n, field: self.GetPerformanceTable(getattr(self, self.df_strategy.loc[n, 'TR']).loc[start:end, 'k' + str(k)], freq=freq)[field] * dict_mult[field]
        for strategy_name in list_of_strategy_names:
            for column_name in column_names:
                field = dict_columns[column_name]
                n = self.df_strategy[self.df_strategy['Name'] == strategy_name].index[0]
                table03_monthly.loc[(strategy_name,), column_name] = [performance_data(k, n, field) for k in self.k_range]
        return table03_monthly

    def _get_fwdpts_curve_bbg(self, ticker_list_curve=None, ini_date_bbg=None, end_date_bbg=None):
        if ticker_list_curve is None:
            ticker_list_curve = ['AUD2W BGN Curncy',
                                 'AUD3M BGN Curncy',
                                 'AUD2M BGN Curncy',
                                 'AUD6M BGN Curncy',
                                 'AUD4M BGN Curncy',
                                 'AUD1M BGN Curncy',
                                 'AUD3W BGN Curncy',
                                 'AUD5M BGN Curncy',
                                 'AUD1W BGN Curncy']
        if ini_date_bbg is None:
            ini_date_bbg = self.ini_date_bbg
        if end_date_bbg is None:
            end_date_bbg = self.end_date_bbg

        # Get data from BBG
        df_fwdpts_curve_bbg = self.con.bdh(ticker_list_curve, self.bbg_field_last, ini_date_bbg, end_date_bbg, elms=[("periodicitySelection", "DAILY")])
        # Clean and organize
        df_fwdpts_curve_bbg.columns = df_fwdpts_curve_bbg.columns.droplevel(level='field')
        df_fwdpts_curve_bbg.columns = [self.fwd_dict[ticker[3:]] for ticker in df_fwdpts_curve_bbg.columns]
        df_fwdpts_curve_bbg[0] = 0.0
        df_fwdpts_curve_bbg.sort_index(axis=1, inplace=True)
        df_fwdpts_curve_bbg.fillna(method='ffill', inplace=True)
        df_fwdpts_curve_bbg.sort_index(axis=1, inplace=True)
        return df_fwdpts_curve_bbg

    def _get_daily_fwdpts_curve_data(self):
        idx = pd.MultiIndex.from_product(iterables=[self.currency_list, self.daily_calendar])
        df_daily_fwdpts = pd.DataFrame(data=None, index=idx, columns=[float(x) for x in range(0, 182)])
        # Build DataFrame with data from bbg
        take_second = lambda s1, s2: s2
        for currency in tqdm(self.currency_list, 'Daily forward points...'):
            list_of_tenors = list(self.df_tickers.loc[currency, {'1w', '2w', '3w', '1m', '2m', '3m', '4m', '5m', '6m'}])
            if currency == 'RUB':
                list_of_tenors = list(self.df_tickers.loc[currency, {'1m', '2m', '3m', '4m', '5m', '6m'}])
            # Get data from bbg
            df_fwdpts_aux_bbg = self._get_fwdpts_curve_bbg(list_of_tenors)
            # Data handling
            idx_aux = pd.MultiIndex.from_product(iterables=[[currency], self.daily_calendar])
            # df_fwdpts_aux is a DataFrame with simple index (axis0=dates, axis1=tenors in actual days)
            df_fwdpts_aux = df_daily_fwdpts.loc[currency].combine(other=df_fwdpts_aux_bbg, func=take_second)
            df_fwdpts_aux.fillna(method='ffill', axis=0, inplace=True)
            # Fill missing tenors with interpolated data
            df_fwdpts_aux.interpolate(method='linear', axis=1, inplace=True)
            # Changing index to match the index of df_daily_fwdpts
            df_fwdpts_aux.set_index(idx_aux, drop=True, inplace=True)
            df_daily_fwdpts.loc[currency] = df_fwdpts_aux
        # Scale adjust for fwd pts
        df_daily_fwdpts_scale_adj = df_daily_fwdpts.copy(deep=True)
        for currency in self.currency_list:
            df_daily_fwdpts_scale_adj.loc[currency] = (df_daily_fwdpts.loc[currency] / self.ticker_scale.loc[currency]).values
        return df_daily_fwdpts_scale_adj

    def _get_daily_fwd_all_tenors(self, df_daily_fwdpts):
        # DataFrame for daily fwds (outright)
        idx = pd.MultiIndex.from_product(iterables=[self.currency_list, self.daily_calendar])
        df_daily_spot = pd.DataFrame(data=None, index=idx, columns=[float(x) for x in range(0, 182)])
        df_daily_fwds = pd.DataFrame(data=None, index=idx, columns=[float(x) for x in range(0, 182)])
        for currency in self.currency_list:
            df_daily_spot.loc[(currency,), 0] = self.spot_last.loc[self.daily_calendar, currency].values
        df_daily_spot.fillna(method='ffill', axis=1, inplace=True)
        df_daily_fwds = df_daily_spot + df_daily_fwdpts
        return df_daily_fwds

    def _get_daily_fwd_XXXUSD(self, df_daily_fwds):
        df_daily_fwds_XXXUSD = df_daily_fwds.copy(deep=True)
        for currency in self.currency_list:
            df_daily_fwds_XXXUSD.loc[currency] = (df_daily_fwds.loc[currency] ** self.ticker_inverse.loc[currency]).values
        return df_daily_fwds_XXXUSD

    @staticmethod
    def eomonth(date):
        return (date - pd.Timedelta(1, unit='d')) + BMonthEnd(1)

    def _price_change_matrix(self):
        # DataFrame with price changes of holdings (eg: price change of a 14 days fwd)
        df_px_change = pd.DataFrame(data=None, index=self.daily_calendar, columns=self.currency_list)
        df_px_change['Maturity'] = self.daily_calendar.copy(deep=True)
        df_px_change['Maturity'] = df_px_change['Maturity'].apply(self.eomonth)
        time_delta_aux = (df_px_change['Maturity'] - df_px_change['Maturity'].index)
        # number of days to maturity (forward)
        df_px_change['days'] = time_delta_aux.astype('timedelta64[D]')
        # corresponding price change of positions
        fwd_aux = self.daily_fwds_XXXUSD.unstack(level=-2)
        for dt, tm1, ndays in zip(self.daily_calendar[1:], self.daily_calendar[:-1], df_px_change['days'][1:]):
            delta_day = (dt - tm1).days
            # price change of a forward position
            df_px_change.loc[dt, self.currency_list] = fwd_aux.loc[dt, ndays][self.currency_list] - fwd_aux.loc[tm1, (ndays + delta_day)][self.currency_list]
        df_px_change.loc['1991-01-01'] = np.NaN  # erasing first row
        return df_px_change

    def _calc_daily_pnl(self, holdings):
        hds_daily = pd.DataFrame(data=np.NaN, index=self.daily_calendar, columns=self.currency_list)
        take_second = lambda s1, s2: s2
        hds_daily = hds_daily.combine(other=holdings, func=take_second)
        hds_daily.fillna(method='ffill', inplace=True)
        # daily_tr_index = pd.DataFrame(data=None, index=self.daily_calendar, columns=self.currency_list)
        daily_tr_index = 100.0  # initial point
        daily_pnl = (hds_daily[self.currency_list].shift(1) * self.px_change_matrix[self.currency_list]).sum(axis=1)
        daily_tr_index = daily_tr_index + daily_pnl.cumsum()
        return daily_tr_index

    def run_daily_pnl(self, df_holdings, strategy_n=0):
        TR_daily_df = pd.DataFrame(index=self.daily_calendar, columns=self.k_list)
        for k in self.k_range:
            TR_daily_df['k' + str(k)] = self._calc_daily_pnl(df_holdings.loc['k' + str(k)])
        #Write results in 'data' folder as Excel file. File will be name according with strategy number
        name_start = self.data_folder + "\\" + "{:02}".format(strategy_n) + '_'
        TR_daily_df.to_excel(name_start + 'TR_Daily.xlsx')
        return TR_daily_df

    def _run_NSiegel_fitting_sequential(self, tenors_greater_than_n_years=0.0):
        # Run Nelson-Siegel fitting only for available dates
        idx = pd.MultiIndex.from_product(iterables=[self.currency_list_curve, self.daily_calendar])
        df_betas = pd.DataFrame(data=np.NaN, index=idx, columns=['b1', 'b2', 'b3'])
        beta0 = np.ones(3)  # b1, b2 e b3

        combination_list = list(product(self.currency_list_curve, self.daily_calendar))
        key_tenors = pd.DataFrame(data=[6.0, 12.0, 24.0, 60.0, 120.0, 180.0],
                                  index=[6.0, 12.0, 24.0, 60.0, 120.0, 180.0])

        def _run_fitting(currency_and_date_tuple):
            d = currency_and_date_tuple[1]
            currency = currency_and_date_tuple[0]
            df_curve_input = self._get_one_curve(currency, str(d), tenors_greater_than_n_years)
            NAN_Value = np.NaN
            if df_curve_input.empty:
                df_betas.loc[currency, d] = NAN_Value
            else:
                if df_curve_input.count() < 2:
                    df_betas.loc[currency, d] = NAN_Value
                else:
                    # Insert longer tenor for better fitting of level
                    max_tenor = df_curve_input.index.max()
                    max_tenor_rate = df_curve_input.loc[max_tenor]
                    if max_tenor < 240.0:
                        for tenor in list(key_tenors.loc[key_tenors[0] > max_tenor].index):
                            df_curve_input.loc[tenor] = max_tenor_rate
                    try:
                        res = minimize(fun=self.NSiegel,
                                   x0=beta0,
                                   args=df_curve_input,
                                   method='SLSQP')
                        if res.success:
                            df_betas.loc[currency, d] = res.x
                            # return res.x
                    except:
                        # print(currency, ' ', d, 'ERROR IN THS CURVE')
                        df_betas.loc[currency, d] = NAN_Value
                        # return beta0 * np.NaN
                        # results = res.x
                        # return results

        for i in tqdm(combination_list, 'NS-Fitting...'):
            _run_fitting(i)

        return df_betas

    def run_USD_index_all(self):
        """
        Calculates dollar index for all currencies in self.currency_list.
        Returns
        -------
        Pandas.DataFrame
        """
        self.build_currency_list_EM()
        USD_index = pd.DataFrame(data=None, index=self.daily_calendar, columns=['USD', 'USD_DM', 'USD_EM'])
        USD_index.loc[self.daily_calendar[0]] = 100.0
        for d, tm1 in zip(self.daily_calendar[1:], self.daily_calendar[:-1]):
            # Performance of USD, so this formula (t_minus_1 / t - 1) is right
            try:
                USD_index.loc[d, 'USD_DM'] = USD_index.loc[tm1, 'USD_DM'] * (1 + (self.spot_last_XXXUSD.loc[tm1, self.currency_list_DM] / self.spot_last_XXXUSD.loc[d, self.currency_list_DM] - 1).mean())
                USD_index.loc[d, 'USD_EM'] = USD_index.loc[tm1, 'USD_EM'] * (1 + (self.spot_last_XXXUSD.loc[tm1, self.currency_list_EM] / self.spot_last_XXXUSD.loc[d, self.currency_list_EM] - 1).mean())
                # For all currencies index, adjust proportion of EM/G10 currencies
                USD_index.loc[d, 'USD'] = USD_index.loc[tm1, 'USD'] * ( 1+ (self.spot_last_XXXUSD.loc[tm1] / self.spot_last_XXXUSD.loc[d] -1).mean())
            except  KeyError:
                pass
        return USD_index

    def build_currency_list_EM(self):
        currency_list_EM = []
        for currency in self.currency_list:
            if currency not in self.currency_list_DM:
                currency_list_EM.append(currency)
        self.currency_list_EM = currency_list_EM
        return currency_list_EM

    def beta_USD(self, prices_df, usd_index_name='USD', window=84):
        # Try reading USD index from all currencies results. It has all three indexes (DM, EM, USD).
        try:
            self.USD_index = pd.read_excel('data_ALL\\USD_index.xlsx')
        except FileNotFoundError:
            pass
        return_index_df = prices_df.copy(deep=True)
        return_index_df.loc[self.USD_index.index, 'USD'] = self.USD_index[str(usd_index_name)]
        covariances = return_index_df.pct_change(1).rolling(window).cov()
        covariance_USD = covariances.xs(key='USD', axis=0, level=1)
        beta_USD_df = pd.DataFrame(data=None, index=covariance_USD.index, columns=covariance_USD.columns)
        for d in beta_USD_df.index:
            beta_USD_df.loc[d] = covariance_USD.loc[d] / covariance_USD['USD'].loc[d]
        return beta_USD_df

    def run_strategy09(self):
        time_start = time.time()
        # Double sorting: first carry, second curvature
        strategy_n = 9
        strategy_name = 'Carry-Curvy DS'
        name_start = self.data_folder + "\\" + "{:02}".format(strategy_n) + '_'
        try:
            # Try to read results from Excel files, otherwise run and write to Excel
            df_excel_tr         = pd.read_excel(name_start + 'TR.xlsx')
            df_excel_holdings   = pd.read_excel(name_start + 'Holdings.xlsx', index_col=[0,1])
            df_excel_weights    = pd.read_excel(name_start + 'Weights.xlsx', index_col=[0,1])
            df_excel_ls_signals = pd.read_excel(name_start + 'LS_Signals.xlsx', index_col=[0,1])
            df_excel_ir         = pd.read_excel(name_start + 'IR.xlsx')
            df_excel_sr         = pd.read_excel(name_start + 'SR.xlsx')
            df_excel_cost       = pd.read_excel(name_start + 'Cost.xlsx')
            df_excel_pnlfx      = pd.read_excel(name_start + 'PctPnlFX.xlsx', index_col=[0,1])
            df_excel_irfx       = pd.read_excel(name_start + 'PctIRFX.xlsx', index_col=[0,1])
            df_excel_srfx       = pd.read_excel(name_start + 'PctSRFX.xlsx', index_col=[0,1])
            df_excel_costfx     = pd.read_excel(name_start + 'PctCostFX.xlsx', index_col=[0,1])
            dt_excel_daily_pnl  = pd.read_excel(name_start + 'TR_Daily.xlsx')
            self.TR_df_09                    = df_excel_tr
            self.holdings_df_09              = df_excel_holdings
            self.weights_df_09               = df_excel_weights
            self.long_short_signals_df_09    = df_excel_ls_signals
            self.ir_09                       = df_excel_ir
            self.sr_09                       = df_excel_sr
            self.cost_09                     = df_excel_cost
            self.pct_pnlfx_09                = df_excel_pnlfx
            self.pct_irfx_09                 = df_excel_irfx
            self.pct_srfx_09                 = df_excel_srfx
            self.pct_costfx_09               = df_excel_costfx
            self.TR_df_daily_09              = dt_excel_daily_pnl
        except FileNotFoundError:
            # this function already writes results do excel (just need strategy_n): self.run_equal_vol_monthly_strategy
            [self.TR_df_09, self.holdings_df_09, self.weights_df_09, self.long_short_signals_df_09, self.ir_09, self.sr_09,
             self.cost_09, self.pct_pnlfx_09, self.pct_irfx_09, self.pct_srfx_09, self.pct_costfx_09, self.blotter_09] = \
                self.run_default_monthly_strategy_double_sorting(df_signals1=self.fwd_discount_last, df_signals2=self.relative_curvature, func_weight=self._ranking_to_wgt_double_sorting, strategy_n=strategy_n, groups=self.double_sorting_groups, subgroups=self.double_sorting_subgroups)
            # self.TR_df_daily_09 = self.run_daily_pnl(self.holdings_df_09, strategy_n)
            [self.TR_df_daily_09, self.TR_df_daily_fx_09] = self.calculate_daily_pnl(df_blotter=self.blotter_09)
            (self.TR_df_daily_09.loc['1991-01-31':]+100).to_excel(name_start + 'TR_Daily.xlsx')
            (self.TR_df_daily_fx_09 + 100).to_excel(name_start + 'TR_DailyFX.xlsx')
        print(strategy_name + ' OK - ', time.time() - time_start, ' seconds')

    def run_strategy10(self):
        time_start = time.time()
        # Double sorting: first carry, second curvature
        strategy_n = 10
        strategy_name = 'Curvy-Carry DS'
        name_start = self.data_folder + "\\" + "{:02}".format(strategy_n) + '_'
        try:
            # Try to read results from Excel files, otherwise run and write to Excel
            df_excel_tr = pd.read_excel(name_start + 'TR.xlsx')
            df_excel_holdings = pd.read_excel(name_start + 'Holdings.xlsx', index_col=[0, 1])
            df_excel_weights = pd.read_excel(name_start + 'Weights.xlsx', index_col=[0, 1])
            df_excel_ls_signals = pd.read_excel(name_start + 'LS_Signals.xlsx', index_col=[0, 1])
            df_excel_ir = pd.read_excel(name_start + 'IR.xlsx')
            df_excel_sr = pd.read_excel(name_start + 'SR.xlsx')
            df_excel_cost = pd.read_excel(name_start + 'Cost.xlsx')
            df_excel_pnlfx = pd.read_excel(name_start + 'PctPnlFX.xlsx', index_col=[0, 1])
            df_excel_irfx = pd.read_excel(name_start + 'PctIRFX.xlsx', index_col=[0, 1])
            df_excel_srfx = pd.read_excel(name_start + 'PctSRFX.xlsx', index_col=[0, 1])
            df_excel_costfx = pd.read_excel(name_start + 'PctCostFX.xlsx', index_col=[0, 1])
            dt_excel_daily_pnl = pd.read_excel(name_start + 'TR_Daily.xlsx')
            self.TR_df_10 = df_excel_tr
            self.holdings_df_10 = df_excel_holdings
            self.weights_df_10 = df_excel_weights
            self.long_short_signals_df_10 = df_excel_ls_signals
            self.ir_10 = df_excel_ir
            self.sr_10 = df_excel_sr
            self.cost_10 = df_excel_cost
            self.pct_pnlfx_10 = df_excel_pnlfx
            self.pct_irfx_10 = df_excel_irfx
            self.pct_srfx_10 = df_excel_srfx
            self.pct_costfx_10 = df_excel_costfx
            self.TR_df_daily_10 = dt_excel_daily_pnl
        except FileNotFoundError:
            # this function already writes results do excel (just need strategy_n): self.run_equal_vol_monthly_strategy
            [self.TR_df_10, self.holdings_df_10, self.weights_df_10, self.long_short_signals_df_10, self.ir_10, self.sr_10,
             self.cost_10, self.pct_pnlfx_10, self.pct_irfx_10, self.pct_srfx_10, self.pct_costfx_10, self.blotter_10] = \
                self.run_default_monthly_strategy_double_sorting(df_signals1=self.relative_curvature, df_signals2=self.fwd_discount_last, func_weight=self._ranking_to_wgt_double_sorting, strategy_n=strategy_n, groups=self.double_sorting_groups, subgroups=self.double_sorting_subgroups)
            # self.TR_df_daily_10 = self.run_daily_pnl(self.holdings_df_10, strategy_n)
            [self.TR_df_daily_10, self.TR_df_daily_fx_10] = self.calculate_daily_pnl(df_blotter=self.blotter_10)
            (self.TR_df_daily_10.loc['1991-01-31':]+100).to_excel(name_start + 'TR_Daily.xlsx')
            (self.TR_df_daily_fx_10 + 100).to_excel(name_start + 'TR_DailyFX.xlsx')
        print(strategy_name + ' OK - ', time.time() - time_start, ' seconds')

    def run_strategy11(self):
        time_start = time.time()
        # Traditional Carry Trade, using carry/vol as criteria
        strategy_n = 11
        name_start = self.data_folder + "\\" + "{:02}".format(strategy_n) + '_'
        strategy_criteria = self.fwd_discount_last / self.vols
        try:
            # Try to read results from Excel files, otherwise run and write to Excel
            df_excel_tr         = pd.read_excel(name_start + 'TR.xlsx')
            df_excel_holdings   = pd.read_excel(name_start + 'Holdings.xlsx', index_col=[0,1])
            df_excel_weights    = pd.read_excel(name_start + 'Weights.xlsx', index_col=[0,1])
            df_excel_ls_signals = pd.read_excel(name_start + 'LS_Signals.xlsx', index_col=[0,1])
            df_excel_ir         = pd.read_excel(name_start + 'IR.xlsx')
            df_excel_sr         = pd.read_excel(name_start + 'SR.xlsx')
            df_excel_cost       = pd.read_excel(name_start + 'Cost.xlsx')
            df_excel_pnlfx      = pd.read_excel(name_start + 'PctPnlFX.xlsx', index_col=[0,1])
            df_excel_irfx       = pd.read_excel(name_start + 'PctIRFX.xlsx', index_col=[0,1])
            df_excel_srfx       = pd.read_excel(name_start + 'PctSRFX.xlsx', index_col=[0,1])
            df_excel_costfx     = pd.read_excel(name_start + 'PctCostFX.xlsx', index_col=[0,1])
            dt_excel_daily_pnl  = pd.read_excel(name_start + 'TR_Daily.xlsx')
            self.TR_df_11                    = df_excel_tr
            self.holdings_df_11              = df_excel_holdings
            self.weights_df_11               = df_excel_weights
            self.long_short_signals_df_11    = df_excel_ls_signals
            self.ir_11                       = df_excel_ir
            self.sr_11                       = df_excel_sr
            self.cost_11                     = df_excel_cost
            self.pct_pnlfx_11                = df_excel_pnlfx
            self.pct_irfx_11                 = df_excel_irfx
            self.pct_srfx_11                 = df_excel_srfx
            self.pct_costfx_11               = df_excel_costfx
            self.TR_df_daily_11              = dt_excel_daily_pnl
        except FileNotFoundError:
            # this function already writes results do excel (just need strategy_n): self.run_equal_vol_monthly_strategy
            [self.TR_df_11, self.holdings_df_11, self.weights_df_11, self.long_short_signals_df_11, self.ir_11, self.sr_11,
             self.cost_11, self.pct_pnlfx_11, self.pct_irfx_11, self.pct_srfx_11, self.pct_costfx_11, self.blotter_11] = \
                self.run_default_monthly_strategy(df_signals=strategy_criteria, func_weight=self._ranking_to_wgt, strategy_n=strategy_n)
            # self.TR_df_daily_11 = self.run_daily_pnl(self.holdings_df_11, strategy_n)
            [self.TR_df_daily_11, self.TR_df_daily_fx_11] = self.calculate_daily_pnl(df_blotter=self.blotter_11)
            (self.TR_df_daily_11.loc['1991-01-31':]+100).to_excel(name_start + 'TR_Daily.xlsx')
            (self.TR_df_daily_fx_11 + 100).to_excel(name_start + 'TR_DailyFX.xlsx')
        print('Carry trade to vol OK - ', time.time() - time_start, ' seconds')

    def run_strategy12(self, strat_number=12, curncy='AUD'):
        time_start = time.time()
        # Traditional Carry Trade, long all currencies Vs USD
        strategy_n = strat_number
        self.currency_strat12 = curncy
        name_start = self.data_folder + "\\" + "{:02}".format(strategy_n) + '_'
        strategy_criteria = self.fwd_discount_last
        # this function already writes results do excel (just need strategy_n): self.run_equal_vol_monthly_strategy
        # Override k_range and k_list
        _k_range = self.k_range
        _k_list = self.k_list
        self.k_range = range(1, 2)
        self.k_list = ['k1']
        [self.TR_df_12, self.holdings_df_12, self.weights_df_12, self.long_short_signals_df_12, self.ir_12, self.sr_12,
         self.cost_12, self.pct_pnlfx_12, self.pct_irfx_12, self.pct_srfx_12, self.pct_costfx_12, self.blotter_12] = \
            self.run_default_monthly_strategy(df_signals=strategy_criteria, func_weight=self._same_wgt, strategy_n=strategy_n)
        # self.TR_df_daily_12 = self.run_daily_pnl(self.holdings_df_12, strategy_n)
        [self.TR_df_daily_12, self.TR_df_daily_fx_12] = self.calculate_daily_pnl(df_blotter=self.blotter_12)
        (self.TR_df_daily_12.loc['1991-01-31':]+100).to_excel(name_start + 'TR_Daily.xlsx')
        (self.TR_df_daily_fx_12 + 100).to_excel(name_start + 'TR_DailyFX.xlsx')
        # returning previous values for parameters
        self.k_range = _k_range
        self.k_list = _k_list
        print('Strategy12 for ' + curncy + ":", time.time() - time_start, ' seconds')

    def run_strategy12_for_all_fx(self):
        strategy_number=12
        curncy_number_start = 0
        n_currencies = len(self.currency_list)
        for currency_n in range(curncy_number_start, n_currencies):
            currency = self.currency_list[currency_n]
            self.run_strategy12(strat_number=strategy_number, curncy=currency)
            strategy_number += 1

    def run_daily_pnl_for_all_strategies(self, strategy_range):
        strat_range = strategy_range
        if strategy_range == None:
            strat_range = range(1,12)
        for strat_number in strat_range:
            time_start = time.time()
            strategy_n = strat_number
            name_start = self.data_folder + "\\" + "{:02}".format(strategy_n) + '_'
            df_blotter = pd.read_excel(name_start + 'Blotter.xlsx')
            [self.TR_df_daily_12, self.TR_df_daily_fx_12] = self.calculate_daily_pnl(df_blotter=df_blotter)
            (self.TR_df_daily_12.loc['1991-01-31':]+100).to_excel(name_start + 'TR_Daily.xlsx')
            (self.TR_df_daily_fx_12 + 100).to_excel(name_start + 'TR_DailyFX.xlsx')
            print('Daily P&L for ' + str(strat_number) + ":", time.time() - time_start, ' seconds')

    def curva_nsiegel(self, Moeda, Data):
        _betasns = self.nsiegel_betas_3month.loc[(Moeda, Data)]
        _lambda = 0.0609
        b1 = _betasns.loc['b1']
        b2 = _betasns.loc['b2']
        b3 = _betasns.loc['b3']
        prazos = list(self.interest_curve.columns)[0:34]
        _result = pd.DataFrame(data=self.interest_curve.loc[(Moeda, Data)][prazos].values, index=prazos,
                               columns=['Actual'])
        _result['NSiegel'] = np.NaN
        for tau in prazos:
            _aux = ((1 - math.exp(-_lambda * tau)) / (_lambda * tau))
            y_tau = b1 + b2 * _aux + b3 * (_aux - math.exp(-_lambda * tau))
            _result.loc[tau, 'NSiegel'] = y_tau
        print(b1, b2, b3)
        return _result

    def curva_nsiegel_given_betas(self, Moeda, Data, b1, b2, b3):
        _betasns = self.nsiegel_betas_3month.loc[(Moeda, Data)]
        _lambda = 0.0609
        prazos = list(self.interest_curve.columns)[0:34]
        _result = pd.DataFrame(data=self.interest_curve.loc[(Moeda, Data)][prazos].values, index=prazos,
                               columns=['Actual'])
        _result['NSiegel'] = np.NaN
        for tau in prazos:
            _aux = ((1 - math.exp(-_lambda * tau)) / (_lambda * tau))
            y_tau = b1 + b2 * _aux + b3 * (_aux - math.exp(-_lambda * tau))
            _result.loc[tau, 'NSiegel'] = y_tau
        print(b1, b2, b3)
        return _result

    def plot_break01(self, strategy_n, k):
        name_strat = '_' + "{:02}".format(strategy_n)
        nome_str = self.df_strategy.loc[strategy_n, 'Name']
        k_str = 'k' + str(k)
        spot_ret = getattr(self, 'pct_srfx' + name_strat).loc[k_str]
        ir_ret = getattr(self, 'pct_irfx' + name_strat).loc[k_str]
        df = pd.DataFrame(data=spot_ret.sum(), columns=['variação spot'])
        df['juros'] = ir_ret.sum()
        df.plot(kind='bar', figsize=(12, 8), title=nome_str + '(' + k_str + ')' + ': decomposição dos retornos')

    def plot_pnl(self, strategy_n, k):
        name_strat = '_' + "{:02}".format(strategy_n)
        nome_str = self.df_strategy.loc[strategy_n, 'Name']
        k_str = 'k' + str(k)
        spot_ret = getattr(self, 'pct_pnlfx' + name_strat).loc[k_str].sum()
        spot_ret = pd.DataFrame(data=spot_ret, columns=['retornos'])
        spot_ret.plot(kind='bar', figsize=(12, 8), title=nome_str + '(' + k_str + ')' + ': decomposição por moeda')

    def calculate_daily_pnl(self, df_blotter):
        first_trade_dt = min(df_blotter['trade_dt'])
        dates_list = pd.DataFrame(data=None, index=self.daily_calendar).loc[first_trade_dt:].index
        list_to_do = tqdm(dates_list)
        df_blotter['ndays'] = 0
        df_blotter['k'] = ['k' + str(i) for i in df_blotter['k']]
        TR_df_daily_new = pd.DataFrame(data=None, index=self.daily_calendar, columns=self.k_list)
        idx = pd.MultiIndex.from_product(iterables=[list(self.k_list), self.daily_calendar])
        TR_df_daily_pnl_byfx = pd.DataFrame(data=None, index=idx, columns=self.currency_list)

        for d in list_to_do:
            # Pnl of not expired trades
            live_trades = (df_blotter['maturity'] >= d) & (df_blotter['trade_dt'] <= d)
            time_delta_aux = (df_blotter.loc[live_trades, 'maturity'] - d)
            df_blotter.loc[live_trades, 'ndays'] = time_delta_aux.astype('timedelta64[D]')
            df_blotter['eval_date'] = d

            col_fx = df_blotter.loc[live_trades, 'fx']
            col_ndays = df_blotter.loc[live_trades, 'ndays']
            col_eval = df_blotter.loc[live_trades, 'eval_date']
            inputs = zip(col_fx, col_ndays, col_eval)
            df_blotter.loc[live_trades, 'price_mtm'] = [self.daily_fwds_XXXUSD.loc[(fx, d), ndays] for fx, ndays, d in
                                                        inputs]
            df_blotter.loc[live_trades, 'pnl'] = (df_blotter.loc[live_trades, 'price_mtm'] - df_blotter.loc[live_trades, 'traded_px']) * df_blotter.loc[live_trades, 'holding']
            trades_till_now = (df_blotter['trade_dt'] <= d)
            pivot_df = df_blotter.loc[trades_till_now].groupby(by=['k', 'eval_date']).sum().loc[(self.k_list, d), 'pnl'].unstack(0)
            pnl_accum = pivot_df.loc[d]
            # Write Pnl
            TR_df_daily_new.loc[d, self.k_list] = pnl_accum
            lst_moedas = df_blotter.loc[trades_till_now].groupby(by=['k', 'eval_date', 'fx']).sum()[
                'pnl'].unstack().columns
            lst_k = list(df_blotter.loc[trades_till_now].groupby(by=['k', 'eval_date', 'fx']).sum().index.get_level_values(0).unique())
            pnl_acum_fx = df_blotter.loc[trades_till_now].groupby(by=['k', 'eval_date', 'fx']).sum()['pnl'].unstack()
            TR_df_daily_pnl_byfx.loc[(lst_k, d), lst_moedas] = pnl_acum_fx
        return TR_df_daily_new, TR_df_daily_pnl_byfx

    def _run_NSiegel_fitting_one_curve(self, FX='AUD', date='2016-04-29', tenors_greater_than_n_years=0.0):
        # Run Nelson-Siegel fitting only for available dates
        # calendar_curves = list(self.interest_curve.index.get_level_values(1).unique())
        idx = pd.MultiIndex.from_product(iterables=[self.currency_list_curve, self.calendar_curves])
        df_betas = pd.DataFrame(data=np.NaN, index=self.calendar_curves, columns=['b1', 'b2', 'b3'])
        beta0 = np.ones(3)  # b1, b2 e b3

        d = date
        currency = FX
        df_curve_input = self._get_one_curve(currency, d, tenors_greater_than_n_years=tenors_greater_than_n_years)
        if df_curve_input is None:
            print(currency, ' ', d, 'ERROR IN THS CURVE')
        try:
            res = minimize(fun=self.NSiegel,
                           x0=beta0,
                           args=df_curve_input,
                           method='SLSQP')
            if res.success:
                df_betas.loc[d, 'b1'] = res.x[0]
                df_betas.loc[d, 'b2'] = res.x[1]
                df_betas.loc[d, 'b3'] = res.x[2]
        except:
            print(currency, ' ', d, 'ERROR IN THS CURVE')
        return df_betas.loc[pd.to_datetime(d)]


    def BZ_curve_data(self):
        try:
            df_bz_bbg_aux = pd.read_excel(self.data_folder + "\\BZ_bbg_aux.xlsx", index_col=0)
        except FileNotFoundError:
            # requesting BDS and BDP data
            member_ticker = list(self.con.bulkref('oda comdty', 'fut chain', ovrds=[("INCLUDE_EXPIRED_CONTRACTS", "Y")])['value'])
            aux_tickers_df = self.con.ref(member_ticker, 'FUT_DLV_DT_FIRST').set_index(keys='value', drop=False)
            aux_tickers_df.index = pd.to_datetime(aux_tickers_df.value)
            aux_tickers_df.value = pd.to_datetime(aux_tickers_df.value)
            # requesting BDH data
            df_bz_bbg = self.con.bdh(member_ticker, self.bbg_field_last, self.ini_date_bbg, self.end_date_bbg)
            # excel backup
            df_bz_bbg.to_excel(self.data_folder + "\\BZ_bbg.xlsx")
            df_bz_bbg.columns = df_bz_bbg.columns.droplevel(level='field')
            ticker_expire_dict = dict(zip(aux_tickers_df['ticker'].values, aux_tickers_df['ticker'].index))
            df_bz_bbg.columns = [ticker_expire_dict[x] for x in df_bz_bbg.columns]
            df_bz_bbg_aux = df_bz_bbg.reindex(sorted(df_bz_bbg.columns), axis=1)
            # excel backup
            df_bz_bbg_aux.to_excel(self.data_folder + "\\BZ_bbg_aux.xlsx")

        # cleaning previous data after getting BBG data
        self.interest_curve.loc['BRL'] = np.NaN
        # list_of_all_tenors = [self.tenor_dict[x] * 12 for x in self.tenor_dict]
        list_of_all_tenors = self.interest_curve.columns
        list_of_all_tenors_aux = pd.Series(data=list_of_all_tenors, index=list_of_all_tenors)
        # calendar_list = self.daily_calendar[self.daily_calendar > '1996-01-02']
        calendar_list = df_bz_bbg_aux.index[df_bz_bbg_aux.index > '1996-01-02']
        for dt in tqdm(calendar_list):
            curve_series_aux = df_bz_bbg_aux.loc[dt].dropna()
            time_deta_aux = (curve_series_aux.index - dt)
            curve_series_aux.index = time_deta_aux.astype('timedelta64[D]') / 365 * 12
            max_tenor_origin = curve_series_aux.index.max()
            min_tenor_origin = curve_series_aux.index.min()
            max_tenor_destination = list_of_all_tenors_aux.loc[list_of_all_tenors_aux >= max_tenor_origin].min()
            tenors_to_fill = list_of_all_tenors_aux[((list_of_all_tenors_aux > 1/30) & (list_of_all_tenors_aux <= max_tenor_destination))].index
            rates_to_fill = pd.Series(data=None, index=tenors_to_fill)
            rates_to_fill[rates_to_fill.index < min_tenor_origin] = curve_series_aux.iloc[0]
            rates_to_fill_aux = pd.DataFrame(rates_to_fill.append(curve_series_aux), columns=['rate'])
            rates_to_fill_aux = rates_to_fill_aux.loc[~rates_to_fill_aux.index.duplicated(keep='last')].rate
            rates_to_fill_aux.sort_index(axis=0, inplace=True)
            rates_to_fill_aux.interpolate(method='index', axis=0, inplace=True)
            rates_to_fill = rates_to_fill_aux.loc[rates_to_fill.index]
            # Write values in self.interest_curve (adjusting only BRL curves)
            for tenor_month in rates_to_fill.index:
                self.interest_curve.loc[('BRL', dt), tenor_month] = rates_to_fill.loc[tenor_month]

    def DEM_curve_data(self):
        try:
            df_bz_bbg_aux = pd.read_excel(self.data_folder + "\\DEM_bbg_aux.xlsx", index_col=0)
        except FileNotFoundError:
            # requesting BDS and BDP data
            member_ticker = ['FD0003M Index','GTDEM2Y Govt','GTDEM3Y Govt','GTDEM4Y Govt','GTDEM5Y Govt','GTDEM7Y Govt','GTDEM10Y Govt','GTDEM15Y Govt','GTDEM20Y Govt','GTDEM30Y Govt']
            # aux_tickers_df = self.con.ref(member_ticker, 'FUT_DLV_DT_FIRST').set_index(keys='value', drop=False)
            # aux_tickers_df.index = pd.to_datetime(aux_tickers_df.value)
            # aux_tickers_df.value = pd.to_datetime(aux_tickers_df.value)
            ticker_expire_dict = {'FD0003M Index': 3.0,
                                  'GTDEM2Y Govt': 24.0,
                                  'GTDEM3Y Govt': 36.0,
                                  'GTDEM4Y Govt': 48.0,
                                  'GTDEM5Y Govt': 60.0,
                                  'GTDEM7Y Govt': 84.0,
                                  'GTDEM10Y Govt': 120.0,
                                  'GTDEM15Y Govt': 180.0,
                                  'GTDEM20Y Govt': 240.0,
                                  'GTDEM30Y Govt': 360.0}
            # requesting BDH data
            df_bz_bbg = self.con.bdh(member_ticker, self.bbg_field_last, self.ini_date_bbg, self.end_date_bbg)
            # excel backup
            df_bz_bbg.to_excel(self.data_folder + "\\DEM_bbg.xlsx")
            df_bz_bbg.columns = df_bz_bbg.columns.droplevel(level='field')
            # ticker_expire_dict = dict(zip(aux_tickers_df['ticker'].values, aux_tickers_df['ticker'].index))
            df_bz_bbg.columns = [ticker_expire_dict[x] for x in df_bz_bbg.columns]
            df_bz_bbg_aux = df_bz_bbg.reindex(sorted(df_bz_bbg.columns), axis=1)
            # excel backup
            df_bz_bbg_aux.to_excel(self.data_folder + "\\DEM_bbg_aux.xlsx")

        # cleaning previous data after getting BBG data
        self.interest_curve.loc['DEM'] = np.NaN
        # list_of_all_tenors = [self.tenor_dict[x] * 12 for x in self.tenor_dict]
        list_of_all_tenors = self.interest_curve.columns
        list_of_all_tenors_aux = pd.Series(data=list_of_all_tenors, index=list_of_all_tenors)
        # calendar_list = self.daily_calendar[self.daily_calendar > '1996-01-02']
        calendar_list = df_bz_bbg_aux.index[df_bz_bbg_aux.index < '1998-12-01']
        for dt in tqdm(calendar_list):
            curve_series_aux = df_bz_bbg_aux.loc[dt].dropna()
            # time_deta_aux = (curve_series_aux.index - dt)
            # curve_series_aux.index = time_deta_aux.astype('timedelta64[D]') / 365 * 12
            max_tenor_origin = curve_series_aux.index.max()
            min_tenor_origin = curve_series_aux.index.min()
            max_tenor_destination = list_of_all_tenors_aux.loc[list_of_all_tenors_aux >= max_tenor_origin].min()
            tenors_to_fill = list_of_all_tenors_aux[((list_of_all_tenors_aux > 1) & (list_of_all_tenors_aux <= max_tenor_destination))].index
            rates_to_fill = pd.Series(data=None, index=tenors_to_fill)
            rates_to_fill[rates_to_fill.index < min_tenor_origin] = curve_series_aux.iloc[0]
            rates_to_fill_aux = pd.DataFrame(rates_to_fill.append(curve_series_aux), columns=['rate'])
            rates_to_fill_aux = rates_to_fill_aux.loc[~rates_to_fill_aux.index.duplicated(keep='last')].rate
            rates_to_fill_aux.sort_index(axis=0, inplace=True)
            rates_to_fill_aux.interpolate(method='index', axis=0, inplace=True)
            rates_to_fill = rates_to_fill_aux.loc[rates_to_fill.index]
            # Write values in self.interest_curve (adjusting only BRL curves)
            for tenor_month in rates_to_fill.index:
                self.interest_curve.loc[('DEM', dt), tenor_month] = rates_to_fill.loc[tenor_month]


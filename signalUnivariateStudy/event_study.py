# still to add:
# rescale_at_index


def adding_leading_lagging_returns(data_df,
                                   factor_to_analyze='returns',
                                   sec_id='SecurityID',
                                   n_pos=12,
                                   n_neg=12,
                                   index_name='r2k'):
    """

    Parameters
    ----------
    data_df
    factor_to_analyze
    n_pos
    n_neg
    index_name

    Returns
    -------

    """
    _data_df = copy.deepcopy(data_df)

    list_lags = []

    for i in np.arange(-(n_neg + 1), 0):  # create neg lags
        list_lags.append(i)
        print(i)
        _data_df[f'{i}'] = _data_df.groupby(sec_id)[factor_to_analyze].shift(-i)

    list_lags.append(0)

    _data_df['0'] = _data_df[factor_to_analyze]

    print(0)

    for i in np.arange(1, n_pos + 1):  # create pos lags
        list_lags.append(i)
        print(i)
        _data_df[f'{i}'] = _data_df.groupby(sec_id)[factor_to_analyze].shift(-i)

    _trim_data_df = _data_df.iloc[:, -(n_pos + n_neg + 2):]
    _trim_data_df['index_name'] = index_name

    return _trim_data_df


def event_study(data_df,
                date_col='Date',
                sec_id='SecurityID',
                event_column,  # must be categorical
                factor_to_analyze='returns',
                cumulative=True,
                n_pos=12,
                n_neg=12,
                verbose=False,
                rescale_at_num=0,  # rescale at 0 at time 0
                ):
    """

    Parameters
    ----------
    data_df - df of [date, secid, factor1, factor2...]
    sec_id - str
    event_column - str - what column defines the event? must be categorical
    factor_to_analyze - str - default returns
    cumulative - boolean - if using returns, makes sense to use cumulative
    n_pos - num pos lags
    n_neg - num neg lags
    verbose
    rescale_at_num - rescale to be 0 at time 0

    Returns
    -------
    3 dfs, [_raw_wealth, _ret_rets_only, _meta_data]

    """

    _data_df = copy.deepcopy(data_df)

    list_lags = []

    for i in np.arange(-(n_neg + 1), 0):  # create neg lags
        list_lags.append(i)
        print(i)
        _data_df[f'{i}'] = _data_df.groupby(sec_id)[factor_to_analyze].shift(-i)

    list_lags.append(0)

    _data_df['0'] = _data_df[factor_to_analyze]

    print(0)

    for i in np.arange(1, n_pos + 1):  # create pos lags
        list_lags.append(i)
        print(i)
        _data_df[f'{i}'] = _data_df.groupby(sec_id)[factor_to_analyze].shift(-i)

    _data_df.set_index([date_col, sec_id], inplace=True)

    all_dts = _data_df.index.get_level_values(date_col).unique()

    # exclude from pos and neg side
    exclude_pos = all_dts[-n_pos:]
    exclude_neg = all_dts[:n_neg]

    raw_rets = _data_df[~_data_df.index.get_level_values(date_col).isin(exclude_pos)]
    raw_rets = raw_rets[~raw_rets.index.get_level_values(date_col).isin(exclude_neg)]

    print("cutting down dates to the following:")
    print(raw_rets.index.get_level_values(date_col).unique())

    # now filter only on the event dates

    raw_rets_events_only = raw_rets[raw_rets[event_column].notnull()]
    print(f"Filtered down to {raw_rets_events_only.shape[0]} events")

    # filter only on lead lag returns
    split_index = (n_neg + n_pos + 2)

    _raw_rets_only = raw_rets_events_only.iloc[:, -split_index:]
    _meta_data = raw_rets_events_only.iloc[:, -split_index:]

    print(f"raw_data_cols = {_raw_rets_only.columns}")
    print(f"meta_data_cols = {_meta_data.columns}")

    if cumulative:
        # fill in missing with zeros
        _raw_rets_only[_raw_rets_only.isnull()] = 0

        # set first column to 0
        print("cumulating returns")
        _raw_rets_only.iloc[:, 0] = 0
        _raw_wealth = np.cumprod(1 + _raw_rets_only.T).T
    else:
        _raw_wealth = _raw_rets_only

    if rescale_at_num is not None:
        print(f"rescaling wealth at {rescale_at_num}")
        try:
            _raw_wealth = rescale_at_index(_raw_wealth,
                                           rescale_at_num=str(rescale_at_num))
        except Exception as e:
            print(f"Error trying to call rescale_at_index {e}")
            import pdb;
            pdb.set_trace()

    if verbose:
        return _raw_wealth, _raw_rets_only, _meta_data, raw_rets
    else:
        return _raw_wealth, _raw_rets_only, _meta_data


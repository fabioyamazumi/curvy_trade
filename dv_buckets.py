
    @staticmethod
    def dv_buckets(buckets, duration, dv) -> pd.DataFrame:
        assert len(duration) == len(dv), 'Duration and dv should have same lenght. Check inputs.'
        _buckets = pd.Series(buckets)
        _duration = pd.Series(duration)
        _dv = pd.Series(dv)
        # _dv_df = pd.DataFrame(data=None, columns=['dur', 'dv'])
        result = pd.DataFrame(data=None, index=list(_buckets), columns=['dv'])
        _dv_lst = []
        for dur, dv in zip(_duration, _dv):
            dur_before = _buckets[_buckets < dur].max()
            dur_after = _buckets[_buckets >= dur].min()
            pct2 = (dur - dur_before) / (dur_after - dur_before)
            pct1 = 1.0 - pct2
            # Append data
            if np.isnan(dur_before):
                # puts all DV on the following bucket
                _dv_lst.append([dur_after, dv * 1.0])
            elif np.isnan(dur_after):
                # puts all DV on the previous bucket
                _dv_lst.append([dur_before, dv * 1.0])
            else:
                # break DVs by default
                _dv_lst.append([dur_before, dv * pct1])
                _dv_lst.append([dur_after, dv * pct2])
        _dv_df = pd.DataFrame(data=_dv_lst, columns=['dur', 'dv'])
        _dv_df = _dv_df.groupby(by='dur').sum()
        result.loc[_dv_df.index, 'dv'] = _dv_df['dv']
        return result
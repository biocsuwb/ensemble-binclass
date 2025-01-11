import pandas as pd

from gprofiler import GProfiler


def get_profile(return_dataframe: bool = True, organism: str = 'hsapiens', query: list | pd.Series = None):
    gp = GProfiler(
        return_dataframe=return_dataframe,
    )

    if isinstance(query, pd.Series):
        query = query.tolist()

    return gp.profile(
        organism=organism,
        query=query,
    )

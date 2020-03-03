def no_preprocess(data, **kwargs):
    return data

def zero_centering(data, **kwargs):
    data = data - data.mean(axis=0)
    return data

def all_but_the_top(data, n_pca=0, **kwargs):
    return data

def get_preprocessor(type_):
    return {
        None: no_preprocess,
        'none': no_preprocess,
        'zero': zero_centering,
        'all-but-the-top': all_but_the_top,
    }[type_.lower() if isinstance(type_, str) else None]

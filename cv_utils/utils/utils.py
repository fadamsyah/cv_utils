def boolean_string(s):
    # Ngambil dari zyolo efficientdet
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'
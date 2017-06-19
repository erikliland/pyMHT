import matplotlib
backend = matplotlib.get_backend()
if backend not in ['Agg', 'WebAgg', 'MacOSX']:
    matplotlib.use('Agg')

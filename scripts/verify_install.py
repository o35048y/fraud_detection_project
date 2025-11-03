import importlib
libs = [('numpy','numpy'),('pandas','pandas'),('scipy','scipy'),('scikit-learn','sklearn'),('tensorflow','tensorflow'),('keras','keras'),('torch','torch'),('torchvision','torchvision'),('torchaudio','torchaudio'),('matplotlib','matplotlib'),('seaborn','seaborn'),('imbalanced-learn','imblearn'),('notebook','notebook'),('ipython','IPython')]
for name, pkg in libs:
    try:
        m = importlib.import_module(pkg)
        ver = getattr(m, '__version__', getattr(getattr(m, 'matplotlib', None), '__version__', 'unknown'))
        print(f'\u2713 {name} version:', ver)
    except Exception as e:
        print(f'\u2717 {name} import failed:', e)

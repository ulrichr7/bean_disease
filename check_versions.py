import importlib.metadata as md
for name in ['tensorflow', 'tensorflow-cpu', 'tensorflow-intel', 'protobuf', 'ml-dtypes']:
    try:
        print(name, md.version(name))
    except Exception:
        print(name, 'NOT INSTALLED')

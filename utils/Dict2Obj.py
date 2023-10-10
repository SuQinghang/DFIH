class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


# Convert 'dict' to 'obj' for using . to access attributes
def Dict2Obj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = Dict()
    for k, v in dictObj.items():
        d[k] = Dict2Obj(v)
    return d

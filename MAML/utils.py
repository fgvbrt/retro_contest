import base64
import pickle
from copy import  deepcopy


def merge_dictionaries(a, b, path_to_root=None, extend_lists=False):
    """
    создает копию словаря `a` и рекурсивно апдейтит ее элементы элементами из `b`
    :param extend_lists:
        if True и в обоих словарях это листы (если в обоих такой элемент есть) то элементы из b добавляются в конец
            к элементам из a, если в одном из словарей это не лист, то бросатеся ValueError
        if False - значения типа list трактуются как обычные значения - заменяют/перетирают друг друга
    """
    res = deepcopy(a)

    if path_to_root is None:
        path_to_root = []

    for key in b:
        if key not in res:
            res[key] = b[key]
            continue
        if isinstance(res[key], dict):
            if isinstance(b[key], dict):
                res[key] = merge_dictionaries(res[key], b[key], path_to_root + [str(key)], extend_lists=extend_lists)
            else:
                raise TypeError('Conflict at {}'.format('.'.join(path_to_root + [str(key)])))
        elif extend_lists and isinstance(res[key], list):
            if isinstance(b[key], list):
                res[key].extend(b[key])
            else:
                raise ValueError(
                    "Cannot extend list with not list. Path: {}".format('.'.join(path_to_root + [str(key)])))
        else:
            if extend_lists and isinstance(b[key], list):
                raise ValueError(
                    "Cannot extend non list with list. Path: {}".format('.'.join(path_to_root + [str(key)])))
            elif not isinstance(b[key], dict):
                res[key] = b[key]
            else:
                raise TypeError('Conflict at {}'.format('.'.join(path_to_root + [str(key)])))
    return res


def conv_out_dim(in_n, k, p, s):
    """
    :param in_n: input dim
    :param k: kernel size
    :param p: padding size
    :param s: stride size
    :return: output dim
    """
    return int((in_n + 2*p - k) / s + 1)


def convs_out_dim(in_n, ks, ps, ss):
    assert len(ks) == len(ps) == len(ss)
    for k, p, s in zip(ks, ps, ss):
        in_n = conv_out_dim(in_n, k, p, s)
    return in_n


def unpickle(data_dict):
    assert isinstance(data_dict, dict) and 'data' in data_dict and 'encoding' in data_dict

    data = data_dict["data"]
    encoding = data_dict["encoding"]

    if encoding == "base64":
        res = pickle.loads(base64.b64decode(data))
    else:
        raise ValueError('unsopported encoding {}'.format(encoding))

    return res
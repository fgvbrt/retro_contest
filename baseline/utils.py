from copy import deepcopy
import yaml


def load_config(fnames):

    config = {}
    for fname in fnames:
        with open(fname) as f:
            config = merge_dictionaries(config, yaml.load(f))

    return config


def add_boolean_flag(parser, name, default=False, help=None):
    """Add a boolean flag to argparse parser.
    Parameters
    ----------
    parser: argparse.Parser
        parser to add the flag to
    name: str
        --<name> will enable the flag, while --no-<name> will disable it
    default: bool or None
        default value of the flag
    help: str
        help string for the flag
    """
    dest = name.replace('-', '_')
    parser.add_argument("--" + name, action="store_true", default=default, dest=dest, help=help)
    parser.add_argument("--no-" + name, action="store_false", dest=dest)


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

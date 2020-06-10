from pkg_resources import get_distribution

__all__ = [
    'algorithm', 'error', 'fitness', 'imputation', 'interface', 'operators',
    'population', 'utils', 'visualization', 'wrapper', 'dataset'
]

__version__ = get_distribution('pywinEA').version
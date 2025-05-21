# -*- coding: utf-8 -*-
"""
Unit testing for utils module.
"""


def test_import_cli():
    from predoc import (
        clean_raw_functions,
        clean_raw_wrapper,
        datasets,
        dictionaries_columns,
        disease_lists_and_groupings,
        dummies_model_funcs,
        generate_dummies,
        model_performance,
        prg_home,
        script_models,
        utils,
    )

    assert True

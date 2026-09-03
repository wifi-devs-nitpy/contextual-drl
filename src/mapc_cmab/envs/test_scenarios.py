import os
os.environ['JAX_ENABLE_X64'] = 'True'

from mapc_cmab.envs.dynamic_scenario import DynamicScenario
from mapc_cmab.envs.scenario_impl import *


SMALL_OFFICE_SCENARIOS = [
    small_office_scenario(d_ap=10.0, d_sta=2.0, n_steps=600),
    DynamicScenario.from_static_scenarios(
        small_office_scenario(d_ap=20.0, d_sta=2.0, n_steps=500),
        small_office_scenario(d_ap=20.0, d_sta=3.0, n_steps=500),
        switch_steps=[500], n_steps=1000
    ),
    DynamicScenario.from_static_scenarios(
        small_office_scenario(d_ap=30.0, d_sta=2.0, n_steps=1500),
        small_office_scenario(d_ap=30.0, d_sta=4.0, n_steps=1500),
        switch_steps=[1500], n_steps=3000
    )
]

EXEMPLARY_SCENARIOS = [
    residential_scenario(seed=100, n_steps=600, x_apartments=2, y_apartments=2, n_sta_per_ap=4, size=10.0),
    DynamicScenario.from_static_scenarios(
        residential_scenario(seed=101, n_steps=600, x_apartments=2, y_apartments=2, n_sta_per_ap=4, size=20.0),
        residential_scenario(seed=102, n_steps=600, x_apartments=2, y_apartments=2, n_sta_per_ap=4, size=20.0),
        switch_steps=[600], n_steps=1200
    ),
    random_scenario(seed=3, d_ap=75., d_sta=4., n_ap=4, n_sta_per_ap=4, n_steps=1200)
]

INDOOR_SCENARIOS = [
    residential_scenario(seed=100, n_steps=1500, x_apartments=2, y_apartments=2, n_sta_per_ap=4, size=10.0),
    residential_scenario(seed=101, n_steps=1500, x_apartments=2, y_apartments=2, n_sta_per_ap=4, size=20.0),
    residential_scenario(seed=102, n_steps=1500, x_apartments=2, y_apartments=2, n_sta_per_ap=4, size=30.0),
    residential_scenario(seed=103, n_steps=3000, x_apartments=2, y_apartments=3, n_sta_per_ap=4, size=10.0),
    residential_scenario(seed=104, n_steps=3000, x_apartments=2, y_apartments=3, n_sta_per_ap=4, size=20.0),
    residential_scenario(seed=105, n_steps=3000, x_apartments=2, y_apartments=3, n_sta_per_ap=4, size=30.0),
    residential_scenario(seed=106, n_steps=20000, x_apartments=3, y_apartments=3, n_sta_per_ap=4, size=10.0),
    residential_scenario(seed=107, n_steps=20000, x_apartments=3, y_apartments=3, n_sta_per_ap=4, size=20.0),
    residential_scenario(seed=108, n_steps=20000, x_apartments=3, y_apartments=3, n_sta_per_ap=4, size=30.0),
    residential_scenario(seed=109, n_steps=200000, x_apartments=3, y_apartments=4, n_sta_per_ap=4, size=10.0),
    residential_scenario(seed=110, n_steps=200000, x_apartments=3, y_apartments=4, n_sta_per_ap=4, size=20.0),
    residential_scenario(seed=111, n_steps=200000, x_apartments=3, y_apartments=4, n_sta_per_ap=4, size=30.0),
    residential_scenario(seed=112, n_steps=2000, x_apartments=4, y_apartments=4, n_sta_per_ap=4, size=10.0),
    residential_scenario(seed=113, n_steps=2000, x_apartments=4, y_apartments=4, n_sta_per_ap=4, size=20.0),
    residential_scenario(seed=114, n_steps=2000, x_apartments=4, y_apartments=4, n_sta_per_ap=4, size=30.0)
]

FREE_SPACE_SCENARIOS = [
    random_scenario(seed=100, d_ap=75., d_sta=8., n_ap=2, n_sta_per_ap=5, n_steps=1000),
    random_scenario(seed=101, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=3, n_steps=1000),
    random_scenario(seed=102, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=4, n_steps=1000),
    random_scenario(seed=103, d_ap=75., d_sta=5., n_ap=4, n_sta_per_ap=3, n_steps=2000),
    random_scenario(seed=104, d_ap=75., d_sta=4., n_ap=4, n_sta_per_ap=4, n_steps=1000),
    random_scenario(seed=105, d_ap=75., d_sta=4., n_ap=5, n_sta_per_ap=3, n_steps=3000),
    random_scenario(seed=106, d_ap=75., d_sta=8., n_ap=2, n_sta_per_ap=5, n_steps=1000),
    random_scenario(seed=107, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=3, n_steps=1000),
    random_scenario(seed=108, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=4, n_steps=1000),
    random_scenario(seed=109, d_ap=75., d_sta=5., n_ap=4, n_sta_per_ap=3, n_steps=2000),
    random_scenario(seed=110, d_ap=75., d_sta=4., n_ap=4, n_sta_per_ap=4, n_steps=1000),
    random_scenario(seed=111, d_ap=75., d_sta=4., n_ap=5, n_sta_per_ap=3, n_steps=3000),
    random_scenario(seed=112, d_ap=75., d_sta=8., n_ap=2, n_sta_per_ap=5, n_steps=1000),
    random_scenario(seed=113, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=3, n_steps=1000),
    random_scenario(seed=114, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=4, n_steps=1000),
    random_scenario(seed=115, d_ap=75., d_sta=5., n_ap=4, n_sta_per_ap=3, n_steps=2000),
    random_scenario(seed=116, d_ap=75., d_sta=4., n_ap=4, n_sta_per_ap=4, n_steps=1000),
    random_scenario(seed=117, d_ap=75., d_sta=4., n_ap=5, n_sta_per_ap=3, n_steps=3000),
    random_scenario(seed=118, d_ap=75., d_sta=8., n_ap=2, n_sta_per_ap=5, n_steps=1000),
    random_scenario(seed=119, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=3, n_steps=1000),
    random_scenario(seed=120, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=4, n_steps=1000),
    random_scenario(seed=121, d_ap=75., d_sta=5., n_ap=4, n_sta_per_ap=3, n_steps=2000),
    random_scenario(seed=122, d_ap=75., d_sta=4., n_ap=4, n_sta_per_ap=4, n_steps=2000),
    random_scenario(seed=123, d_ap=75., d_sta=4., n_ap=5, n_sta_per_ap=3, n_steps=3000),
]

CLUSTERED_SCENARIOS = [
    residential_scenario(seed=110, n_steps=200000, x_apartments=3, y_apartments=4, n_sta_per_ap=4, size=20.0),
    residential_scenario(seed=113, n_steps=20000, x_apartments=4, y_apartments=4, n_sta_per_ap=4, size=20.0),
    symm_residential_scenario(seed=42, n_steps=2000, x_apartments=1, y_apartments=4, n_sta_per_ap=4, size=30, d_sta=2),
    symm_residential_scenario(seed=42, n_steps=2000, x_apartments=2, y_apartments=4, n_sta_per_ap=4, size=30, d_sta=2),
    symm_residential_scenario(seed=42, n_steps=2000, x_apartments=3, y_apartments=4, n_sta_per_ap=4, size=30, d_sta=2),
    symm_residential_scenario(seed=42, n_steps=2000, x_apartments=4, y_apartments=4, n_sta_per_ap=4, size=30, d_sta=2)
]

ALL_SCENARIOS = EXEMPLARY_SCENARIOS + INDOOR_SCENARIOS + FREE_SPACE_SCENARIOS

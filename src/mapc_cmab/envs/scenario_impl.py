from itertools import product

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey, Scalar

from mapc_cmab.envs.scenario import Scenario
from mapc_cmab.envs.static_scenario import StaticScenario
from mapc_cmab.envs.dynamic_scenario import DynamicScenario


def toy_scenario_1(d: Scalar = 20., n_steps: int = 600, channel_width: int = None, **kwargs) -> StaticScenario:
    """
    STA 1     AP A     STA 2     STA 3     AP B     STA 4
    """

    pos = jnp.array([
        [0 * d, 0.],  # STA 1
        [1 * d, 0.],  # AP A
        [2 * d, 0.],  # STA 2
        [3 * d, 0.],  # STA 3
        [4 * d, 0.],  # AP B
        [5 * d, 0.]   # STA 4
    ])

    associations = {
        1: [0, 2],
        4: [3, 5]
    }

    return StaticScenario(pos, associations, n_steps, str_repr="toy_scenario_1", channel_width=channel_width, **kwargs)


def toy_scenario_2(d_ap: Scalar = 50., d_sta: Scalar = 2., n_steps: int = 600, channel_width: int = None, **kwargs) -> StaticScenario:
    """
    STA 16   STA 15                  STA 12   STA 11

         AP D                             AP C

    STA 13   STA 14                  STA 9    STA 10



    STA 4    STA 3                   STA 8    STA 7

         AP A                             AP B

    STA 1    STA 2                   STA 5    STA 6
    """

    ap_pos = [
        [0 * d_ap, 0 * d_ap],  # AP A
        [1 * d_ap, 0 * d_ap],  # AP B
        [1 * d_ap, 1 * d_ap],  # AP C
        [0 * d_ap, 1 * d_ap],  # AP D
    ]

    dx = jnp.array([-1, 1, 1, -1]) * d_sta / jnp.sqrt(2)
    dy = jnp.array([-1, -1, 1, 1]) * d_sta / jnp.sqrt(2)

    sta_pos = [[x + dx[i], y + dy[i]] for x, y in ap_pos for i in range(len(dx))]
    pos = jnp.array(ap_pos + sta_pos)

    associations = {
        0: [4, 5, 6, 7],
        1: [8, 9, 10, 11],
        2: [12, 13, 14, 15],
        3: [16, 17, 18, 19]
    }

    return StaticScenario(pos, associations, n_steps, str_repr="toy_scenario_2", channel_width=channel_width, **kwargs)


def small_office_scenario(d_ap: Scalar, d_sta: Scalar, n_steps: int = float('inf'), channel_width: int = None, **kwargs) -> StaticScenario:
    """
    STA 16   STA 15         |        STA 12   STA 11
                            |
         AP D               |              AP C
                            |
    STA 13   STA 14         |         STA 9    STA 10
                            |
    ------------------------+------------------------

    STA 4    STA 3                    STA 8    STA 7

         AP A                              AP B

    STA 1    STA 2                    STA 5    STA 6
    """

    str_repr = f"small_office_{d_ap}_{d_sta}"

    ap_pos = [
        [0 * d_ap, 0 * d_ap],  # AP A
        [1 * d_ap, 0 * d_ap],  # AP B
        [1 * d_ap, 1 * d_ap],  # AP C
        [0 * d_ap, 1 * d_ap],  # AP D
    ]

    dx = jnp.array([-1, 1, 1, -1]) * d_sta / jnp.sqrt(2)
    dy = jnp.array([-1, -1, 1, 1]) * d_sta / jnp.sqrt(2)

    sta_pos = [[x + dx[i], y + dy[i]] for x, y in ap_pos for i in range(len(dx))]
    pos = jnp.array(ap_pos + sta_pos)

    associations = {
        0: [4, 5, 6, 7],
        1: [8, 9, 10, 11],
        2: [12, 13, 14, 15],
        3: [16, 17, 18, 19]
    }

    aps = associations.keys()

    # Setup walls in between each BSS
    walls = jnp.zeros((20, 20))
    walls = walls.at[4:, 4:].set(True)
    for i in range(20):
        for j in range(20):

            # If both are APs
            if i in aps and j in aps:
                walls = walls.at[i, j].set(i != j)

            # If i is an AP
            elif i in aps:
                for ap_j in set(aps) - {i}:
                    for sta in associations[ap_j]:
                        walls = walls.at[i, sta].set(True)

            # If j is an AP
            elif j in aps:
                for ap_i in set(aps) - {j}:
                    for sta in associations[ap_i]:
                        walls = walls.at[sta, j].set(True)

            # If both are STAs
            else:
                for ap in aps:
                    if i in associations[ap] and j in associations[ap]:
                        walls = walls.at[i, j].set(False)

    # - Remove wall between AP A and AP B
    walls = walls.at[:2, :2].set(False)
    walls = walls.at[1, 4:8].set(False)
    walls = walls.at[4:8, 1].set(False)
    walls = walls.at[0, 8:12].set(False)
    walls = walls.at[8:12, 0].set(False)
    walls = walls.at[4:12, 4:12].set(False)

    # Walls positions
    walls_pos = jnp.array([
        [-d_ap / 2, d_ap / 2, d_ap + d_ap / 2, d_ap / 2],
        [d_ap / 2, d_ap / 2, d_ap / 2, d_ap + d_ap / 2],
    ])

    return StaticScenario(pos, associations, n_steps, walls=walls, walls_pos=walls_pos, str_repr=str_repr, channel_width=channel_width, **kwargs)


def openwifi_scenario(channel_width: int = None):
    class OpenWifiScenario(StaticScenario):
        def __call__(self, key, tx, tx_power):
            _, reward = super().__call__(key, tx, tx_power)
            return int(10 * reward), reward

    pos = jnp.asarray([
        [-47., 0.],  # AP1
        [0., 39.],   # AP2
        [34., 0.],   # AP3
        [-29., 0.],  # STA1
        [-1., 0.],   # STA2
        [0., 25.],   # STA3
        [0., 1.],    # STA4
        [19., 0.],   # STA5
        [1., 0.],    # STA6
    ])

    associations = {
        0: [3, 4],
        1: [5, 6],
        2: [7, 8],
    }

    return OpenWifiScenario(pos, associations, 500, str_repr="openwifi", channel_width=channel_width)


def random_scenario(
        seed: int,
        d_ap: float,
        n_ap: int,
        d_sta: float,
        n_sta_per_ap: int,
        n_steps: int = float('inf'),
        randomize: bool = True,
        channel_width: int = None,
        **kwargs
) -> Scenario:
    def _draw_positions(key: PRNGKey) -> Array:
        ap_key, key = jax.random.split(key)
        ap_pos = jax.random.uniform(ap_key, (n_ap, 2)) * d_ap
        sta_pos = []

        for pos in ap_pos:
            sta_key, key = jax.random.split(key)
            center = jnp.repeat(pos[None, :], n_sta_per_ap, axis=0)
            stations = center + jax.random.normal(sta_key, (n_sta_per_ap, 2)) * d_sta
            sta_pos += stations.tolist()

        pos = jnp.array(ap_pos.tolist() + sta_pos)
        return pos

    str_repr = f"random_{seed}_{d_ap}_{n_ap}_{d_sta}_{n_sta_per_ap}"

    associations = {i: list(range(n_ap + i * n_sta_per_ap, n_ap + (i + 1) * n_sta_per_ap)) for i in range(n_ap)}

    key_first, key_sec = jax.random.split(jax.random.PRNGKey(seed), 2)
    pos_first = _draw_positions(key_first)
    pos_sec = _draw_positions(key_sec)

    if randomize:
        return DynamicScenario(pos_first, associations, n_steps, pos_sec=pos_sec, switch_steps=[n_steps // 2], str_repr=str_repr, channel_width=channel_width)
    else:
        return StaticScenario(pos_first, associations, n_steps, str_repr=str_repr, channel_width=channel_width, **kwargs)


def residential_scenario(
        seed: int,
        x_apartments: int = 10,
        y_apartments: int = 2,
        n_sta_per_ap: int = 2,
        size: Scalar = 10,
        n_steps: int = float('inf'),
        channel_width: int = None,
        **kwargs
) -> StaticScenario:
    """
    Implementation of the Residential Scenario from S. Merlin et al. "TGax Simulation Scenarios", IEEE 802.11-14/0980r16

    The path loss model of this scenario requires:

    BREAKING_POINT = 5
    WALL_LOSS = 5

    to apply the required path loss model, set the `path_loss_fn` parameter of the `StaticScenario`
    to `path_loss_fn=mapc_sim_mlo.utils.residential_tgax_path_loss`

    Suggested parameter ranges:

    x_apartments: 2..10
    y_apartments: 2
    n_sta_per_ap: 1..10
    size: 5..10

    """

    key = jax.random.PRNGKey(seed)
    str_repr = f"residential_{seed}_{x_apartments}_{y_apartments}_{n_sta_per_ap}_{size}"
    associations, pos, walls_pos = {}, [], []
    rooms = {}

    for x, y in product(range(x_apartments), range(y_apartments)):
        ap, stas = len(pos), list(range(len(pos) + 1, len(pos) + n_sta_per_ap + 1))
        associations[ap] = stas
        rooms.update({node: (x, y) for node in stas + [ap]})

        walls_pos.append([x * size, y * size, (x + 1) * size, y * size])
        walls_pos.append([x * size, y * size, x * size, (y + 1) * size])

        pos_key, key = jax.random.split(key)
        pos += (jax.random.uniform(pos_key, (n_sta_per_ap + 1, 2)) * size + jnp.array([x * size, y * size])).tolist()

    walls_pos.append([x_apartments * size, 0, x_apartments * size, y_apartments * size])
    walls_pos.append([0, y_apartments * size, x_apartments * size, y_apartments * size])
    walls = jnp.zeros((len(pos), len(pos)))

    for i, j in product(rooms.keys(), repeat=2):
        xi, yi = rooms[i]
        xj, yj = rooms[j]

        walls = walls.at[i, j].set(jnp.abs(xi - xj) + jnp.abs(yi - yj))
        walls = walls.at[j, i].set(jnp.abs(xi - xj) + jnp.abs(yi - yj))

    return StaticScenario(
        jnp.array(pos),
        associations,
        n_steps=n_steps,
        walls=walls,
        walls_pos=jnp.array(walls_pos),
        str_repr=str_repr,
        channel_width=channel_width,
        **kwargs
    )


def distance_scenario(d: Scalar, n_steps: int = float('inf'), channel_width: int = None, **kwargs) -> StaticScenario:
    """
    There is a single AP with a single STA placed at distance `d`. 
    """

    return StaticScenario(jnp.array([[0., 0.], [d, 0.]]), {0: [1]}, n_steps, str_repr=f"distance_{d}", channel_width=channel_width)


def hidden_station_scenario(d: Scalar, n_steps: int = float('inf'), channel_width: int = None, **kwargs) -> StaticScenario:
    """
    There are two APs 2 distance units `d` apart. Both APs have a single
    station placed in between them in the same place.

    AP_A <--d--> STA_1, STA_2 <--d--> AP_B 
    """

    pos = jnp.array([
        [0., 0.],  # AP A
        [d, 0.],  # STA 1
        [d, 0.],  # STA 2
        [2 * d, 0.]  # AP B
    ])

    associations = {
        0: [1],
        3: [2]
    }

    return StaticScenario(pos, associations, n_steps, str_repr=f"hidden_station_{d}", channel_width=channel_width, **kwargs)


def flow_in_the_middle_scenario(d: Scalar, n_steps: int = float('inf'), channel_width: int = None, **kwargs) -> StaticScenario:
    """
    There are three APs placed in line spaced `d` units apart. Each AP is associated with a single STA,
    placed in the same place as the AP.

    AP_A, STA_1 <--d--> AP_B, STA_2 <--d--> AP_C, STA_3
    """

    pos = jnp.array([
        [0., 0.],     # AP A
        [0., 0.],     # STA 1
        [1 * d, 0.],  # AP B
        [1 * d, 0.],  # STA 2
        [2 * d, 0.],  # AP C
        [2 * d, 0.],  # STA 3
    ])

    associations = {
        0: [1],
        2: [3],
        4: [5]
    }

    return StaticScenario(pos, associations, n_steps, str_repr=f"flow_in_the_middle_{d}", channel_width=channel_width, **kwargs)


def dense_point_scenario(n_ap: int, n_associations: int, n_steps: int = float('inf'), channel_width: int = None, **kwargs) -> StaticScenario:
    """
    There is `n_ap` APs with `n_associations` STAs each. All of the devices are placed at the same point. 
    """

    pos = jnp.array([[0., 0.] for _ in range(n_ap * (n_associations + 1))])

    associations = {i: [n_ap + i * n_associations + j for j in range(n_associations)] for i in range(n_ap)}

    return StaticScenario(pos, associations, n_steps, str_repr=f"dense_point_{n_ap}_{n_associations}", channel_width=channel_width, **kwargs)


def spatial_reuse_scenario(d_ap: Scalar, d_sta: Scalar, n_steps: int = 600, channel_width: int = None, **kwargs) -> StaticScenario:
    """
    STA 1 <--d_sta--> AP A <--d_ap--> AP B <--d_sta--> STA 4
    """

    pos = jnp.array([
        [0., 0.],               # STA 1
        [d_sta, 0.],            # AP A
        [d_sta + d_ap, 0.],     # AP B
        [2 * d_sta + d_ap, 0.]  # STA 2
    ])

    associations = {
        1: [0],
        2: [3]
    }

    return StaticScenario(pos, associations, n_steps, str_repr="spatial_reuse_scenario", channel_width=channel_width, **kwargs)


def test_scenario(scale: float = 1.0, channel_width: int = None, **kwargs) -> StaticScenario:
    """

            STA 1    AP A    STA 2


    --------------------------------------



    AP B    STA 3            STA 4    AP C

    """

    pos = scale * jnp.array([
        [  0.,  1.],  # AP A
        [ -1., -1.],  # AP B
        [  1., -1.],  # AP C
        [-0.5,  1.],  # STA 1
        [ 0.5,  1.],  # STA 2
        [-0.5, -1.],  # STA 3
        [ 0.5, -1.],  # STA 4
    ])

    associations = {
        0: [3, 4],
        1: [5],
        2: [6]
    }

    walls_pos = scale * jnp.array([
        [-2.0, 0.0, 2.0, 0.0]
    ])

    return StaticScenario(pos, associations, walls_pos=walls_pos, channel_width=channel_width, **kwargs)


def enterprise_scenario(
        seed: int,
        x_offices: int = 4,
        y_offices: int = 2,
        x_cubicles: int = 8,
        y_cubicles: int = 8,
        n_sta_per_cubicle: int = 4,
        n_cubicle_per_ap: int = 16,
        n_ap_per_office: int = 4,
        size_office: Scalar = 20,
        size_cubicle: Scalar = 2,
        n_steps: int = float('inf'),
        channel_width: int = None
) -> StaticScenario:
    """
    Implementation of the Enterprise Scenario from S. Merlin et al. "TGax Simulation Scenarios", IEEE 802.11-14/0980r16

    As per Figure 5, we assume that APs 1-4 and 5-8 share a managing entity, hence the default x_offices, y_offices values

    The path loss model of this scenario requires:

    BREAKING_POINT = 10
    WALL_LOSS = 7

    Suggested parameter ranges:

    x_offices: 1..4
    y_offices: 1..2
    x_cubicles: 8
    y_cubicles: 8
    n_sta_per_cubicle: 1..4
    n_cubicle_per_ap: 16
    n_ap_per_office: 4
    size_office: 20
    size_cubicle: 2

    """

    # Additional variables describing the scenario
    inner_corridor_width = 1
    # outer_corridor_width = 0.5
    size_quad = 2 * size_cubicle

    key = jax.random.PRNGKey(seed)
    str_repr = f"enterprise_{seed}_{x_offices}_{y_offices}_{n_cubicle_per_ap}_{n_sta_per_cubicle}_{size_office}_{size_cubicle}"
    associations, pos, walls_pos = {}, [], []
    offices = {}

    for x, y in product(range(x_offices), range(y_offices)):
        aps = list(range(len(pos), len(pos) + n_ap_per_office))
        for ap in range(n_ap_per_office):
            associations[aps[ap]] = list(range(
                len(pos) + len(aps) + ap * n_cubicle_per_ap * n_sta_per_cubicle,
                len(pos) + len(aps) + ap * n_cubicle_per_ap * n_sta_per_cubicle + n_cubicle_per_ap * n_sta_per_cubicle
            ))
            offices.update({node: (x, y) for node in associations[aps[ap]] + [aps[ap]]})

        ap_pos = (
            jnp.array([[5, 5], [15, 5], [5, 15], [15, 15]]) + jnp.array([x * size_office, y * size_office])
        )

        walls_pos.append([x * size_office, y * size_office, (x + 1) * size_office, y * size_office])
        walls_pos.append([x * size_office, y * size_office, x * size_office, (y + 1) * size_office])

        pos += ap_pos.tolist()

        for ap in range(n_ap_per_office):
            x_ap, y_ap = ap_pos[ap]

            for x_q, y_q in [
                (x_ap - (size_quad + inner_corridor_width / 2), y_ap - (size_quad + inner_corridor_width / 2)),
                (x_ap - (size_quad + inner_corridor_width / 2), y_ap + (inner_corridor_width / 2)),
                (x_ap + (inner_corridor_width / 2), y_ap - (size_quad + inner_corridor_width / 2)),
                (x_ap + (inner_corridor_width / 2), y_ap + (inner_corridor_width / 2))
            ]:
                for x_c, y_c in [
                    (x_q, y_q),
                    (x_q + size_cubicle, y_q),
                    (x_q, y_q + size_cubicle),
                    (x_q + size_cubicle, y_q + size_cubicle)
                ]:
                    pos_key, key = jax.random.split(key)
                    pos += (jax.random.uniform(pos_key, (n_sta_per_cubicle, 2)) * size_cubicle + jnp.array([x_c, y_c])).tolist()

    walls_pos.append([x_offices * size_office, 0, x_offices * size_office, y_offices * size_office])
    walls_pos.append([0, y_offices * size_office, x_offices * size_office, y_offices * size_office])
    walls = jnp.zeros((len(pos), len(pos)))

    for i, j in product(offices.keys(), repeat=2):
        xi, yi = offices[i]
        xj, yj = offices[j]

        walls = walls.at[i, j].set(jnp.abs(xi - xj) + jnp.abs(yi - yj))
        walls = walls.at[j, i].set(jnp.abs(xi - xj) + jnp.abs(yi - yj))

    return StaticScenario(
        jnp.array(pos),
        associations,
        n_steps=n_steps,
        walls=walls,
        walls_pos=jnp.array(walls_pos),
        str_repr=str_repr,
        channel_width=channel_width
    )


def indoor_small_bsss_scenario(
        seed: int,
        grid_layers: int = 3,
        n_sta_per_ap: int = 30,
        frequency_reuse: int = 1,
        bss_radius: int = 10,
        n_steps: int = float('inf'),
        channel_width: int = None,
        **kwargs
) -> StaticScenario:
    """
    Implementation of the Indoor Small BSSs Scenario from S. Merlin et al. "TGax Simulation Scenarios", IEEE 802.11-14/0980r16

    The path loss model of this scenario requires:

    BREAKING_POINT = 10
    WALL_LOSS = 7

    Suggested parameter ranges:

    grid_layers: 3 or 5
    n_sta_per_ap: 5..30
    frequency_reuse: 1 or 3
    bss_radius: 10

    """

    def is_point_in_hexagon(x, y, hexagon):
        n = len(hexagon)
        inside = False
        p1x, p1y = hexagon[0]

        for i in range(n + 1):
            p2x, p2y = hexagon[i % n]

            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside

            p1x, p1y = p2x, p2y

        return inside

    inter_bss_distance = 2 * (bss_radius ** 2 - bss_radius ** 2 / 4) ** 0.5

    key = jax.random.PRNGKey(seed)
    str_repr = f"indoor_small_bsss_{seed}_{bss_radius}_{n_sta_per_ap}_{frequency_reuse}_{grid_layers}"
    associations, pos = {}, []

    # Calculate the number of rows and columns needed to fill the outer_grid_layer_hexcenter_radius
    outer_grid_layer_hexcenter_radius = (grid_layers - 1) * inter_bss_distance
    rows = int(outer_grid_layer_hexcenter_radius / (bss_radius * jnp.sqrt(3))) + 1
    cols = int(outer_grid_layer_hexcenter_radius / (bss_radius * 3 / 2)) + 1

    ap_pos = []  # List to store the centers of hexagons where the APs are

    for row in range(-rows, rows + 1):
        for col in range(-cols, cols + 1):
            x_offset = bss_radius * 3 / 2 * col
            y_offset = bss_radius * jnp.sqrt(3) * (row + 0.5 * (col % 2))
            if jnp.sqrt(x_offset ** 2 + y_offset ** 2) <= outer_grid_layer_hexcenter_radius:
                ap_pos.append((x_offset, y_offset))  # Add center to the list

    if frequency_reuse == 3:
        if grid_layers == 3:
            ap_pos = [ap_pos[i] for i in [1, 3, 7, 10, 13, 17, 18]]
        elif grid_layers == 5:
            ap_pos = [ap_pos[i] for i in [4, 6, 8, 11, 13, 15, 17, 28, 30, 32, 34, 36, 38, 40, 42, 44, 55, 57, 59]]
        else:
            print("grid_layer value not supported")

    pos += ap_pos

    aps = list(range(len(ap_pos)))

    # Create hexagon vertices
    theta = jnp.linspace(0, 2 * jnp.pi, 7)
    x_hex = jnp.cos(theta)
    y_hex = jnp.sin(theta)

    for ap in range(len(aps)):
        associations[ap] = list(range(len(pos), len(pos) + n_sta_per_ap))
        x_ap, y_ap = ap_pos[ap]
        sta_pos = []

        hexagon = list(zip(x_hex * bss_radius + x_ap, y_hex * bss_radius + y_ap))

        for _ in range(n_sta_per_ap):
            while True:
                pos_key, key = jax.random.split(key)
                x_sta = jax.random.uniform(pos_key, minval=x_ap - bss_radius, maxval=x_ap + bss_radius)

                pos_key, key = jax.random.split(key)
                y_sta = jax.random.uniform(pos_key, minval=y_ap - bss_radius * jnp.sqrt(3) / 2, maxval=y_ap + bss_radius * jnp.sqrt(3) / 2)

                if is_point_in_hexagon(x_sta, y_sta, hexagon):
                    sta_pos.append((x_sta, y_sta))
                    break

        pos += sta_pos

    # No walls in this scenario
    walls = jnp.zeros((len(pos), len(pos)))

    return StaticScenario(
        jnp.array(pos),
        associations,
        n_steps=n_steps,
        walls=walls,
        str_repr=str_repr,
        channel_width=channel_width,
        **kwargs
    )


def position_stations_in_circle(key, n_sta_per_ap, radius, ap_position):
    pos_key, key = jax.random.split(key)

    angles = jax.random.uniform(pos_key, (n_sta_per_ap,), minval=0, maxval=2 * jnp.pi)
    radii = jax.random.uniform(key, (n_sta_per_ap,), minval=0, maxval=radius)

    x_positions = ap_position[0] + radii * jnp.cos(angles)
    y_positions = ap_position[1] + radii * jnp.sin(angles)

    positions = jnp.stack((x_positions, y_positions), axis=-1)

    return positions.tolist()


def position_stations_on_ring(key, n_sta_per_ap, radius, ap_position):
    angles = jnp.linspace(0, 2 * jnp.pi, n_sta_per_ap, endpoint=False)

    x_positions = ap_position[0] + radius * jnp.cos(angles)
    y_positions = ap_position[1] + radius * jnp.sin(angles)

    positions = jnp.stack((x_positions, y_positions), axis=-1)

    return positions.tolist()


def symm_residential_scenario(
        seed: int,
        n_steps: int,
        x_apartments: int = 2,
        y_apartments: int = 2,
        n_sta_per_ap: int = 4,
        size: Scalar = 10,
        d_sta: Scalar = 5,
        sta_positioning: int = 0,
        channel_width: int = None,
        **kwargs
) -> StaticScenario:
    """
    Symmetrical version of the residential scenario. Set `sta_positioning` to 0 for ring placement
    and 1 for circle placement of STAs. The APs are placed in the center of each apartment, and the STAs
    are placed around the APs.

    The path loss model of this scenario requires:

    BREAKING_POINT = 5
    WALL_LOSS = 5

    to apply the required path loss model, set the `path_loss_fn` parameter of the `StaticScenario`
    to `path_loss_fn=mapc_sim_mlo.utils.residential_tgax_path_loss`

    """

    key = jax.random.PRNGKey(seed)
    str_repr = f"symm_residential_{seed}_{x_apartments}_{y_apartments}_{n_sta_per_ap}_{size}_{d_sta}"
    associations, pos, walls_pos = {}, [], []
    rooms = {}
    ap_pos = []

    for x, y in product(range(x_apartments), range(y_apartments)):
        ap, stas = len(pos), list(range(len(pos) + 1, len(pos) + n_sta_per_ap + 1))
        associations[ap] = stas
        rooms.update({node: (x, y) for node in stas + [ap]})

        walls_pos.append([x * size, y * size, (x + 1) * size, y * size])
        walls_pos.append([x * size, y * size, x * size, (y + 1) * size])
        ap_pos.append([(x + 0.5) * size, (y + 0.5) * size])

        pos_key, key = jax.random.split(key)
        pos.append(ap_pos[len(ap_pos)-1])

        if sta_positioning ==0:
            pos += position_stations_on_ring(key, n_sta_per_ap, d_sta, ap_pos[len(ap_pos) - 1])
        elif sta_positioning ==1:
            pos += position_stations_in_circle(key, n_sta_per_ap, d_sta, ap_pos[len(ap_pos) - 1])
        else:
            raise ValueError("sta_positioning should be either 0 or 1")

    walls_pos.append([x_apartments * size, 0, x_apartments * size, y_apartments * size])
    walls_pos.append([0, y_apartments * size, x_apartments * size, y_apartments * size])
    walls = jnp.zeros((len(pos), len(pos)))

    for i, j in product(rooms.keys(), repeat=2):
        xi, yi = rooms[i]
        xj, yj = rooms[j]

        walls = walls.at[i, j].set(jnp.abs(xi - xj) + jnp.abs(yi - yj))
        walls = walls.at[j, i].set(jnp.abs(xi - xj) + jnp.abs(yi - yj))


    return StaticScenario(
        jnp.array(pos), associations, n_steps,
        walls=walls,
        walls_pos=jnp.array(walls_pos),
        str_repr=str_repr,
        channel_width=channel_width,
        **kwargs
    )

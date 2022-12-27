from enum import IntEnum

import matplotlib.pyplot as plt
import numpy as np

NEED_VALID = False
STRICT_ADD = False


class SpaceTypes(IntEnum):
    WATER = 0
    LAND = 1
    ISLAND = 2
    JOIN = 3


def has_inner_island(island: list[tuple[int, int]]) -> bool:
    """Isolate island to own world, then see if any landlocked water islands within that, check if landlocked if they don't include a world boundary."""
    if len(island) < 2:
        return False
    island_world = add_island(island)
    water_islands = [
        get_island(
            island_world,
            r,
            c,
            land_spaces=[SpaceTypes.WATER],
            need_valid=False,
            strict_add=False,
        )
        for r in range(island_world.shape[0])
        for c in range(island_world.shape[1])
    ]

    for water_island in water_islands:
        if len(water_island) == 0:
            continue
        r_vals = [p_row for (p_row, _) in water_island]
        c_vals = [p_col for (_, p_col) in water_island]
        min_row, max_row = min(r_vals), max(r_vals)
        min_col, max_col = min(c_vals), max(c_vals)

        if (
            min_row > 0
            and max_row < island_world.shape[0] - 1
            and min_col > 0
            and max_col < island_world.shape[1] - 1
        ):
            return True
    return False


def add_island(
    island: list[tuple[int, int]],
    val: int = SpaceTypes.ISLAND,
    use_world: np.ndarray = None,
) -> np.ndarray:
    """Put island in world. If no world, then put island in its own world defined by max boudaries"""
    if use_world is None:
        # Define new_island as shifted so upperleft point is (0,0) get max rows cols by taking max point row/col - offset row/col
        r_off = min(p_row for (p_row, _) in island)
        c_off = min(p_col for (_, p_col) in island)
        new_island = [(p_row - r_off, p_col - c_off) for (p_row, p_col) in island]
        rmax = max(p_row for (p_row, _) in new_island)
        cmax = max(p_col for (_, p_col) in new_island)
        new_world = np.zeros((rmax + 1, cmax + 1))
    else:
        new_world = use_world.copy()
        new_island = [isl_p for isl_p in island]

    for point in new_island:
        new_world[point] = val
    return new_world


def get_island(
    use_world: np.ndarray,
    row: int,
    col: int,
    land_spaces: list[int] = [SpaceTypes.LAND, SpaceTypes.ISLAND],
    need_valid: bool = NEED_VALID,
    strict_add: bool = STRICT_ADD,
) -> list[tuple[int, int]]:
    """Construct island by searching UDLR neighbors for island, then keep searching neighbors of those points until no neighbors left.
    If strict_add true, add all contiguous land no matter what which may invalidate whole island, if off add one neighbor at time and discard those that invalidate island."""

    if use_world[row, col] not in land_spaces:
        return []

    def _is_point(
        row: int, col: int, idxs: list[tuple[int, int]], use_world: np.ndarray
    ) -> bool:
        """If row and column exist on grid and spot has an island, return true, else false"""
        return (
            row >= 0
            and row < use_world.shape[0]
            and col >= 0
            and col < use_world.shape[1]
            and use_world[row, col] in land_spaces
            and (row, col) not in idxs
        )

    origin = (row, col)
    all_points = [origin]
    banned: list[tuple[int, int]] = []
    latest_points = [origin]
    while len(latest_points) > 0:
        # unique points not already ruled out
        latest_points = list(set(latest_points) - set(banned))
        latest_points = [
            (rp, cp)
            for (lp_row, lp_col) in latest_points
            for (rp, cp) in [
                (lp_row - 1, lp_col),  # up
                (lp_row + 1, lp_col),  # down
                (lp_row, lp_col - 1),  # left
                (lp_row, lp_col + 1),  # right
            ]
            if _is_point(rp, cp, all_points, use_world)
        ]
        if not need_valid:
            # just add points, who cares
            all_points = all_points + latest_points
        elif strict_add:
            # add all points as long as island stays valid
            tmp_points = all_points + latest_points
            if has_inner_island(tmp_points):
                # invalidates island
                return []
            all_points = tmp_points
        else:
            # add one point at time, add any bad points to list and don't add to island to invalidate it
            for latest_point in latest_points:
                tmp_points = [allp for allp in all_points] + [latest_point]
                if has_inner_island(tmp_points):
                    banned.append(latest_point)
                    if origin in banned:
                        return []
                else:
                    all_points = tmp_points
    return all_points


def max_island(
    world: np.ndarray, mod_point: bool = False
) -> tuple[int, list[tuple[int, int]], np.ndarray]:
    """Go through all indexes for biggest island."""
    biggest_island_size = 0
    best_island = []
    best_world = world.copy()
    for row in range(world.shape[0]):
        for col in range(world.shape[1]):
            mod_world = world.copy()
            if mod_point and mod_world[row, col] == SpaceTypes.WATER:
                mod_world[row, col] = SpaceTypes.ISLAND
            island = get_island(mod_world, row, col)
            if len(island) > biggest_island_size:
                biggest_island_size = len(island)
                best_island = island
                best_world = add_island(best_island, use_world=mod_world)
                if mod_point:
                    best_world[row, col] = SpaceTypes.JOIN
    return biggest_island_size, best_island, best_world


def main():
    # make world
    WORLD_SIZE_X = 100
    WORLD_SIZE_Y = 100

    world = np.random.choice(
        [SpaceTypes.WATER, SpaceTypes.LAND],
        p=[0.7, 0.3],
        size=(WORLD_SIZE_X, WORLD_SIZE_Y),
    )

    # test_world = np.array(
    #     [
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 1, 1, 1, 1, 1, 0],
    #         [0, 1, 0, 0, 0, 0, 0],
    #         [0, 1, 0, 1, 1, 1, 0],
    #         [0, 1, 0, 1, 0, 1, 0],
    #         [0, 1, 1, 1, 1, 1, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 1, 1, 1, 0],
    #         [0, 0, 0, 1, 1, 1, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #     ]
    # )

    # get current islands max size, and max size adding one island
    size, _, upd_world = max_island(world, mod_point=False)
    # get islands max size  adding one water to land
    new_size, _, new_world = max_island(world, mod_point=True)

    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title(f"Size {size}")
    im1 = ax1.imshow(upd_world.copy(), vmin=0, vmax=3, cmap="viridis", aspect="auto")
    ax2.imshow(new_world.copy(), vmin=0, vmax=3, cmap="viridis", aspect="auto")
    ax2.set_title(f"Size {new_size}")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cb = fig.colorbar(im1, ticks=[0, 1, 2, 3], cax=cbar_ax)
    cb.ax.set_yticklabels(["water", "land", "island", "join"])
    plt.show()


if __name__ == "__main__":
    np.random.seed(123456)
    for _ in range(10):
        main()

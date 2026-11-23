"""Write the radtrax values that radiomax's tests compare against.

radtrax imports radiomax, so it cannot be a test dependency; its values are
stored once, with the commit in the file's name. Run it in a checkout of
radtrax at that commit, from radiomax's root:

    uv run --project <radtrax> python tests/data/make_radtrax_values.py fadf2a6
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def values() -> dict[str, np.ndarray]:
    """Every stored array by name, float64, inputs next to outputs."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from radtrax.atmosphere import ViewingGeometry
    from radtrax.atmosphere.geometry import _shell_air_mass, air_mass_factor
    from radtrax.instrument import GaussianInstrument
    from radtrax.radiometry import (
        brightness_temperature,
        log_planck_radiance,
        planck_radiance,
    )
    from radtrax.radiometry.planck import to_per_ghz, to_per_um
    from radtrax.spectroscopy.solar import earth_sun_distance
    from radtrax.surface.ross_li import li_sparse_reciprocal, ross_thick

    out: dict[str, np.ndarray] = {}
    # --- air masses ---
    zenith = np.linspace(0.0, 89.0, 24)
    out["air_mass_solar_zenith"], out["air_mass_view_zenith"] = np.meshgrid(
        zenith, np.linspace(0.0, 70.0, 8), indexing="ij"
    )
    out["air_mass_factor"] = air_mass_factor(
        out["air_mass_solar_zenith"], out["air_mass_view_zenith"]
    )
    edges = np.asarray([0.0, 10.0, 500.0, 2e3, 1e4, 3e4, 1e5])
    out["shell_bottom"], out["shell_top"] = edges[:-1], edges[1:]
    out["shell_cosine"] = np.cos(np.deg2rad(np.append(zenith, 90.0)))
    out["shell_air_mass"] = np.stack(
        [
            _shell_air_mass(jnp.asarray(mu), edges[:-1], edges[1:], 6_371_008.771415059)
            for mu in out["shell_cosine"]
        ]
    )
    # --- Planck and its inverse ---
    out["planck_wavenumber_cm"] = 1e4 / np.geomspace(0.4, 15.0, 24)
    out["planck_temperature"] = np.geomspace(150.0, 2500.0, 12)
    nu = out["planck_wavenumber_cm"][:, None]
    t = out["planck_temperature"][None, :]
    out["planck_radiance"] = planck_radiance(nu, t)
    out["log_planck_radiance"] = log_planck_radiance(nu, t)
    over_t = jax.vmap(jax.grad(planck_radiance, argnums=1), (None, 0))
    out["planck_dt"] = jax.vmap(over_t, (0, None))(nu[:, 0], t[0])
    out["planck_per_um"] = to_per_um(out["planck_radiance"], nu)
    out["planck_per_ghz"] = to_per_ghz(out["planck_radiance"])
    out["brightness_temperature"] = brightness_temperature(nu, out["planck_radiance"])
    # --- Ross–Li kernels: random angles, the hotspot, nadir ---
    rng = np.random.default_rng(0)
    cosines = rng.uniform(0.1, 1.0, (2, 60))
    azimuth = rng.uniform(-np.pi, np.pi, 60)
    cosines[1, :10], azimuth[:10] = cosines[0, :10], 0.0
    cosines[:, 10:14] = 1.0
    out["kernel_cos_solar"], out["kernel_cos_view"] = cosines
    out["kernel_relative_azimuth"] = azimuth
    for name, kernel in (("ross", ross_thick), ("li", li_sparse_reciprocal)):

        def of_angles(mu0, mu, phi, kernel=kernel):
            return kernel(ViewingGeometry(mu0, mu, phi))

        out[f"kernel_{name}"] = jax.vmap(of_angles)(*cosines, azimuth)
        gradient = jax.vmap(jax.grad(of_angles, argnums=(0, 1, 2)))
        out[f"kernel_{name}_gradient"] = np.stack(gradient(*cosines, azimuth))
    # --- Gaussian responses ---
    out["response_wavenumber_cm"] = np.linspace(4250.0, 4350.0, 201)
    out["response_centre_nm"] = np.asarray([2310.0, 2325.0, 2340.0])
    out["response_fwhm_nm"] = np.asarray([6.0, 8.5, 12.0])
    instrument = GaussianInstrument(
        out["response_wavenumber_cm"],
        out["response_centre_nm"],
        out["response_fwhm_nm"],
    )
    out["response_matrix"] = instrument.response()
    # --- the Earth–Sun distance formula that radiomax does not use ---
    out["distance_day_of_year"] = np.linspace(1.0, 366.0, 74)
    out["distance"] = earth_sun_distance(out["distance_day_of_year"])
    return {name: np.asarray(array, dtype=np.float64) for name, array in out.items()}


def main(commit: str) -> None:
    """Write ``radtrax_<commit>.npz`` next to this file.

    Args:
        commit: The short hash of the radtrax commit this runs in.
    """
    from loguru import logger

    path = Path(__file__).parent / f"radtrax_{commit}.npz"
    stored = values()
    np.savez_compressed(path, **stored)
    logger.info("{}: {} arrays, {} bytes", path, len(stored), path.stat().st_size)


if __name__ == "__main__":
    import cyclopts

    cyclopts.run(main)

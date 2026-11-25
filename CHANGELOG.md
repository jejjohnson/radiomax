# Changelog

## [0.0.10](https://github.com/jejjohnson/radiomax/compare/v0.0.9...v0.0.10) (2026-11-25)


### Bug Fixes

* **deps:** pipekit-array 0.1.6 and geophysix 0.0.8, the tags of the stack ([#80](https://github.com/jejjohnson/radiomax/issues/80)) ([911712b](https://github.com/jejjohnson/radiomax/commit/911712bc72deba31dc33fb3c9aa8fe956fa2ad51))

## [0.0.9](https://github.com/jejjohnson/radiomax/compare/v0.0.8...v0.0.9) (2026-11-24)


### Bug Fixes

* **deps:** pipekit-array 0.1.4 and geophysix 0.0.7, the tags of the stack ([#77](https://github.com/jejjohnson/radiomax/issues/77)) ([f26843c](https://github.com/jejjohnson/radiomax/commit/f26843cf1c2806816fb6abbf749cdcd02112bc03))
* **xarray:** a DataArray's coordinates keep their attributes ([1baa444](https://github.com/jejjohnson/radiomax/commit/1baa444b2ac104e66ff5c5249b305efb7df13004)), closes [#64](https://github.com/jejjohnson/radiomax/issues/64)

## [0.0.8](https://github.com/jejjohnson/radiomax/compare/v0.0.7...v0.0.8) (2026-11-23)


### Features

* **altimetry:** tropospheric, ionospheric and inverse-barometer range corrections ([#74](https://github.com/jejjohnson/radiomax/issues/74)) ([07d1f68](https://github.com/jejjohnson/radiomax/commit/07d1f68d35eadea34ebb8ca14aa2942fa39bd2fc))
* **sar:** calibration to σ⁰, β⁰ and γ⁰ ([#72](https://github.com/jejjohnson/radiomax/issues/72)) ([7525a33](https://github.com/jejjohnson/radiomax/commit/7525a33a858732e9d662f06117b471bf5a65b716))

## [0.0.7](https://github.com/jejjohnson/radiomax/compare/v0.0.6...v0.0.7) (2026-11-23)


### Bug Fixes

* **geometry:** the functions that call geophysix coordinates stay on their device ([#67](https://github.com/jejjohnson/radiomax/issues/67)) ([a37db36](https://github.com/jejjohnson/radiomax/commit/a37db368e5f9050d7f076daf62ae5dc01bfdfae5))

## [0.0.6](https://github.com/jejjohnson/radiomax/compare/v0.0.5...v0.0.6) (2026-11-23)


### Features

* **geometry:** the on-Earth mask of a geostationary disk, alone ([#66](https://github.com/jejjohnson/radiomax/issues/66)) ([486ad55](https://github.com/jejjohnson/radiomax/commit/486ad554ec96b00438dcc6278b203bdacae06e83))


### Bug Fixes

* **labelled:** an output that ignores an input keeps its dimension ([#62](https://github.com/jejjohnson/radiomax/issues/62)) ([8e2137b](https://github.com/jejjohnson/radiomax/commit/8e2137b235de1181de006dacb79f55db47a3277a))

## [0.0.5](https://github.com/jejjohnson/radiomax/compare/v0.0.4...v0.0.5) (2026-11-23)


### Features

* **retrievals:** the Ross–Li kernels from cosines, for radtrax's geometry ([#60](https://github.com/jejjohnson/radiomax/issues/60)) ([965fe71](https://github.com/jejjohnson/radiomax/commit/965fe71e8291736b8d2b1198c4b03fb356d1f52a))


### Bug Fixes

* **backend:** host values go to the input's device, and torch tensors keep their gradients ([#59](https://github.com/jejjohnson/radiomax/issues/59)) ([ff34156](https://github.com/jejjohnson/radiomax/commit/ff34156bf127a62eb2603c418ce622bd89141bf9))

## [0.0.4](https://github.com/jejjohnson/radiomax/compare/v0.0.3...v0.0.4) (2026-11-23)


### Features

* **geometry:** geostationary scan angles to longitude and latitude, and back ([#52](https://github.com/jejjohnson/radiomax/issues/52)) ([468e15e](https://github.com/jejjohnson/radiomax/commit/468e15ef767ef9a07cabcdae21897c1aab9de74e))
* **geometry:** the apparent position of an elevated target, and the ray–ellipsoid intersection ([#50](https://github.com/jejjohnson/radiomax/issues/50)) ([c60b7ce](https://github.com/jejjohnson/radiomax/commit/c60b7ce82d567606ac3a7c95b8372de1e58ca8a7))
* **geometry:** whiskbroom scan angles, pixel growth and bowtie detector positions ([#53](https://github.com/jejjohnson/radiomax/issues/53)) ([1ed8a37](https://github.com/jejjohnson/radiomax/commit/1ed8a3737569c562e39aa15988d9a781ecd7afe2))
* **retrievals:** kernel BRDF reflectance and the nadir adjustment ([#55](https://github.com/jejjohnson/radiomax/issues/55)) ([f2d7ecd](https://github.com/jejjohnson/radiomax/commit/f2d7ecd99544755074d237dd56d8d3a4313e59f4))
* **retrievals:** ocean-colour chlorophyll from the maximum band ratio, with NASA's published coefficients ([#57](https://github.com/jejjohnson/radiomax/issues/57)) ([2498dca](https://github.com/jejjohnson/radiomax/commit/2498dcab81594a941e037ecfc83e59b5d42d8b19))
* **retrievals:** split-window sea- and land-surface temperature from given coefficients ([#56](https://github.com/jejjohnson/radiomax/issues/56)) ([71c5744](https://github.com/jejjohnson/radiomax/commit/71c57440f90dea11a9b3d7fda8f8650d66cf38c2))
* **retrievals:** the RossThick and LiSparse-Reciprocal kernels, moved from radtrax ([#54](https://github.com/jejjohnson/radiomax/issues/54)) ([bfcb1e2](https://github.com/jejjohnson/radiomax/commit/bfcb1e2a5712eedba93ec44aa67914626986d136))
* **thermal:** band brightness temperatures from published coefficients (GOES, Landsat, Himawari) ([#49](https://github.com/jejjohnson/radiomax/issues/49)) ([a15b5a3](https://github.com/jejjohnson/radiomax/commit/a15b5a3b62ab4051081902fa3add4935bfff7345))
* **thermal:** planck's law per wavenumber, moved from radtrax, and its views per µm and per GHz ([#46](https://github.com/jejjohnson/radiomax/issues/46)) ([fdeee0a](https://github.com/jejjohnson/radiomax/commit/fdeee0a1c888471fe665ead37e4ca9c8c5a7617a))
* **thermal:** the monochromatic brightness temperature, moved from radtrax ([#47](https://github.com/jejjohnson/radiomax/issues/47)) ([d3ed4bd](https://github.com/jejjohnson/radiomax/commit/d3ed4bd88058cce47f74c885bb7b32e13b3fb4eb))

## [0.0.3](https://github.com/jejjohnson/radiomax/compare/v0.0.2...v0.0.3) (2026-11-23)


### Features

* **geometry:** solar zenith and azimuth per pixel, and the Earth–Sun distance ([#28](https://github.com/jejjohnson/radiomax/issues/28)) ([9aefa63](https://github.com/jejjohnson/radiomax/commit/9aefa63f0cf8c886104dd1b6a1fb677ed1ad42e0)), closes [#18](https://github.com/jejjohnson/radiomax/issues/18)
* **geometry:** the plane-parallel and spherical-shell air masses, moved from radtrax ([#31](https://github.com/jejjohnson/radiomax/issues/31)) ([00670c2](https://github.com/jejjohnson/radiomax/commit/00670c2897488a803f95567680b34462302ff9cb)), closes [#20](https://github.com/jejjohnson/radiomax/issues/20)
* **geometry:** viewing angles from the satellite's position, relative azimuth and scattering angle ([#29](https://github.com/jejjohnson/radiomax/issues/29)) ([679bbe9](https://github.com/jejjohnson/radiomax/commit/679bbe99d0812ffe774c62d63bf4bfaa7fc9b49d)), closes [#19](https://github.com/jejjohnson/radiomax/issues/19)
* **radiometry:** band averages over a spectral response, the band solar irradiance and spectral densities ([#33](https://github.com/jejjohnson/radiomax/issues/33)) ([427220b](https://github.com/jejjohnson/radiomax/commit/427220bac132749e59fef8070c2a972a11309e70)), closes [#22](https://github.com/jejjohnson/radiomax/issues/22)
* **radiometry:** counts, radiance and top-of-atmosphere reflectance, and back ([#32](https://github.com/jejjohnson/radiomax/issues/32)) ([bfd3dee](https://github.com/jejjohnson/radiomax/commit/bfd3dee88a88c20e577e5e6ee11f4d56998d17df)), closes [#21](https://github.com/jejjohnson/radiomax/issues/21)
* **radiometry:** spectral response matrices, exact on any grid, with their adjoint and coverage ([#34](https://github.com/jejjohnson/radiomax/issues/34)) ([f8ffa75](https://github.com/jejjohnson/radiomax/commit/f8ffa7570dc3ba285cae7a70ac5111bb68bc3e19)), closes [#23](https://github.com/jejjohnson/radiomax/issues/23)

## [0.0.2](https://github.com/jejjohnson/radiomax/compare/v0.0.1...v0.0.2) (2026-11-23)


### Features

* **backend:** one namespace and dtype per call, and DataArrays keep their labels ([#25](https://github.com/jejjohnson/radiomax/issues/25)) ([7a27c71](https://github.com/jejjohnson/radiomax/commit/7a27c712ed08576921e06761b73e76ae3d979eda)), closes [#16](https://github.com/jejjohnson/radiomax/issues/16)

## Changelog
